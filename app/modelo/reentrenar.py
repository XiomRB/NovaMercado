"""Reentrenamiento orquestado por Airflow (etapa 7, DECISIONES.md §16).

Uso:  python -m app.modelo.reentrenar --ejecucion <id>

Lo ejecuta la tarea ``reentrenar`` del DAG ``novamercado_reentrenamiento`` a través del
lanzador y del arranque verificado (``docker/airflow``). Pasos:

1. **Procedencia** antes de importar el resto del proyecto (como en la etapa 6). Si
   falla, código 2 sin escribir nada (el cierre lo hace ``registrar_resultado``).
2. Lee **t0** (``tareas_reentrenamiento.iniciada_en``, escrito por ``marcar_inicio``)
   y calcula todos los plazos con el reloj de PostgreSQL: hijo detenido a t0+8:00,
   promoción solo si el hijo terminó antes de t0+7:15, reconciliación final omitida
   después de t0+8:05, resultado escrito a más tardar a t0+8:20.
3. ``ejecutar_entrenamiento(modo="reentrenamiento")``: concesión (espera 60 s),
   reclamación atómica (código 7 si ya había resultado, veredicto o reclamación),
   hijo con R2 y promoción ``si_mejora``.
4. Escribe el resultado con el rol ``nm_cierre`` (``app.db.cierre.cerrar``): códigos
   internos 0 y 3 → ``exitoso``; cualquier otro → ``fallido``. Nunca después de un
   veredicto de ausencia; ``promovido`` y ``version`` salen de PostgreSQL.
5. ``SIGTERM`` (``execution_timeout`` de Airflow a t0+8:30): detiene al hijo, libera la
   concesión y escribe ``fallido`` si aún queda tiempo antes de t0+8:43.

Código de salida: 0 si el resultado de negocio es ``exitoso`` (internos 0 y 3); si no,
el código interno (1, 2, 4, 5, 6, 7, 8) o 143 si llegó ``SIGTERM``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import time

# Solo módulos sin dependencias del resto del proyecto hasta verificar la procedencia
from app.modelo.codigos_salida import (
    ERROR,
    INDETERMINADA,
    OCUPADO,
    OK,
    PERDIDA,
    PROCEDENCIA,
    RECLAMADA,
    SIN_MODELO,
    TIEMPO,
)
from app.modelo.git_estado import RepositorioNoValido

# Desplazamientos desde t0 en segundos (plan v6–v12, §6)
PROMOCION_S = 7 * 60 + 15
HIJO_S = 8 * 60
RECONCILIACION_S = 8 * 60 + 5
RESULTADO_S = 8 * 60 + 20
RESULTADO_TRAS_SIGTERM_S = 8 * 60 + 43
EXITOSOS = (OK, SIN_MODELO)
ESCRITO_POR = "reentrenar"

logger = logging.getLogger("app.modelo.reentrenar")


class Terminado(BaseException):
    """SIGTERM recibido (Airflow detiene la tarea)."""


def _al_recibir_sigterm(_signo, _marco):
    raise Terminado()


def leer_t0(url_bd: str, id_: str) -> float | None:
    """t0 como instante de ``time.monotonic()``, calculado con el reloj de PostgreSQL."""
    from sqlalchemy import text

    from app.db.conexion import crear_motor
    motor = crear_motor(url_bd, timeout_consulta=10, timeout_conexion=5)
    try:
        with motor.connect() as c:
            d = c.execute(text("SELECT EXTRACT(EPOCH FROM iniciada_en - clock_timestamp()) "
                               "FROM tareas_reentrenamiento WHERE id = :i"), {"i": id_}).scalar()
            ahora = time.monotonic()
    finally:
        motor.dispose()
    return None if d is None else ahora + float(d)


def escribir(id_: str, estado: str, motivo: str, limite: float, nombre: str) -> str:
    from app.config import parametros_rol
    from app.db.cierre import CIERRE_ESCRITURA_S, CIERRE_TOTAL_S, cerrar
    total = min(CIERRE_TOTAL_S, limite - time.monotonic())
    if total < 1.0:
        logger.error("sin tiempo para escribir el resultado: lo hará registrar_resultado")
        return "sin_tiempo"
    try:
        params = parametros_rol("nm_cierre", "NM_CIERRE_PASSWORD")
    except Exception as exc:  # noqa: BLE001
        logger.error("no se puede escribir el resultado: %s", exc)
        return "sin_configuracion"
    r = cerrar(params, id_, estado=estado, motivo=motivo, escrito_por=ESCRITO_POR, nombre_modelo=nombre,
               total=total, escritura=min(CIERRE_ESCRITURA_S, total * 2 / 3))
    logger.info("resultado %s: %s (%s) %s", estado, r.codigo, r.estado_persistido, r.detalle)
    return r.codigo


def ejecutar(id_: str) -> int:
    from app.modelo.procedencia import verificar_procedencia
    try:
        proc = verificar_procedencia()                 # antes de leer la base o tocar MLflow
    except RepositorioNoValido as exc:
        print(str(exc), file=sys.stderr)
        return PROCEDENCIA
    os.chdir(proc.raiz)

    from app.config import mlflow_alias, mlflow_modelo_nombre, mlflow_tracking_uri, url_base_datos
    from app.db.esquema import ID_REENTRENAMIENTO_RE
    from app.modelo.bloqueo import BloqueoOcupado, ConcesionPerdida, ReclamacionRechazada
    from app.modelo.ciclo_entrenamiento import PlazoExcedido, PlazosReentrenamiento, ejecutar_entrenamiento
    from app.modelo.promocion import PromocionIndeterminada, ReconciliacionIncompleta
    if not re.fullmatch(ID_REENTRENAMIENTO_RE, id_):
        print("identificador de reentrenamiento inválido", file=sys.stderr)
        return ERROR
    nombre = mlflow_modelo_nombre()
    try:
        url = url_base_datos()
        t0 = leer_t0(url, id_)
    except Exception as exc:  # noqa: BLE001
        print(f"no se pudo leer el inicio de la tarea: {type(exc).__name__}", file=sys.stderr)
        return ERROR
    if t0 is None:
        print(f"la solicitud {id_} no tiene inicio registrado (marcar_inicio)", file=sys.stderr)
        return ERROR
    plazos = PlazosReentrenamiento(id_, t0 + HIJO_S, t0 + PROMOCION_S, t0 + RECONCILIACION_S)
    limite_resultado = t0 + RESULTADO_S
    logger.info("reentrenamiento %s: %.0f s desde t0", id_, time.monotonic() - t0)

    codigo, motivo = ERROR, ""
    try:
        try:
            r = ejecutar_entrenamiento(url_bd=url, tracking_uri=mlflow_tracking_uri(), nombre=nombre,
                                       alias=mlflow_alias(), modo="reentrenamiento", promover="si_mejora",
                                       git_commit=proc.commit, etiquetas=proc.etiquetas(), plazos=plazos)
            codigo, motivo = r.codigo, r.mensaje
            print(r.como_json(), flush=True)
        except ReclamacionRechazada as exc:
            print(f"no se entrena: {exc}", file=sys.stderr)
            return RECLAMADA                           # nunca escribe: otro ya decidió
        except BloqueoOcupado as exc:
            codigo, motivo = OCUPADO, f"otro entrenamiento en curso: {exc}"
        except ConcesionPerdida as exc:
            codigo, motivo = PERDIDA, f"se perdió la concesión de entrenamiento: {exc}"
        except PlazoExcedido as exc:
            codigo, motivo = TIEMPO, f"tiempo límite: {exc}"
        except PromocionIndeterminada as exc:
            codigo, motivo = INDETERMINADA, str(exc)
        except ReconciliacionIncompleta as exc:
            codigo = INDETERMINADA if exc.resultado == "indeterminada" else ERROR
            motivo = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("el reentrenamiento falló")
            codigo, motivo = ERROR, f"error: {type(exc).__name__}: {exc}"
    except Terminado:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        motivo = "tiempo límite: Airflow detuvo la tarea"
        print(motivo, file=sys.stderr)
        escribir(id_, "fallido", motivo, t0 + RESULTADO_TRAS_SIGTERM_S, nombre)
        return 128 + signal.SIGTERM

    estado = "exitoso" if codigo in EXITOSOS else "fallido"
    if codigo != OK and motivo:
        print(motivo, file=sys.stderr)
    escribir(id_, estado, motivo or estado, limite_resultado, nombre)
    return OK if estado == "exitoso" else codigo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reentrenamiento orquestado por Airflow (etapa 7)")
    ap.add_argument("--ejecucion", required=True, help="id de la solicitud (dag_run_id)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _al_recibir_sigterm)
    try:
        return ejecutar(a.ejecucion)
    except Terminado:
        return 128 + signal.SIGTERM


if __name__ == "__main__":
    sys.exit(main())
