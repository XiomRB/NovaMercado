"""Supervisor del entrenamiento (etapa 6, plan §4.2): **único** punto que adquiere la
concesión de entrenamiento. Lo usan ``entrenar_mlflow`` (modo ``manual``),
``arranque`` (modo ``arranque``) y, en la etapa 7, el DAG (``reentrenamiento``).

1. Toma la concesión (``app.modelo.bloqueo``) con el plazo del modo.
2. Reconciliación (anula lo huérfano y refleja el vigente en el alias).
3. En el arranque, si PostgreSQL ya tiene un vigente confirmado, termina sin entrenar.
4. Lanza el **proceso hijo** (``app.modelo.trabajo_entrenamiento``) y lo vigila:
   sondea la concesión cada 5 s; si se pierde, ``SIGTERM`` al hijo, ``SIGKILL`` a
   los 10 s y código 6. El hijo se lanza desde el hilo principal (PDEATHSIG).
5. ``autorizar_y_promover`` con el resultado del hijo (solo el supervisor autoriza).
6. Arranque: respaldo baseline si el candidato falló **solo** C1–C3 y sigue sin
   haber vigente (§6), como otro hijo bajo la misma concesión.
7. Reconciliación final y liberación de la concesión (también ante errores).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import configurar_cliente_mlflow
from app.modelo.bloqueo import SONDEO_S, ConcesionEntrenamiento, ConcesionPerdida
from app.modelo.promocion import autorizar_y_promover, reconciliar
from app.modelo.registro import ResultadoRegistro, califica_para_respaldo

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
GRACIA_TERMINACION_S = 10
ESPERA_POR_MODO = {"manual": 0.0, "arranque": 900.0}
from app.modelo.codigos_salida import OK, PERDIDA, SIN_MODELO  # noqa: E402


class ErrorHijo(RuntimeError):
    def __init__(self, codigo: int):
        self.codigo = codigo
        super().__init__(f"el proceso de entrenamiento terminó con código {codigo}")


@dataclass
class ResultadoCiclo:
    codigo: int
    mensaje: str
    version_vigente: str | None = None
    promovido: bool = False
    detalles: list[dict] = field(default_factory=list)

    def como_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


def lanzar_hijo(c: ConcesionEntrenamiento, url_bd: str, entrada: dict, *, sondeo: float = SONDEO_S,
                gracia: float = GRACIA_TERMINACION_S) -> ResultadoRegistro:
    """Ejecuta el hijo y lo vigila. Debe llamarse desde el hilo que vive toda la ejecución
    (``PDEATHSIG`` se dispara cuando muere el *hilo* que creó el proceso)."""
    with tempfile.TemporaryDirectory(prefix="nm_hijo_") as tmp:
        ruta_entrada, ruta_salida = Path(tmp) / "entrada.json", Path(tmp) / "salida.json"
        ruta_entrada.write_text(json.dumps({**entrada, "epoca": c.epoca}), encoding="utf-8")
        env = {**os.environ, "NM_PID_SUPERVISOR": str(os.getpid()), "NM_URL_BD_ENTRENAMIENTO": url_bd,
               "PYTHONPATH": str(RAIZ)}
        proc = subprocess.Popen([sys.executable, "-m", "app.modelo.trabajo_entrenamiento",
                                 str(ruta_entrada), str(ruta_salida)], env=env, cwd=str(RAIZ))
        try:
            while True:
                try:
                    proc.wait(timeout=sondeo)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if not c.sondear():
                    logger.error("concesión perdida: se detiene el entrenamiento (pid %s)", proc.pid)
                    _detener(proc, gracia)
                    raise ConcesionPerdida(f"se perdió la concesión de la época {c.epoca} durante el entrenamiento")
        except BaseException:
            if proc.poll() is None:
                _detener(proc, gracia)
            raise
        if proc.returncode != 0:
            if proc.returncode == PERDIDA:
                raise ConcesionPerdida("el proceso de entrenamiento detectó la pérdida de la concesión")
            raise ErrorHijo(proc.returncode)
        c.verificar()
        return ResultadoRegistro.desde_dict(json.loads(ruta_salida.read_text(encoding="utf-8")))


def _detener(proc: subprocess.Popen, gracia: float) -> None:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=gracia)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except ProcessLookupError:
        pass


def ejecutar_entrenamiento(*, url_bd: str, tracking_uri: str, nombre: str, alias: str, modo: str,
                           promover: str = "si_no_hay", git_commit: str, etiquetas: dict | None = None,
                           espera: float | None = None, config: dict | None = None,
                           evaluacion: dict | None = None) -> ResultadoCiclo:
    configurar_cliente_mlflow()
    from mlflow import MlflowClient
    cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    origen = {"manual": "manual", "arranque": "arranque"}.get(modo, modo)
    if modo == "arranque":
        promover = "si_no_hay"
    base = {"tracking_uri": tracking_uri, "nombre": nombre, "git_commit": git_commit,
            "etiquetas": etiquetas or {}}
    espera = ESPERA_POR_MODO.get(modo, 0.0) if espera is None else espera

    with ConcesionEntrenamiento(url_bd, modo=modo, espera=espera) as c:
        reconciliar(c, cliente, nombre, alias)
        if modo == "arranque":
            vigente = c.leer_vigente(nombre)
            if vigente is not None:
                return ResultadoCiclo(OK, f"ya existe un vigente confirmado (versión {vigente.version}); "
                                          "no se entrena", vigente.version)
        r = lanzar_hijo(c, url_bd, {**base, "tarea": "candidato", "origen": origen, "config": config,
                                    "evaluacion": evaluacion})
        p = autorizar_y_promover(c, cliente, r, nombre=nombre, alias=alias, promover=promover, origen=origen)
        detalles = [{"criterio": "candidato", "run_id": r.run_id, "version": r.version,
                     "condiciones": r.condiciones, "autorizada": p.autorizada, "promovido": p.promovido,
                     "motivo": p.motivo}]
        if modo == "arranque" and not p.promovido:
            if califica_para_respaldo(r) and c.leer_vigente(nombre) is None:
                logger.warning("el candidato falló solo criterios comerciales (%s): se entrena el respaldo "
                               "baseline", r.motivo)
                rb = lanzar_hijo(c, url_bd, {**base, "tarea": "respaldo", "run_candidato": r.run_id})
                pb = autorizar_y_promover(c, cliente, rb, nombre=nombre, alias=alias, promover="si_no_hay",
                                          origen="arranque_respaldo")
                detalles.append({"criterio": "respaldo", "run_id": rb.run_id, "version": rb.version,
                                 "condiciones": rb.condiciones, "autorizada": pb.autorizada,
                                 "promovido": pb.promovido, "motivo": pb.motivo})
                p = pb
            else:
                logger.error("el candidato no es utilizable y no califica para el respaldo: %s", p.motivo)
        reconciliar(c, cliente, nombre, alias)
        vigente = c.leer_vigente(nombre)
    if modo == "arranque":
        codigo = OK if vigente is not None else SIN_MODELO
    else:
        codigo = OK if p.autorizada else SIN_MODELO
    return ResultadoCiclo(codigo, p.motivo, vigente.version if vigente else None, p.promovido, detalles)
