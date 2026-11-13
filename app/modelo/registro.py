"""Entrenamiento, evaluación y registro en MLflow (DECISIONES.md §14).

Flujo (el alias se mueve **solo al final** y solo con un candidato válido):

1. Git limpio (lo verifica el comando ``entrenar_mlflow`` antes de llamar aquí).
2. Consolidar datos de PostgreSQL → escribir la copia inmutable → **releerla y
   verificarla**: desde aquí nada lee la base.
3. Backtest de K cortes (candidato y baseline) y criterio D1, T2, C1–C3.
4. Modelo definitivo con todo lo anterior al corte → ``modelo.joblib`` → SHA-256.
5. ``log_model`` (pyfunc + ``modelo.joblib``), registro de la versión con
   ``estado=pendiente`` y los tags de trazabilidad.
6. T4b: descargar ``runs:/<run>/<ruta_artefacto>``; SHA, formato, humo.
7. T4a: descargar la copia de datos de MLflow, verificarla y repetir los cortes;
   métricas idénticas (enteros) o dentro de la tolerancia (reales).
8. ``estado=valido|invalido``, tags de la versión y de la ejecución, métricas
   finales y ``criterio.json``; la ejecución termina ``FINISHED``.
9. Solo entonces, si es válido y ``promover`` lo permite, se asigna el alias:
   es la **última** operación (después no se escribe nada en MLflow).

Cualquier excepción antes del paso 9: la versión (si existe) queda
``estado=invalido``, la ejecución termina ``FAILED`` y el alias no se toca. Si
la propia asignación del alias lanza una excepción, el servidor pudo haberla
aplicado igual (respuesta perdida, timeout), así que se **consulta** el alias:

* apunta a la nueva versión → la promoción se realizó (se conserva todo);
* apunta a otra versión o a ninguna → candidato válido sin promover (se retiran
  ``promovido_por``/``alias_anterior``, ``promovido=false``) y se propaga el error;
* la consulta también falla → ``PromocionIndeterminada``: no se afirma nada
  sobre el alias (tags ``promocion=indeterminada``). La versión es válida, así
  que en ningún caso el alias queda en una versión inválida.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from app.config import configurar_cliente_mlflow
from app.modelo.artefacto import NOMBRE_ARCHIVO, sha256_bytes, validar_archivo
from app.modelo.copia_datos import DatosEntrenamiento, desde_bd, escribir_copia, leer_copia, sha256
from app.modelo.entrenamiento import CONFIG_APROBADA, empaquetar
from app.modelo.evaluacion import (
    ConfigEvaluacion,
    ResultadoEvaluacion,
    comparar_metricas,
    evaluar,
    evaluar_criterio,
    leer_tabla_csv,
)
from app.modelo.mlflow_util import descargar_de_ejecucion
from app.modelo.proveedores import versiones_librerias
from ml.pronosticador import ConfigEntrenamiento, Pronosticador

logger = logging.getLogger(__name__)

EXPERIMENTO = "novamercado-entrenamiento"
RUTA_MODELO = "modelo"
RUTA_DATOS = "datos"
RUTA_BACKTEST = "backtest"
ARCHIVO_BACKTEST = "resultados_por_corte.csv"
MODOS_PROMOVER = ("si_no_hay", "siempre", "nunca")
RAIZ = Path(__file__).resolve().parents[2]
# Hilos de OpenMP fijos para entrenar y evaluar (DECISIONES.md §14). Con varios
# hilos, el orden de las reducciones paralelas del HGB cambia de una ejecución a
# otra y algunos valores varían en el último bit (≈1e-17): los bytes de
# modelo.joblib no se reproducen y, en el límite, una cantidad podría redondearse
# distinto. Con 1 hilo el entrenamiento es determinista byte a byte (medido: 5 de
# 5 entrenamientos idénticos con los datos reales) a cambio de ~40 % más de tiempo.
HILOS_POR_DEFECTO = 1


def hilos_entrenamiento() -> int:
    return int(os.environ.get("HILOS_ENTRENAMIENTO", HILOS_POR_DEFECTO))


def hilos_fijos(n: int):
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=int(n), user_api="openmp")


class PromocionIndeterminada(RuntimeError):
    """Falló la asignación del alias y tampoco se pudo consultar su valor: no se sabe
    si apunta a la nueva versión (válida) o sigue en la anterior."""

    def __init__(self, nombre: str, alias: str, version: str, anterior: str | None, causa: BaseException):
        self.nombre, self.alias, self.version, self.anterior = nombre, alias, version, anterior
        super().__init__(
            f"Promoción en estado indeterminado: la asignación del alias '{alias}' de '{nombre}' a la "
            f"versión {version} falló ({type(causa).__name__}) y no se pudo consultar el alias después. "
            f"Puede apuntar a la versión {version} (válida) o seguir en "
            f"{'la versión ' + anterior if anterior else 'ninguna versión'}; no está garantizado. "
            f"Consulte el alias en MLflow antes de reintentar.")


@dataclass
class ResultadoRegistro:
    run_id: str
    version: str | None
    valido: bool
    motivo: str
    promovido: bool
    alias_anterior: str | None
    resumen: dict


def entorno() -> dict:
    import joblib
    import mlflow
    import scipy
    return {"python": platform.python_version(), "plataforma": platform.platform(), "cpus": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "librerias": {**versiones_librerias(), "scipy": scipy.__version__, "joblib": joblib.__version__,
                          "mlflow": mlflow.__version__}}


def requisitos_pip() -> list[str]:
    import mlflow
    lineas = []
    for archivo in ("comun.txt",):
        for ln in (RAIZ / "requirements" / archivo).read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln and not ln.startswith("-r"):
                lineas.append(ln)
    return lineas + [f"mlflow=={mlflow.__version__}"]


def entrenar_definitivo(datos: DatosEntrenamiento, cfg: ConfigEntrenamiento, nombre: str):
    p = Pronosticador.entrenar(datos.ventas, datos.promociones, datos.tiendas, datos.feriados, datos.costos,
                               datos.familia_ids, datos.corte, cfg)
    return p, empaquetar(p, nombre)


def resolver_asignacion_alias(cliente, nombre: str, alias: str, version: str) -> tuple[str, str | None]:
    """Tras una excepción de ``set_registered_model_alias``, consulta el alias una vez.

    Devuelve ``("aplicada", version)``, ``("no_aplicada", observado)`` (otra versión o
    ``None``) o ``("indeterminada", None)`` si la consulta también falla.
    """
    try:
        observado = _alias_actual(cliente, nombre, alias)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de la consulta deja el resultado abierto
        logger.error("no se pudo consultar el alias %s de %s tras el fallo: %s", alias, nombre, type(exc).__name__)
        return "indeterminada", None
    return ("aplicada", observado) if observado == str(version) else ("no_aplicada", observado)


def _alias_actual(cliente, nombre: str, alias: str) -> str | None:
    from mlflow.exceptions import MlflowException
    try:
        return str(cliente.get_model_version_by_alias(nombre, alias).version)
    except MlflowException as exc:
        if getattr(exc, "error_code", None) in ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"):
            return None
        raise


def _log_metricas_evaluacion(mlflow, ev: ResultadoEvaluacion) -> None:
    for i, corte in enumerate(ev.cortes):
        for fila in ev.por_corte[ev.por_corte["corte"] == corte].itertuples():
            for m in ("costo_total", "mae", "wape", "tasa_quiebre"):
                mlflow.log_metric(f"{m}_{fila.modelo}", float(getattr(fila, m)), step=i)


def entrenar_y_registrar(motor, *, tracking_uri: str, nombre: str, alias: str, promover: str,
                         git_commit: str, config: ConfigEntrenamiento = CONFIG_APROBADA,
                         evaluacion: ConfigEvaluacion | None = None) -> ResultadoRegistro:
    if promover not in MODOS_PROMOVER:
        raise ValueError(f"promover debe ser uno de {MODOS_PROMOVER}")
    evaluacion = evaluacion or ConfigEvaluacion()
    configurar_cliente_mlflow()
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    mlflow.set_experiment(EXPERIMENTO)
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="nm_entrenamiento_") as tmp:
        tmp = Path(tmp)
        datos_bd = desde_bd(motor)
        hilos = hilos_entrenamiento()
        extra = {"config": asdict(config), "evaluacion": evaluacion.como_dict(), "git_commit": git_commit,
                 "entorno": entorno(), "hilos_entrenamiento": hilos}
        manifiesto, huella = escribir_copia(datos_bd, tmp / RUTA_DATOS, extra)
        datos, manifiesto_leido = leer_copia(tmp / RUTA_DATOS, huella)   # desde aquí, solo la copia
        # La configuración también sale del manifiesto releído, igual que en `reproducir`:
        # pickle depende de la identidad de los objetos (p. ej. cadenas internadas frente a
        # cadenas leídas de JSON), así que ambos caminos deben construir los mismos objetos
        # para que modelo.joblib tenga los mismos bytes.
        config, evaluacion = config_desde_manifiesto(manifiesto_leido)

        version: str | None = None
        run = mlflow.start_run(tags={"origen": "manual", "git_commit": git_commit, "git_limpio": "true",
                                     "huella_datos": huella, "modelo": config.modelo,
                                     "estrategia": config.estrategia})
        run_id = run.info.run_id
        try:
            mlflow.log_params({**{f"config.{k}": v for k, v in asdict(config).items() if k != "params_hgb"},
                               **{f"hgb.{k}": v for k, v in config.params_hgb.items()},
                               **{f"evaluacion.{k}": v for k, v in evaluacion.como_dict().items()},
                               "corte": manifiesto["corte"], "hilos_entrenamiento": hilos})
            mlflow.log_artifacts(str(tmp / RUTA_DATOS), RUTA_DATOS)

            t = time.monotonic()
            with hilos_fijos(hilos):
                ev = evaluar(datos, evaluacion, config)
            crit = evaluar_criterio(ev, evaluacion)
            mlflow.log_metric("segundos_backtest", time.monotonic() - t)
            for k, v in crit.resumen.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_dict({"condiciones": crit.condiciones, "motivo": crit.motivo,
                             "cortes": [c.strftime("%Y-%m-%d") for c in ev.cortes]}, "criterio.json")
            if not crit.condiciones.get("D1", False):
                mlflow.set_tags({"valido": "false", "motivo": crit.motivo, "promovido": "false"})
                mlflow.end_run(status="FINISHED")
                return ResultadoRegistro(run_id, None, False, crit.motivo, False, None, crit.resumen)
            _log_metricas_evaluacion(mlflow, ev)
            tabla = ev.tabla_csv()
            (tmp / ARCHIVO_BACKTEST).write_bytes(tabla)
            mlflow.log_artifact(str(tmp / ARCHIVO_BACKTEST), RUTA_BACKTEST)
            sha_backtest = sha256(tabla)

            with hilos_fijos(hilos):
                p, art = entrenar_definitivo(datos, config, nombre)
            ruta_local = tmp / NOMBRE_ARCHIVO
            ruta_local.write_bytes(art.artefacto)
            sha_art = sha256_bytes(art.artefacto)

            from ml.mlflow_modelo import ModeloNovaMercado
            mlflow.pyfunc.log_model(RUTA_MODELO, python_model=ModeloNovaMercado(),
                                    artifacts={"modelo": str(ruta_local)},
                                    code_paths=[str(RAIZ / "ml")], pip_requirements=requisitos_pip())
            ruta_artefacto = f"{RUTA_MODELO}/artifacts/{NOMBRE_ARCHIVO}"
            mv = mlflow.register_model(f"runs:/{run_id}/{RUTA_MODELO}", nombre)
            version = str(mv.version)
            for k, v in {"estado": "pendiente", "sha256_artefacto": sha_art, "ruta_artefacto": ruta_artefacto,
                         "huella_datos": huella, "sha256_backtest": sha_backtest,
                         "corte": manifiesto["corte"], "git_commit": git_commit}.items():
                cliente.set_model_version_tag(nombre, version, k, v)
            mlflow.set_tags({"version_registrada": version, "sha256_artefacto": sha_art,
                             "sha256_backtest": sha_backtest})

            t4b = verificar_t4b(tracking_uri, run_id, ruta_artefacto, sha_art)
            t = time.monotonic()
            with hilos_fijos(hilos):
                t4a = verificar_t4a(tracking_uri, run_id, huella, sha_backtest, evaluacion, config)
            mlflow.log_metric("segundos_t4a", time.monotonic() - t)
            crit.condiciones["T4a"], crit.condiciones["T4b"] = not t4a, not t4b
            fallas = ([f"T4b: {t4b}"] if t4b else []) + ([f"T4a: {t4a[0]}"] if t4a else [])
            valido = crit.valido and not fallas
            motivo = crit.motivo if not fallas else "; ".join(fallas)
            cliente.set_model_version_tag(nombre, version, "estado", "valido" if valido else "invalido")
            cliente.set_model_version_tag(nombre, version, "motivo", motivo)

            anterior = _alias_actual(cliente, nombre, alias)
            promover_ahora = valido and (promover == "siempre" or (promover == "si_no_hay" and anterior is None))
            if promover_ahora:
                cliente.set_model_version_tag(nombre, version, "promovido_por",
                                              "manual" if promover == "siempre" else "si_no_hay")
                if anterior is not None:
                    cliente.set_model_version_tag(nombre, version, "alias_anterior", anterior)
            mlflow.log_metric("segundos_total", time.monotonic() - t0)
            mlflow.log_dict({"condiciones": crit.condiciones, "motivo": motivo}, "criterio.json")
            mlflow.set_tags({"valido": str(valido).lower(), "motivo": motivo,
                             "promovido": str(promover_ahora).lower()})
            mlflow.end_run(status="FINISHED")
        except BaseException as exc:
            if version is not None:
                try:
                    cliente.set_model_version_tag(nombre, version, "estado", "invalido")
                    cliente.set_model_version_tag(nombre, version, "motivo", f"fallo: {type(exc).__name__}")
                except Exception:  # noqa: BLE001 - se informa el error original
                    logger.error("no se pudo marcar la versión %s como inválida", version)
            # Por run_id (no por la ejecución activa): si el fallo ocurrió al cerrarla,
            # `mlflow.set_tag` abriría una ejecución nueva.
            try:
                cliente.set_tag(run_id, "valido", "false")
            except Exception:  # noqa: BLE001 - se informa el error original
                logger.error("no se pudo marcar la ejecución %s como inválida", run_id)
            if mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
            else:
                cliente.set_terminated(run_id, status="FAILED")
            raise

    # Promoción: la asignación del alias es la ÚLTIMA operación. Todo lo anterior
    # (tags, métricas, criterio y cierre de la ejecución) ya terminó sin errores.
    if promover_ahora:
        try:
            cliente.set_registered_model_alias(nombre, alias, version)
        except BaseException as exc:
            # El servidor pudo aplicar el cambio aunque el cliente reciba un error
            # (respuesta perdida, timeout): se consulta antes de deshacer nada.
            resultado, observado = resolver_asignacion_alias(cliente, nombre, alias, version)
            if resultado == "aplicada":
                logger.warning("la asignación del alias %s lanzó %s, pero el alias ya apunta a la versión %s: "
                               "la promoción se realizó", alias, type(exc).__name__, version)
                if not isinstance(exc, Exception):         # p. ej. KeyboardInterrupt: se respeta
                    raise
            elif resultado == "no_aplicada":
                logger.error("la asignación del alias %s falló; sigue en %s; la versión %s queda válida sin "
                             "promover", alias, observado or "ninguna versión", version)
                _deshacer_marcas_de_promocion(cliente, nombre, version, run_id)
                raise
            else:
                _marcar_promocion_indeterminada(cliente, nombre, version, run_id)
                raise PromocionIndeterminada(nombre, alias, version, anterior, exc) from exc
    return ResultadoRegistro(run_id, version, valido, motivo, promover_ahora, anterior, crit.resumen)


def _marcar_promocion_indeterminada(cliente, nombre: str, version: str, run_id: str) -> None:
    """Mejor esfuerzo: MLflow puede seguir sin responder. Se conservan las marcas de
    promoción (pueden ser ciertas) y se agrega ``promocion=indeterminada``."""
    for escribir in (lambda: cliente.set_model_version_tag(nombre, version, "promocion", "indeterminada"),
                     lambda: cliente.set_tag(run_id, "promovido", "indeterminado")):
        try:
            escribir()
        except Exception:  # noqa: BLE001
            logger.error("no se pudo marcar la promoción indeterminada de la versión %s", version)


def _deshacer_marcas_de_promocion(cliente, nombre: str, version: str, run_id: str) -> None:
    """La asignación del alias falló: el alias anterior no cambió; se retiran las marcas
    escritas antes (mejor esfuerzo, sin ocultar el error original)."""
    for tag in ("promovido_por", "alias_anterior"):
        try:
            cliente.delete_model_version_tag(nombre, version, tag)
        except Exception:  # noqa: BLE001
            pass
    try:
        cliente.set_tag(run_id, "promovido", "false")
    except Exception:  # noqa: BLE001
        logger.error("la asignación del alias falló y no se pudo marcar la ejecución %s", run_id)


def verificar_t4b(tracking_uri: str, run_id: str, ruta_artefacto: str, sha: str) -> str | None:
    """Integridad del modelo definitivo registrado. Devuelve el motivo del fallo o None."""

    from app.modelo.proveedores import ModeloNoDisponible
    with tempfile.TemporaryDirectory(prefix="nm_t4b_") as tmp:
        try:
            ruta = descargar_de_ejecucion(tracking_uri, run_id, ruta_artefacto, tmp)
            validar_archivo(ruta, sha)
        except ModeloNoDisponible as exc:
            return str(exc)
    return None


def descargar_copia(tracking_uri: str, run_id: str, destino: Path) -> Path:
    return Path(descargar_de_ejecucion(tracking_uri, run_id, RUTA_DATOS, str(destino)))


def descargar_backtest(tracking_uri: str, run_id: str, destino: Path, sha_esperado: str) -> pd.DataFrame:
    ruta = descargar_de_ejecucion(tracking_uri, run_id, f"{RUTA_BACKTEST}/{ARCHIVO_BACKTEST}", str(destino))
    crudo = Path(ruta).read_bytes()
    if sha256(crudo) != sha_esperado:
        raise ValueError("los resultados del backtest registrados no coinciden con su SHA-256")
    return leer_tabla_csv(crudo)


def verificar_t4a(tracking_uri: str, run_id: str, huella: str, sha_backtest: str, evaluacion: ConfigEvaluacion,
                  config: ConfigEntrenamiento) -> list[str]:
    """Repite los cortes desde la copia **descargada de MLflow**. Devuelve las diferencias."""
    from app.modelo.copia_datos import CopiaDatosInvalida
    with tempfile.TemporaryDirectory(prefix="nm_t4a_") as tmp:
        tmp = Path(tmp)
        try:
            datos, _ = leer_copia(descargar_copia(tracking_uri, run_id, tmp), huella)
            registrada = descargar_backtest(tracking_uri, run_id, tmp, sha_backtest)
        except (CopiaDatosInvalida, ValueError) as exc:
            return [str(exc)]
        reproducida = evaluar(datos, evaluacion, config).por_corte
    return comparar_metricas(registrada, reproducida)


def config_desde_manifiesto(manifiesto: dict) -> tuple[ConfigEntrenamiento, ConfigEvaluacion]:
    return ConfigEntrenamiento(**manifiesto["config"]), ConfigEvaluacion(**manifiesto["evaluacion"])


def resumen_json(r: ResultadoRegistro) -> str:
    return json.dumps({"run_id": r.run_id, "version": r.version, "valido": r.valido, "motivo": r.motivo,
                       "promovido": r.promovido, "alias_anterior": r.alias_anterior}, ensure_ascii=False)


__all__ = ["EXPERIMENTO", "MODOS_PROMOVER", "ResultadoRegistro", "entrenar_y_registrar"]
