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
8. ``estado=valido|invalido`` y, si es válido, alias según ``promover``.

Cualquier excepción: la versión (si existe) queda ``estado=invalido``, la
ejecución termina ``FAILED`` y el alias no se toca.
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
        with mlflow.start_run(tags={"origen": "manual", "git_commit": git_commit, "git_limpio": "true",
                                    "huella_datos": huella, "modelo": config.modelo,
                                    "estrategia": config.estrategia}) as run:
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
                    mlflow.set_tags({"valido": "false", "motivo": crit.motivo})
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

                promovido, anterior = False, _alias_actual(cliente, nombre, alias)
                if valido and (promover == "siempre" or (promover == "si_no_hay" and anterior is None)):
                    cliente.set_model_version_tag(nombre, version, "promovido_por",
                                                  "manual" if promover == "siempre" else "si_no_hay")
                    if anterior is not None:
                        cliente.set_model_version_tag(nombre, version, "alias_anterior", anterior)
                    cliente.set_registered_model_alias(nombre, alias, version)    # último paso
                    promovido = True
                mlflow.log_metric("segundos_total", time.monotonic() - t0)
                mlflow.log_dict({"condiciones": crit.condiciones, "motivo": motivo}, "criterio.json")
                mlflow.set_tags({"valido": str(valido).lower(), "motivo": motivo, "promovido": str(promovido).lower()})
                return ResultadoRegistro(run_id, version, valido, motivo, promovido, anterior, crit.resumen)
            except BaseException as exc:
                if version is not None:
                    try:
                        cliente.set_model_version_tag(nombre, version, "estado", "invalido")
                        cliente.set_model_version_tag(nombre, version, "motivo", f"fallo: {type(exc).__name__}")
                    except Exception:  # noqa: BLE001 - se informa el error original
                        logger.error("no se pudo marcar la versión %s como inválida", version)
                mlflow.set_tag("valido", "false")
                mlflow.end_run(status="FAILED")
                raise


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
