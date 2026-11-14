"""Entrenamiento, evaluación y registro en MLflow (DECISIONES.md §14 y §15).

Desde la etapa 6 este módulo es el **trabajo del proceso hijo**: registra versiones
en estado ``pendiente`` (o ``invalido`` si una validación falla) y **nunca** las
autoriza ni asigna el alias. La autorización (``estado=valido``), la decisión en
PostgreSQL y el alias los hace solo el supervisor
(``app.modelo.ciclo_entrenamiento.autorizar_y_promover``).

Flujo de :func:`registrar_version` (candidato):

1. Consolidar los datos de PostgreSQL → escribir la copia inmutable → **releerla y
   verificarla**: desde aquí nada lee la base.
2. Backtest de K cortes (candidato y baseline) y criterio T2, D1, C1–C3.
3. Modelo definitivo con todo lo anterior al corte → ``modelo.joblib`` → SHA-256.
4. ``log_model`` y registro de la versión con ``estado=pendiente``,
   ``epoca_bloqueo`` y los tags de trazabilidad.
5. T4b (identidad: SHA local = tag = SHA descargado; ruta y ``run_id``) y T3
   (artefacto descargado: SHA antes de deserializar, formato, bibliotecas y humo),
   registrados **por separado**.
6. T4a: reproducción del backtest desde la copia descargada de MLflow.
7. Reentrenamiento (etapa 7): si el supervisor pasa el ``campeon`` (decisión vigente
   confirmada en PostgreSQL), criterio R2 (``app.modelo.r2``): el campeón se descarga y
   verifica **antes** de escribir la copia (su ``corte_v`` queda en el manifiesto) y R2
   se calcula después de T4a, con ``r2/resultados_por_bloque.csv`` (tag ``sha256_r2``).
   R2 no es requisito de validez: lo exige solo la promoción ``si_mejora``.
8. ``verificacion=superada|fallida`` y ``condiciones``; si algo falla, la versión
   pasa a ``invalido`` (solo baja de categoría). La ejecución termina ``FINISHED``.

:func:`registrar_respaldo` hace lo mismo con ``baseline_dow4`` sobre la copia del
candidato (§6): T1, T2, T3, D1, T4a, T4b y R1 (coherencia con las filas del
baseline en el backtest del candidato).

``comprobar`` (obligatorio) se llama al empezar y antes de registrar en MLflow:
en el hijo verifica que la concesión de su época sigue vigente; en las pruebas,
``ConcesionEntrenamiento.verificar``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from app.config import configurar_cliente_mlflow
from app.modelo.artefacto import NOMBRE_ARCHIVO, sha256_archivo, sha256_bytes, validar_archivo
from app.modelo.copia_datos import DatosEntrenamiento, desde_bd, escribir_copia, leer_copia, sha256
from app.modelo.entrenamiento import CONFIG_APROBADA, empaquetar
from app.modelo.evaluacion import (
    BASELINE,
    MODELOS_CANDIDATO,
    MODELOS_RESPALDO,
    ConfigEvaluacion,
    ResultadoEvaluacion,
    comparar_metricas,
    evaluar,
    evaluar_criterio,
    evaluar_criterio_respaldo,
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
MODOS_PROMOVER = ("si_no_hay", "siempre", "nunca", "si_mejora")
RAIZ = Path(__file__).resolve().parents[2]
# Criterios que debe superar una versión para poder autorizarse (§5.1 y §6).
REQUERIDOS = {
    "candidato": ("T1", "T2", "T3", "D1", "T4a", "T4b", "C1", "C2", "C3"),
    "respaldo": ("T1", "T2", "T3", "D1", "T4a", "T4b", "R1"),
}
TECNICOS = ("T1", "T2", "T3", "D1", "T4a", "T4b")
COMERCIALES = ("C1", "C2", "C3")
# Hilos de OpenMP fijos para entrenar y evaluar (DECISIONES.md §14). Con varios
# hilos, el orden de las reducciones paralelas del HGB cambia de una ejecución a
# otra y algunos valores varían en el último bit (≈1e-17): los bytes de
# modelo.joblib no se reproducen. Con 1 hilo el entrenamiento es determinista byte
# a byte (medido: 5 de 5 entrenamientos idénticos con los datos reales).
HILOS_POR_DEFECTO = 1


def hilos_entrenamiento() -> int:
    return int(os.environ.get("HILOS_ENTRENAMIENTO", HILOS_POR_DEFECTO))


def hilos_fijos(n: int):
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=int(n), user_api="openmp")


@dataclass
class ResultadoRegistro:
    """Lo que el hijo informa al supervisor (también se guarda como JSON)."""
    run_id: str
    version: str | None
    verificacion: bool
    motivo: str
    condiciones: dict[str, bool]
    resumen: dict = field(default_factory=dict)
    criterio: str = "candidato"
    epoca: int | None = None
    sha256: str | None = None
    ruta_artefacto: str | None = None
    huella: str | None = None
    r2: dict | None = None            # etapa 7: resultado de R2 y la decisión del campeón

    def como_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def desde_dict(d: dict) -> ResultadoRegistro:
        return ResultadoRegistro(**d)


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


def _log_metricas_evaluacion(mlflow, ev: ResultadoEvaluacion) -> None:
    for i, corte in enumerate(ev.cortes):
        for fila in ev.por_corte[ev.por_corte["corte"] == corte].itertuples():
            for m in ("costo_total", "mae", "wape", "tasa_quiebre"):
                mlflow.log_metric(f"{m}_{fila.modelo}", float(getattr(fila, m)), step=i)


def config_desde_manifiesto(manifiesto: dict) -> tuple[ConfigEntrenamiento, ConfigEvaluacion]:
    return ConfigEntrenamiento(**manifiesto["config"]), ConfigEvaluacion(**manifiesto["evaluacion"])


def modelos_de_manifiesto(manifiesto: dict) -> tuple[str, ...]:
    return tuple(manifiesto.get("modelos_evaluados", MODELOS_CANDIDATO))


# ----------------------------------------------------------------- verificaciones
def verificar_t3_t4b(cliente, tracking_uri: str, nombre: str, version: str, run_id: str, ruta_artefacto: str,
                     sha_local: str) -> tuple[str | None, str | None]:
    """Descarga ``modelo.joblib`` de la versión registrada una vez y devuelve
    ``(motivo T3, motivo T4b)`` (``None`` = superado).

    * T4b, identidad: el tag ``sha256_artefacto``, la ruta y el ``run_id`` registrados
      coinciden con lo entrenado, y el SHA del archivo descargado es el local.
    * T3, uso: SHA verificado **antes** de deserializar, formato, bibliotecas y humo.
    """
    from app.modelo.proveedores import ModeloNoDisponible
    t4b = None
    try:
        mv = cliente.get_model_version(nombre, version)
        tags = dict(mv.tags or {})
        if mv.run_id != run_id:
            t4b = f"la versión {version} pertenece a otra ejecución"
        elif tags.get("sha256_artefacto") != sha_local:
            t4b = "el tag sha256_artefacto no coincide con el modelo entrenado"
        elif tags.get("ruta_artefacto") != ruta_artefacto:
            t4b = "el tag ruta_artefacto no coincide con el registrado"
    except Exception as exc:  # noqa: BLE001
        motivo = f"no se pudo consultar la versión: {type(exc).__name__}"
        return motivo, motivo
    with tempfile.TemporaryDirectory(prefix="nm_t3_") as tmp:
        try:
            ruta = descargar_de_ejecucion(tracking_uri, run_id, ruta_artefacto, tmp)
        except Exception as exc:  # noqa: BLE001
            motivo = f"no se pudo descargar el artefacto registrado: {type(exc).__name__}"
            return motivo, motivo
        if t4b is None and sha256_archivo(ruta) != sha_local:
            t4b = "el artefacto descargado no es el que se entrenó (SHA-256 distinto)"
        try:
            validar_archivo(ruta, sha_local)
            t3 = None
        except ModeloNoDisponible as exc:
            t3 = str(exc)
    return t3, t4b


def descargar_copia(tracking_uri: str, run_id: str, destino: Path) -> Path:
    return Path(descargar_de_ejecucion(tracking_uri, run_id, RUTA_DATOS, str(destino)))


def descargar_backtest(tracking_uri: str, run_id: str, destino: Path, sha_esperado: str) -> pd.DataFrame:
    ruta = descargar_de_ejecucion(tracking_uri, run_id, f"{RUTA_BACKTEST}/{ARCHIVO_BACKTEST}", str(destino))
    crudo = Path(ruta).read_bytes()
    if sha256(crudo) != sha_esperado:
        raise ValueError("los resultados del backtest registrados no coinciden con su SHA-256")
    return leer_tabla_csv(crudo)


def verificar_t4a(tracking_uri: str, run_id: str, huella: str, sha_backtest: str, evaluacion: ConfigEvaluacion,
                  config: ConfigEntrenamiento, modelos: tuple[str, ...] = MODELOS_CANDIDATO) -> list[str]:
    """Repite los cortes desde la copia **descargada de MLflow**. Devuelve las diferencias."""
    from app.modelo.copia_datos import CopiaDatosInvalida
    with tempfile.TemporaryDirectory(prefix="nm_t4a_") as tmp:
        tmp = Path(tmp)
        try:
            datos, _ = leer_copia(descargar_copia(tracking_uri, run_id, tmp), huella)
            registrada = descargar_backtest(tracking_uri, run_id, tmp, sha_backtest)
        except (CopiaDatosInvalida, ValueError) as exc:
            return [str(exc)]
        reproducida = evaluar(datos, evaluacion, config, modelos).por_corte
    return comparar_metricas(registrada, reproducida)


# --------------------------------------------------------------------- registro
def _cliente(tracking_uri: str):
    configurar_cliente_mlflow()
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENTO)
    return mlflow, MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _registrar(*, datos_bd: DatosEntrenamiento, tracking_uri: str, nombre: str, git_commit: str, epoca: int,
               origen: str, criterio: str, config: ConfigEntrenamiento, evaluacion: ConfigEvaluacion,
               comprobar: Callable[[], None], etiquetas: dict, tabla_candidato: pd.DataFrame | None = None,
               campeon: dict | None = None) -> ResultadoRegistro:
    mlflow, cliente = _cliente(tracking_uri)
    modelos = MODELOS_CANDIDATO if criterio == "candidato" else MODELOS_RESPALDO
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="nm_entrenamiento_") as tmp:
        tmp = Path(tmp)
        hilos = hilos_entrenamiento()
        extra = {"config": asdict(config), "evaluacion": evaluacion.como_dict(), "git_commit": git_commit,
                 "entorno": entorno(), "hilos_entrenamiento": hilos, "modelos_evaluados": list(modelos),
                 "criterio": criterio}
        campeon_p, r2_previo = None, None
        if criterio == "candidato" and campeon is not None:
            campeon_p, corte_v, r2_previo = _preparar_campeon(tracking_uri, campeon, tmp)
            extra["campeon"] = {k: campeon.get(k) for k in CAMPOS_CAMPEON} | {"corte_v": corte_v}
        manifiesto, huella = escribir_copia(datos_bd, tmp / RUTA_DATOS, extra)
        datos, manifiesto_leido = leer_copia(tmp / RUTA_DATOS, huella)   # desde aquí, solo la copia
        # La configuración también sale del manifiesto releído, igual que en `reproducir`:
        # pickle depende de la identidad de los objetos (p. ej. cadenas internadas frente a
        # cadenas leídas de JSON), así que ambos caminos deben construir los mismos objetos.
        config, evaluacion = config_desde_manifiesto(manifiesto_leido)

        comprobar()
        version: str | None = None
        tags_run = {"origen": origen, "git_commit": git_commit, "git_limpio": "true", "huella_datos": huella,
                    "modelo": config.modelo, "estrategia": config.estrategia, "epoca_bloqueo": str(epoca),
                    "criterio": criterio, **{k: str(v) for k, v in etiquetas.items()}}
        run = mlflow.start_run(tags=tags_run)
        run_id = run.info.run_id
        try:
            mlflow.log_params({**{f"config.{k}": v for k, v in asdict(config).items() if k != "params_hgb"},
                               **{f"hgb.{k}": v for k, v in config.params_hgb.items()},
                               **{f"evaluacion.{k}": v for k, v in evaluacion.como_dict().items()},
                               "corte": manifiesto["corte"], "hilos_entrenamiento": hilos})
            mlflow.log_artifacts(str(tmp / RUTA_DATOS), RUTA_DATOS)

            t = time.monotonic()
            with hilos_fijos(hilos):
                ev = evaluar(datos, evaluacion, config, modelos)
            crit = (evaluar_criterio(ev, evaluacion) if criterio == "candidato"
                    else evaluar_criterio_respaldo(ev, evaluacion, tabla_candidato))
            mlflow.log_metric("segundos_backtest", time.monotonic() - t)
            for k, v in crit.resumen.items():
                mlflow.log_metric(k, float(v))
            if not crit.condiciones.get("D1", False):
                cond = {**crit.condiciones, "T1": True}
                mlflow.log_dict({"condiciones": cond, "motivo": crit.motivo}, "criterio.json")
                mlflow.set_tags({"verificacion": "fallida", "motivo": crit.motivo})
                mlflow.end_run(status="FINISHED")
                return ResultadoRegistro(run_id, None, False, crit.motivo, cond, crit.resumen, criterio, epoca,
                                         huella=huella)
            _log_metricas_evaluacion(mlflow, ev)
            tabla = ev.tabla_csv()
            (tmp / ARCHIVO_BACKTEST).write_bytes(tabla)
            mlflow.log_artifact(str(tmp / ARCHIVO_BACKTEST), RUTA_BACKTEST)
            sha_backtest = sha256(tabla)

            with hilos_fijos(hilos):
                _, art = entrenar_definitivo(datos, config, nombre)
            ruta_local = tmp / NOMBRE_ARCHIVO
            ruta_local.write_bytes(art.artefacto)
            sha_art = sha256_bytes(art.artefacto)

            comprobar()
            from ml.mlflow_modelo import ModeloNovaMercado
            mlflow.pyfunc.log_model(RUTA_MODELO, python_model=ModeloNovaMercado(),
                                    artifacts={"modelo": str(ruta_local)},
                                    code_paths=[str(RAIZ / "ml")], pip_requirements=requisitos_pip())
            ruta_artefacto = f"{RUTA_MODELO}/artifacts/{NOMBRE_ARCHIVO}"
            mv = mlflow.register_model(f"runs:/{run_id}/{RUTA_MODELO}", nombre)
            version = str(mv.version)
            tags_version = {"estado": "pendiente", "sha256_artefacto": sha_art, "ruta_artefacto": ruta_artefacto,
                            "huella_datos": huella, "sha256_backtest": sha_backtest, "corte": manifiesto["corte"],
                            "git_commit": git_commit, "epoca_bloqueo": str(epoca), "criterio": criterio,
                            "origen": origen, **{k: str(v) for k, v in etiquetas.items()}}
            for k, v in tags_version.items():
                cliente.set_model_version_tag(nombre, version, k, v)
            mlflow.set_tags({"version_registrada": version, "sha256_artefacto": sha_art,
                             "sha256_backtest": sha_backtest})

            t3, t4b = verificar_t3_t4b(cliente, tracking_uri, nombre, version, run_id, ruta_artefacto, sha_art)
            t = time.monotonic()
            with hilos_fijos(hilos):
                t4a = verificar_t4a(tracking_uri, run_id, huella, sha_backtest, evaluacion, config, modelos)
            mlflow.log_metric("segundos_t4a", time.monotonic() - t)
            cond = {"T1": True, **crit.condiciones, "T3": not t3, "T4a": not t4a, "T4b": not t4b}
            r2 = None
            if criterio == "candidato" and campeon is not None:
                r2 = _registrar_r2(mlflow, cliente, tmp, nombre, version, datos, config, hilos, campeon,
                                   campeon_p, r2_previo)
                cond["R2"] = r2["resultado"]
            fallas = ([f"T3: {t3}"] if t3 else []) + ([f"T4b: {t4b}"] if t4b else []) + \
                     ([f"T4a: {t4a[0]}"] if t4a else [])
            superada = all(cond.get(k, False) for k in REQUERIDOS[criterio])
            motivo = crit.motivo if not fallas else "; ".join(fallas)
            # Solo se baja de categoría: el hijo nunca escribe "valido" (§5.2)
            cliente.set_model_version_tag(nombre, version, "verificacion", "superada" if superada else "fallida")
            cliente.set_model_version_tag(nombre, version, "condiciones", json.dumps(cond, sort_keys=True))
            cliente.set_model_version_tag(nombre, version, "motivo", motivo)
            if not superada:
                cliente.set_model_version_tag(nombre, version, "estado", "invalido")
            mlflow.log_metric("segundos_total", time.monotonic() - t0)
            mlflow.log_dict({"condiciones": cond, "motivo": motivo}, "criterio.json")
            mlflow.set_tags({"verificacion": "superada" if superada else "fallida", "motivo": motivo})
            mlflow.end_run(status="FINISHED")
            return ResultadoRegistro(run_id, version, superada, motivo, cond, crit.resumen, criterio, epoca,
                                     sha_art, ruta_artefacto, huella, r2)
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
                cliente.set_tag(run_id, "verificacion", "fallida")
            except Exception:  # noqa: BLE001
                logger.error("no se pudo marcar la ejecución %s", run_id)
            if mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
            else:
                cliente.set_terminated(run_id, status="FAILED")
            raise


def registrar_version(motor, *, tracking_uri: str, nombre: str, git_commit: str, epoca: int,
                      comprobar: Callable[[], None], origen: str = "manual",
                      config: ConfigEntrenamiento = CONFIG_APROBADA, evaluacion: ConfigEvaluacion | None = None,
                      etiquetas: dict | None = None, campeon: dict | None = None) -> ResultadoRegistro:
    """Candidato desde PostgreSQL. Deja la versión ``pendiente`` o ``invalido``.
    ``campeon`` (solo reentrenamiento con vigente): decisión confirmada contra la que se
    calcula R2."""
    comprobar()
    return _registrar(datos_bd=desde_bd(motor), tracking_uri=tracking_uri, nombre=nombre, git_commit=git_commit,
                      epoca=epoca, origen=origen, criterio="candidato", config=config,
                      evaluacion=evaluacion or ConfigEvaluacion(), comprobar=comprobar, etiquetas=etiquetas or {},
                      campeon=campeon)


def registrar_respaldo(*, run_candidato: str, tracking_uri: str, nombre: str, git_commit: str, epoca: int,
                       comprobar: Callable[[], None], etiquetas: dict | None = None) -> ResultadoRegistro:
    """Baseline de respaldo (§6) sobre la copia **del candidato** descargada de MLflow y
    verificada por su huella, con la misma configuración salvo el modelo."""
    comprobar()
    _, cliente = _cliente(tracking_uri)
    run = cliente.get_run(run_candidato)
    versiones = cliente.search_model_versions(f"run_id='{run_candidato}'")
    if not versiones:
        raise ValueError(f"la ejecución {run_candidato} no registró una versión")
    vt = dict(versiones[0].tags or {})
    with tempfile.TemporaryDirectory(prefix="nm_respaldo_") as tmp:
        tmp = Path(tmp)
        datos, manifiesto = leer_copia(descargar_copia(tracking_uri, run_candidato, tmp / "copia"),
                                       run.data.tags["huella_datos"])
        tabla = descargar_backtest(tracking_uri, run_candidato, tmp / "bt", vt["sha256_backtest"])
    cfg, ev = config_desde_manifiesto(manifiesto)
    cfg_baseline = ConfigEntrenamiento(**{**asdict(cfg), "modelo": BASELINE})
    return _registrar(datos_bd=datos, tracking_uri=tracking_uri, nombre=nombre, git_commit=git_commit,
                      epoca=epoca, origen="arranque_respaldo", criterio="respaldo", config=cfg_baseline,
                      evaluacion=ev, comprobar=comprobar, tabla_candidato=tabla,
                      etiquetas={"tipo": "respaldo_baseline", "run_candidato": run_candidato,
                                 **(etiquetas or {})})


def califica_para_respaldo(r: ResultadoRegistro) -> bool:
    """§6: el candidato superó T1, T2, T3, D1, T4a y T4b, falló **solo** uno o más de
    C1–C3 y quedó registrado. Cualquier otra falla no activa el respaldo."""
    if r.criterio != "candidato" or r.version is None:
        return False
    if not all(r.condiciones.get(k, False) for k in TECNICOS):
        return False
    fallas = {k for k, v in r.condiciones.items() if not v and k != "R2"}     # R2 (etapa 7) no cuenta
    return bool(fallas) and fallas <= set(COMERCIALES)


# ------------------------------------------------------------------ R2 (etapa 7)
CAMPOS_CAMPEON = ("decision_id", "secuencia", "version", "run_id", "ruta_artefacto", "sha256_artefacto", "origen")


def _preparar_campeon(tracking_uri: str, campeon: dict, tmp: Path):
    """Antes de la copia: descarga y verifica el campeón. Devuelve ``(pronosticador,
    corte_v, None)`` o ``(None, None, ResultadoR2 no_disponible)``."""
    from app.modelo.r2 import CampeonNoDisponible, no_disponible, obtener_campeon
    try:
        p, corte_v = obtener_campeon(tracking_uri, campeon, tmp / "r2_campeon")
        return p, corte_v.strftime("%Y-%m-%d"), None
    except CampeonNoDisponible as exc:
        logger.warning("R2 no disponible: %s", exc)
        return None, None, no_disponible(str(exc))


def _registrar_r2(mlflow, cliente, tmp: Path, nombre: str, version: str, datos: DatosEntrenamiento,
                  config: ConfigEntrenamiento, hilos: int, campeon: dict, campeon_p, r2_previo) -> dict:
    from app.modelo.r2 import ARCHIVO_R2, RUTA_R2, calcular
    t = time.monotonic()
    if r2_previo is not None:
        r2 = r2_previo
    else:
        try:
            with hilos_fijos(hilos):
                r2 = calcular(datos, campeon_p, pd.Timestamp(campeon_p.info["corte"]), config)
        except ValueError as exc:
            from app.modelo.r2 import ResultadoR2
            r2 = ResultadoR2(False, f"R2 falla cerrado: {exc}")
    mlflow.log_metric("segundos_r2", time.monotonic() - t)
    tabla = r2.tabla_csv()
    (tmp / ARCHIVO_R2).write_bytes(tabla)
    mlflow.log_artifact(str(tmp / ARCHIVO_R2), RUTA_R2)
    sha_r2 = sha256(tabla)
    for k, v in r2.metricas().items():
        mlflow.log_metric(k, v)
    resultado = r2.resultado if isinstance(r2.resultado, str) else bool(r2.resultado)
    etiquetas = {"sha256_r2": sha_r2, "r2": json.dumps(resultado), "r2_motivo": r2.motivo[:500],
                 "r2_decision_id": str(campeon.get("decision_id"))}
    for k, v in etiquetas.items():
        cliente.set_model_version_tag(nombre, version, k, v)
    mlflow.set_tags(etiquetas)
    return {"resultado": resultado, "motivo": r2.motivo, "decision_id": campeon.get("decision_id"),
            "secuencia": campeon.get("secuencia"), "version": campeon.get("version"), "corte_v": r2.corte_v,
            "costo_retador": r2.costo_retador, "costo_vigente": r2.costo_vigente, "dias": r2.dias,
            "filas": r2.filas, "sha256_r2": sha_r2}


def resumen_json(r: ResultadoRegistro) -> str:
    return json.dumps({"run_id": r.run_id, "version": r.version, "verificacion": r.verificacion,
                       "motivo": r.motivo, "condiciones": r.condiciones}, ensure_ascii=False)


__all__ = ["EXPERIMENTO", "MODOS_PROMOVER", "ResultadoRegistro", "registrar_version", "registrar_respaldo",
           "califica_para_respaldo"]
