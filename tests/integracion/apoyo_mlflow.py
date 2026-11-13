"""Ayudantes de las pruebas de integración con MLflow (datos sintéticos, configuración rápida)."""

from __future__ import annotations

import io

import joblib
import pandas as pd
from mlflow import MlflowClient

from app.db.carga import cargar_datos_publicados
from app.modelo.evaluacion import ConfigEvaluacion
from app.modelo.registro import entrenar_y_registrar
from ml.pronosticador import ConfigEntrenamiento
from tests.ml.sinteticos import escribir_carpeta, generar_contrato

CONFIG = ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=40))
EVALUACION = ConfigEvaluacion(cortes=3, horizonte=14)
DATOS = generar_contrato()
FECHA = "2026-04-14"


def cargar(motor, carpeta, datos=DATOS):
    escribir_carpeta(datos, carpeta)
    cargar_datos_publicados(motor, carpeta)


def registrar(motor, uri, nombre, promover="si_no_hay", alias="vigente", config=CONFIG, evaluacion=EVALUACION):
    return entrenar_y_registrar(motor, tracking_uri=uri, nombre=nombre, alias=alias, promover=promover,
                                git_commit="commit-de-prueba", config=config, evaluacion=evaluacion)


def cliente(uri):
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def alias_actual(uri, nombre, alias="vigente"):
    try:
        return str(cliente(uri).get_model_version_by_alias(nombre, alias).version)
    except Exception:  # noqa: BLE001
        return None


def tags_version(uri, nombre, version):
    return dict(cliente(uri).get_model_version(nombre, version).tags)


def pronosticador_de(uri, nombre, version):
    """Descarga modelo.joblib de una versión (para comparar predicciones en las pruebas)."""
    from app.modelo.mlflow_util import descargar_de_ejecucion
    t = tags_version(uri, nombre, version)
    mv = cliente(uri).get_model_version(nombre, version)
    ruta = descargar_de_ejecucion(uri, mv.run_id, t["ruta_artefacto"])
    return joblib.load(io.BytesIO(open(ruta, "rb").read()))["pronosticador"]


def objetivos(tienda=5, fecha=FECHA):
    fams = list(DATOS.familias["familia_id"])
    return pd.DataFrame({"fecha": pd.Timestamp(fecha), "tienda_id": tienda, "familia_id": fams, "en_promocion": 0})
