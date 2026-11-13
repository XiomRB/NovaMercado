"""Ayudantes de las pruebas de integración con MLflow (datos sintéticos, configuración rápida)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import joblib
import pandas as pd
from mlflow import MlflowClient

from app.db.carga import cargar_datos_publicados
from app.db.promociones import leer_vigente
from app.modelo.bloqueo import ConcesionEntrenamiento
from app.modelo.evaluacion import ConfigEvaluacion
from app.modelo.promocion import ResultadoPromocion, _escribir_alias, autorizar_y_promover
from app.modelo.registro import ResultadoRegistro, registrar_version
from ml.pronosticador import ConfigEntrenamiento
from tests.ml.sinteticos import escribir_carpeta, generar_contrato

CONFIG = ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=40))
EVALUACION = ConfigEvaluacion(cortes=3, horizonte=14)
DATOS = generar_contrato()
FECHA = "2026-04-14"


def cargar(motor, carpeta, datos=DATOS):
    escribir_carpeta(datos, carpeta)
    cargar_datos_publicados(motor, carpeta)


def url_de(motor) -> str:
    return motor.url.render_as_string(hide_password=False)


@dataclass
class Registro:
    """Resultado combinado del hijo (registro) y del supervisor (autorización)."""
    r: ResultadoRegistro
    p: ResultadoPromocion

    run_id = property(lambda self: self.r.run_id)
    version = property(lambda self: self.r.version)
    condiciones = property(lambda self: self.r.condiciones)
    valido = property(lambda self: self.p.autorizada)
    promovido = property(lambda self: self.p.promovido)
    motivo = property(lambda self: self.p.motivo)


def registrar(motor, uri, nombre, promover="si_no_hay", alias="vigente", config=CONFIG, evaluacion=EVALUACION,
              origen="manual") -> Registro:
    """Mismo camino que el supervisor (concesión real, registro en pendiente y autorización)
    pero en el proceso de la prueba, para poder inyectar fallos con monkeypatch."""
    with ConcesionEntrenamiento(url_de(motor), modo="prueba") as c:
        r = registrar_version(motor, tracking_uri=uri, nombre=nombre, git_commit="commit-de-prueba",
                              epoca=c.epoca, comprobar=c.verificar, origen=origen, config=config,
                              evaluacion=evaluacion)
        p = autorizar_y_promover(c, cliente(uri), r, nombre=nombre, alias=alias, promover=promover, origen=origen)
    return Registro(r, p)


def vigente_pg(motor, nombre):
    with motor.connect() as conn:
        return leer_vigente(conn, nombre)


def promover_existente(motor, uri, nombre, version, alias="vigente"):
    """Confirma en PostgreSQL una versión ya válida (nueva decisión) y refleja el alias,
    como hace autorizar_y_promover. Solo para preparar escenarios de prueba."""
    mv = cliente(uri).get_model_version(nombre, str(version))
    t = dict(mv.tags)
    with ConcesionEntrenamiento(url_de(motor), modo="prueba") as c:
        d = c.insertar_decision(nombre=nombre, version=str(version), run_id=mv.run_id, ruta=t["ruta_artefacto"],
                                sha256=t["sha256_artefacto"], origen="prueba")
        assert _escribir_alias(cliente(uri), nombre, alias, str(version))[0] == "aplicada"
        return d, c.insertar_resolucion(d, "confirmada", "prueba")


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
