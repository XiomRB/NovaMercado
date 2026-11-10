"""Aplicación de prueba: persistencia en memoria y un modelo real pequeño entrenado con
datos sintéticos del catálogo del contrato. Sin base de datos, sin MLflow, sin hilos."""

from __future__ import annotations

import io

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.dependencias import Dependencias, construir_monitor
from app.main import create_app
from app.modelo.entrenamiento import empaquetar
from app.modelo.gestor import GestorModelo
from app.modelo.proveedores import ModeloCargado, ModeloNoDisponible
from app.orquestador import OrquestadorNoConfigurado
from ml.pronosticador import ConfigEntrenamiento, Pronosticador
from tests.api.memoria import PersistenciaMemoria
from tests.ml.sinteticos import generar_contrato

CONFIG_RAPIDA = ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=40))
DATOS = generar_contrato()
FECHA = "2026-04-14"        # día siguiente al último dato sintético (= fecha de los ejemplos)


@pytest.fixture(scope="session")
def artefacto():
    h = DATOS.historico
    p = Pronosticador.entrenar(h, h, DATOS.tiendas, DATOS.feriados, DATOS.costos,
                               list(DATOS.familias["familia_id"]), pd.Timestamp(FECHA), CONFIG_RAPIDA)
    return empaquetar(p)


class ProveedorMemoria:
    """Proveedor de prueba: entrega el artefacto empaquetado (mismo formato que el archivo)."""

    alias = "local"

    def __init__(self, artefacto=None):
        self.artefacto = artefacto

    def huella(self):
        return None if self.artefacto is None else self.artefacto.version

    def obtener(self):
        if self.artefacto is None:
            raise ModeloNoDisponible("no hay un modelo disponible")
        a = joblib.load(io.BytesIO(self.artefacto.artefacto))
        return ModeloCargado(a["pronosticador"], a["nombre"], self.alias, a["version"])


def construir(artefacto=None, persistencia=None, requeridos=("api", "base_datos", "modelo")):
    persistencia = persistencia or PersistenciaMemoria(DATOS)
    gestor = GestorModelo(ProveedorMemoria(artefacto))
    gestor.refrescar()
    orq = OrquestadorNoConfigurado()
    monitor = construir_monitor(persistencia, gestor, orq, requeridos=requeridos, intervalo=1, limite=1)
    monitor.ejecutar_ciclo()
    return Dependencias(persistencia, gestor, monitor, orq)


@pytest.fixture
def deps(artefacto):
    d = construir(artefacto)
    yield d
    d.monitor.detener()


@pytest.fixture
def cliente(deps):
    with TestClient(create_app(deps, iniciar_hilos=False), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def cliente_sin_modelo():
    d = construir(None)
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        yield c, d
    d.monitor.detener()
