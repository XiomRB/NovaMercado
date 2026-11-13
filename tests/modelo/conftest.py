"""Artefactos y copias de datos sintéticos para las pruebas de la etapa 5 (sin MLflow ni base)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.modelo.artefacto import sha256_bytes
from app.modelo.copia_datos import DatosEntrenamiento
from app.modelo.entrenamiento import empaquetar
from ml.datos import COLUMNAS_SOLICITUDES, consolidar_promociones, consolidar_ventas
from ml.pronosticador import ConfigEntrenamiento, Pronosticador
from tests.ml.sinteticos import generar_contrato

CONFIG_RAPIDA = ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=40))
D = generar_contrato()


def datos_entrenamiento(historico=None) -> DatosEntrenamiento:
    h = D.historico if historico is None else historico
    obs = h.iloc[0:0][["fecha", "tienda_id", "familia_id", "ventas"]]
    sol = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        COLUMNAS_SOLICITUDES, ["datetime64[ns]", int, str, int, bool, "datetime64[ns, UTC]", int])})
    return DatosEntrenamiento(consolidar_ventas(h), consolidar_promociones(h), obs, sol, D.tiendas, D.familias,
                              D.costos, D.feriados, h["fecha"].max() + pd.Timedelta(days=1))


def entrenar(datos: DatosEntrenamiento | None = None, corte=None):
    datos = datos or datos_entrenamiento()
    p = Pronosticador.entrenar(datos.ventas, datos.promociones, datos.tiendas, datos.feriados, datos.costos,
                               datos.familia_ids, corte or datos.corte, CONFIG_RAPIDA)
    return empaquetar(p)


@pytest.fixture(scope="session")
def artefacto():
    return entrenar()


@pytest.fixture(scope="session")
def artefacto_b():
    """Otro modelo válido (corte anterior): sirve como 'versión anterior'."""
    return entrenar(corte=D.historico["fecha"].max() - pd.Timedelta(days=6))


@pytest.fixture
def archivo(tmp_path, artefacto):
    ruta = tmp_path / "modelo.joblib"
    ruta.write_bytes(artefacto.artefacto)
    return ruta, sha256_bytes(artefacto.artefacto)
