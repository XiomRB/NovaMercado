"""Lectura y validación de una carpeta con el formato de raw/ (generada en tmp_path)."""

import pandas as pd
import pytest

from ml.datos import COLUMNAS, cargar_raw
from tests.ml.sinteticos import escribir_carpeta, generar


@pytest.fixture
def carpeta(tmp_path):
    escribir_carpeta(generar(dias=40), tmp_path)
    return tmp_path


def test_carga_y_tipos(carpeta):
    d = cargar_raw(carpeta)
    assert pd.api.types.is_datetime64_any_dtype(d.historico["fecha"])
    assert pd.api.types.is_datetime64_any_dtype(d.feriados["fecha"])
    assert pd.api.types.is_datetime64_any_dtype(d.tiendas["fecha_apertura"])
    assert list(d.historico.columns) == COLUMNAS["historico_ventas.csv"]


def test_falta_columna(carpeta):
    pd.read_csv(carpeta / "costos.csv").drop(columns="costo_sobrante").to_csv(carpeta / "costos.csv", index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        cargar_raw(carpeta)


@pytest.mark.parametrize("dano, mensaje", [
    (lambda h: pd.concat([h, h.head(1)]), "duplicadas"),
    (lambda h: h.assign(ventas=h["ventas"].where(h.index != 0, -1)), "negativas"),
])
def test_historico_invalido(carpeta, dano, mensaje):
    h = pd.read_csv(carpeta / "historico_ventas.csv")
    dano(h).to_csv(carpeta / "historico_ventas.csv", index=False)
    with pytest.raises(ValueError, match=mensaje):
        cargar_raw(carpeta)


def test_costos_deben_cubrir_las_familias(carpeta):
    pd.read_csv(carpeta / "costos.csv").head(2).to_csv(carpeta / "costos.csv", index=False)
    with pytest.raises(ValueError, match="mismas familias"):
        cargar_raw(carpeta)
