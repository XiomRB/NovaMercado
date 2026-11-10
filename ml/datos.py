"""Carga y validación de los archivos publicados (``raw/``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLUMNAS = {
    "historico_ventas.csv": ["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"],
    "tiendas.csv": ["tienda_id", "nombre", "ciudad", "departamento", "tipo", "fecha_apertura"],
    "familias.csv": ["familia_id", "nombre"],
    "costos.csv": ["familia_id", "costo_faltante", "costo_sobrante"],
    "feriados.csv": ["fecha", "nombre", "tipo", "ambito"],
}


@dataclass
class DatosBase:
    historico: pd.DataFrame   # fecha, tienda_id, familia_id, ventas, en_promocion
    tiendas: pd.DataFrame     # tienda_id, ciudad, tipo, fecha_apertura, ...
    familias: pd.DataFrame    # familia_id, nombre
    costos: pd.DataFrame      # familia_id, costo_faltante, costo_sobrante
    feriados: pd.DataFrame    # fecha, nombre, tipo, ambito


def _leer(carpeta: Path, archivo: str) -> pd.DataFrame:
    df = pd.read_csv(carpeta / archivo)
    faltan = [c for c in COLUMNAS[archivo] if c not in df.columns]
    if faltan:
        raise ValueError(f"{archivo}: faltan columnas {faltan}")
    return df


def cargar_raw(carpeta: str | Path) -> DatosBase:
    carpeta = Path(carpeta)
    historico = _leer(carpeta, "historico_ventas.csv")
    historico["fecha"] = pd.to_datetime(historico["fecha"], format="%Y-%m-%d")
    tiendas = _leer(carpeta, "tiendas.csv")
    tiendas["fecha_apertura"] = pd.to_datetime(tiendas["fecha_apertura"], format="%Y-%m-%d")
    familias = _leer(carpeta, "familias.csv")
    costos = _leer(carpeta, "costos.csv")
    feriados = _leer(carpeta, "feriados.csv")
    feriados["fecha"] = pd.to_datetime(feriados["fecha"], format="%Y-%m-%d")

    if historico[["fecha", "tienda_id", "familia_id"]].duplicated().any():
        raise ValueError("historico_ventas.csv tiene claves duplicadas")
    if (historico["ventas"] < 0).any() or historico["ventas"].isna().any():
        raise ValueError("historico_ventas.csv tiene ventas negativas o nulas")
    if set(costos["familia_id"]) != set(familias["familia_id"]):
        raise ValueError("costos.csv y familias.csv no cubren las mismas familias")
    return DatosBase(historico, tiendas, familias, costos, feriados)
