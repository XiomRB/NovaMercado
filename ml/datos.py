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


# ---------------------------------------------------------------- reentrenamiento
CLAVE = ["fecha", "tienda_id", "familia_id"]


def _exigir_clave_unica(df: pd.DataFrame, nombre: str) -> None:
    if df.duplicated(CLAVE).any():
        raise ValueError(f"{nombre}: hay claves (fecha, tienda_id, familia_id) duplicadas")


def consolidar_promociones(historicas: pd.DataFrame, solicitudes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una promoción por (fecha, tienda, familia) para entrenar.

    * ``historicas``: ``en_promocion`` de ``historico_ventas.csv`` (dato oficial;
      tiene prioridad).
    * ``solicitudes``: promociones recibidas en ``POST /v1/prediccion``, con
      ``registrado_en`` y ``promocion_supuesta``. Se descartan las supuestas (no
      son dato real) y, si una tienda pidió varias veces el mismo día, se usa la
      última solicitud registrada.

    Las observaciones de ventas reales no traen promoción: una fecha sin
    promoción conocida queda fuera de esta tabla.
    """
    h = historicas[CLAVE + ["en_promocion"]].copy()
    h["fecha"] = pd.to_datetime(h["fecha"])
    _exigir_clave_unica(h, "promociones históricas")
    h["origen"] = "historico"
    if solicitudes is None or solicitudes.empty:
        return h.reset_index(drop=True)
    s = solicitudes.copy()
    s["fecha"] = pd.to_datetime(s["fecha"])
    if "promocion_supuesta" in s:
        s = s[~s["promocion_supuesta"].astype(bool)]
    orden = ["registrado_en"] if "registrado_en" in s else []
    s = s.sort_values(orden + CLAVE, kind="stable").drop_duplicates(CLAVE, keep="last")
    s = s[CLAVE + ["en_promocion"]].assign(origen="solicitud")
    s = s.merge(h[CLAVE], on=CLAVE, how="left", indicator=True)
    s = s[s["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([h, s], ignore_index=True)


def filas_supervisadas(ventas: pd.DataFrame, promociones: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Filas con venta real **y** promoción conocida: las únicas que se usan como
    ejemplos de entrenamiento. Devuelve también cuántas ventas quedaron fuera
    (siguen sirviendo para las variables históricas)."""
    v = ventas[CLAVE + ["ventas"]].copy()
    v["fecha"] = pd.to_datetime(v["fecha"])
    p = promociones[CLAVE + ["en_promocion"]].copy()
    p["fecha"] = pd.to_datetime(p["fecha"])
    _exigir_clave_unica(v, "ventas")
    _exigir_clave_unica(p, "promociones")
    if v["ventas"].isna().any() or (v["ventas"] < 0).any():
        raise ValueError("ventas: hay valores nulos o negativos")
    sup = v.merge(p, on=CLAVE, how="inner").sort_values(CLAVE).reset_index(drop=True)
    return sup, len(v) - len(sup)
