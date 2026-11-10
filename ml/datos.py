"""Carga y validación de los archivos publicados (``raw/``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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


COLUMNAS_SOLICITUDES = CLAVE + ["en_promocion", "promocion_supuesta", "registrado_en", "solicitud_id"]


def _ultima_solicitud_real(solicitudes: pd.DataFrame) -> pd.DataFrame:
    """Una fila por clave: la última solicitud con promoción **real** (no supuesta).

    "Última" = mayor ``registrado_en`` (momento en que el servicio registró la
    solicitud); empate -> mayor ``solicitud_id`` (identificador creciente asignado
    por la base de datos). No depende del orden de las filas de entrada.
    """
    faltan = [c for c in COLUMNAS_SOLICITUDES if c not in solicitudes.columns]
    if faltan:
        raise ValueError(f"solicitudes: faltan columnas {faltan}")
    s = solicitudes[COLUMNAS_SOLICITUDES].copy()
    if s[["registrado_en", "solicitud_id", "promocion_supuesta"]].isna().any().any():
        raise ValueError("solicitudes: registrado_en, solicitud_id y promocion_supuesta no admiten nulos")
    if s.duplicated(CLAVE + ["solicitud_id"]).any():
        raise ValueError("solicitudes: la misma solicitud trae dos veces una familia")
    s["fecha"] = pd.to_datetime(s["fecha"])
    s["registrado_en"] = pd.to_datetime(s["registrado_en"])
    s = s[~s["promocion_supuesta"].astype(bool)]
    s = s.sort_values(CLAVE + ["registrado_en", "solicitud_id"], kind="mergesort")
    return s.drop_duplicates(CLAVE, keep="last")


def consolidar_promociones(historicas: pd.DataFrame, solicitudes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una promoción por (fecha, tienda, familia) para entrenar.

    * ``historicas``: ``en_promocion`` de ``historico_ventas.csv``. Es la promoción
      **observada** registrada junto con la venta real, por lo que tiene prioridad.
    * ``solicitudes``: promociones recibidas en ``POST /v1/prediccion``
      (columnas :data:`COLUMNAS_SOLICITUDES`). Las ``promocion_supuesta`` se
      descartan (las completó la capa de calidad, no son dato) y, si hubo varias
      solicitudes para la misma clave, se usa la última (:func:`_ultima_solicitud_real`).

    Columnas de salida: ``en_promocion`` (valor usado), ``origen`` (historico |
    solicitud), ``en_promocion_solicitud`` (valor de la última solicitud real o
    NA) y ``discrepancia`` (True si histórico y solicitud existen y difieren).
    El valor de la solicitud **nunca se descarta en silencio**: queda en
    ``en_promocion_solicitud`` y la discrepancia se marca y se registra en el log.

    Las observaciones de ventas reales no traen promoción: una clave sin
    promoción conocida no aparece en la tabla.
    """
    h = historicas[CLAVE + ["en_promocion"]].copy()
    h["fecha"] = pd.to_datetime(h["fecha"])
    _exigir_clave_unica(h, "promociones históricas")
    h["origen"] = "historico"
    if solicitudes is None or solicitudes.empty:
        t = h.assign(en_promocion_solicitud=pd.array([pd.NA] * len(h), dtype="Int64"), discrepancia=False)
        return t.sort_values(CLAVE).reset_index(drop=True)
    sol = _ultima_solicitud_real(solicitudes)[CLAVE + ["en_promocion"]]
    sol = sol.rename(columns={"en_promocion": "en_promocion_solicitud"})

    t = h.merge(sol, on=CLAVE, how="outer")
    solo_sol = t["en_promocion"].isna()
    t.loc[solo_sol, "en_promocion"] = t.loc[solo_sol, "en_promocion_solicitud"]
    t.loc[solo_sol, "origen"] = "solicitud"
    t["en_promocion"] = t["en_promocion"].astype(int)
    t["en_promocion_solicitud"] = t["en_promocion_solicitud"].astype("Int64")
    t["discrepancia"] = (~solo_sol & t["en_promocion_solicitud"].notna()
                         & (t["en_promocion_solicitud"] != t["en_promocion"])).astype(bool)
    n = int(t["discrepancia"].sum())
    if n:
        logger.warning("promociones: %d claves con histórico y solicitud distintos; se usa el histórico", n)
    return t.sort_values(CLAVE).reset_index(drop=True)


def discrepancias_promocion(promociones: pd.DataFrame) -> pd.DataFrame:
    """Claves donde la promoción del histórico y la de la solicitud difieren."""
    if "discrepancia" not in promociones:
        return promociones.iloc[0:0]
    return promociones[promociones["discrepancia"]].reset_index(drop=True)


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
