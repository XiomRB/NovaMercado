"""Construcción de variables para predecir la venta de (tienda, familia, fecha F).

Garantía anti-fuga: toda variable derivada de ventas se calcula sobre la matriz
diaria de ventas **desplazada al menos 1 día** (``shift(k)``, k >= 1). Por lo
tanto, la fila de fecha F solo ve ventas con fecha <= F-1, aunque ``ventas``
contenga datos de F o posteriores (como ocurre en el backtesting). La misma
función se usa para entrenar, para el backtesting y en producción.

Información que usa el modelo al predecir F:
* ventas reales de días anteriores a F (histórico + observaciones);
* ``en_promocion`` del propio día F (llega en la solicitud);
* calendario y eventos de ``feriados.csv`` para F, F-1 y F+1 (conocidos de antemano);
* atributos fijos de la tienda (tipo) y la familia.

Nunca se usan promociones de días pasados: las observaciones de ventas reales
no las incluyen, así que en producción no estarían disponibles de forma uniforme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.calendario import COLUMNAS_CALENDARIO, Calendario

LAGS = (1, 2, 3, 7, 14, 21, 28)
LAGS_DIA_SEMANA = (7, 14, 21, 28)
VENTANA_MAX = max(LAGS)

COLUMNAS_VENTAS = [f"lag_{k}" for k in LAGS] + [
    "media_dow_4", "media_7", "media_28", "std_28", "ultimo_valor",
]
COLUMNAS_CONTEO = ["n_obs_28", "n_dow_4", "dias_desde_ultimo"]
COLUMNAS_CATEGORICAS = ["tienda_id", "familia_cod", "tipo_cod", "ev_codigo"]
COLUMNAS_MODELO = (["en_promocion"] + COLUMNAS_VENTAS + COLUMNAS_CONTEO
                   + [c for c in COLUMNAS_CALENDARIO if c != "ev_codigo"] + COLUMNAS_CATEGORICAS)


def _matriz_diaria(ventas: pd.DataFrame, series: pd.MultiIndex, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    if len(ventas):
        wide = ventas.pivot_table(index="fecha", columns=["tienda_id", "familia_id"],
                                  values="ventas", aggfunc="first")
    else:
        wide = pd.DataFrame(index=pd.DatetimeIndex([], name="fecha"))
    return wide.reindex(index=fechas, columns=series).astype(float)


def construir_features(
    ventas: pd.DataFrame,
    objetivos: pd.DataFrame,
    tiendas: pd.DataFrame,
    calendario: Calendario,
    familias: list[str],
) -> pd.DataFrame:
    """Devuelve ``objetivos`` (mismo orden) con las variables del modelo.

    ``ventas``: fecha, tienda_id, familia_id, ventas (ventas reales conocidas).
    ``objetivos``: fecha, tienda_id, familia_id, en_promocion (filas a predecir).
    ``familias``: orden de familias del catálogo (define ``familia_cod`` estable).
    """
    obj = objetivos[["fecha", "tienda_id", "familia_id", "en_promocion"]].copy().reset_index(drop=True)
    obj["fecha"] = pd.to_datetime(obj["fecha"])
    v = ventas[["fecha", "tienda_id", "familia_id", "ventas"]].copy()
    v["fecha"] = pd.to_datetime(v["fecha"])

    series = pd.MultiIndex.from_frame(obj[["tienda_id", "familia_id"]].drop_duplicates()
                                      .sort_values(["tienda_id", "familia_id"]))
    inicio = obj["fecha"].min() - pd.Timedelta(days=VENTANA_MAX)
    if len(v):
        inicio = min(inicio, v["fecha"].min())
    fechas = pd.date_range(inicio, obj["fecha"].max(), freq="D")
    # Solo hacen falta ventas anteriores a la última fecha objetivo
    v = v[v["fecha"] < obj["fecha"].max()]
    wide = _matriz_diaria(v, series, fechas)

    previo = wide.shift(1)                       # fila F = venta de F-1
    calc: dict[str, pd.DataFrame] = {f"lag_{k}": wide.shift(k) for k in LAGS}
    calc["media_7"] = previo.rolling(7, min_periods=1).mean()
    calc["media_28"] = previo.rolling(28, min_periods=1).mean()
    calc["std_28"] = previo.rolling(28, min_periods=2).std()
    calc["n_obs_28"] = previo.notna().astype(float).rolling(28, min_periods=1).sum()
    dow = np.stack([calc[f"lag_{k}"].to_numpy() for k in LAGS_DIA_SEMANA])
    with np.errstate(all="ignore"):
        n_dow = np.sum(~np.isnan(dow), axis=0)
        media_dow = np.where(n_dow > 0, np.nansum(dow, axis=0) / np.maximum(n_dow, 1), np.nan)
    calc["media_dow_4"] = pd.DataFrame(media_dow, index=wide.index, columns=wide.columns)
    calc["n_dow_4"] = pd.DataFrame(n_dow.astype(float), index=wide.index, columns=wide.columns)
    calc["ultimo_valor"] = previo.ffill()
    ordinal = np.arange(len(fechas), dtype=float)[:, None]
    pos_ultimo = pd.DataFrame(np.where(wide.notna(), ordinal, np.nan),
                              index=wide.index, columns=wide.columns).shift(1).ffill()
    calc["dias_desde_ultimo"] = ordinal - pos_ultimo   # NaN si nunca hubo venta

    fila = fechas.get_indexer(obj["fecha"])
    col = series.get_indexer(pd.MultiIndex.from_frame(obj[["tienda_id", "familia_id"]]))
    for nombre, matriz in calc.items():
        obj[nombre] = matriz.to_numpy()[fila, col]

    t = tiendas[["tienda_id", "ciudad", "tipo"]]
    obj = obj.merge(t, how="left", on="tienda_id")
    if obj["ciudad"].isna().any():
        raise ValueError("hay tiendas objetivo que no están en el catálogo")
    cal = calendario.features(obj[["fecha", "ciudad"]])
    obj = pd.concat([obj, cal.drop(columns=["fecha", "ciudad"])], axis=1)

    obj["familia_cod"] = obj["familia_id"].map({f: i for i, f in enumerate(familias)})
    tipos = sorted(tiendas["tipo"].unique())
    obj["tipo_cod"] = obj["tipo"].map({t: i for i, t in enumerate(tipos)})
    obj["en_promocion"] = obj["en_promocion"].astype(float)
    return obj
