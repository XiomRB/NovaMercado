"""Métricas técnicas (de la venta esperada) y de negocio (de la cantidad sugerida)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.costos import costo


def agregar_costo(pred: pd.DataFrame, costos: pd.DataFrame) -> pd.DataFrame:
    c = costos.set_index("familia_id")
    d = pred.copy()
    d["costo"] = costo(d["real"], d["cantidad_sugerida"],
                       d["familia_id"].map(c["costo_faltante"]), d["familia_id"].map(c["costo_sobrante"]))
    d["error"] = d["venta_esperada"] - d["real"]
    d["faltante"] = np.maximum(d["real"] - d["cantidad_sugerida"], 0)
    d["sobrante"] = np.maximum(d["cantidad_sugerida"] - d["real"], 0)
    return d


def _resumen(g: pd.DataFrame) -> pd.Series:
    real = g["real"].sum()
    return pd.Series({
        "n": len(g),
        "costo_total": g["costo"].sum(),
        "costo_por_dia_tienda": g["costo"].sum() / max(g.groupby(["fecha", "tienda_id"]).ngroups, 1),
        "mae": g["error"].abs().mean(),
        "rmse": float(np.sqrt((g["error"] ** 2).mean())),
        "wape": g["error"].abs().sum() / real if real else np.nan,
        "sesgo": g["error"].sum() / real if real else np.nan,
        "tasa_quiebre": (g["faltante"] > 0).mean(),
        "unidades_faltantes": g["faltante"].sum(),
        "unidades_sobrantes": g["sobrante"].sum(),
    })


def resumir(d: pd.DataFrame, por: list[str] | None = None) -> pd.DataFrame:
    """``d`` = salida de :func:`agregar_costo`. Agrupa por modelo, estrategia y ``por``."""
    claves = ["modelo", "estrategia", *(por or [])]
    return d.groupby(claves).apply(_resumen).reset_index()
