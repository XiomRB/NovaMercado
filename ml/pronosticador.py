"""Procedimiento de entrenamiento compartido por producción y backtesting.

Dada una fecha de corte C (se entrena con datos de fechas < C):

1. ``modelo_cal`` se ajusta con fechas < C − V y predice la ventana [C − V, C).
   Esas predicciones son **fuera de muestra**: sus residuos calibran el cuantil τ.
2. ``modelo_final`` se ajusta con todas las fechas < C (incluye lo más reciente).
3. La cantidad sugerida = calibrador(τ por familia) aplicado a la predicción.

Ningún paso usa datos con fecha >= C. El backtesting llama exactamente a
:func:`ajustar_con_calibracion` y :func:`sugerir`, igual que producción.

Reentrenamiento con observaciones: las **ventas** (histórico + observaciones)
alimentan todas las variables históricas; los **ejemplos de entrenamiento** son
solo las filas con promoción conocida (histórico o solicitud de predicción
registrada; ver :func:`ml.datos.consolidar_promociones`). Una observación sin
promoción no se usa como ejemplo, pero su venta sí cuenta en los rezagos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ml.calendario import Calendario
from ml.costos import CalibradorCuantil, a_entero_no_negativo, cantidad_redondeo, taus
from ml.datos import discrepancias_promocion, filas_supervisadas
from ml.features import construir_features
from ml.modelos import ModeloHGB, crear_modelo

ESTRATEGIAS = {
    "redondeo": None,  # referencia: ignora los costos
    "mult_familia": dict(modo="multiplicativo", grupos=("familia_id",)),
    "mult_tipo_familia": dict(modo="multiplicativo", grupos=("tipo", "familia_id")),
    "raiz_familia": dict(modo="raiz", grupos=("familia_id",)),
    "raiz_tipo_familia": dict(modo="raiz", grupos=("tipo", "familia_id")),
}


@dataclass
class ConfigEntrenamiento:
    modelo: str = "hgb_relativo"
    estrategia: str = "mult_familia"
    dias_calibracion: int = 56
    dias_minimos_historia: int = 28
    semilla: int = 0
    params_hgb: dict = field(default_factory=dict)


def preparar(ventas: pd.DataFrame, promociones: pd.DataFrame, tiendas: pd.DataFrame,
             calendario: Calendario, familias: list[str]) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Variables y objetivo de cada ejemplo supervisado (venta y promoción conocidas).

    Las variables se calculan con **todas** las ``ventas``, incluidas las que no
    tienen promoción. Devuelve ``(X, y, filas_sin_promocion)``.
    """
    sup, sin_promo = filas_supervisadas(ventas, promociones)
    X = construir_features(ventas, sup, tiendas, calendario, familias)
    return X, sup["ventas"].to_numpy(float), sin_promo


def ajustar_con_calibracion(X: pd.DataFrame, y: np.ndarray, corte: pd.Timestamp,
                            config: ConfigEntrenamiento):
    corte = pd.Timestamp(corte)
    inicio = X["fecha"].min() + pd.Timedelta(days=config.dias_minimos_historia)
    limite_cal = corte - pd.Timedelta(days=config.dias_calibracion)
    entren_cal = ((X["fecha"] >= inicio) & (X["fecha"] < limite_cal)).to_numpy()
    ventana_cal = ((X["fecha"] >= limite_cal) & (X["fecha"] < corte)).to_numpy()
    entren = ((X["fecha"] >= inicio) & (X["fecha"] < corte)).to_numpy()
    if not entren_cal.any() or not ventana_cal.any():
        raise ValueError("historia insuficiente para entrenar y calibrar antes del corte")

    m_cal = crear_modelo(config.modelo, config.semilla, config.params_hgb).ajustar(X[entren_cal], y[entren_cal])
    Xc = X[ventana_cal]
    residuos = Xc[["fecha", "tienda_id", "familia_id", "tipo"]].assign(
        real=y[ventana_cal], pred=m_cal.predecir(Xc))
    m_final = crear_modelo(config.modelo, config.semilla, config.params_hgb).ajustar(X[entren], y[entren])
    info = {
        "corte": corte,
        "fecha_max_entrenamiento": X.loc[entren, "fecha"].max(),
        "fecha_max_entrenamiento_cal": X.loc[entren_cal, "fecha"].max(),
        "ventana_calibracion": (Xc["fecha"].min(), Xc["fecha"].max()),
        "filas_entrenamiento": int(entren.sum()),
    }
    return m_final, residuos, info


def crear_calibrador(estrategia: str, residuos: pd.DataFrame, taus_familia: dict[str, float]):
    cfg = ESTRATEGIAS[estrategia]
    if cfg is None:
        return None
    return CalibradorCuantil(**cfg).ajustar(residuos, taus_familia)


def sugerir(calibrador, pred: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    if calibrador is None:
        return cantidad_redondeo(pred)
    return calibrador.sugerir(pred, X)


class CuantilesHGB:
    """Alternativa sin calibrador: un HGB con pérdida cuantílica por cada τ distinto."""

    def __init__(self, taus_familia: dict[str, float], semilla: int = 0, params: dict | None = None):
        self.taus = taus_familia
        self.semilla, self.params = semilla, params
        self.modelos: dict[float, ModeloHGB] = {}

    def ajustar(self, X, y) -> "CuantilesHGB":
        for t in sorted(set(round(v, 6) for v in self.taus.values())):
            self.modelos[t] = ModeloHGB("relativo", loss="quantile", quantile=t,
                                        semilla=self.semilla, params=self.params).ajustar(X, y)
        return self

    def sugerir(self, X: pd.DataFrame) -> np.ndarray:
        t = X["familia_id"].map(lambda f: round(self.taus[f], 6)).to_numpy()
        s = np.zeros(len(X))
        for valor, m in self.modelos.items():
            sel = t == valor
            if sel.any():
                s[sel] = m.predecir(X[sel])
        return a_entero_no_negativo(s)


@dataclass
class Pronosticador:
    """Artefacto entrenado: modelo + calibrador + metadatos. Es lo que se registrará en MLflow."""

    config: ConfigEntrenamiento
    modelo: object
    calibrador: CalibradorCuantil | None
    calendario: Calendario
    tiendas: pd.DataFrame
    familias: list[str]
    maximos_promocion: dict[str, int]
    info: dict

    @classmethod
    def entrenar(cls, ventas: pd.DataFrame, promociones: pd.DataFrame, tiendas: pd.DataFrame,
                 feriados: pd.DataFrame, costos: pd.DataFrame, familias: list[str],
                 fecha_corte: date | pd.Timestamp,
                 config: ConfigEntrenamiento | None = None) -> "Pronosticador":
        """``ventas``: ventas reales (histórico + observaciones). ``promociones``: una por
        clave (:func:`ml.datos.consolidar_promociones`). Solo se usan fechas < ``fecha_corte``."""
        config = config or ConfigEntrenamiento()
        corte = pd.Timestamp(fecha_corte)
        cal = Calendario(feriados)
        v = ventas[pd.to_datetime(ventas["fecha"]) < corte]
        p = promociones[pd.to_datetime(promociones["fecha"]) < corte]
        X, y, sin_promo = preparar(v, p, tiendas, cal, familias)
        modelo, residuos, info = ajustar_con_calibracion(X, y, corte, config)
        calibrador = crear_calibrador(config.estrategia, residuos, taus(costos))
        # Máximo de promociones por familia en los ejemplos de entrenamiento (< corte)
        maximos = {f: int(m) for f, m in X.groupby("familia_id")["en_promocion"].max().items()}
        info.update(filas_sin_promocion=int(sin_promo), filas_ventas=int(len(v)),
                    maximos_promocion=maximos,
                    discrepancias_promocion=int(len(discrepancias_promocion(p))))
        return cls(config, modelo, calibrador, cal, tiendas.copy(), list(familias), maximos, info)

    def predecir(self, objetivos: pd.DataFrame, ventas: pd.DataFrame) -> pd.DataFrame:
        """``objetivos``: fecha, tienda_id, familia_id, en_promocion. ``ventas``: ventas
        reales conocidas (solo se usan las anteriores a cada fecha objetivo)."""
        X = construir_features(ventas, objetivos, self.tiendas, self.calendario, self.familias)
        pred = self.modelo.predecir(X)
        out = X[["fecha", "tienda_id", "familia_id"]].copy()
        out["venta_esperada"] = pred
        out["cantidad_sugerida"] = sugerir(self.calibrador, pred, X)
        return out
