"""Modelos de venta esperada con una interfaz común: ``ajustar(X, y)`` y ``predecir(X)``.

``X`` es la salida de :func:`ml.features.construir_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.features import COLUMNAS_CATEGORICAS, COLUMNAS_MODELO, COLUMNAS_VENTAS

PARAMS_HGB = dict(
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,   # sin partición aleatoria interna: reproducible y sin mezclar fechas
)


class _Respaldo:
    """Nivel de respaldo cuando una serie no tiene historia: media reciente de la
    misma familia en tiendas del mismo tipo (y, si no hay, de la familia)."""

    def ajustar(self, X: pd.DataFrame, y: np.ndarray, dias: int = 28) -> "_Respaldo":
        d = X[["fecha", "familia_id", "tipo"]].assign(y=np.asarray(y, float))
        d = d[d["fecha"] > d["fecha"].max() - pd.Timedelta(days=dias)]
        self.tipo_familia = d.groupby(["tipo", "familia_id"])["y"].mean().to_dict()
        self.familia = d.groupby("familia_id")["y"].mean().to_dict()
        self.global_ = float(d["y"].mean()) if len(d) else 0.0
        return self

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        tf = [self.tipo_familia.get(k) for k in zip(X["tipo"], X["familia_id"])]
        fam = X["familia_id"].map(self.familia)
        out = pd.Series(tf, index=X.index, dtype=float).fillna(fam).fillna(self.global_)
        return out.to_numpy(float)


class BaselineDiaSemana:
    """Promedio de la venta del mismo día de la semana en las últimas 4 semanas.

    Respaldo (series con pocos datos): media de los últimos 28 días disponibles ->
    último valor -> media de la familia en tiendas del mismo tipo.
    """

    nombre = "baseline_dow4"

    def ajustar(self, X: pd.DataFrame, y) -> "BaselineDiaSemana":
        self.respaldo = _Respaldo().ajustar(X, y)
        return self

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        p = X["media_dow_4"].fillna(X["media_28"]).fillna(X["ultimo_valor"])
        p = p.fillna(pd.Series(self.respaldo.predecir(X), index=X.index))
        return np.maximum(p.to_numpy(float), 0.0)


class ModeloHGB:
    """HistGradientBoostingRegressor global (todas las tiendas y familias).

    * ``objetivo="directo"``: predice la venta con pérdida Poisson.
    * ``objetivo="relativo"``: predice venta / nivel, con nivel = media de los
      28 días previos; las variables de ventas también se expresan relativas al
      nivel. Es invariante a la escala de cada serie y se adapta a cambios de
      nivel de la demanda sin reentrenar.
    """

    def __init__(self, objetivo: str = "relativo", loss: str | None = None,
                 quantile: float | None = None, params: dict | None = None, semilla: int = 0):
        if objetivo not in ("directo", "relativo"):
            raise ValueError(objetivo)
        self.objetivo = objetivo
        self.loss = loss or ("poisson" if objetivo == "directo" else "squared_error")
        self.quantile = quantile
        self.params = {**PARAMS_HGB, **(params or {})}
        self.semilla = semilla
        self.nombre = f"hgb_{objetivo}"

    def _matriz(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        M = X[COLUMNAS_MODELO].astype(float).copy()
        nivel = X["media_28"].to_numpy(float)
        if self.objetivo == "relativo":
            with np.errstate(all="ignore"):
                for c in COLUMNAS_VENTAS:
                    M[c] = np.where(nivel > 0, M[c].to_numpy(float) / nivel, np.nan)
            M["nivel_log"] = np.log1p(nivel)
        return M, nivel

    def ajustar(self, X: pd.DataFrame, y) -> "ModeloHGB":
        y = np.asarray(y, float)
        self.respaldo = _Respaldo().ajustar(X, y)
        M, nivel = self._matriz(X)
        if self.objetivo == "relativo":
            ok = np.isfinite(nivel) & (nivel > 0)
            M, y = M[ok], y[ok] / nivel[ok]
        kwargs = dict(loss=self.loss, random_state=self.semilla,
                      categorical_features=[c in COLUMNAS_CATEGORICAS for c in M.columns], **self.params)
        if self.loss == "quantile":
            kwargs["quantile"] = self.quantile
        self.columnas = list(M.columns)
        self.modelo = HistGradientBoostingRegressor(**kwargs).fit(M, y)
        return self

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        M, nivel = self._matriz(X)
        p = self.modelo.predict(M[self.columnas])
        if self.objetivo == "relativo":
            ok = np.isfinite(nivel) & (nivel > 0)
            p = np.where(ok, p * np.where(ok, nivel, 0), self.respaldo.predecir(X))
        return np.maximum(p, 0.0)


def crear_modelo(nombre: str, semilla: int = 0, params: dict | None = None):
    if nombre == "baseline_dow4":
        return BaselineDiaSemana()
    if nombre == "hgb_directo":
        return ModeloHGB("directo", semilla=semilla, params=params)
    if nombre == "hgb_relativo":
        return ModeloHGB("relativo", semilla=semilla, params=params)
    raise ValueError(f"modelo desconocido: {nombre}")
