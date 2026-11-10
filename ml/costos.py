"""Costo de negocio y cantidad sugerida (problema del vendedor de periódicos).

Costo por tienda, familia y día:  c(s, d) = cf·max(d − s, 0) + cs·max(s − d, 0)

Dividido entre (cf + cs) es la pérdida cuantílica (pinball) con τ = cf/(cf+cs):
    c(s, d)/(cf + cs) = τ·max(d − s, 0) + (1 − τ)·max(s − d, 0),
así que minimizar el costo equivale a minimizar esa pérdida y solo depende de τ.

Para una distribución F de la demanda, E[c(s, D)] es convexa en s y su
derivada por la derecha es (cf + cs)·(F(s) − τ): es negativa mientras F(s) < τ y
no negativa desde que F(s) ≥ τ. Por tanto:

* sobre los reales, el mínimo está en q = cuantil τ de F (para la distribución
  empírica de unas muestras: la menor muestra con F ≥ τ, método "inverted_cdf");
* como el costo es convexo, sobre los **enteros** el mínimo está en floor(q) o
  ceil(q); :func:`cantidad_optima` evalúa ambos y elige el de menor costo;
* si la demanda es entera (q entero) ambos coinciden y s* = menor entero con
  F(s) ≥ τ, la regla clásica del vendedor de periódicos.

``tests/ml/test_costos.py`` lo comprueba contra la búsqueda exhaustiva del
mínimo y contra el costo esperado exacto de una Poisson.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Tolerancia para evitar que errores de punto flotante (p. ej. 640.0000001)
# suban una unidad completa al aplicar ceil.
_EPS = 1e-9


def taus(costos: pd.DataFrame) -> dict[str, float]:
    """τ = cf / (cf + cs) por familia."""
    cf, cs = costos["costo_faltante"].astype(float), costos["costo_sobrante"].astype(float)
    if (cf <= 0).any() or (cs <= 0).any():
        raise ValueError("los costos deben ser positivos")
    return dict(zip(costos["familia_id"], cf / (cf + cs)))


def costo(real, sugerido, cf, cs) -> np.ndarray:
    real, sugerido = np.asarray(real, float), np.asarray(sugerido, float)
    return np.asarray(cf, float) * np.maximum(real - sugerido, 0) + \
        np.asarray(cs, float) * np.maximum(sugerido - real, 0)


def cuantil(muestras, tau: float) -> float:
    m = np.sort(np.asarray(muestras, float))
    if m.size == 0:
        raise ValueError("sin muestras")
    k = max(math.ceil(tau * m.size - _EPS), 1)   # menor k con k/n >= τ
    return float(m[k - 1])


def perdida_cuantil(s, muestras, tau: float) -> np.ndarray:
    """Pérdida media τ·(d−s)⁺ + (1−τ)·(s−d)⁺ = costo medio / (cf + cs).

    ``s`` de forma (m,) y ``muestras`` de forma (n,) o (m, n): devuelve (m,).
    """
    s = np.asarray(s, float)
    d = np.asarray(muestras, float)
    if d.ndim == 1:
        d = d[None, :]
    dif = d - s[:, None]
    return (tau * np.maximum(dif, 0) + (1 - tau) * np.maximum(-dif, 0)).mean(axis=1)


def _elegir_entero(q: np.ndarray, muestras: np.ndarray, tau: float) -> np.ndarray:
    """Entre floor(q) y ceil(q) (>= 0), el de menor pérdida empírica por fila."""
    lo = np.maximum(np.floor(q), 0)
    hi = np.maximum(np.ceil(q), 0)
    distinto = hi != lo
    s = lo.copy()
    if distinto.any():
        m = muestras if muestras.ndim == 1 else muestras[distinto]
        p_lo = perdida_cuantil(lo[distinto], m, tau)
        p_hi = perdida_cuantil(hi[distinto], m, tau)
        s[distinto] = np.where(p_hi < p_lo, hi[distinto], lo[distinto])
    return s.astype(int)


def cantidad_optima(muestras, tau: float) -> int:
    """Entero s >= 0 que minimiza el costo empírico de la demanda ``muestras``."""
    m = np.maximum(np.asarray(muestras, float), 0)
    return int(_elegir_entero(np.array([cuantil(m, tau)]), m, tau)[0])


def a_entero_no_negativo(x) -> np.ndarray:
    return np.maximum(np.ceil(np.asarray(x, float) - _EPS), 0).astype(int)


class CalibradorCuantil:
    """Convierte la venta esperada en la cantidad entera de menor costo esperado.

    Se ajusta con residuos **fuera de muestra** (predicciones de una ventana que
    el modelo no vio al entrenar). Para cada fila con venta esperada p, la
    demanda se representa con las muestras d_i = g(p, r_i) de su grupo:

    * ``multiplicativo``: r = real/pred,  d_i = p·r_i  (error proporcional al nivel);
    * ``raiz``: z = (real − pred)/sqrt(pred),  d_i = p + z_i·sqrt(p) (varianza ∝ nivel).

    La cantidad es el entero que minimiza el costo empírico de esas muestras
    con el τ de la familia (:func:`_elegir_entero` sobre el cuantil τ).

    ``grupos``: columnas que definen cada grupo de calibración (p. ej.
    ("tipo", "familia_id")). Un grupo con menos de ``min_muestras`` usa los
    residuos de la familia completa.
    """

    def __init__(self, modo: str = "multiplicativo", grupos: tuple[str, ...] = ("familia_id",),
                 min_muestras: int = 50, pred_minima: float = 1.0):
        if modo not in ("multiplicativo", "raiz"):
            raise ValueError(f"modo desconocido: {modo}")
        if "familia_id" not in grupos:
            grupos = (*grupos, "familia_id")
        self.modo = modo
        self.grupos = tuple(grupos)
        self.min_muestras = min_muestras
        self.pred_minima = pred_minima
        self.residuos: dict[tuple, np.ndarray] = {}
        self.residuos_familia: dict[str, np.ndarray] = {}
        self.taus: dict[str, float] = {}
        self.n_muestras = 0

    def _residuo(self, real: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if self.modo == "multiplicativo":
            return real / pred
        return (real - pred) / np.sqrt(pred)

    def _muestras(self, pred: np.ndarray, r: np.ndarray) -> np.ndarray:
        if self.modo == "multiplicativo":
            d = pred[:, None] * r[None, :]
        else:
            d = pred[:, None] + r[None, :] * np.sqrt(pred)[:, None]
        return np.maximum(d, 0)

    def ajustar(self, df: pd.DataFrame, taus_familia: dict[str, float]) -> "CalibradorCuantil":
        """``df``: columnas ``real``, ``pred`` y las de ``grupos``."""
        self.taus = dict(taus_familia)
        d = df[df["pred"] >= self.pred_minima].copy()
        d["r"] = self._residuo(d["real"].to_numpy(float), d["pred"].to_numpy(float))
        self.n_muestras = len(d)
        for fam, g in d.groupby("familia_id"):
            self.residuos_familia[fam] = np.sort(g["r"].to_numpy(float))
        if self.grupos != ("familia_id",):
            for clave, g in d.groupby(list(self.grupos)):
                if len(g) >= self.min_muestras:
                    self.residuos[clave] = np.sort(g["r"].to_numpy(float))
        faltan = set(self.taus) - set(self.residuos_familia)
        if faltan:
            raise ValueError(f"sin residuos de calibración para: {sorted(faltan)}")
        return self

    def _clave(self, fila: tuple) -> tuple:
        if self.grupos == ("familia_id",):
            return ("familia", fila[0])
        fam = fila[self.grupos.index("familia_id")]
        return ("grupo", fila) if fila in self.residuos else ("familia", fam)

    def _residuos_de(self, clave: tuple) -> np.ndarray:
        return self.residuos[clave[1]] if clave[0] == "grupo" else self.residuos_familia[clave[1]]

    def factor(self, df: pd.DataFrame) -> np.ndarray:
        """Cuantil τ del residuo de cada fila (diagnóstico)."""
        filas = df[list(self.grupos)].itertuples(index=False, name=None)
        fams = df["familia_id"].to_numpy()
        return np.array([cuantil(self._residuos_de(self._clave(f)), self.taus[fam])
                         for f, fam in zip(filas, fams)], float)

    def sugerir(self, pred: np.ndarray, df: pd.DataFrame, bloque: int = 2048) -> np.ndarray:
        pred = np.maximum(np.asarray(pred, float), 0.0)
        claves = [self._clave(f) for f in df[list(self.grupos)].itertuples(index=False, name=None)]
        fams = df["familia_id"].to_numpy()
        out = np.zeros(len(pred), dtype=int)
        indices: dict[tuple, list[int]] = {}
        for i, c in enumerate(claves):
            indices.setdefault(c, []).append(i)
        for clave, idx in indices.items():
            idx = np.asarray(idx)
            r = self._residuos_de(clave)
            tau = self.taus[fams[idx[0]]]
            for ini in range(0, len(idx), bloque):
                sel = idx[ini:ini + bloque]
                muestras = self._muestras(pred[sel], r)
                # el cuantil de g(p, r) es g(p, cuantil de r): g es creciente en r
                q = self._muestras(pred[sel], np.array([cuantil(r, tau)]))[:, 0]
                out[sel] = _elegir_entero(q, muestras, tau)
        return out


def cantidad_redondeo(pred) -> np.ndarray:
    """Referencia ingenua: sugerir la venta esperada redondeada (ignora los costos)."""
    return np.maximum(np.floor(np.asarray(pred, float) + 0.5), 0).astype(int)
