"""Backtesting temporal con origen móvil (rolling origin).

Para cada corte C: se entrena con fechas < C (ver :mod:`ml.pronosticador`) y se
predicen los días [C, C + horizonte) **sin reentrenar**, como en producción
entre dos reentrenamientos. Las variables de cada día objetivo F usan las
ventas reales de días < F (en operación, la observación de F−1 llega antes de la
solicitud de F). Todos los modelos y estrategias se evalúan sobre exactamente
las mismas filas.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.calendario import Calendario
from ml.costos import taus
from ml.datos import DatosBase
from ml.modelos import crear_modelo  # noqa: F401  (re-export útil en notebooks)
from ml.pronosticador import (ESTRATEGIAS, ConfigEntrenamiento, CuantilesHGB, ajustar_con_calibracion,
                              crear_calibrador, preparar, sugerir)


@dataclass
class ResultadoBacktest:
    predicciones: pd.DataFrame   # una fila por (corte, modelo, estrategia, fecha, tienda, familia)
    info: pd.DataFrame           # metadatos por (corte, modelo): fechas usadas para entrenar/calibrar


def cortes_consecutivos(fecha_fin: pd.Timestamp, n: int, horizonte: int) -> list[pd.Timestamp]:
    """``n`` cortes cuyas ventanas de prueba consecutivas terminan en ``fecha_fin`` (inclusive)."""
    ultimo = pd.Timestamp(fecha_fin) - pd.Timedelta(days=horizonte - 1)
    return [ultimo - pd.Timedelta(days=horizonte * i) for i in reversed(range(n))]


def ejecutar_backtest(datos: DatosBase, cortes: list, horizonte: int = 28,
                      modelos: tuple[str, ...] = ("baseline_dow4", "hgb_directo", "hgb_relativo"),
                      estrategias: tuple[str, ...] = tuple(ESTRATEGIAS),
                      cuantil_directo: bool = True,
                      config: ConfigEntrenamiento | None = None) -> ResultadoBacktest:
    config = config or ConfigEntrenamiento()
    cal = Calendario(datos.feriados)
    familias = list(datos.familias["familia_id"])
    t = taus(datos.costos)
    # Las variables de cada fila solo usan ventas anteriores a su fecha, así que se
    # pueden calcular una vez para todo el histórico (equivalencia probada en tests).
    X, y = preparar(datos.historico, datos.tiendas, cal, familias)

    filas, infos = [], []
    for corte in map(pd.Timestamp, cortes):
        prueba = ((X["fecha"] >= corte) & (X["fecha"] < corte + pd.Timedelta(days=horizonte))).to_numpy()
        Xp = X[prueba]
        base = Xp[["fecha", "tienda_id", "familia_id", "tipo"]].assign(real=y[prueba], corte=corte)
        for nombre in modelos:
            cfg = ConfigEntrenamiento(**{**config.__dict__, "modelo": nombre})
            modelo, residuos, info = ajustar_con_calibracion(X, y, corte, cfg)
            infos.append({"modelo": nombre, **info,
                          "fecha_min_prueba": Xp["fecha"].min(), "fecha_max_prueba": Xp["fecha"].max()})
            pred = modelo.predecir(Xp)
            for est in estrategias:
                calib = crear_calibrador(est, residuos, t)
                filas.append(base.assign(modelo=nombre, estrategia=est, venta_esperada=pred,
                                         cantidad_sugerida=sugerir(calib, pred, Xp)))
            if cuantil_directo and nombre == "hgb_relativo":
                entren = ((X["fecha"] >= X["fecha"].min() + pd.Timedelta(days=cfg.dias_minimos_historia))
                          & (X["fecha"] < corte)).to_numpy()
                cq = CuantilesHGB(t, cfg.semilla, cfg.params_hgb).ajustar(X[entren], y[entren])
                filas.append(base.assign(modelo=nombre, estrategia="cuantil_hgb", venta_esperada=pred,
                                         cantidad_sugerida=cq.sugerir(Xp)))
    return ResultadoBacktest(pd.concat(filas, ignore_index=True), pd.DataFrame(infos))
