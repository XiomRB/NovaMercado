"""Evaluación del candidato frente al baseline y criterio mínimo de validez (M6).

* Backtest con origen móvil de la etapa 2 (``ml.backtest.ejecutar_backtest``,
  sin fuga): K cortes consecutivos de ``horizonte`` días que terminan en la última
  fecha con venta y promoción reales; candidato (``hgb_relativo + mult_familia``)
  y baseline (``baseline_dow4 + mult_familia``) sobre las mismas filas.
* Criterio (todas): T2 salidas dentro del contrato; D1 al menos ``minimo_cortes``
  cortes evaluables; C1 costo total ≤ (1 − mejora_minima) × baseline; C2 gana en
  ≥ ⌈fracción × n⌉ cortes; C3 ningún corte supera al baseline en más de
  ``perdida_maxima_corte``.
* Comparación de métricas reproducidas: enteros exactos; punto flotante con
  ``math.isclose(rel_tol=1e-9, abs_tol=1e-9)``.
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.modelo.copia_datos import DatosEntrenamiento
from ml.backtest import cortes_consecutivos, ejecutar_backtest
from ml.datos import CLAVE, DatosBase, filas_supervisadas
from ml.metricas import agregar_costo, resumir
from ml.pronosticador import ConfigEntrenamiento

CANDIDATO = "hgb_relativo"
BASELINE = "baseline_dow4"
ESTRATEGIA = "mult_familia"
TOL_REL = 1e-9
TOL_ABS = 1e-9
COLUMNAS_ENTERAS = ("n", "unidades_faltantes", "unidades_sobrantes")
COLUMNAS_REALES = ("costo_total", "costo_por_dia_tienda", "mae", "rmse", "wape", "sesgo", "tasa_quiebre")


@dataclass(frozen=True)
class ConfigEvaluacion:
    cortes: int = 7
    horizonte: int = 28
    minimo_cortes: int = 3
    mejora_minima: float = 0.05
    fraccion_cortes_ganados: float = 2 / 3
    perdida_maxima_corte: float = 0.10

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoEvaluacion:
    cortes: list[pd.Timestamp]
    por_corte: pd.DataFrame          # corte, modelo, métricas
    salidas_validas: bool            # T2
    detalle_salidas: str = ""

    def tabla_csv(self) -> bytes:
        d = self.por_corte.copy()
        d["corte"] = pd.to_datetime(d["corte"]).dt.strftime("%Y-%m-%d")
        return d.to_csv(index=False, lineterminator="\n").encode("utf-8")


@dataclass
class ResultadoCriterio:
    valido: bool
    condiciones: dict[str, bool]
    motivo: str
    resumen: dict = field(default_factory=dict)


def cortes_evaluables(datos: DatosEntrenamiento, ev: ConfigEvaluacion, cfg: ConfigEntrenamiento) -> list:
    sup, _ = filas_supervisadas(datos.ventas, datos.promociones)
    if sup.empty:
        return []
    cortes = cortes_consecutivos(sup["fecha"].max(), ev.cortes, ev.horizonte)
    # cada corte necesita historia previa para entrenar (mínima) y calibrar (ventana)
    minimo = sup["fecha"].min() + pd.Timedelta(days=cfg.dias_minimos_historia + cfg.dias_calibracion)
    return [c for c in cortes if c > minimo]


def evaluar(datos: DatosEntrenamiento, ev: ConfigEvaluacion, cfg: ConfigEntrenamiento) -> ResultadoEvaluacion:
    cortes = cortes_evaluables(datos, ev, cfg)
    if len(cortes) < ev.minimo_cortes:
        return ResultadoEvaluacion(cortes, pd.DataFrame(), True, "")
    base = DatosBase(datos.ventas[CLAVE + ["ventas"]], datos.tiendas, datos.familias, datos.costos, datos.feriados)
    r = ejecutar_backtest(base, cortes, ev.horizonte, modelos=(BASELINE, CANDIDATO), estrategias=(ESTRATEGIA,),
                          cuantil_directo=False, config=cfg, promociones=datos.promociones)
    p = r.predicciones
    venta, cant = p["venta_esperada"].to_numpy(float), p["cantidad_sugerida"].to_numpy()
    validas = bool(np.isfinite(venta).all() and (venta >= 0).all() and cant.dtype.kind in "iu" and (cant >= 0).all())
    d = agregar_costo(p, datos.costos)
    por_corte = resumir(d, ["corte"]).drop(columns="estrategia").sort_values(["corte", "modelo"])
    por_corte[list(COLUMNAS_ENTERAS)] = por_corte[list(COLUMNAS_ENTERAS)].astype(int)
    detalle = "" if validas else "el backtest produjo predicciones fuera del contrato"
    return ResultadoEvaluacion(cortes, por_corte.reset_index(drop=True), validas, detalle)


def evaluar_criterio(res: ResultadoEvaluacion, ev: ConfigEvaluacion) -> ResultadoCriterio:
    n = len(res.cortes)
    c = {"T2": res.salidas_validas, "D1": n >= ev.minimo_cortes}
    if not c["D1"]:
        return ResultadoCriterio(False, c, f"D1: solo {n} cortes evaluables (mínimo {ev.minimo_cortes})",
                                 {"n_cortes": n})
    t = res.por_corte.pivot(index="corte", columns="modelo", values="costo_total")
    cand, base = t[CANDIDATO], t[BASELINE]
    total_c, total_b = float(cand.sum()), float(base.sum())
    ganados = int((cand < base).sum())
    requeridos = math.ceil(ev.fraccion_cortes_ganados * n - 1e-12)
    peor = float((cand / base - 1).max())
    c["C1"] = total_c <= (1 - ev.mejora_minima) * total_b
    c["C2"] = ganados >= requeridos
    c["C3"] = peor <= ev.perdida_maxima_corte
    resumen = {"n_cortes": n, "costo_total_candidato": total_c, "costo_total_baseline": total_b,
               "mejora_pct": 100 * (1 - total_c / total_b), "cortes_ganados": ganados,
               "cortes_requeridos": requeridos, "peor_corte_pct": 100 * peor}
    fallas = [k for k, v in c.items() if not v]
    if fallas:
        motivo = "no cumple " + ", ".join(fallas) + (
            f" (mejora {resumen['mejora_pct']:.1f} %, gana {ganados}/{n}, peor corte {100 * peor:+.1f} %)")
    else:
        motivo = (f"costo {total_c:,.0f} vs {total_b:,.0f} del baseline ({resumen['mejora_pct']:.1f} % menos); "
                  f"gana {ganados}/{n} cortes; peor corte {100 * peor:+.1f} %")
    return ResultadoCriterio(not fallas, c, motivo, resumen)


def leer_tabla_csv(datos: bytes) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(datos))
    d["corte"] = pd.to_datetime(d["corte"])
    return d


def comparar_metricas(registrada: pd.DataFrame, reproducida: pd.DataFrame) -> list[str]:
    """Diferencias entre dos tablas por corte y modelo: enteros exactos, reales con tolerancia."""
    a = registrada.sort_values(["corte", "modelo"]).reset_index(drop=True)
    b = reproducida.sort_values(["corte", "modelo"]).reset_index(drop=True)
    if len(a) != len(b) or list(a[["corte", "modelo"]].astype(str).itertuples(index=False)) != \
            list(b[["corte", "modelo"]].astype(str).itertuples(index=False)):
        return ["los cortes o modelos evaluados no coinciden"]
    difs = []
    for col in COLUMNAS_ENTERAS:
        for i in np.flatnonzero(a[col].to_numpy() != b[col].to_numpy()):
            difs.append(f"{a.at[i, 'modelo']} {a.at[i, 'corte'].date()} {col}: {a.at[i, col]} != {b.at[i, col]}")
    for col in COLUMNAS_REALES:
        for i in range(len(a)):
            x, y = float(a.at[i, col]), float(b.at[i, col])
            if not (math.isclose(x, y, rel_tol=TOL_REL, abs_tol=TOL_ABS) or (math.isnan(x) and math.isnan(y))):
                difs.append(f"{a.at[i, 'modelo']} {a.at[i, 'corte'].date()} {col}: {x!r} != {y!r}")
    return difs
