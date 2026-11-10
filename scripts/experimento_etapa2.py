"""Experimento de la etapa 2: backtesting temporal de modelos y estrategias de cantidad.

Uso:  python -m scripts.experimento_etapa2 --raw raw --salida reportes/etapa2

No registra nada en MLflow ni promueve modelos: solo produce tablas para revisión.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from ml.backtest import cortes_consecutivos, ejecutar_backtest
from ml.costos import taus
from ml.datos import cargar_raw
from ml.metricas import agregar_costo, resumir


def tabla_md(df: pd.DataFrame, decimales: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{decimales}f}" if abs(v) < 1000 else f"{v:,.0f}"
        if isinstance(v, pd.Timestamp):
            return v.date().isoformat()
        return str(v)
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--salida", default="reportes/etapa2")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--horizonte", type=int, default=28)
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    datos = cargar_raw(a.raw)
    fin = datos.historico["fecha"].max()
    cortes = cortes_consecutivos(fin, a.folds, a.horizonte)
    t0 = time.time()
    res = ejecutar_backtest(datos, cortes, a.horizonte)
    segundos = time.time() - t0

    d = agregar_costo(res.predicciones, datos.costos)
    glob = resumir(d).sort_values("costo_total").reset_index(drop=True)
    por_corte = resumir(d, ["corte"])
    por_tienda = resumir(d, ["tienda_id"])
    por_familia = resumir(d, ["familia_id"])
    for nombre, df in [("resumen_global", glob), ("resumen_por_corte", por_corte),
                       ("por_tienda", por_tienda), ("por_familia", por_familia)]:
        df.to_csv(salida / f"{nombre}.csv", index=False)
    res.info.to_csv(salida / "folds.csv", index=False)

    mejor = glob.iloc[0]
    base = glob[(glob.modelo == "baseline_dow4")].iloc[0]
    clave_m = (mejor.modelo, mejor.estrategia)
    clave_b = (base.modelo, base.estrategia)

    pc = por_corte.set_index(["modelo", "estrategia", "corte"])["costo_total"]
    duelo = pd.DataFrame({"corte": cortes,
                          "baseline": [pc[(*clave_b, c)] for c in cortes],
                          "mejor": [pc[(*clave_m, c)] for c in cortes]})
    duelo["mejora_%"] = 100 * (1 - duelo["mejor"] / duelo["baseline"])

    def comparar(por: str) -> pd.DataFrame:
        r = resumir(d, [por]).set_index(["modelo", "estrategia", por])
        out = pd.DataFrame({
            "costo_baseline": r.loc[clave_b, "costo_total"],
            "costo_mejor": r.loc[clave_m, "costo_total"],
            "wape_baseline": r.loc[clave_b, "wape"],
            "wape_mejor": r.loc[clave_m, "wape"],
        })
        out["mejora_%"] = 100 * (1 - out["costo_mejor"] / out["costo_baseline"])
        return out.reset_index()

    tf = resumir(d[d.set_index(["modelo", "estrategia"]).index.isin([clave_m, clave_b])],
                 ["tienda_id", "familia_id"])
    tf.to_csv(salida / "por_tienda_familia.csv", index=False)
    tf_piv = tf.pivot_table(index=["tienda_id", "familia_id"], columns="modelo", values="costo_total")
    peores = (tf_piv.assign(mejora_pct=100 * (1 - tf_piv[mejor.modelo] / tf_piv["baseline_dow4"]))
              .sort_values("mejora_pct").head(10).reset_index())

    tau = pd.DataFrame({"familia_id": list(taus(datos.costos)), "tau": list(taus(datos.costos).values())})
    tau = tau.merge(datos.costos, on="familia_id")

    h = datos.historico
    promo = h.groupby("familia_id")["en_promocion"].agg(
        maximo="max", p99=lambda s: s.quantile(.99), p999=lambda s: s.quantile(.999),
        frac_con_promo=lambda s: (s > 0).mean()).reset_index()

    cols = ["modelo", "estrategia", "costo_total", "costo_por_dia_tienda", "mae", "rmse", "wape", "sesgo",
            "tasa_quiebre", "unidades_faltantes", "unidades_sobrantes"]
    md = [
        "# Resultados del backtesting — etapa 2",
        "",
        f"Generado por `scripts/experimento_etapa2.py` en {segundos:,.0f} s "
        f"(Python {platform.python_version()}, scikit-learn {sklearn.__version__}).",
        "",
        f"* Histórico: {h.fecha.min().date()} a {fin.date()}.",
        f"* {a.folds} cortes con ventanas de prueba consecutivas de {a.horizonte} días: "
        f"{cortes[0].date()} a {fin.date()} ({a.folds * a.horizonte} días, {len(d) // d.groupby(['modelo','estrategia']).ngroups:,} filas por combinación).",
        "* En cada corte C: entrenamiento con fechas < C, calibración con residuos fuera de muestra de "
        "[C−56, C), prueba en [C, C+28) sin reentrenar.",
        "",
        "## Resumen global (ordenado por costo de negocio)",
        "",
        tabla_md(glob[cols]),
        "",
        f"**Mejor combinación:** `{mejor.modelo}` + `{mejor.estrategia}` — costo {mejor.costo_total:,.0f} vs "
        f"{base.costo_total:,.0f} del mejor baseline (`{base.estrategia}`): "
        f"{100 * (1 - mejor.costo_total / base.costo_total):.1f} % menos.",
        "",
        "## Costo por corte: mejor combinación vs baseline",
        "",
        tabla_md(duelo, 1),
        "",
        f"La mejor combinación gana en {(duelo['mejora_%'] > 0).sum()} de {len(duelo)} cortes.",
        "",
        "## Desglose por tienda",
        "",
        tabla_md(comparar("tienda_id"), 3),
        "",
        "## Desglose por familia",
        "",
        tabla_md(comparar("familia_id").merge(tau[["familia_id", "tau"]], on="familia_id"), 3),
        "",
        "## Series tienda×familia donde menos mejora (o empeora) el modelo",
        "",
        tabla_md(peores, 1),
        "",
        "## Cuantil objetivo por familia",
        "",
        tabla_md(tau, 3),
        "",
        "## Distribución de `en_promocion` en el histórico",
        "",
        tabla_md(promo, 3),
    ]
    (salida / "resultados.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:20]))


if __name__ == "__main__":
    main()
