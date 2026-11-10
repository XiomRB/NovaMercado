import numpy as np
import pandas as pd
import pytest

from ml.backtest import cortes_consecutivos, ejecutar_backtest
from ml.calendario import Calendario
from ml.metricas import agregar_costo, resumir
from ml.modelos import BaselineDiaSemana, ModeloHGB
from ml.pronosticador import ConfigEntrenamiento, Pronosticador, preparar
from tests.ml.sinteticos import FAMILIAS, generar

DATOS = generar()
CAL = Calendario(DATOS.feriados)
RAPIDO = dict(max_iter=30)
CONFIG = ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14, params_hgb=RAPIDO)
CORTES = [pd.Timestamp("2025-04-26"), pd.Timestamp("2025-05-10")]
HORIZONTE = 14


@pytest.fixture(scope="module")
def resultado():
    return ejecutar_backtest(DATOS, CORTES, HORIZONTE, config=CONFIG)


# ------------------------------------------------------------------ baseline
def test_baseline_promedio_mismo_dia_ultimas_cuatro_semanas():
    X, y, _ = preparar(DATOS.historico, DATOS.historico, DATOS.tiendas, CAL, FAMILIAS)
    b = BaselineDiaSemana().ajustar(X, y)
    fila = X[(X.fecha == pd.Timestamp("2025-05-20")) & (X.tienda_id == 1) & (X.familia_id == "fam_a")]
    H = DATOS.historico.set_index(["fecha", "tienda_id", "familia_id"])["ventas"]
    esperado = np.mean([H[(pd.Timestamp("2025-05-20") - pd.Timedelta(days=k), 1, "fam_a")]
                        for k in (7, 14, 21, 28)])
    assert b.predecir(fila)[0] == pytest.approx(esperado)


def test_baseline_con_pocos_datos_y_sin_historia():
    X, y, _ = preparar(DATOS.historico, DATOS.historico, DATOS.tiendas, CAL, FAMILIAS)
    b = BaselineDiaSemana().ajustar(X, y)
    # tienda 3 abre el día 130: sus primeras filas tienen poca o ninguna historia
    t3 = X[X.tienda_id == 3].sort_values("fecha")
    p = b.predecir(t3)
    assert np.isfinite(p).all() and (p >= 0).all()
    nuevo = t3.head(3).copy()
    for c in ["media_dow_4", "media_28", "ultimo_valor"]:
        nuevo[c] = np.nan
    esperado = X[(X.tipo == "express") & (X.fecha > X.fecha.max() - pd.Timedelta(days=28))]
    assert b.predecir(nuevo)[0] == pytest.approx(
        y[esperado.index][esperado.familia_id == nuevo.familia_id.iloc[0]].mean())


@pytest.mark.parametrize("objetivo", ["directo", "relativo"])
def test_hgb_reproducible_y_no_negativo(objetivo):
    X, y, _ = preparar(DATOS.historico, DATOS.historico, DATOS.tiendas, CAL, FAMILIAS)
    a = ModeloHGB(objetivo, params=RAPIDO).ajustar(X, y).predecir(X)
    b = ModeloHGB(objetivo, params=RAPIDO).ajustar(X, y).predecir(X)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and (a >= 0).all()


# ------------------------------------------------------------------ backtest
def test_sin_fuga_en_fechas_de_entrenamiento_y_calibracion(resultado):
    for r in resultado.info.itertuples():
        assert r.fecha_max_entrenamiento < r.corte
        assert r.ventana_calibracion[1] < r.corte
        assert r.fecha_max_entrenamiento_cal < r.ventana_calibracion[0]
        assert r.fecha_min_prueba == r.corte


def test_mismas_filas_para_todos_los_modelos_y_estrategias(resultado):
    p = resultado.predicciones
    claves = {k: set(map(tuple, g[["corte", "fecha", "tienda_id", "familia_id"]].to_numpy()))
              for k, g in p.groupby(["modelo", "estrategia"])}
    referencia = next(iter(claves.values()))
    assert all(v == referencia for v in claves.values())
    assert {"baseline_dow4", "hgb_directo", "hgb_relativo"} <= set(p["modelo"])
    reales = p.groupby(["corte", "fecha", "tienda_id", "familia_id"])["real"].nunique()
    assert (reales == 1).all()


def test_cantidades_enteras_no_negativas(resultado):
    p = resultado.predicciones
    assert p["cantidad_sugerida"].dtype.kind == "i" and (p["cantidad_sugerida"] >= 0).all()
    assert np.isfinite(p["venta_esperada"]).all() and (p["venta_esperada"] >= 0).all()


def test_tienda_con_pocos_datos_se_evalua(resultado):
    t3 = resultado.predicciones[resultado.predicciones.tienda_id == 3]
    assert len(t3) > 0 and np.isfinite(t3["venta_esperada"]).all()


def test_backtest_reproducible(resultado):
    otra = ejecutar_backtest(DATOS, CORTES, HORIZONTE, config=CONFIG)
    pd.testing.assert_frame_equal(resultado.predicciones, otra.predicciones)


def test_alterar_ventas_futuras_no_cambia_predicciones_previas(resultado):
    """Cambiar las ventas desde F no altera ninguna predicción de fechas <= F."""
    F = CORTES[0] + pd.Timedelta(days=6)
    alterado = generar()
    m = alterado.historico["fecha"] >= F
    alterado.historico.loc[m, "ventas"] *= 10
    otra = ejecutar_backtest(alterado, CORTES[:1], HORIZONTE, config=CONFIG).predicciones
    a = resultado.predicciones
    a = a[(a.corte == CORTES[0]) & (a.fecha <= F)].reset_index(drop=True)
    b = otra[otra.fecha <= F].reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns="real"), b.drop(columns="real"))


@pytest.mark.parametrize("modelo, estrategia", [("hgb_relativo", "mult_familia"),
                                                ("baseline_dow4", "raiz_tipo_familia")])
def test_pronosticador_de_produccion_coincide_con_backtest(resultado, modelo, estrategia):
    """El camino de producción (datos truncados en el corte) da lo mismo que el backtest."""
    corte = CORTES[1]
    cfg = ConfigEntrenamiento(**{**CONFIG.__dict__, "modelo": modelo, "estrategia": estrategia})
    H = DATOS.historico
    p = Pronosticador.entrenar(H, H, DATOS.tiendas, DATOS.feriados, DATOS.costos, FAMILIAS, corte, cfg)
    objetivo = H[(H.fecha >= corte) & (H.fecha < corte + pd.Timedelta(days=HORIZONTE))]
    prod = p.predecir(objetivo[["fecha", "tienda_id", "familia_id", "en_promocion"]], H)
    bt = resultado.predicciones
    bt = bt[(bt.corte == corte) & (bt.modelo == modelo) & (bt.estrategia == estrategia)]
    m = prod.merge(bt, on=["fecha", "tienda_id", "familia_id"], suffixes=("_prod", "_bt"))
    assert len(m) == len(objetivo)
    np.testing.assert_allclose(m.venta_esperada_prod, m.venta_esperada_bt, rtol=1e-9)
    assert (m.cantidad_sugerida_prod == m.cantidad_sugerida_bt).all()
    assert p.info["fecha_max_entrenamiento"] < corte


def test_metricas(resultado):
    d = agregar_costo(resultado.predicciones, DATOS.costos)
    r = resumir(d)
    assert {"costo_total", "mae", "rmse", "wape", "sesgo"} <= set(r.columns)
    assert (r["costo_total"] > 0).all()
    fila = d.iloc[0]
    c = DATOS.costos.set_index("familia_id").loc[fila.familia_id]
    esperado = c.costo_faltante * max(fila.real - fila.cantidad_sugerida, 0) + \
        c.costo_sobrante * max(fila.cantidad_sugerida - fila.real, 0)
    assert fila.costo == pytest.approx(esperado)


def test_cortes_consecutivos():
    c = cortes_consecutivos(pd.Timestamp("2026-02-28"), 3, 28)
    assert c == [pd.Timestamp("2025-12-07"), pd.Timestamp("2026-01-04"), pd.Timestamp("2026-02-01")]
