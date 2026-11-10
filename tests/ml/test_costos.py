"""La cantidad sugerida minimiza el costo esperado (no solo el error de predicción)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ml.costos import (
    CalibradorCuantil,
    a_entero_no_negativo,
    cantidad_optima,
    cantidad_redondeo,
    costo,
    cuantil,
    perdida_cuantil,
    taus,
)

COSTOS_LAB = pd.DataFrame({
    "familia_id": ["abarrotes", "lacteos", "carnes", "panaderia", "frutas_verduras", "licores"],
    "costo_faltante": [1.0, 1.0, 1.5, 1.0, 1.0, 2.0],
    "costo_sobrante": [0.2, 1.2, 2.0, 2.5, 2.0, 0.1],
})


def test_costo_asimetrico():
    assert costo(10, 7, 1.5, 2.0) == pytest.approx(4.5)    # faltan 3 × 1.5
    assert costo(10, 12, 1.5, 2.0) == pytest.approx(4.0)   # sobran 2 × 2.0
    assert costo(10, 10, 1.5, 2.0) == 0
    np.testing.assert_allclose(costo([5, 5], [3, 8], [1, 1], [0.1, 0.1]), [2.0, 0.3])


def test_tau_por_familia():
    t = taus(COSTOS_LAB)
    assert t["abarrotes"] == pytest.approx(1 / 1.2)
    assert t["licores"] == pytest.approx(2 / 2.1)
    assert t["panaderia"] == pytest.approx(1 / 3.5)
    assert t["lacteos"] == pytest.approx(1 / 2.2)
    assert t["licores"] > t["abarrotes"] > t["lacteos"] > t["panaderia"]


def test_tau_rechaza_costos_no_positivos():
    with pytest.raises(ValueError):
        taus(pd.DataFrame({"familia_id": ["x"], "costo_faltante": [0.0], "costo_sobrante": [1.0]}))


def _costo_medio(muestras, s, cf, cs):
    return costo(muestras, s, cf, cs).mean()


@pytest.mark.parametrize("semilla", range(300))
def test_cantidad_optima_es_el_minimo_exhaustivo(semilla):
    """Contra búsqueda exhaustiva en todos los enteros: ningún s tiene menor costo empírico."""
    rng = np.random.default_rng(semilla)
    n = int(rng.integers(1, 80))
    muestras = rng.gamma(2.0, rng.uniform(1, 100), n)
    if semilla % 2:
        muestras = np.round(muestras)          # también con demanda entera
    cf, cs = rng.uniform(0.1, 3), rng.uniform(0.1, 3)
    tau = cf / (cf + cs)
    s = cantidad_optima(muestras, tau)
    candidatos = np.arange(0, int(muestras.max()) + 3)
    costos = np.array([_costo_medio(muestras, c, cf, cs) for c in candidatos])
    assert isinstance(s, int) and s >= 0
    assert _costo_medio(muestras, s, cf, cs) <= costos.min() + 1e-9


@pytest.mark.parametrize("lam", [0.5, 3, 20, 150, 600])
@pytest.mark.parametrize("familia", COSTOS_LAB["familia_id"])
def test_cuantil_minimiza_costo_esperado_poisson(lam, familia):
    """Con la distribución exacta (Poisson): el menor s con F(s) >= τ minimiza E[costo]."""
    c = COSTOS_LAB.set_index("familia_id").loc[familia]
    cf, cs = c.costo_faltante, c.costo_sobrante
    tau = cf / (cf + cs)
    soporte = np.arange(0, int(lam * 3 + 30))
    pmf = stats.poisson.pmf(soporte, lam)
    esperado = np.array([(pmf * costo(soporte, s, cf, cs)).sum() for s in soporte])
    s_teorico = int(np.argmax(stats.poisson.cdf(soporte, lam) >= tau))
    assert esperado[s_teorico] <= esperado.min() + 1e-9
    # y la versión empírica converge al óptimo teórico
    muestras = np.random.default_rng(0).poisson(lam, 200_000)
    assert abs(cantidad_optima(muestras, tau) - s_teorico) <= 1


def test_tau_propio_supera_a_otros_cuantiles_fuera_de_muestra():
    """Calibrar con el τ de cada familia da menor costo en datos nuevos que usar la
    mediana, el redondeo de la media o el τ de otra familia."""
    rng = np.random.default_rng(1)
    pred_cal, pred_test = rng.uniform(20, 400, 4000), rng.uniform(20, 400, 4000)
    real_cal = rng.poisson(pred_cal * rng.lognormal(0, 0.25, 4000))
    real_test = rng.poisson(pred_test * rng.lognormal(0, 0.25, 4000))
    for _, c in COSTOS_LAB.iterrows():
        f, cf, cs = c.familia_id, c.costo_faltante, c.costo_sobrante
        df_cal = pd.DataFrame({"familia_id": f, "real": real_cal, "pred": pred_cal})
        df_test = pd.DataFrame({"familia_id": [f] * len(pred_test)})
        propio = CalibradorCuantil().ajustar(df_cal, {f: cf / (cf + cs)}).sugerir(pred_test, df_test)
        costo_propio = costo(real_test, propio, cf, cs).sum()
        for tau_otro in (0.5, 0.2, 0.95):
            if abs(tau_otro - cf / (cf + cs)) < 0.1:
                continue
            otro = CalibradorCuantil().ajustar(df_cal, {f: tau_otro}).sugerir(pred_test, df_test)
            assert costo_propio < costo(real_test, otro, cf, cs).sum(), (f, tau_otro)
        assert costo_propio < costo(real_test, cantidad_redondeo(pred_test), cf, cs).sum()


def test_cuantil_inverted_cdf():
    assert cuantil([1, 2, 3, 4], 0.5) == 2
    assert cuantil([1, 2, 3, 4], 0.51) == 3
    assert cuantil([5], 0.99) == 5
    with pytest.raises(ValueError):
        cuantil([], 0.5)


def test_enteros_no_negativos():
    s = a_entero_no_negativo([-3.2, 0, 0.1, 640.0000000001, 12.5])
    assert s.dtype.kind == "i"
    assert s.tolist() == [0, 0, 1, 640, 13]
    assert cantidad_redondeo([-1, 2.5, 2.49]).tolist() == [0, 3, 2]


def test_calibrador_multiplicativo_recupera_el_cuantil():
    rng = np.random.default_rng(2)
    pred = rng.uniform(50, 300, 20000)
    ratio = rng.lognormal(0, 0.3, 20000)
    df = pd.DataFrame({"familia_id": "x", "pred": pred, "real": pred * ratio})
    cal = CalibradorCuantil().ajustar(df, {"x": 0.9})
    f = cal.factor(pd.DataFrame({"familia_id": ["x"]}))[0]
    assert f == pytest.approx(np.exp(0.3 * stats.norm.ppf(0.9)), rel=0.02)
    s = cal.sugerir(np.array([0.0, 100.0]), pd.DataFrame({"familia_id": ["x", "x"]}))
    assert s[0] == 0 and s[1] in (int(np.floor(100 * f)), int(np.ceil(100 * f)))


@pytest.mark.parametrize("modo", ["multiplicativo", "raiz"])
@pytest.mark.parametrize("tau", [0.286, 0.455, 0.833, 0.952])
def test_calibrador_sugiere_el_minimo_exhaustivo_por_fila(modo, tau):
    """Para cada fila, la cantidad sugerida minimiza el costo sobre sus muestras de demanda."""
    rng = np.random.default_rng(int(tau * 1000))
    pred_cal = rng.uniform(1, 200, 300)
    df = pd.DataFrame({"familia_id": "x", "pred": pred_cal,
                       "real": rng.poisson(pred_cal * rng.lognormal(0, 0.3, 300))})
    cal = CalibradorCuantil(modo=modo).ajustar(df, {"x": tau})
    pred = np.concatenate([[0.0, 0.3], rng.uniform(0, 400, 40)])
    s = cal.sugerir(pred, pd.DataFrame({"familia_id": ["x"] * len(pred)}))
    r = cal.residuos_familia["x"]
    for p, si in zip(pred, s):
        muestras = cal._muestras(np.array([p]), r)[0]
        candidatos = np.arange(0, int(muestras.max()) + 3)
        perdidas = [perdida_cuantil(np.array([c]), muestras, tau)[0] for c in candidatos]
        assert perdida_cuantil(np.array([si]), muestras, tau)[0] <= min(perdidas) + 1e-12
        assert si == cantidad_optima(muestras, tau)


def test_calibrador_por_grupo_usa_respaldo_de_familia():
    df = pd.DataFrame({"familia_id": ["x"] * 100 + ["x"] * 5, "tipo": ["a"] * 100 + ["b"] * 5,
                       "pred": 100.0, "real": [100.0] * 100 + [500.0] * 5})
    cal = CalibradorCuantil(grupos=("tipo", "familia_id"), min_muestras=50).ajustar(df, {"x": 0.5})
    assert ("a", "x") in cal.residuos and ("b", "x") not in cal.residuos
    f = cal.factor(pd.DataFrame({"familia_id": ["x", "x", "x"], "tipo": ["a", "b", "nuevo"]}))
    familia = cuantil(cal.residuos_familia["x"], 0.5)
    assert f[0] == 1.0 and f[1] == f[2] == familia


def test_calibrador_exige_residuos_de_todas_las_familias():
    df = pd.DataFrame({"familia_id": ["x"], "pred": [10.0], "real": [9.0]})
    with pytest.raises(ValueError, match="sin residuos"):
        CalibradorCuantil().ajustar(df, {"x": 0.5, "y": 0.5})
