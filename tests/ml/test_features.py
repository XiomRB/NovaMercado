"""Ninguna variable para la fecha F usa información de fechas >= F."""

import numpy as np
import pandas as pd
import pytest

from ml.calendario import Calendario
from ml.features import COLUMNAS_MODELO, LAGS, construir_features

from tests.ml.sinteticos import FAMILIAS, generar

DATOS = generar()
CAL = Calendario(DATOS.feriados)
H = DATOS.historico


def feats(ventas, objetivos):
    return construir_features(ventas, objetivos, DATOS.tiendas, CAL, FAMILIAS)


def objetivos_de(fecha):
    o = H[H["fecha"] == pd.Timestamp(fecha)][["fecha", "tienda_id", "familia_id", "en_promocion"]]
    return o.reset_index(drop=True)


FECHAS = ["2025-02-10", "2025-03-11", "2025-05-20", "2025-06-01"]


@pytest.mark.parametrize("fecha", FECHAS)
def test_alterar_el_futuro_no_cambia_las_variables(fecha):
    F = pd.Timestamp(fecha)
    obj = objetivos_de(F)
    base = feats(H, obj)
    futuro = H.copy()
    m = futuro["fecha"] >= F
    futuro.loc[m, "ventas"] = futuro.loc[m, "ventas"] * 1000 + 7
    futuro.loc[m, "en_promocion"] = 99
    extra = futuro[m].assign(fecha=futuro.loc[m, "fecha"] + pd.Timedelta(days=400))
    alterado = feats(pd.concat([futuro, extra]), obj)
    pd.testing.assert_frame_equal(base[COLUMNAS_MODELO], alterado[COLUMNAS_MODELO])


@pytest.mark.parametrize("fecha", FECHAS)
def test_truncar_en_F_da_las_mismas_variables(fecha):
    """Calcular con todo el histórico (backtest) == calcular solo con datos < F (producción)."""
    F = pd.Timestamp(fecha)
    obj = objetivos_de(F)
    pd.testing.assert_frame_equal(feats(H, obj)[COLUMNAS_MODELO],
                                  feats(H[H["fecha"] < F], obj)[COLUMNAS_MODELO])


def test_calcular_todo_junto_equivale_a_calcular_por_fecha():
    obj_todo = H[["fecha", "tienda_id", "familia_id", "en_promocion"]]
    todo = feats(H, obj_todo).set_index(["fecha", "tienda_id", "familia_id"], drop=False)
    for fecha in FECHAS:
        uno = feats(H[H["fecha"] < pd.Timestamp(fecha)], objetivos_de(fecha)).set_index(
            ["fecha", "tienda_id", "familia_id"], drop=False)
        pd.testing.assert_frame_equal(todo.loc[uno.index, COLUMNAS_MODELO], uno[COLUMNAS_MODELO])


def test_promociones_pasadas_no_se_usan():
    F = pd.Timestamp("2025-05-20")
    obj = objetivos_de(F)
    otra = H.copy()
    otra.loc[otra["fecha"] != F, "en_promocion"] = 50
    pd.testing.assert_frame_equal(feats(H, obj)[COLUMNAS_MODELO], feats(otra, obj)[COLUMNAS_MODELO])


def test_valores_contra_calculo_directo():
    rng = np.random.default_rng(0)
    muestra = H[H["fecha"] >= pd.Timestamp("2025-02-15")].sample(60, random_state=1)
    X = feats(H, muestra).reset_index(drop=True)
    for i, fila in enumerate(muestra.itertuples(index=False)):
        serie = H[(H.tienda_id == fila.tienda_id) & (H.familia_id == fila.familia_id)].set_index("fecha")["ventas"]
        previas = serie[serie.index < fila.fecha]
        for k in LAGS:
            esperado = serie.get(fila.fecha - pd.Timedelta(days=k), np.nan)
            assert X.loc[i, f"lag_{k}"] == pytest.approx(esperado, nan_ok=True)
        ult7 = previas[previas.index >= fila.fecha - pd.Timedelta(days=7)]
        ult28 = previas[previas.index >= fila.fecha - pd.Timedelta(days=28)]
        dows = [serie.get(fila.fecha - pd.Timedelta(days=k)) for k in (7, 14, 21, 28)]
        dows = [v for v in dows if v is not None]
        assert X.loc[i, "media_7"] == pytest.approx(ult7.mean())
        assert X.loc[i, "media_28"] == pytest.approx(ult28.mean())
        assert X.loc[i, "std_28"] == pytest.approx(ult28.std(), rel=1e-9)
        assert X.loc[i, "media_dow_4"] == pytest.approx(np.mean(dows))
        assert X.loc[i, "ultimo_valor"] == previas.iloc[-1]
        assert X.loc[i, "en_promocion"] == fila.en_promocion
    assert rng is not None


def test_observaciones_desactualizadas():
    """Si faltan los últimos 3 días, los rezagos cortos quedan vacíos y se mide la antigüedad."""
    F = pd.Timestamp("2025-05-20")
    sin_recientes = H[~((H["fecha"] >= F - pd.Timedelta(days=3)) & (H["fecha"] < F))]
    X = feats(sin_recientes, objetivos_de(F))
    assert X[["lag_1", "lag_2", "lag_3"]].isna().all().all()
    assert (X["dias_desde_ultimo"] == 4).all()
    assert X["lag_7"].notna().all()
    esperado = H[H["fecha"] == F - pd.Timedelta(days=4)].set_index(["tienda_id", "familia_id"])["ventas"]
    obtenido = X.set_index(["tienda_id", "familia_id"])["ultimo_valor"]
    assert (obtenido == esperado.loc[obtenido.index]).all()


def test_serie_sin_historia():
    """Tienda 3 antes de abrir: sin ventas previas, las variables de ventas son NaN."""
    obj = pd.DataFrame({"fecha": pd.Timestamp("2025-03-01"), "tienda_id": 3,
                        "familia_id": FAMILIAS, "en_promocion": 0})
    X = feats(H, obj)
    assert X[["lag_1", "media_28", "media_dow_4", "ultimo_valor"]].isna().all().all()
    assert (X["n_obs_28"] == 0).all() and X["dias_desde_ultimo"].isna().all()


def test_calendario_local_y_nacional():
    claves = pd.DataFrame({"fecha": pd.to_datetime(["2025-03-10", "2025-03-10", "2025-03-09",
                                                    "2025-03-11", "2025-04-17", "2025-02-14"]),
                           "ciudad": ["Cobán", "Guatemala", "Cobán", "Cobán", "Guatemala", "Cobán"]})
    c = CAL.features(claves)
    assert c["ev_feriado"].tolist() == [1, 0, 0, 0, 1, 0]
    assert c["vispera_feriado"].tolist() == [0, 0, 1, 0, 0, 0]
    assert c["post_feriado"].tolist() == [0, 0, 0, 1, 0, 0]
    assert c["ev_comercial"].tolist() == [0, 0, 0, 0, 0, 1]
    assert c["ev_codigo"].isna().tolist() == [False, True, True, True, False, False]
    assert c["dias_para_fin_mes"].iloc[0] == 21


def test_orden_de_objetivos_se_conserva():
    obj = objetivos_de("2025-05-20").sample(frac=1, random_state=3).reset_index(drop=True)
    X = feats(H, obj)
    assert X[["tienda_id", "familia_id"]].equals(obj[["tienda_id", "familia_id"]])


@pytest.mark.parametrize("fecha", FECHAS)
def test_alterar_el_futuro_con_objetivos_de_varias_fechas(fecha):
    """Objetivos en [F−5, F+5]: alterar ventas desde F no cambia las filas de fechas <= F.

    Con varias fechas objetivo la función no puede filtrar todo el futuro de
    antemano, así que esta prueba ejercita los desplazamientos de cada variable.
    """
    F = pd.Timestamp(fecha)
    ventana = (H["fecha"] >= F - pd.Timedelta(days=5)) & (H["fecha"] <= F + pd.Timedelta(days=5))
    obj = H[ventana][["fecha", "tienda_id", "familia_id", "en_promocion"]].reset_index(drop=True)
    futuro = H.copy()
    m = futuro["fecha"] >= F
    futuro.loc[m, "ventas"] = futuro.loc[m, "ventas"] * 1000 + 7
    hasta_F = (obj["fecha"] <= F).to_numpy()
    a = feats(H, obj)[COLUMNAS_MODELO][hasta_F]
    b = feats(futuro, obj)[COLUMNAS_MODELO][hasta_F]
    pd.testing.assert_frame_equal(a, b)
