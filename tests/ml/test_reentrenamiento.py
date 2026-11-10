"""Reentrenamiento con observaciones: promociones desde solicitudes y filas sin promoción."""

import numpy as np
import pandas as pd
import pytest

from ml.calendario import Calendario
from ml.datos import CLAVE, consolidar_promociones, filas_supervisadas
from ml.pronosticador import ConfigEntrenamiento, Pronosticador, preparar

from tests.ml.sinteticos import FAMILIAS, generar

DATOS = generar()
CAL = Calendario(DATOS.feriados)
H = DATOS.historico
CONFIG = ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=30))
CORTE = pd.Timestamp("2025-05-20")
# Simulación de operación: desde DESDE las "observaciones" no traen promoción
DESDE = pd.Timestamp("2025-05-01")
VENTAS = H[CLAVE + ["ventas"]]
PROMOS_HIST = H[H["fecha"] < DESDE][CLAVE + ["en_promocion"]]


def solicitudes(df, registrado_en="2025-01-01T08:00", supuesta=False):
    return df[CLAVE + ["en_promocion"]].assign(registrado_en=pd.Timestamp(registrado_en),
                                              promocion_supuesta=supuesta)


# -------------------------------------------------------- consolidar_promociones
def test_solo_historico():
    p = consolidar_promociones(PROMOS_HIST)
    assert len(p) == len(PROMOS_HIST) and set(p["origen"]) == {"historico"}


def test_promocion_de_la_solicitud_completa_las_fechas_sin_historico():
    nuevas = H[(H["fecha"] >= DESDE) & (H["fecha"] < DESDE + pd.Timedelta(days=3))]
    p = consolidar_promociones(PROMOS_HIST, solicitudes(nuevas))
    desde_sol = p[p["origen"] == "solicitud"]
    assert len(desde_sol) == len(nuevas)
    m = desde_sol.merge(nuevas, on=CLAVE, suffixes=("", "_real"))
    assert (m["en_promocion"] == m["en_promocion_real"]).all()


def test_varias_solicitudes_del_mismo_dia_usa_la_ultima():
    fila = H[(H["fecha"] == DESDE)].head(1)
    primera = solicitudes(fila.assign(en_promocion=1), "2025-05-01T07:00")
    ultima = solicitudes(fila.assign(en_promocion=6), "2025-05-01T09:00")
    p = consolidar_promociones(PROMOS_HIST, pd.concat([ultima, primera]))
    assert p[p["origen"] == "solicitud"]["en_promocion"].tolist() == [6]


def test_promocion_supuesta_no_se_usa():
    fila = H[(H["fecha"] == DESDE)].head(1)
    real = solicitudes(fila.assign(en_promocion=4), "2025-05-01T07:00")
    supuesta = solicitudes(fila.assign(en_promocion=0), "2025-05-01T09:00", supuesta=True)
    p = consolidar_promociones(PROMOS_HIST, pd.concat([real, supuesta]))
    assert p[p["origen"] == "solicitud"]["en_promocion"].tolist() == [4]
    solo_supuesta = consolidar_promociones(PROMOS_HIST, supuesta)
    assert (solo_supuesta["origen"] == "historico").all()


def test_el_historico_tiene_prioridad_sobre_la_solicitud():
    fila = H[H["fecha"] == DESDE - pd.Timedelta(days=1)].head(1)
    p = consolidar_promociones(PROMOS_HIST, solicitudes(fila.assign(en_promocion=99)))
    v = p.merge(fila[CLAVE], on=CLAVE)
    assert v["origen"].tolist() == ["historico"] and v["en_promocion"].tolist() == fila["en_promocion"].tolist()


def test_claves_duplicadas_se_rechazan():
    with pytest.raises(ValueError, match="duplicadas"):
        consolidar_promociones(pd.concat([PROMOS_HIST, PROMOS_HIST.head(1)]))
    with pytest.raises(ValueError, match="duplicadas"):
        filas_supervisadas(pd.concat([VENTAS, VENTAS.head(1)]), PROMOS_HIST)


# -------------------------------------------------------------- filas supervisadas
def test_sin_promocion_se_excluye_del_entrenamiento_pero_cuenta_en_rezagos():
    v = VENTAS[VENTAS["fecha"] < CORTE]
    X, y, sin_promo = preparar(v, consolidar_promociones(PROMOS_HIST), DATOS.tiendas, CAL, FAMILIAS)
    sin = v[v["fecha"] >= DESDE]
    assert sin_promo == len(sin)
    assert X["fecha"].max() < DESDE and len(X) == len(y) == len(v) - len(sin)
    # Una predicción para CORTE usa como lag_1 la venta (sin promoción) de CORTE-1
    p = Pronosticador.entrenar(v, consolidar_promociones(PROMOS_HIST), DATOS.tiendas, DATOS.feriados,
                               DATOS.costos, FAMILIAS, CORTE, CONFIG)
    obj = H[H["fecha"] == CORTE][CLAVE + ["en_promocion"]]
    from ml.features import construir_features
    Xp = construir_features(v, obj, DATOS.tiendas, CAL, FAMILIAS).set_index(["tienda_id", "familia_id"])
    ayer = v[v["fecha"] == CORTE - pd.Timedelta(days=1)].set_index(["tienda_id", "familia_id"])["ventas"]
    assert (Xp.loc[ayer.index, "lag_1"] == ayer).all()
    assert p.info["filas_sin_promocion"] == len(sin)
    assert p.info["fecha_max_entrenamiento"] < DESDE


def test_con_promociones_de_solicitudes_esas_filas_si_se_entrenan():
    v = VENTAS[VENTAS["fecha"] < CORTE]
    sol = solicitudes(H[(H["fecha"] >= DESDE) & (H["fecha"] < CORTE)])
    X, _, sin_promo = preparar(v, consolidar_promociones(PROMOS_HIST, sol), DATOS.tiendas, CAL, FAMILIAS)
    assert sin_promo == 0 and X["fecha"].max() == CORTE - pd.Timedelta(days=1)


def test_mismos_datos_que_el_historico_completo_dan_el_mismo_modelo():
    """Promociones del histórico + solicitudes que replican las reales == histórico completo."""
    sol = solicitudes(H[H["fecha"] >= DESDE])
    a = Pronosticador.entrenar(VENTAS, consolidar_promociones(PROMOS_HIST, sol), DATOS.tiendas,
                               DATOS.feriados, DATOS.costos, FAMILIAS, CORTE, CONFIG)
    b = Pronosticador.entrenar(H, H, DATOS.tiendas, DATOS.feriados, DATOS.costos, FAMILIAS, CORTE, CONFIG)
    obj = H[H["fecha"] == CORTE][CLAVE + ["en_promocion"]]
    pd.testing.assert_frame_equal(a.predecir(obj, H), b.predecir(obj, H))


# --------------------------------------------------------------- máximos de promoción
def test_maximos_de_promocion_solo_con_datos_de_entrenamiento():
    alterado = H.copy()
    alterado.loc[alterado["fecha"] >= CORTE, "en_promocion"] = 99   # posterior al corte
    p = Pronosticador.entrenar(alterado, alterado, DATOS.tiendas, DATOS.feriados, DATOS.costos,
                               FAMILIAS, CORTE, CONFIG)
    esperado = H[H["fecha"] < CORTE].groupby("familia_id")["en_promocion"].max().to_dict()
    assert p.maximos_promocion == esperado
    assert max(p.maximos_promocion.values()) < 99


def test_maximos_ignoran_filas_sin_promocion_conocida():
    v = VENTAS[VENTAS["fecha"] < CORTE]
    p = Pronosticador.entrenar(v, consolidar_promociones(PROMOS_HIST), DATOS.tiendas, DATOS.feriados,
                               DATOS.costos, FAMILIAS, CORTE, CONFIG)
    esperado = PROMOS_HIST.groupby("familia_id")["en_promocion"].max().to_dict()
    assert p.maximos_promocion == esperado
    assert all(isinstance(v, int) for v in p.maximos_promocion.values())
    assert np.isfinite(list(p.maximos_promocion.values())).all()
