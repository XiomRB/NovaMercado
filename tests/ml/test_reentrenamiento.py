"""Reentrenamiento con observaciones: promociones desde solicitudes y filas sin promoción."""

import numpy as np
import pandas as pd
import pytest

from ml.calendario import Calendario
from ml.datos import CLAVE, consolidar_promociones, consolidar_ventas, discrepancias_promocion, filas_supervisadas
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


_ID = iter(range(1, 10**9))


def solicitudes(df, registrado_en="2025-01-01T08:00", supuesta=False, solicitud_id=None):
    """Filas como las que entregará la persistencia (una por familia de cada solicitud)."""
    sid = next(_ID) if solicitud_id is None else solicitud_id
    return df[CLAVE + ["en_promocion"]].assign(registrado_en=pd.Timestamp(registrado_en),
                                              promocion_supuesta=supuesta, solicitud_id=sid)


def de_solicitud(p):
    return p[p["origen"] == "solicitud"]


FILA = H[H["fecha"] == DESDE].head(1)                       # sin promoción histórica
FILA_HIST = H[H["fecha"] == DESDE - pd.Timedelta(days=1)].head(1)   # con promoción histórica


# -------------------------------------------------------- consolidar_promociones
def test_solo_historico():
    p = consolidar_promociones(PROMOS_HIST)
    assert len(p) == len(PROMOS_HIST) and set(p["origen"]) == {"historico"}
    assert not p["discrepancia"].any() and p["en_promocion_solicitud"].isna().all()


def test_promocion_de_la_solicitud_completa_las_fechas_sin_historico():
    nuevas = H[(H["fecha"] >= DESDE) & (H["fecha"] < DESDE + pd.Timedelta(days=3))]
    p = consolidar_promociones(PROMOS_HIST, solicitudes(nuevas))
    assert len(de_solicitud(p)) == len(nuevas)
    m = de_solicitud(p).merge(nuevas, on=CLAVE, suffixes=("", "_real"))
    assert (m["en_promocion"] == m["en_promocion_real"]).all()
    assert not de_solicitud(p)["discrepancia"].any()


# ------------------------------------------- última solicitud del mismo día
def test_ultima_por_registrado_en_sin_importar_el_orden_de_entrada():
    primera = solicitudes(FILA.assign(en_promocion=1), "2025-05-01T07:00", solicitud_id=20)
    ultima = solicitudes(FILA.assign(en_promocion=6), "2025-05-01T09:00", solicitud_id=10)
    for orden in ([primera, ultima], [ultima, primera]):
        p = consolidar_promociones(PROMOS_HIST, pd.concat(orden))
        assert de_solicitud(p)["en_promocion"].tolist() == [6]   # gana la hora, no el id


def test_empate_en_registrado_en_se_resuelve_por_solicitud_id():
    a = solicitudes(FILA.assign(en_promocion=2), "2025-05-01T08:00", solicitud_id=7)
    b = solicitudes(FILA.assign(en_promocion=5), "2025-05-01T08:00", solicitud_id=8)
    c = solicitudes(FILA.assign(en_promocion=3), "2025-05-01T08:00", solicitud_id=6)
    for orden in ([a, b, c], [c, b, a], [b, a, c]):
        p = consolidar_promociones(PROMOS_HIST, pd.concat(orden))
        assert de_solicitud(p)["en_promocion"].tolist() == [5]


def test_varias_tiendas_y_familias_del_mismo_dia_se_resuelven_por_separado():
    dia = H[H["fecha"] == DESDE]
    temprano = solicitudes(dia.assign(en_promocion=1), "2025-05-01T07:00")
    tarde = solicitudes(dia[dia["tienda_id"] == 1].assign(en_promocion=9), "2025-05-01T10:00")
    p = de_solicitud(consolidar_promociones(PROMOS_HIST, pd.concat([tarde, temprano])))
    assert len(p) == len(dia)
    assert set(p.loc[p["tienda_id"] == 1, "en_promocion"]) == {9}
    assert set(p.loc[p["tienda_id"] != 1, "en_promocion"]) == {1}


@pytest.mark.parametrize("columna", ["registrado_en", "solicitud_id", "promocion_supuesta"])
def test_columnas_de_orden_obligatorias(columna):
    s = solicitudes(FILA)
    with pytest.raises(ValueError, match="faltan columnas"):
        consolidar_promociones(PROMOS_HIST, s.drop(columns=columna))
    with pytest.raises(ValueError, match="no admiten nulos"):
        consolidar_promociones(PROMOS_HIST, s.assign(**{columna: None}))


def test_misma_solicitud_con_familia_repetida_se_rechaza():
    s = solicitudes(FILA, solicitud_id=3)
    with pytest.raises(ValueError, match="dos veces"):
        consolidar_promociones(PROMOS_HIST, pd.concat([s, s.assign(en_promocion=4)]))


# ------------------------------------------------------ promociones supuestas
def test_promocion_supuesta_no_se_usa_aunque_sea_la_ultima():
    real = solicitudes(FILA.assign(en_promocion=4), "2025-05-01T07:00")
    supuesta = solicitudes(FILA.assign(en_promocion=0), "2025-05-01T09:00", supuesta=True)
    p = consolidar_promociones(PROMOS_HIST, pd.concat([real, supuesta]))
    assert de_solicitud(p)["en_promocion"].tolist() == [4]


def test_solo_supuesta_equivale_a_promocion_desconocida():
    supuesta = solicitudes(FILA.assign(en_promocion=0), supuesta=True)
    p = consolidar_promociones(PROMOS_HIST, supuesta)
    assert de_solicitud(p).empty
    assert p.merge(FILA[CLAVE], on=CLAVE).empty


def test_supuesta_no_genera_discrepancia_con_el_historico():
    supuesta = solicitudes(FILA_HIST.assign(en_promocion=FILA_HIST["en_promocion"] + 5), supuesta=True)
    p = consolidar_promociones(PROMOS_HIST, supuesta)
    assert not p["discrepancia"].any()


# ------------------------------------------- histórico vs solicitud
def test_historico_y_solicitud_coinciden():
    p = consolidar_promociones(PROMOS_HIST, solicitudes(FILA_HIST))
    v = p.merge(FILA_HIST[CLAVE], on=CLAVE).iloc[0]
    assert v["origen"] == "historico" and not v["discrepancia"]
    assert v["en_promocion"] == v["en_promocion_solicitud"] == FILA_HIST["en_promocion"].iloc[0]
    assert discrepancias_promocion(p).empty


def test_conflicto_usa_el_historico_y_conserva_el_valor_de_la_solicitud(caplog):
    real = int(FILA_HIST["en_promocion"].iloc[0])
    with caplog.at_level("WARNING", logger="ml.datos"):
        p = consolidar_promociones(PROMOS_HIST, solicitudes(FILA_HIST.assign(en_promocion=real + 3)))
    v = p.merge(FILA_HIST[CLAVE], on=CLAVE).iloc[0]
    assert v["origen"] == "historico" and v["en_promocion"] == real
    assert v["en_promocion_solicitud"] == real + 3 and v["discrepancia"]
    d = discrepancias_promocion(p)
    assert len(d) == 1 and d.iloc[0]["en_promocion_solicitud"] == real + 3
    assert "1 claves con histórico y solicitud distintos" in caplog.text


def test_sin_historico_se_usa_la_solicitud_sin_discrepancia():
    p = consolidar_promociones(PROMOS_HIST, solicitudes(FILA.assign(en_promocion=6)))
    v = p.merge(FILA[CLAVE], on=CLAVE).iloc[0]
    assert v["origen"] == "solicitud" and v["en_promocion"] == 6 and not v["discrepancia"]


def test_el_entrenamiento_informa_las_discrepancias():
    sol = solicitudes(FILA_HIST.assign(en_promocion=FILA_HIST["en_promocion"] + 3))
    p = Pronosticador.entrenar(VENTAS[VENTAS["fecha"] < CORTE], consolidar_promociones(PROMOS_HIST, sol),
                               DATOS.tiendas, DATOS.feriados, DATOS.costos, FAMILIAS, CORTE, CONFIG)
    assert p.info["discrepancias_promocion"] == 1


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


def test_supuesta_se_excluye_como_ejemplo_pero_su_venta_cuenta_en_rezagos():
    """El día DESDE solo tiene una solicitud con promociones supuestas: su venta no es
    ejemplo de entrenamiento, pero es el lag_1 del día siguiente."""
    v = VENTAS[VENTAS["fecha"] <= DESDE]
    sup = solicitudes(H[H["fecha"] == DESDE].assign(en_promocion=0), supuesta=True)
    promos = consolidar_promociones(PROMOS_HIST, sup)
    X, _, sin_promo = preparar(v, promos, DATOS.tiendas, CAL, FAMILIAS)
    assert X["fecha"].max() == DESDE - pd.Timedelta(days=1)
    assert sin_promo == int((v["fecha"] == DESDE).sum())
    from ml.features import construir_features
    manana = H[H["fecha"] == DESDE + pd.Timedelta(days=1)][CLAVE + ["en_promocion"]]
    Xm = construir_features(v, manana, DATOS.tiendas, CAL, FAMILIAS).set_index(["tienda_id", "familia_id"])
    hoy = v[v["fecha"] == DESDE].set_index(["tienda_id", "familia_id"])["ventas"]
    assert (Xm.loc[hoy.index, "lag_1"] == hoy).all()


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


# ------------------------------------------------------- consolidar_ventas
def test_ventas_observacion_completa_fechas_sin_historico():
    nuevas = H[H["fecha"] >= DESDE][CLAVE + ["ventas"]]
    v = consolidar_ventas(H[H["fecha"] < DESDE], nuevas)
    assert len(v) == len(H) and set(v.loc[v["fecha"] >= DESDE, "origen"]) == {"observacion"}
    assert not v["discrepancia"].any()


def test_ventas_coincidencia_y_conflicto_con_el_historico(caplog):
    fila = H[H["fecha"] == DESDE - pd.Timedelta(days=1)].head(2)[CLAVE + ["ventas"]].reset_index(drop=True)
    obs = fila.copy()
    obs.loc[1, "ventas"] = obs.loc[1, "ventas"] + 10          # la segunda difiere
    with caplog.at_level("WARNING", logger="ml.datos"):
        v = consolidar_ventas(H[H["fecha"] < DESDE], obs).merge(fila[CLAVE], on=CLAVE)
    assert (v["origen"] == "historico").all()
    assert v["ventas"].tolist() == fila["ventas"].tolist()          # manda el histórico
    assert v["ventas_observacion"].tolist() == obs["ventas"].tolist()  # no se descarta en silencio
    assert v["discrepancia"].tolist() == [False, True]
    assert "1 claves con histórico y observación distintos" in caplog.text


def test_ventas_sin_observaciones():
    v = consolidar_ventas(H)
    assert len(v) == len(H) and not v["discrepancia"].any() and v["ventas_observacion"].isna().all()


def test_ventas_consolidadas_sirven_para_entrenar():
    obs = H[H["fecha"] >= DESDE][CLAVE + ["ventas"]]
    ventas = consolidar_ventas(H[H["fecha"] < DESDE], obs)
    X, y, sin_promo = preparar(ventas, consolidar_promociones(PROMOS_HIST), DATOS.tiendas, CAL, FAMILIAS)
    assert sin_promo == len(obs) and len(X) == len(PROMOS_HIST)


def test_consolidacion_de_ventas_es_determinista_ante_el_orden_de_entrada():
    obs = H[H["fecha"] >= DESDE - pd.Timedelta(days=3)][CLAVE + ["ventas"]].copy()
    obs.loc[obs.index[:5], "ventas"] += 1                    # algunas discrepancias
    base = consolidar_ventas(H[H["fecha"] < DESDE], obs)
    for semilla in range(5):
        otra = consolidar_ventas(H[H["fecha"] < DESDE].sample(frac=1, random_state=semilla),
                                 obs.sample(frac=1, random_state=semilla + 100))
        pd.testing.assert_frame_equal(base, otra)
    assert base["discrepancia"].sum() == 5


def test_consolidacion_de_promociones_es_determinista_ante_el_orden_de_entrada():
    sol = pd.concat([solicitudes(H[H["fecha"] >= DESDE].assign(en_promocion=1), "2025-05-01T07:00"),
                     solicitudes(H[H["fecha"] >= DESDE].assign(en_promocion=2), "2025-05-01T09:00")])
    base = consolidar_promociones(PROMOS_HIST, sol)
    for semilla in range(5):
        otra = consolidar_promociones(PROMOS_HIST.sample(frac=1, random_state=semilla),
                                      sol.sample(frac=1, random_state=semilla))
        pd.testing.assert_frame_equal(base, otra)
