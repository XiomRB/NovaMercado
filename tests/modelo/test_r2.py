"""Criterio R2 (etapa 7, plan v2 §5, v3 §5.8, v4 §5.3) con datos sintéticos, sin MLflow
ni base: ventana y bloques, mismas filas, sin fuga temporal, empate, retador peor y
mejor, datos insuficientes, filas no comparables, reproducción y decisión ``si_mejora``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.modelo import r2
from app.modelo.promocion import decidir_si_mejora
from app.modelo.registro import ResultadoRegistro, califica_para_respaldo
from ml.datos import CLAVE
from ml.pronosticador import Pronosticador
from tests.modelo.conftest import CONFIG_RAPIDA, datos_entrenamiento

DATOS = datos_entrenamiento()
CORTE_C = DATOS.corte


def campeon_entrenado(corte_v):
    return Pronosticador.entrenar(DATOS.ventas, DATOS.promociones, DATOS.tiendas, DATOS.feriados, DATOS.costos,
                                  DATOS.familia_ids, corte_v, CONFIG_RAPIDA)


class CampeonFijo:
    """Campeón de prueba: cantidad = venta real (costo 0) o siempre 0."""

    def __init__(self, modo, quitar=0):
        self.modo, self.quitar = modo, quitar
        v = DATOS.ventas[CLAVE + ["ventas"]].copy()
        v["fecha"] = pd.to_datetime(v["fecha"])
        self.reales = v.set_index(CLAVE)["ventas"]

    def predecir(self, objetivos, _historia):
        out = objetivos[CLAVE].copy().iloc[self.quitar:]
        reales = self.reales.reindex(pd.MultiIndex.from_frame(out[CLAVE])).to_numpy()
        out["venta_esperada"] = reales.astype(float)
        out["cantidad_sugerida"] = reales.astype(int) if self.modo == "oraculo" else 0
        return out


def test_ventana_y_bloques():
    cv, cc = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")
    assert r2.ventana(cv, cc) == (cc - pd.Timedelta(days=28), cc)               # como mucho 28 días
    assert r2.ventana(cc - pd.Timedelta(days=10), cc)[0] == cc - pd.Timedelta(days=10)
    b = r2.bloques(cc - pd.Timedelta(days=10), cc)
    assert [(x.day, y.day) for x, y in b] == [(19, 26), (26, 1)] and b[-1][1] == cc       # último parcial
    assert r2.bloques(cc, cc) == []


def test_menos_de_7_dias_no_es_evaluable():
    res = r2.calcular(DATOS, CampeonFijo("cero"), CORTE_C - pd.Timedelta(days=6), CONFIG_RAPIDA)
    assert res.resultado == r2.NO_EVALUABLE and res.dias == 6 and "6 < 7" in res.motivo


def test_retador_peor_que_un_campeon_perfecto():
    res = r2.calcular(DATOS, CampeonFijo("oraculo"), CORTE_C - pd.Timedelta(days=14), CONFIG_RAPIDA)
    assert res.resultado is False and res.costo_vigente == 0 and res.costo_retador > 0
    assert res.dias == 14 and len(res.por_bloque) == 2
    assert "Costo en validación" in res.motivo and "del vigente (14 días" in res.motivo


def test_retador_mejor_que_un_campeon_sin_pedidos():
    res = r2.calcular(DATOS, CampeonFijo("cero"), CORTE_C - pd.Timedelta(days=40), CONFIG_RAPIDA)
    assert res.resultado is True and res.costo_retador < res.costo_vigente
    assert res.dias == 28 and len(res.por_bloque) == 4                         # ventana limitada a 28 días


def test_empate_con_el_mismo_procedimiento_cumple():
    """Un solo bloque que empieza en corte_v: el retador se entrena con los mismos datos y
    la misma configuración que el campeón; sus costos son iguales y R2 se cumple."""
    cv = CORTE_C - pd.Timedelta(days=7)
    res = r2.calcular(DATOS, campeon_entrenado(cv), cv, CONFIG_RAPIDA)
    assert res.resultado is True and res.costo_retador == pytest.approx(res.costo_vigente, rel=1e-12)


def test_mismas_filas_y_misma_tabla_para_ambos():
    cv = CORTE_C - pd.Timedelta(days=21)
    res = r2.calcular(DATOS, campeon_entrenado(CORTE_C - pd.Timedelta(days=35)), cv, CONFIG_RAPIDA)
    from ml.datos import filas_supervisadas
    sup, _ = filas_supervisadas(DATOS.ventas, DATOS.promociones)
    esperadas = sup[(sup["fecha"] >= cv) & (sup["fecha"] < CORTE_C)]
    assert res.filas == len(esperadas) == int(res.por_bloque["filas"].sum())
    assert res.corte_v == cv.strftime("%Y-%m-%d")


def test_filas_no_comparables_fallan_cerrado():
    with pytest.raises(ValueError, match="filas no comparables"):
        r2.calcular(DATOS, CampeonFijo("cero", quitar=1), CORTE_C - pd.Timedelta(days=14), CONFIG_RAPIDA)


def test_prediccion_por_bloque_sin_fuga_temporal():
    """Predecir un bloque de una vez con la historia hasta su fin da lo mismo que predecir
    cada fecha F con solo las ventas anteriores a F (como en producción)."""
    cv = CORTE_C - pd.Timedelta(days=7)
    p = campeon_entrenado(cv)
    from ml.datos import filas_supervisadas
    sup, _ = filas_supervisadas(DATOS.ventas, DATOS.promociones)
    obj = sup[(sup["fecha"] >= cv) & (sup["fecha"] < CORTE_C)].sort_values(CLAVE).reset_index(drop=True)
    hist = DATOS.ventas[CLAVE + ["ventas"]].assign(fecha=lambda d: pd.to_datetime(d["fecha"]))
    junto = p.predecir(obj[CLAVE + ["en_promocion"]], hist[hist["fecha"] < CORTE_C]).sort_values(CLAVE)
    partes = []
    for f, g in obj.groupby("fecha"):
        partes.append(p.predecir(g[CLAVE + ["en_promocion"]], hist[hist["fecha"] < f]))
    separado = pd.concat(partes).sort_values(CLAVE)
    pd.testing.assert_frame_equal(junto.reset_index(drop=True), separado.reset_index(drop=True))


def test_reproduccion_identica_y_deteccion_de_diferencias():
    cv = CORTE_C - pd.Timedelta(days=14)
    a = r2.calcular(DATOS, campeon_entrenado(CORTE_C - pd.Timedelta(days=21)), cv, CONFIG_RAPIDA)
    b = r2.calcular(DATOS, campeon_entrenado(CORTE_C - pd.Timedelta(days=21)), cv, CONFIG_RAPIDA)
    registrada = r2.leer_tabla_csv(a.tabla_csv())
    assert a.tabla_csv() == b.tabla_csv() and r2.comparar(registrada, b.por_bloque) == []
    otra = b.por_bloque.copy()
    otra.loc[0, "costo_retador"] += 1
    otra.loc[1, "filas"] += 1
    difs = r2.comparar(registrada, otra)
    assert any("costo_retador" in d for d in difs) and any("filas" in d for d in difs)


def test_metricas():
    res = r2.calcular(DATOS, CampeonFijo("cero"), CORTE_C - pd.Timedelta(days=7), CONFIG_RAPIDA)
    m = res.metricas()
    assert set(m) == {"r2_dias", "r2_filas", "r2_costo_retador", "r2_costo_vigente"}
    assert set(r2.no_disponible("x").metricas()) == {"r2_dias", "r2_filas"}


# ------------------------------------------------------------- decisión si_mejora
class ConcesionFalsa:
    def __init__(self, origen):
        self.origen = origen

    def origen_de_decision(self, _decision_id):
        return self.origen


def resultado(r2_resultado, decision_id=10, **cond):
    base = {k: True for k in ("T1", "T2", "T3", "D1", "T4a", "T4b", "C1", "C2", "C3")}
    rr = {"resultado": r2_resultado, "decision_id": decision_id, "motivo": "Costo en validación 1 vs 2"}
    return ResultadoRegistro("run", "5", True, "ok", {**base, "R2": r2_resultado, **cond}, r2=rr)


VIGENTE = SimpleNamespace(decision_id=10, version="4")


@pytest.mark.parametrize("r2_res, origen, promueve", [
    (True, "reentrenamiento", True),
    (False, "reentrenamiento", False),
    ("no_evaluable", "arranque", False),
    ("no_disponible", "manual", False),
    (False, "arranque_respaldo", True),               # E7-6b: basta C1–C3
    ("no_evaluable", "arranque_respaldo", True),
    ("no_disponible", "arranque_respaldo", True),
])
def test_decidir_si_mejora(r2_res, origen, promueve):
    ok, motivo = decidir_si_mejora(ConcesionFalsa(origen), resultado(r2_res), VIGENTE)
    assert ok is promueve and motivo


def test_si_mejora_sin_vigente_promueve():
    assert decidir_si_mejora(ConcesionFalsa(None), resultado("no_disponible"), None)[0] is True


def test_si_mejora_exige_r2_contra_el_vigente_actual():
    ok, motivo = decidir_si_mejora(ConcesionFalsa("reentrenamiento"), resultado(True, decision_id=9), VIGENTE)
    assert ok is False and "vigente actual" in motivo
    r = resultado(True)
    r.r2 = None
    assert decidir_si_mejora(ConcesionFalsa("reentrenamiento"), r, VIGENTE)[0] is False


def test_r2_no_afecta_el_criterio_del_respaldo():
    r = resultado(False, C1=False)
    assert califica_para_respaldo(r) is True
