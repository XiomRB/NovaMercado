"""Copia inmutable de datos, criterio de validez y tolerancias de comparación."""

import gzip
import json

import pandas as pd
import pytest

from app.modelo.copia_datos import CopiaDatosInvalida, escribir_copia, leer_copia
from app.modelo.evaluacion import (
    BASELINE,
    CANDIDATO,
    ConfigEvaluacion,
    ResultadoEvaluacion,
    comparar_metricas,
    evaluar,
    evaluar_criterio,
)
from tests.modelo.conftest import CONFIG_RAPIDA, D, datos_entrenamiento


# ---------------------------------------------------------------- copia de datos
def test_ida_y_vuelta_identica(tmp_path):
    datos = datos_entrenamiento()
    _, huella = escribir_copia(datos, tmp_path / "c", {"x": 1})
    leidos, manifiesto = leer_copia(tmp_path / "c", huella)
    pd.testing.assert_frame_equal(leidos.ventas[["fecha", "tienda_id", "familia_id", "ventas"]],
                                  datos.ventas[["fecha", "tienda_id", "familia_id", "ventas"]], check_dtype=False)
    pd.testing.assert_frame_equal(leidos.promociones[["fecha", "tienda_id", "familia_id", "en_promocion"]],
                                  datos.promociones[["fecha", "tienda_id", "familia_id", "en_promocion"]],
                                  check_dtype=False)
    assert leidos.familia_ids == datos.familia_ids and leidos.corte == datos.corte and manifiesto["x"] == 1


def test_escritura_determinista(tmp_path):
    datos = datos_entrenamiento()
    m1, h1 = escribir_copia(datos, tmp_path / "a")
    m2, h2 = escribir_copia(datos, tmp_path / "b")
    assert h1 == h2 and m1 == m2
    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()


@pytest.mark.parametrize("dano", ["contenido", "manifiesto", "falta", "huella"])
def test_alteraciones_se_detectan(tmp_path, dano):
    _, huella = escribir_copia(datos_entrenamiento(), tmp_path / "c")
    c = tmp_path / "c"
    if dano == "contenido":
        crudo = gzip.decompress((c / "ventas_consolidadas.csv.gz").read_bytes()).replace(b",1,", b",2,", 1)
        (c / "ventas_consolidadas.csv.gz").write_bytes(gzip.compress(crudo, mtime=0))
    elif dano == "manifiesto":
        m = json.loads((c / "manifiesto.json").read_text())
        m["corte"] = "2030-01-01"
        (c / "manifiesto.json").write_text(json.dumps(m))
    elif dano == "falta":
        (c / "costos.csv.gz").unlink()
    else:
        huella = "0" * 64
    with pytest.raises(CopiaDatosInvalida):
        leer_copia(c, huella)


# ----------------------------------------------------------------------- criterio
def _tabla(costos_cand, costos_base):
    filas = []
    for i, (c, b) in enumerate(zip(costos_cand, costos_base)):
        corte = pd.Timestamp("2026-01-01") + pd.Timedelta(days=28 * i)
        for modelo, costo in ((CANDIDATO, c), (BASELINE, b)):
            filas.append({"corte": corte, "modelo": modelo, "costo_total": float(costo)})
    t = pd.DataFrame(filas)
    return ResultadoEvaluacion(sorted(t["corte"].unique()), t, True)


EV = ConfigEvaluacion(cortes=7)


def test_criterio_valido():
    r = evaluar_criterio(_tabla([80] * 7, [100] * 7), EV)
    assert r.valido and all(r.condiciones.values()) and r.resumen["cortes_ganados"] == 7


@pytest.mark.parametrize("cand, falla", [
    ([97] * 7, "C1"),                                   # solo 3 % de mejora total
    ([60, 60, 60, 60, 101, 101, 101], "C2"),            # gana 4 de 7 (< 5)
    ([60, 60, 60, 60, 60, 60, 115], "C3"),              # un corte 15 % peor
])
def test_criterio_no_cumplido(cand, falla):
    r = evaluar_criterio(_tabla(cand, [100] * 7), EV)
    assert not r.valido and not r.condiciones[falla] and falla in r.motivo


def test_datos_insuficientes():
    r = evaluar_criterio(_tabla([50, 50], [100, 100]), EV)
    assert not r.valido and not r.condiciones["D1"] and "D1" in r.motivo


def test_salidas_fuera_de_contrato_invalidan():
    t = _tabla([80] * 7, [100] * 7)
    t.salidas_validas = False
    r = evaluar_criterio(t, EV)
    assert not r.valido and not r.condiciones["T2"]


def test_evaluacion_real_con_datos_sinteticos():
    ev = ConfigEvaluacion(cortes=3, horizonte=14)
    r = evaluar(datos_entrenamiento(), ev, CONFIG_RAPIDA)
    assert len(r.cortes) == 3 and r.salidas_validas
    assert set(r.por_corte["modelo"]) == {CANDIDATO, BASELINE}
    assert evaluar_criterio(r, ev).valido


def test_pocos_dias_no_alcanzan_los_cortes():
    corto = D.historico[D.historico["fecha"] >= D.historico["fecha"].max() - pd.Timedelta(days=40)]
    ev = ConfigEvaluacion(cortes=3, horizonte=14)
    r = evaluar(datos_entrenamiento(corto), ev, CONFIG_RAPIDA)
    assert len(r.cortes) < 3 and not evaluar_criterio(r, ev).valido


# --------------------------------------------------------------------- tolerancias
def _metricas():
    r = evaluar(datos_entrenamiento(), ConfigEvaluacion(cortes=3, horizonte=14), CONFIG_RAPIDA)
    return r.por_corte


def test_tolerancia_acepta_diferencias_de_representacion():
    a = _metricas()
    b = a.copy()
    b["costo_total"] = b["costo_total"] * (1 + 1e-12)
    b["mae"] = b["mae"] + 1e-12
    assert comparar_metricas(a, b) == []


@pytest.mark.parametrize("cambio", ["real", "entero"])
def test_tolerancia_rechaza_diferencias_reales(cambio):
    a = _metricas()
    b = a.copy()
    if cambio == "real":
        b.loc[0, "costo_total"] = b.loc[0, "costo_total"] + 1e-3
    else:
        b.loc[0, "unidades_faltantes"] = b.loc[0, "unidades_faltantes"] + 1
    assert comparar_metricas(a, b) != []
