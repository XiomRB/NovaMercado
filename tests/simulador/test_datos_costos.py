"""Preparación de datos (S1), evidencia inmutable y costo frente a baseline_dow4 (S9).
Datos sintéticos: nunca se lee raw/."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.calendario import Calendario
from ml.costos import cantidad_redondeo, costo
from ml.features import construir_features
from ml.modelos import BaselineDiaSemana
from simulador import costos, datos, defectos, evidencia
from simulador.escenario import CAMBIO_DEMANDA, COMPLETO, CORTO
from tests.ml.sinteticos import escribir_carpeta, generar_contrato

ESC = replace(CORTO, dias=10, corte="2026-04-05")


@pytest.fixture
def fuente(tmp_path):
    d = generar_contrato(fin="2026-04-27", dias=120)
    escribir_carpeta(d, tmp_path / "fuente")
    return tmp_path / "fuente"


@pytest.fixture
def preparada(tmp_path, fuente):
    carpeta = tmp_path / "ej"
    carpeta.mkdir()
    resumen = datos.preparar(ESC, fuente, carpeta, "a" * 40)
    return carpeta, resumen


def _filas(ruta):
    with open(ruta, newline="") as fh:
        return list(csv.DictReader(fh))


def test_sin_informacion_posterior_al_corte(preparada, fuente):
    carpeta, resumen = preparada
    h = _filas(carpeta / "datos" / "historico_ventas.csv")
    assert max(r["fecha"] for r in h) == ESC.corte == resumen["max_fecha_historico"]
    assert all(r["fecha"] <= ESC.corte for r in h)
    original = [r for r in _filas(fuente / "historico_ventas.csv") if r["fecha"] <= ESC.corte]
    assert h == original                                           # mismas filas, mismo orden
    for a in ("tiendas.csv", "familias.csv", "costos.csv", "feriados.csv"):
        assert (carpeta / "datos" / a).read_bytes() == (fuente / a).read_bytes()


def test_dias_simulados_y_referencia(preparada):
    carpeta, _ = preparada
    sim = _filas(carpeta / "trabajo" / "dias_simulados.csv")
    assert {r["fecha"] for r in sim} == {(ESC.inicio + timedelta(days=i)).isoformat() for i in range(ESC.dias)}
    assert min(r["fecha"] for r in sim) == "2026-04-06"             # el día 1 es corte + 1
    ref = _filas(carpeta / "evidencia" / "ventas_referencia.csv")
    assert min(r["fecha"] for r in ref) == (ESC.inicio - timedelta(days=28)).isoformat()
    assert max(r["fecha"] for r in ref) == ESC.fin.isoformat()


def test_primer_dia_sin_observacion_del_corte():
    """El día 1 no envía observaciones (las del corte ya están en el histórico); la primera
    observación es la del día 1, que se envía el día 2; el cierre envía la del último día."""
    enviadas = [COMPLETO.fecha(d - 1) for d in range(2, COMPLETO.dias + 2)]
    assert enviadas[0] == date(2025, 12, 1) and enviadas[-1] == date(2026, 2, 28)
    assert date(2025, 11, 30) not in enviadas and len(enviadas) == 90


def test_calendario_de_reentrenamientos():
    assert COMPLETO.reentrenos == (8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85)
    assert COMPLETO.fecha(8) == date(2025, 12, 8) and COMPLETO.fecha(47) == date(2026, 1, 16)
    assert COMPLETO.dia_mlflow not in COMPLETO.reentrenos
    assert (COMPLETO.inicio, COMPLETO.fin) == (date(2025, 12, 1), date(2026, 2, 28))


def test_cambio_demanda_es_independiente():
    assert CAMBIO_DEMANDA.huella() != COMPLETO.huella()
    assert not CAMBIO_DEMANDA.criterio_costo and COMPLETO.cambio_demanda is None


def test_cambio_demanda_solo_desde_su_dia(tmp_path, fuente):
    esc = replace(ESC, cambio_demanda={"familia": "bebidas", "factor": 1.3, "desde_dia": 4})
    carpeta = tmp_path / "cd"
    carpeta.mkdir()
    datos.preparar(esc, fuente, carpeta, "a" * 40)
    base = {(r["fecha"], r["tienda_id"], r["familia_id"]): int(r["ventas"])
            for r in _filas(fuente / "historico_ventas.csv")}
    for r in _filas(carpeta / "trabajo" / "dias_simulados.csv"):
        b = base[(r["fecha"], r["tienda_id"], r["familia_id"])]
        cambia = r["familia_id"] == "bebidas" and r["fecha"] >= esc.fecha(4).isoformat()
        assert int(r["ventas"]) == (datos.redondear(Decimal(b) * Decimal("1.3")) if cambia else b)


def test_cuadricula_incompleta_se_rechaza(tmp_path, fuente):
    lineas = (fuente / "historico_ventas.csv").read_text().splitlines()
    quitar = next(i for i, ln in enumerate(lineas) if ln.startswith(ESC.fin.isoformat()))
    (fuente / "historico_ventas.csv").write_text("\n".join(lineas[:quitar] + lineas[quitar + 1:]) + "\n")
    carpeta = tmp_path / "x"
    carpeta.mkdir()
    with pytest.raises(datos.DatosInvalidos, match="faltan"):
        datos.preparar(ESC, fuente, carpeta, "a" * 40)


# ------------------------------------------------------------------ evidencia
def _respuestas_sinteticas(carpeta: Path, rng=random.Random(0)) -> Path:
    conj = json.loads((carpeta / "evidencia" / "escenario.json").read_text())["conjunto"]
    dias = datos.leer_dias(carpeta / "trabajo")
    lineas = []
    seq = 0
    for f in sorted(dias):
        for t in sorted(dias[f]):
            seq += 1
            preds = [{"familia_id": fam, "venta_esperada": v * 1.0, "cantidad_sugerida": max(v + rng.randint(-5, 5), 0)}
                     for fam, (v, _) in dias[f][t].items()]
            lineas.append({"fase": "resultado", "tipo": "prediccion", "codigo": 200, "seq": seq, "fecha": f.isoformat(),
                           "tienda": t, "respuesta": {"version_modelo": "1", "predicciones": preds}})
    assert set(conj["familias"]) == set(next(iter(next(iter(dias.values())).values())))
    ruta = carpeta / "trabajo" / "respuestas.jsonl"
    ruta.write_text("".join(json.dumps(x) + "\n" for x in lineas))
    return ruta


def test_evidencia_se_escribe_una_sola_vez(preparada):
    carpeta, _ = preparada
    ev = carpeta / "evidencia"
    assert not evidencia.sellada(ev)                     # el manifiesto no existe antes del sellado
    with pytest.raises(evidencia.EvidenciaInvalida):
        evidencia.escribir_unica(ev / "costos.csv", b"otro")
    assert oct(os.stat(ev / "costos.csv").st_mode & 0o777) == "0o444"
    evidencia.sellar(ev, _respuestas_sinteticas(carpeta))
    with pytest.raises(evidencia.EvidenciaInvalida):
        evidencia.sellar(ev, carpeta / "trabajo" / "respuestas.jsonl")
    assert set(evidencia.verificar(ev)) == {"costos.csv", "escenario.json", "origen.json", "ventas_referencia.csv",
                                            "respuestas.jsonl"}


def test_recalcular_sin_datos_ni_raw(preparada, fuente):
    carpeta, _ = preparada
    resp = _respuestas_sinteticas(carpeta)
    vivo = costos.calcular(*costos.leer_evidencia(carpeta / "evidencia", resp))
    evidencia.sellar(carpeta / "evidencia", resp)
    import shutil
    shutil.rmtree(carpeta / "datos")
    shutil.rmtree(fuente)                                # raw/ ausente
    rec = costos.recalcular(carpeta / "evidencia")
    assert (rec["costo_servicio"], rec["costo_referencia"], rec["filas"]) == \
        (vivo["costo_servicio"], vivo["costo_referencia"], vivo["filas"])
    assert rec["filas"] == ESC.dias * 3 * 12


@pytest.mark.parametrize("archivo", ["costos.csv", "ventas_referencia.csv", "respuestas.jsonl", "escenario.json",
                                     "evidencia.sha256"])
def test_evidencia_alterada_se_niega(preparada, archivo):
    carpeta, _ = preparada
    ev = carpeta / "evidencia"
    evidencia.sellar(ev, _respuestas_sinteticas(carpeta))
    os.chmod(ev, 0o755)
    ruta = ev / archivo
    os.chmod(ruta, 0o644)
    contenido = bytearray(ruta.read_bytes())
    contenido[-2] ^= 1
    ruta.write_bytes(bytes(contenido))
    with pytest.raises(evidencia.EvidenciaInvalida):
        costos.recalcular(ev)


def test_evidencia_con_archivo_de_mas_se_niega(preparada):
    carpeta, _ = preparada
    ev = carpeta / "evidencia"
    evidencia.sellar(ev, _respuestas_sinteticas(carpeta))
    os.chmod(ev, 0o755)
    (ev / "extra.csv").write_text("x")
    with pytest.raises(evidencia.EvidenciaInvalida):
        evidencia.verificar(ev)


def test_borrar_evidencia_exige_confirmar(preparada):
    carpeta, _ = preparada
    with pytest.raises(evidencia.EvidenciaInvalida):
        evidencia.borrar(carpeta / "evidencia", False)
    assert (carpeta / "evidencia").exists()


# ------------------------------------------------------------------ costo y referencia
def test_costo_fila_coincide_con_ml_costos():
    rng = np.random.default_rng(1)
    for _ in range(500):
        real, q = int(rng.integers(0, 300)), int(rng.integers(0, 300))
        cf, cs = float(rng.choice([1.0, 1.5, 2.0])), float(rng.choice([0.1, 0.2, 2.5]))
        assert costos.costo_fila(real, q, cf, cs) == pytest.approx(float(costo(real, q, cf, cs)), abs=0)


def test_referencia_coincide_con_baseline_dow4():
    d = generar_contrato(fin="2026-04-27", dias=120)
    h = d.historico
    objetivo = pd.Timestamp("2026-04-20")
    obj = h[h["fecha"] == objetivo]
    X = construir_features(h[h["fecha"] < objetivo], obj, d.tiendas, Calendario(d.feriados),
                           list(d.familias["familia_id"]))
    pred = BaselineDiaSemana().ajustar(X, obj["ventas"]).predecir(X)
    esperado = cantidad_redondeo(pred)
    ventas = {(f.date(), int(t), fam): int(v) for f, t, fam, v in h[["fecha", "tienda_id", "familia_id", "ventas"]]
              .itertuples(index=False)}
    obtenido = [costos.referencia(ventas, int(t), fam, objetivo.date()) for t, fam in
                zip(obj["tienda_id"], obj["familia_id"])]
    assert obtenido == list(esperado)


def test_redondeo_aritmetico_de_la_referencia():
    v = {(date(2026, 1, 1) - timedelta(days=k), 1, "a"): x for k, x in zip((7, 14, 21, 28), (1, 1, 0, 0))}
    assert costos.referencia(v, 1, "a", date(2026, 1, 1)) == 1      # 0,5 -> 1 (no redondeo bancario)


def test_primera_respuesta_200_por_tienda_y_fecha():
    def r(seq, codigo, q):
        return {"fase": "resultado", "tipo": "prediccion", "codigo": codigo, "seq": seq, "fecha": "2026-01-01",
                "tienda": 1, "respuesta": {"version_modelo": "1", "predicciones": [
                    {"familia_id": "a", "cantidad_sugerida": q}]} if codigo == 200 else {"detail": "x"}}
    s = costos.sugerencias([r(3, 200, 9), r(1, 422, 0), r(2, 200, 5)])
    assert s[(date(2026, 1, 1), 1)]["cantidades"] == {"a": 5}


def test_defectos_deterministas_con_la_semilla():
    cat = defectos.Catalogo({1: date(2019, 1, 1), 2: date(2019, 1, 1)}, ["abarrotes", "lacteos", "frutas_verduras"],
                            {"abarrotes": "Abarrotes", "lacteos": "Lácteos", "frutas_verduras": "Frutas y verduras"})
    vals = {"abarrotes": 3, "lacteos": 0, "frutas_verduras": 1}

    def una(semilla):
        rng = random.Random(semilla)
        nombres = defectos.elegir(rng, "prediccion")
        s = defectos.generar("prediccion", 1, date(2026, 1, 5), dict(vals), cat, rng, nombres)
        return nombres, s.bytes(), s.esperado.como_dict()
    assert [una(i) for i in range(50)] == [una(i) for i in range(50)]
    assert len({una(i)[1] for i in range(50)}) > 20
