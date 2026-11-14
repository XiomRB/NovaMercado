"""El oráculo del simulador (DECISIONES.md §3) frente a la API real con persistencia en
memoria (``tests/api``): cada defecto, solo y combinado, debe producir exactamente el
resultado que declara. Una discrepancia aquí es una discrepancia entre el documento y el
producto: se informa, no se ajusta el oráculo."""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

import pytest

from simulador import contrato, defectos
from tests.api.conftest import DATOS, FECHA

FIX = Path(__file__).resolve().parents[1] / "fixtures"
CATALOGO = defectos.Catalogo(
    tiendas={int(t.tienda_id): t.fecha_apertura.date() for t in DATOS.tiendas.itertuples()},
    familias=list(DATOS.familias["familia_id"]),
    nombres=dict(zip(DATOS.familias["familia_id"], DATOS.familias["nombre"])))
MAXIMOS = DATOS.historico.groupby("familia_id")["en_promocion"].max().astype(int).to_dict()
PRED = json.loads((FIX / "ejemplos" / "prediccion_valida.json").read_text())
OBS = json.loads((FIX / "ejemplos" / "observacion_valida.json").read_text())
PROMOS = {f["familia_id"]: f["en_promocion"] for f in PRED["familias"]}
VENTAS = {v["familia_id"]: v["ventas"] for v in OBS["ventas"]} | {"abarrotes": 1730}
NOMBRES = sorted(defectos.DEFECTOS)


def _enviar(cliente, tipo, sol):
    ruta = "/v1/prediccion" if tipo == "prediccion" else "/v1/observaciones"
    r = cliente.post(ruta, content=sol.bytes(), headers={"Content-Type": "application/json"})
    return r.status_code, r.json()


def _verificar(cliente, tipo, nombres, semilla):
    rng = random.Random(semilla)
    valores = PROMOS if tipo == "prediccion" else VENTAS
    sol = defectos.generar(tipo, PRED["tienda_id"], date.fromisoformat(FECHA), dict(valores), CATALOGO, rng, nombres)
    codigo, cuerpo = _enviar(cliente, tipo, sol)
    req, opc = (defectos.contexto_prediccion(sol.esperado, MAXIMOS, MAXIMOS, False)
                if tipo == "prediccion" else ([], []))
    errores = defectos.comparar(sol.esperado, codigo, cuerpo, req, opc)
    if codigo == 200:
        errores += (contrato.prediccion(cuerpo, PRED["tienda_id"], FECHA, CATALOGO.familias)
                    if tipo == "prediccion" else contrato.observacion(cuerpo))
    elif codigo == 422:
        errores += contrato.rechazo(cuerpo)
    assert not errores, (nombres, sol.cuerpo if sol.texto is None else sol.texto, cuerpo, errores)
    return sol, cuerpo


CASOS = [(t, n) for n in NOMBRES for t in ("prediccion", "observacion")
         if not (t == "observacion" and n in defectos.SOLO_PREDICCION)]


@pytest.mark.parametrize("tipo, nombre", CASOS)
def test_cada_defecto(cliente, tipo, nombre):
    for semilla in range(3):
        sol, cuerpo = _verificar(cliente, tipo, [nombre], semilla)
    if tipo == "observacion" and sol.esperado.codigo == 200:
        assert cuerpo["registros_guardados"] >= 0


@pytest.mark.parametrize("tipo", ["prediccion", "observacion"])
def test_combinaciones(cliente, tipo):
    rng = random.Random(7)
    for i in range(150):
        _verificar(cliente, tipo, defectos.elegir(rng, tipo), 1000 + i)


def test_limpia_es_ok(cliente):
    sol, cuerpo = _verificar(cliente, "prediccion", [], 0)
    assert cuerpo["calidad"] == {"estado": "ok", "observaciones": []}
    assert sol.esperado.valores == PROMOS


def test_observacion_guardadas(cliente):
    sol = defectos.generar("observacion", 5, date.fromisoformat(FECHA), dict(VENTAS), CATALOGO,
                           random.Random(0), ["valor_negativo"])
    descartada = next(f for f in VENTAS if f not in sol.esperado.valores)
    codigo, cuerpo = _enviar(cliente, "observacion", sol)
    assert codigo == 200 and cuerpo["registros_guardados"] == len(VENTAS) - 1
    assert descartada not in sol.esperado.valores and len(sol.esperado.valores) == len(VENTAS) - 1
