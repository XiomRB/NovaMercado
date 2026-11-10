from datetime import date

import pytest

from app.calidad import SolicitudRechazada, limpiar_solicitud_prediccion
from app.schemas.contrato import Calidad

from tests.conftest import FAMILIAS


def limpiar(payload, catalogo):
    solicitud, registro = limpiar_solicitud_prediccion(payload, catalogo)
    Calidad.model_validate(registro.como_dict())  # siempre coherente con el contrato
    return solicitud, registro


def acciones(registro):
    return [(o.campo, o.accion) for o in registro.observaciones]


def promos(solicitud):
    return {f.familia_id: f.en_promocion for f in solicitud.familias}


def test_valida_estado_ok(catalogo, prediccion_valida):
    s, r = limpiar(prediccion_valida, catalogo)
    assert r.estado == "ok" and r.observaciones == []
    assert s.tienda_id == 5 and s.fecha == date(2025, 11, 17)
    assert [f.familia_id for f in s.familias] == list(FAMILIAS)


# ---------------------------------------------------------------- E01
@pytest.mark.parametrize("valor", ["T05", "5", " 5 ", "T-05", "tienda 5", "#5", 5.0])
def test_e01_tienda_reparada(catalogo, prediccion_valida, valor):
    prediccion_valida["tienda_id"] = valor
    s, r = limpiar(prediccion_valida, catalogo)
    assert s.tienda_id == 5
    assert r.estado == "corregido" and acciones(r) == [("tienda_id", "reparar")]


@pytest.mark.parametrize("valor", ["T5-2", "abc", "", "-5", 5.5, True, None, [5], 99, 0, -1])
def test_e01_tienda_rechazada(catalogo, prediccion_valida, valor):
    prediccion_valida["tienda_id"] = valor
    with pytest.raises(SolicitudRechazada) as exc:
        limpiar(prediccion_valida, catalogo)
    assert exc.value.campo == "tienda_id"


# ---------------------------------------------------------------- E02
def test_e02_familias_otra_forma(catalogo, prediccion_valida):
    prediccion_valida["familias"][2]["familia_id"] = "LÁCTEOS"
    prediccion_valida["familias"][5]["familia_id"] = "Frutas y Verduras"
    prediccion_valida["familias"][0]["familia_id"] = "abarrotes "
    s, r = limpiar(prediccion_valida, catalogo)
    assert set(promos(s)) == set(FAMILIAS)
    assert r.estado == "corregido"
    assert sorted(acciones(r)) == sorted([("familias[0].familia_id", "reparar"),
                                          ("familias[2].familia_id", "reparar"),
                                          ("familias[5].familia_id", "reparar")])


def test_e02_familia_por_similitud_advierte(catalogo, prediccion_valida):
    prediccion_valida["familias"][2]["familia_id"] = "lacteo"
    s, r = limpiar(prediccion_valida, catalogo)
    assert "lacteos" in promos(s)
    assert r.estado == "advertencia"


def test_familia_desconocida_se_ignora_y_se_completa(catalogo, prediccion_valida):
    prediccion_valida["familias"][2]["familia_id"] = "electronica"
    s, r = limpiar(prediccion_valida, catalogo)
    assert "electronica" not in promos(s) and promos(s)["lacteos"] == 0
    assert ("familias[2].familia_id", "advertir") in acciones(r)
    assert ("familias", "advertir") in acciones(r)


# ---------------------------------------------------------------- E03
@pytest.mark.parametrize("valor, esperado", [("3", 3), (2.0, 2), (" 4 ", 4), ("1,000", 1000), ("8.0", 8)])
def test_e03_promocion_reparada(catalogo, prediccion_valida, valor, esperado):
    prediccion_valida["familias"][1]["en_promocion"] = valor
    s, r = limpiar(prediccion_valida, catalogo)
    assert promos(s)["bebidas"] == esperado
    assert acciones(r) == [("familias[1].en_promocion", "reparar")]


@pytest.mark.parametrize("valor, esperado", [
    (2.5, 3), ("2,5", 3), (-3, 0), ("tres", 0), (None, 0), (True, 0), ([1], 0), ({}, 0),
])
def test_e03_promocion_con_supuesto(catalogo, prediccion_valida, valor, esperado):
    prediccion_valida["familias"][1]["en_promocion"] = valor
    s, r = limpiar(prediccion_valida, catalogo)
    assert promos(s)["bebidas"] == esperado
    assert acciones(r) == [("familias[1].en_promocion", "advertir")]
    assert r.estado == "advertencia"


def test_e03_promocion_ausente(catalogo, prediccion_valida):
    del prediccion_valida["familias"][1]["en_promocion"]
    s, r = limpiar(prediccion_valida, catalogo)
    assert promos(s)["bebidas"] == 0 and r.estado == "advertencia"


# ---------------------------------------------------------------- E04
@pytest.mark.parametrize("valor, accion", [
    ("2025-11-17T00:00:00", "reparar"), ("17/11/2025", "reparar"), ("05/11/2025", "advertir"),
])
def test_e04_fecha(catalogo, prediccion_valida, valor, accion):
    prediccion_valida["fecha"] = valor
    _, r = limpiar(prediccion_valida, catalogo)
    assert acciones(r) == [("fecha", accion)]


@pytest.mark.parametrize("valor", ["2025-02-30", "ayer", 20251117, None, ""])
def test_e04_fecha_rechazada(catalogo, prediccion_valida, valor):
    prediccion_valida["fecha"] = valor
    with pytest.raises(SolicitudRechazada) as exc:
        limpiar(prediccion_valida, catalogo)
    assert exc.value.campo == "fecha"


def test_fecha_anterior_a_apertura_rechazada(catalogo, prediccion_valida):
    prediccion_valida.update(tienda_id=10, fecha="2025-03-14")
    with pytest.raises(SolicitudRechazada, match="apertura"):
        limpiar(prediccion_valida, catalogo)
    prediccion_valida["fecha"] = "2025-03-15"
    assert limpiar(prediccion_valida, catalogo)[1].estado == "ok"


@pytest.mark.parametrize("fecha", ["2024-03-01", "2026-03-01", "2026-12-31", "2030-01-01"])
def test_fechas_dentro_y_despues_del_historico_se_aceptan(catalogo, prediccion_valida, fecha):
    prediccion_valida["fecha"] = fecha
    assert limpiar(prediccion_valida, catalogo)[0].fecha == date.fromisoformat(fecha)


# ---------------------------------------------------------------- E05 (D1)
def test_e05_faltan_familias_se_completan(catalogo, prediccion_valida):
    prediccion_valida["familias"] = [f for f in prediccion_valida["familias"]
                                     if f["familia_id"] not in ("bebe", "mascotas")]
    s, r = limpiar(prediccion_valida, catalogo)
    assert len(s.familias) == 12
    assert promos(s)["bebe"] == 0 and promos(s)["mascotas"] == 0
    assert promos(s)["bebidas"] == 8
    assert acciones(r) == [("familias", "advertir")] * 2
    assert r.estado == "advertencia"


# ---------------------------------------------------------------- E06 (D2)
def test_e06_repetida_con_valores_distintos_se_rechaza(catalogo, prediccion_valida):
    prediccion_valida["familias"].append({"familia_id": "abarrotes", "en_promocion": 3})
    with pytest.raises(SolicitudRechazada, match="repetida con valores distintos"):
        limpiar(prediccion_valida, catalogo)


@pytest.mark.parametrize("duplicado", [
    {"familia_id": "abarrotes", "en_promocion": 0},
    {"familia_id": "ABARROTES", "en_promocion": "0"},   # igual tras normalizar
])
def test_e06_repetida_identica_se_elimina(catalogo, prediccion_valida, duplicado):
    prediccion_valida["familias"].append(duplicado)
    s, r = limpiar(prediccion_valida, catalogo)
    assert len(s.familias) == 12
    assert ("familias[12].familia_id", "reparar") in acciones(r)
    assert r.estado == "corregido"


def test_e06_repetida_por_otra_forma_con_valor_distinto(catalogo, prediccion_valida):
    prediccion_valida["familias"].append({"familia_id": "Bebidas", "en_promocion": 1})
    with pytest.raises(SolicitudRechazada):
        limpiar(prediccion_valida, catalogo)


# ---------------------------------------------------------------- E07
def test_e07_campo_renombrado(catalogo, prediccion_valida):
    prediccion_valida["tienda"] = prediccion_valida.pop("tienda_id")
    s, r = limpiar(prediccion_valida, catalogo)
    assert s.tienda_id == 5 and acciones(r) == [("tienda_id", "reparar")]


def test_e07_campos_de_item_renombrados(catalogo, prediccion_valida):
    prediccion_valida["familias"][0] = {"familia": "abarrotes", "promocion": 2}
    s, r = limpiar(prediccion_valida, catalogo)
    assert promos(s)["abarrotes"] == 2
    assert acciones(r) == [("familias[0].familia_id", "reparar"), ("familias[0].en_promocion", "reparar")]


@pytest.mark.parametrize("campo", ["tienda_id", "fecha", "familias"])
def test_falta_campo_obligatorio(catalogo, prediccion_valida, campo):
    del prediccion_valida[campo]
    with pytest.raises(SolicitudRechazada, match="falta el campo obligatorio"):
        limpiar(prediccion_valida, catalogo)


def test_campos_contradictorios_se_rechazan(catalogo, prediccion_valida):
    prediccion_valida["tienda"] = 1
    with pytest.raises(SolicitudRechazada):
        limpiar(prediccion_valida, catalogo)


# --------------------------------------------------- fuera del catálogo
@pytest.mark.parametrize("payload", [None, [], "texto", 5, [{"tienda_id": 5}]])
def test_cuerpo_no_objeto(catalogo, payload):
    with pytest.raises(SolicitudRechazada, match="objeto JSON"):
        limpiar(payload, catalogo)


@pytest.mark.parametrize("familias", [[], {}, "abarrotes", 3, [1, 2], [{"familia_id": "xyz"}]])
def test_sin_familias_utilizables_se_rechaza(catalogo, prediccion_valida, familias):
    prediccion_valida["familias"] = familias
    with pytest.raises(SolicitudRechazada) as exc:
        limpiar(prediccion_valida, catalogo)
    assert exc.value.campo == "familias"


def test_familias_como_objeto_se_convierte(catalogo, prediccion_valida):
    prediccion_valida["familias"] = {f["familia_id"]: f["en_promocion"] for f in prediccion_valida["familias"]}
    s, r = limpiar(prediccion_valida, catalogo)
    assert promos(s)["bebidas"] == 8 and acciones(r) == [("familias", "reparar")]


def test_elemento_no_objeto_se_ignora(catalogo, prediccion_valida):
    prediccion_valida["familias"].append("abarrotes")
    s, r = limpiar(prediccion_valida, catalogo)
    assert len(s.familias) == 12 and ("familias[12]", "advertir") in acciones(r)


def test_campo_extra_se_advierte(catalogo, prediccion_valida):
    prediccion_valida["origen"] = "pos-v2"
    prediccion_valida["familias"][0]["precio"] = 10
    _, r = limpiar(prediccion_valida, catalogo)
    assert acciones(r) == [("origen", "advertir"), ("familias[0].precio", "advertir")]


def test_varios_problemas_a_la_vez(catalogo, prediccion_valida):
    prediccion_valida["tienda"] = "T05"
    del prediccion_valida["tienda_id"]
    prediccion_valida["fecha"] = "17/11/2025"
    prediccion_valida["familias"][2]["familia_id"] = "LÁCTEOS"
    prediccion_valida["familias"][1]["en_promocion"] = "8"
    s, r = limpiar(prediccion_valida, catalogo)
    assert s.tienda_id == 5 and s.fecha == date(2025, 11, 17) and promos(s)["bebidas"] == 8
    assert r.estado == "corregido" and len(r.observaciones) == 5


def test_no_modifica_el_payload_original(catalogo, prediccion_valida):
    import copy
    original = copy.deepcopy(prediccion_valida)
    prediccion_valida["fecha"] = "17/11/2025"
    copia = copy.deepcopy(prediccion_valida)
    limpiar(prediccion_valida, catalogo)
    assert prediccion_valida == copia and original != copia
