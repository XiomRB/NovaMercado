import pytest

from app.calidad import SolicitudRechazada, limpiar_solicitud_observacion
from app.schemas.contrato import Calidad


def limpiar(payload, catalogo):
    solicitud, registro = limpiar_solicitud_observacion(payload, catalogo)
    Calidad.model_validate(registro.como_dict())
    return solicitud, registro


def ventas(s):
    return {v.familia_id: v.ventas for v in s.ventas}


def acciones(r):
    return [(o.campo, o.accion) for o in r.observaciones]


def test_valida(catalogo, observacion_valida):
    s, r = limpiar(observacion_valida, catalogo)
    assert len(s.ventas) == 12 and r.estado == "ok"


# ---------------------------------------------------------------- E08
def test_e08_venta_negativa_no_se_guarda(catalogo, observacion_valida):
    observacion_valida["ventas"][5]["ventas"] = -4
    s, r = limpiar(observacion_valida, catalogo)
    assert "frutas_verduras" not in ventas(s) and len(s.ventas) == 11
    assert acciones(r) == [("ventas[5].ventas", "advertir")]
    assert r.estado == "advertencia"


@pytest.mark.parametrize("valor", [None, "abc", True, [], "-4", "-1,234"])
def test_venta_no_confiable_no_se_guarda(catalogo, observacion_valida, valor):
    observacion_valida["ventas"][0]["ventas"] = valor
    s, r = limpiar(observacion_valida, catalogo)
    assert "abarrotes" not in ventas(s)
    assert acciones(r) == [("ventas[0].ventas", "advertir")]


def test_todas_invalidas_se_rechaza(catalogo, observacion_valida):
    for v in observacion_valida["ventas"]:
        v["ventas"] = -1
    with pytest.raises(SolicitudRechazada, match="ningún registro"):
        limpiar(observacion_valida, catalogo)


def test_venta_cero_es_valida(catalogo, observacion_valida):
    observacion_valida["ventas"][0]["ventas"] = 0
    s, r = limpiar(observacion_valida, catalogo)
    assert ventas(s)["abarrotes"] == 0 and r.estado == "ok"


# ---------------------------------------------------------------- E09
def test_e09_envio_repetido_produce_la_misma_solicitud(catalogo, observacion_valida):
    """La idempotencia se garantiza en la persistencia; la limpieza debe ser determinista."""
    a, _ = limpiar(observacion_valida, catalogo)
    b, _ = limpiar(observacion_valida, catalogo)
    assert a == b


# ---------------------------------------------------------------- E10
@pytest.mark.parametrize("valor, esperado, accion", [
    ("1,234", 1234, "reparar"),
    ("1,292", 1292, "reparar"),
    ("812,0", 812, "reparar"),
    ("50,0", 50, "reparar"),
    ("1.234,0", 1234, "reparar"),
    (812.0, 812, "reparar"),
    ("994", 994, "reparar"),
    ("812,5", 813, "advertir"),
    (812.4, 812, "advertir"),
])
def test_e10_numero_con_formato(catalogo, observacion_valida, valor, esperado, accion):
    observacion_valida["ventas"][0]["ventas"] = valor
    s, r = limpiar(observacion_valida, catalogo)
    assert ventas(s)["abarrotes"] == esperado
    assert acciones(r) == [("ventas[0].ventas", accion)]


# ------------------------------------------------- reglas compartidas
def test_familia_y_tienda_y_fecha_se_normalizan(catalogo, observacion_valida):
    observacion_valida["tienda_id"] = "T05"
    observacion_valida["fecha"] = "17/11/2025"
    observacion_valida["ventas"][2]["familia_id"] = "LÁCTEOS"
    s, r = limpiar(observacion_valida, catalogo)
    assert s.tienda_id == 5 and "lacteos" in ventas(s) and r.estado == "corregido"


def test_repetida_identica_se_elimina_distinta_se_rechaza(catalogo, observacion_valida):
    observacion_valida["ventas"].append(dict(observacion_valida["ventas"][0]))
    s, r = limpiar(observacion_valida, catalogo)
    assert len(s.ventas) == 12 and r.estado == "corregido"
    observacion_valida["ventas"][-1]["ventas"] = 1
    with pytest.raises(SolicitudRechazada, match="repetida"):
        limpiar(observacion_valida, catalogo)


def test_familias_faltantes_se_advierten_sin_inventar(catalogo, observacion_valida):
    observacion_valida["ventas"] = observacion_valida["ventas"][:10]
    s, r = limpiar(observacion_valida, catalogo)
    assert len(s.ventas) == 10
    assert acciones(r) == [("ventas", "advertir")]


def test_campos_renombrados(catalogo, observacion_valida):
    observacion_valida["observaciones"] = observacion_valida.pop("ventas")
    observacion_valida["observaciones"][0] = {"familia": "abarrotes", "unidades_vendidas": 5}
    s, r = limpiar(observacion_valida, catalogo)
    assert ventas(s)["abarrotes"] == 5
    assert all(a == "reparar" for _, a in acciones(r))


def test_fecha_previa_a_apertura(catalogo, observacion_valida):
    observacion_valida.update(tienda_id=10, fecha="2025-01-01")
    with pytest.raises(SolicitudRechazada, match="apertura"):
        limpiar(observacion_valida, catalogo)


@pytest.mark.parametrize("campo", ["tienda_id", "fecha", "ventas"])
def test_falta_campo_obligatorio(catalogo, observacion_valida, campo):
    del observacion_valida[campo]
    with pytest.raises(SolicitudRechazada):
        limpiar(observacion_valida, catalogo)
