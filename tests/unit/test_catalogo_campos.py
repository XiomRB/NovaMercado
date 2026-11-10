import pytest

from app.calidad.campos import ESQUEMA_PREDICCION, normalizar_campos
from app.calidad.resultado import RegistroCalidad, SolicitudRechazada
from app.calidad.texto import normalizar_clave


@pytest.mark.parametrize("entrada, esperado", [
    ("  Frutas y Verduras ", "frutas_y_verduras"),
    ("LÁCTEOS", "lacteos"),
    ("tiendaId", "tienda_id"),
    ("Cuidado-Personal", "cuidado_personal"),
])
def test_normalizar_clave(entrada, esperado):
    assert normalizar_clave(entrada) == esperado


@pytest.mark.parametrize("entrada, esperado, accion", [
    ("abarrotes", "abarrotes", None),
    ("LÁCTEOS", "lacteos", "reparar"),                  # E02
    ("lacteos ", "lacteos", "reparar"),                 # E02
    ("Frutas y Verduras", "frutas_verduras", "reparar"),  # E02: nombre del catálogo
    ("frutas-verduras", "frutas_verduras", "reparar"),
    ("FrutasVerduras", "frutas_verduras", "reparar"),
    ("Snacks y dulces", "snacks", "reparar"),
    ("Bebé", "bebe", "reparar"),
    ("cuidado personal", "cuidado_personal", "reparar"),
    ("lacteo", "lacteos", "advertir"),                  # errata: supuesto
    ("carne", "carnes", "advertir"),
    ("bebida", "bebidas", "advertir"),
    ("panaderias", "panaderia", "advertir"),
])
def test_resolver_familia(catalogo, entrada, esperado, accion):
    fid, acc, detalle = catalogo.resolver_familia(entrada)
    assert fid == esperado and acc == accion
    assert (detalle is None) == (accion is None)


@pytest.mark.parametrize("entrada", ["electronica", "xyz", "", "   ", None, 3, ["abarrotes"], "pan"])
def test_familia_no_reconocida(catalogo, entrada):
    assert catalogo.resolver_familia(entrada) is None


def test_campos_exactos_sin_observaciones():
    r = RegistroCalidad()
    datos = normalizar_campos({"tienda_id": 5, "fecha": "x", "familias": []}, ESQUEMA_PREDICCION, "", r)
    assert set(datos) == {"tienda_id", "fecha", "familias"} and r.observaciones == []


@pytest.mark.parametrize("clave, accion", [
    ("tienda", "reparar"),        # E07
    ("Tienda_ID", "reparar"),
    ("tiendaId", "reparar"),
    ("store_id", "reparar"),
    ("tienda_di", "advertir"),    # errata
])
def test_campo_renombrado(clave, accion):
    r = RegistroCalidad()
    datos = normalizar_campos({clave: 5}, ESQUEMA_PREDICCION, "", r)
    assert datos == {"tienda_id": 5}
    assert [o.accion for o in r.observaciones] == [accion]


def test_campo_desconocido_se_ignora_con_advertencia():
    r = RegistroCalidad()
    datos = normalizar_campos({"tienda_id": 5, "comentario": "hola"}, ESQUEMA_PREDICCION, "", r)
    assert datos == {"tienda_id": 5}
    assert r.observaciones[0].accion == "advertir" and r.observaciones[0].campo == "comentario"


def test_campo_duplicado_igual_se_repara_distinto_se_rechaza():
    r = RegistroCalidad()
    assert normalizar_campos({"tienda_id": 5, "tienda": 5}, ESQUEMA_PREDICCION, "", r) == {"tienda_id": 5}
    assert r.observaciones[0].accion == "reparar"
    with pytest.raises(SolicitudRechazada):
        normalizar_campos({"tienda_id": 5, "tienda": 6}, ESQUEMA_PREDICCION, "", RegistroCalidad())


@pytest.mark.parametrize("clave", ["id", "dia", "day", "items", "detalle", "productos", "categoria"])
def test_nombres_genericos_no_son_sinonimos(clave):
    """Solo se reparan sinónimos inequívocos; los genéricos se ignoran con advertencia."""
    from app.calidad.campos import ESQUEMA_PREDICCION_ITEM
    for esquema in (ESQUEMA_PREDICCION, ESQUEMA_PREDICCION_ITEM):
        r = RegistroCalidad()
        assert normalizar_campos({clave: 5}, esquema, "", r) == {}
        assert [o.accion for o in r.observaciones] == ["advertir"]
        assert "se ignoró" in r.observaciones[0].detalle


def test_id_no_identifica_tienda(catalogo, prediccion_valida):
    from app.calidad import limpiar_solicitud_prediccion
    prediccion_valida["id"] = prediccion_valida.pop("tienda_id")
    with pytest.raises(SolicitudRechazada, match="tienda_id: falta el campo obligatorio"):
        limpiar_solicitud_prediccion(prediccion_valida, catalogo)
