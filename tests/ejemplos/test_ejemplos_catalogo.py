"""Política sobre un ejemplo por cada caso del catálogo de errores (E01–E10).

Los ejemplos de ``tests/fixtures/ejemplos`` son sintéticos: reproducen la
estructura y el problema de cada caso publicado, con valores inventados, para
que las pruebas se ejecuten desde un clon limpio sin ``raw/``.
"""

import json

import pytest

from app.calidad import SolicitudRechazada, limpiar_solicitud_observacion, limpiar_solicitud_prediccion
from tests.conftest import FIXTURES

EJEMPLOS = FIXTURES / "ejemplos"

# archivo -> (función, estado esperado o "rechazo", acciones esperadas)
ESPERADO = {
    "prediccion_valida.json": ("p", "ok", []),
    "errores/E01_tienda_texto.json": ("p", "corregido", [("tienda_id", "reparar")]),
    "errores/E02_familia_otra_forma.json": ("p", "corregido", [("familias[5].familia_id", "reparar"),
                                                               ("familias[6].familia_id", "reparar")]),
    "errores/E03_promocion_texto.json": ("p", "corregido", [("familias[2].en_promocion", "reparar"),
                                                            ("familias[11].en_promocion", "reparar")]),
    "errores/E04_fecha_formato.json": ("p", "corregido", [("fecha", "reparar")]),
    "errores/E05_faltan_familias.json": ("p", "advertencia", [("familias", "advertir")] * 2),
    "errores/E06_familia_repetida.json": ("p", "rechazo", None),
    "errores/E07_campo_renombrado.json": ("p", "corregido", [("tienda_id", "reparar")]),
    "observacion_valida.json": ("o", "ok", []),
    "errores/E08_ventas_negativas.json": ("o", "advertencia", [("ventas[5].ventas", "advertir")]),
    "errores/E09_envio_repetido.json": ("o", "ok", []),
    "errores/E10_numero_formato.json": ("o", "corregido", [("ventas[0].ventas", "reparar"),
                                                           ("ventas[1].ventas", "reparar")]),
}


def cargar(archivo):
    return json.loads((EJEMPLOS / archivo).read_text(encoding="utf-8"))


def test_hay_un_ejemplo_por_caso_del_catalogo():
    presentes = {str(p.relative_to(EJEMPLOS)) for p in EJEMPLOS.rglob("*.json")}
    assert presentes == set(ESPERADO)
    assert {f"E{i:02d}" for i in range(1, 11)} == {p.split("/")[1][:3] for p in presentes if "/" in p}


@pytest.mark.parametrize("archivo", sorted(ESPERADO))
def test_ejemplo(catalogo, archivo):
    tipo, estado, acciones = ESPERADO[archivo]
    fn = limpiar_solicitud_prediccion if tipo == "p" else limpiar_solicitud_observacion
    if estado == "rechazo":
        with pytest.raises(SolicitudRechazada, match="repetida con valores distintos"):
            fn(cargar(archivo), catalogo)
        return
    _, registro = fn(cargar(archivo), catalogo)
    assert registro.estado == estado
    assert [(o.campo, o.accion) for o in registro.observaciones] == acciones


def test_valores_reparados(catalogo):
    s, _ = limpiar_solicitud_observacion(cargar("errores/E10_numero_formato.json"), catalogo)
    v = {x.familia_id: x.ventas for x in s.ventas}
    assert v["abarrotes"] == 1305 and v["bebe"] == 41
    s, _ = limpiar_solicitud_prediccion(cargar("errores/E04_fecha_formato.json"), catalogo)
    assert s.fecha.isoformat() == "2026-04-14"
    s, _ = limpiar_solicitud_prediccion(cargar("errores/E01_tienda_texto.json"), catalogo)
    assert s.tienda_id == 5


def test_e08_no_guarda_la_venta_negativa(catalogo):
    s, _ = limpiar_solicitud_observacion(cargar("errores/E08_ventas_negativas.json"), catalogo)
    assert "frutas_verduras" not in {x.familia_id for x in s.ventas} and len(s.ventas) == 11


def test_e09_reenvio_da_la_misma_solicitud(catalogo):
    a, _ = limpiar_solicitud_observacion(cargar("observacion_valida.json"), catalogo)
    b, _ = limpiar_solicitud_observacion(cargar("errores/E09_envio_repetido.json"), catalogo)
    assert a == b
