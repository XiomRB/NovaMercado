"""Comprueba la política sobre los ejemplos publicados en ``raw/ejemplos``.

``raw/`` no forma parte del repositorio: si los archivos no están, se omite.
"""

import json
from pathlib import Path

import pytest

from app.calidad import (Catalogo, SolicitudRechazada, limpiar_solicitud_observacion,
                         limpiar_solicitud_prediccion)

RAW = Path(__file__).resolve().parents[2] / "raw"
EJEMPLOS = RAW / "ejemplos"

pytestmark = pytest.mark.skipif(not (EJEMPLOS / "errores").is_dir(), reason="raw/ejemplos no disponible")

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
    "errores/E10_numero_formato.json": ("o", "corregido", [("ventas[1].ventas", "reparar"),
                                                           ("ventas[2].ventas", "reparar")]),
}


@pytest.fixture(scope="module")
def catalogo_real():
    return Catalogo.desde_csv(RAW)


def test_todos_los_ejemplos_estan_cubiertos():
    publicados = {str(p.relative_to(EJEMPLOS)) for p in EJEMPLOS.rglob("*.json")}
    assert publicados == set(ESPERADO), "hay ejemplos publicados sin caso de prueba"


@pytest.mark.parametrize("archivo", sorted(ESPERADO))
def test_ejemplo(catalogo_real, archivo):
    tipo, estado, acciones = ESPERADO[archivo]
    payload = json.loads((EJEMPLOS / archivo).read_text(encoding="utf-8"))
    fn = limpiar_solicitud_prediccion if tipo == "p" else limpiar_solicitud_observacion
    if estado == "rechazo":
        with pytest.raises(SolicitudRechazada):
            fn(payload, catalogo_real)
        return
    _, registro = fn(payload, catalogo_real)
    assert registro.estado == estado
    assert [(o.campo, o.accion) for o in registro.observaciones] == acciones


def test_e10_valores(catalogo_real):
    payload = json.loads((EJEMPLOS / "errores/E10_numero_formato.json").read_text(encoding="utf-8"))
    s, _ = limpiar_solicitud_observacion(payload, catalogo_real)
    v = {x.familia_id: x.ventas for x in s.ventas}
    assert v["bebe"] == 50 and v["bebidas"] == 1292
