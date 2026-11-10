"""Fixtures compartidas.

Ninguna prueba lee ``raw/`` (los datos publicados no forman parte del
repositorio): todo sale de ``tests/fixtures/`` o de datos generados. El catálogo
de prueba es sintético: los identificadores de familia son los del contrato y
las tiendas cubren los casos de borde (la tienda 10 abre el 2025-03-15).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.calidad.catalogo import Catalogo

FIXTURES = Path(__file__).resolve().parent / "fixtures"

FAMILIAS = {
    "abarrotes": "Abarrotes",
    "bebidas": "Bebidas",
    "lacteos": "Lácteos",
    "carnes": "Carnes",
    "panaderia": "Panadería",
    "frutas_verduras": "Frutas y verduras",
    "limpieza": "Limpieza",
    "cuidado_personal": "Cuidado personal",
    "licores": "Licores",
    "snacks": "Snacks y dulces",
    "bebe": "Bebé",
    "mascotas": "Mascotas",
}

PROMOS_VALIDAS = {f: 0 for f in FAMILIAS} | {"bebidas": 8, "snacks": 9}
VENTAS_VALIDAS = {f: 100 + i for i, f in enumerate(FAMILIAS)}


@pytest.fixture(scope="session")
def catalogo() -> Catalogo:
    """Catálogo sintético cargado con el mismo lector que usa el servicio."""
    c = Catalogo.desde_csv(FIXTURES / "catalogo")
    assert c.familias == FAMILIAS
    return c


@pytest.fixture
def prediccion_valida() -> dict:
    return copy.deepcopy({
        "tienda_id": 5,
        "fecha": "2025-11-17",
        "familias": [{"familia_id": f, "en_promocion": p} for f, p in PROMOS_VALIDAS.items()],
    })


@pytest.fixture
def observacion_valida() -> dict:
    return copy.deepcopy({
        "tienda_id": 5,
        "fecha": "2025-11-17",
        "ventas": [{"familia_id": f, "ventas": v} for f, v in VENTAS_VALIDAS.items()],
    })
