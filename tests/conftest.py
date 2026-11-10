"""Fixtures compartidas.

Las pruebas unitarias usan un catálogo *sintético* (no leen ``raw/``, que no
forma parte del repositorio). Los identificadores de familia son los del
contrato; las tiendas se eligen para cubrir los casos de borde (tienda 10 con
apertura posterior).
"""

from __future__ import annotations

import copy
from datetime import date

import pytest

from app.calidad.catalogo import Catalogo, Tienda

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
    tiendas = {
        1: Tienda(1, "Guatemala", "hipermercado", date(2019, 1, 1)),
        5: Tienda(5, "Quetzaltenango", "hipermercado", date(2019, 1, 1)),
        10: Tienda(10, "Guatemala", "express", date(2025, 3, 15)),
    }
    return Catalogo(tiendas, FAMILIAS)


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
