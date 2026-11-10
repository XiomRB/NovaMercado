"""Mutaciones aleatorias (deterministas) de solicitudes válidas.

Garantía que se prueba: cualquier entrada termina en una solicitud válida con
``calidad`` coherente, o en :class:`SolicitudRechazada`. Nunca en otra excepción
(que en la API sería un 500).
"""

import copy
import random

import pytest

from app.calidad import SolicitudRechazada, limpiar_solicitud_observacion, limpiar_solicitud_prediccion
from app.schemas.contrato import Calidad

VALORES_RAROS = [
    None, True, False, 0, -1, 1.5, float("nan"), float("inf"), 10**30, "", " ", "x", "T05",
    "1,234", "12,5", "2025-11-17", "17/11/2025", [], [1], {}, {"a": 1}, "LÁCTEOS", "ñ" * 500,
]


def _mutar(obj, rng):
    obj = copy.deepcopy(obj)
    for _ in range(rng.randint(1, 4)):
        destino = obj
        lista_clave = "familias" if "familias" in obj else "ventas"
        if rng.random() < 0.6 and isinstance(obj.get(lista_clave), list) and obj[lista_clave]:
            destino = rng.choice(obj[lista_clave])
            if not isinstance(destino, dict):
                continue
        if not destino:
            continue
        clave = rng.choice(list(destino))
        op = rng.random()
        if op < 0.5:
            destino[clave] = rng.choice(VALORES_RAROS)
        elif op < 0.7:
            del destino[clave]
        elif op < 0.85:
            destino[clave.upper() + "_X"] = destino.pop(clave)
        elif isinstance(obj.get(lista_clave), list):
            obj[lista_clave].append(rng.choice(VALORES_RAROS + [copy.deepcopy(destino)]))
    return obj


@pytest.mark.parametrize("semilla", range(400))
def test_prediccion_nunca_falla_sin_control(catalogo, prediccion_valida, semilla):
    payload = _mutar(prediccion_valida, random.Random(semilla))
    try:
        s, r = limpiar_solicitud_prediccion(payload, catalogo)
    except SolicitudRechazada as exc:
        assert exc.detail
        return
    Calidad.model_validate(r.como_dict())
    assert {f.familia_id for f in s.familias} == set(catalogo.familia_ids)


@pytest.mark.parametrize("semilla", range(400))
def test_observacion_nunca_falla_sin_control(catalogo, observacion_valida, semilla):
    payload = _mutar(observacion_valida, random.Random(semilla))
    try:
        s, r = limpiar_solicitud_observacion(payload, catalogo)
    except SolicitudRechazada as exc:
        assert exc.detail
        return
    Calidad.model_validate(r.como_dict())
    assert all(v.ventas >= 0 for v in s.ventas)
