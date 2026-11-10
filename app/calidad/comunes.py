"""Reglas compartidas por la limpieza de predicciones y de observaciones."""

from __future__ import annotations

import math
import re
from datetime import date

from app.calidad.catalogo import Catalogo, Tienda
from app.calidad.fechas import FechaInvalida, interpretar_fecha
from app.calidad.resultado import RegistroCalidad, SolicitudRechazada

_DIGITOS = re.compile(r"\d+")


def exigir_objeto(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise SolicitudRechazada(None, "el cuerpo debe ser un objeto JSON")
    return payload


def exigir_campo(datos: dict, campo: str) -> object:
    if campo not in datos or datos[campo] is None:
        raise SolicitudRechazada(campo, "falta el campo obligatorio")
    return datos[campo]


def limpiar_tienda(valor: object, catalogo: Catalogo, registro: RegistroCalidad) -> Tienda:
    """E01. Entero -> ok; '5', 5.0, 'T05', 'tienda 5' -> reparar; resto -> rechazo."""
    campo = "tienda_id"
    tienda_id: int | None = None
    if isinstance(valor, bool):
        raise SolicitudRechazada(campo, f"{valor!r} no es un identificador de tienda")
    if isinstance(valor, int):
        tienda_id = valor
    elif isinstance(valor, float):
        if not (math.isfinite(valor) and valor.is_integer()):
            raise SolicitudRechazada(campo, f"{valor!r} no es un identificador entero")
        tienda_id = int(valor)
        registro.reparar(campo, f"{valor!r} se interpretó como {tienda_id}")
    elif isinstance(valor, str):
        grupos = _DIGITOS.findall(valor)
        if re.match(r"^\s*-\s*\d", valor) or len(grupos) != 1:
            raise SolicitudRechazada(campo, f"'{valor}' no contiene un identificador de tienda inequívoco")
        tienda_id = int(grupos[0])
        registro.reparar(campo, f"'{valor}' se interpretó como {tienda_id}")
    else:
        raise SolicitudRechazada(campo, f"tipo {type(valor).__name__} no es un identificador de tienda")

    if tienda_id not in catalogo.tiendas:
        raise SolicitudRechazada(campo, f"la tienda {tienda_id} no existe en el catálogo")
    return catalogo.tiendas[tienda_id]


def limpiar_fecha(valor: object, tienda: Tienda, registro: RegistroCalidad) -> date:
    """E04. Ver :mod:`app.calidad.fechas`. Además rechaza fechas previas a la apertura."""
    try:
        fecha, accion, detalle = interpretar_fecha(valor)
    except FechaInvalida as exc:
        raise SolicitudRechazada("fecha", str(exc)) from exc
    if fecha < tienda.fecha_apertura:
        raise SolicitudRechazada(
            "fecha",
            f"{fecha.isoformat()} es anterior a la apertura de la tienda {tienda.tienda_id} "
            f"({tienda.fecha_apertura.isoformat()})",
        )
    registro.registrar("fecha", accion, detalle)
    return fecha


def normalizar_lista(valor: object, campo: str, clave_valor: str,
                     registro: RegistroCalidad) -> list:
    """La lista de ítems debe ser una lista no vacía.

    Un objeto ``{"abarrotes": 3, ...}`` se convierte a lista (reparar): la
    correspondencia familia -> valor es inequívoca.
    """
    if isinstance(valor, list):
        if not valor:
            raise SolicitudRechazada(campo, "la lista está vacía")
        return valor
    if isinstance(valor, dict) and valor:
        items = []
        for k, v in valor.items():
            if isinstance(v, dict):
                items.append({"familia_id": k, **v})
            else:
                items.append({"familia_id": k, clave_valor: v})
        registro.reparar(campo, "se recibió un objeto en lugar de una lista y se convirtió")
        return items
    raise SolicitudRechazada(campo, "se esperaba una lista no vacía")

