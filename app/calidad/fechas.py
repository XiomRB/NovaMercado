"""Interpretación general de fechas. Nunca usa el reloj del sistema.

Resultado: ``(fecha, accion, detalle)`` con ``accion`` en {None, reparar, advertir}.

* ``AAAA-MM-DD``                         -> sin cambios
* ISO con hora / zona, ``AAAA/MM/DD``, ``AAAAMMDD``, sin ceros -> reparar
* ``DD/MM/AAAA`` con día > 12, o día == mes  -> reparar (única lectura)
* ``MM/DD/AAAA`` con el segundo número > 12  -> reparar (única lectura)
* ``NN/NN/AAAA`` ambos <= 12 y distintos     -> advertir, se asume DD/MM (uso local)
* año de dos dígitos                        -> advertir, se asume 20AA
* mes con nombre (``17 de noviembre de 2025``, ``17-nov-2025``) -> reparar
* cualquier otra cosa o fecha inexistente   -> FechaInvalida (rechazo)
"""

from __future__ import annotations

import re
from datetime import date

from app.calidad.resultado import ADVERTIR, REPARAR
from app.calidad.texto import quitar_tildes


class FechaInvalida(ValueError):
    pass


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_HORA = r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?)?"
_AMD = re.compile(rf"^(\d{{4}})[-/.](\d{{1,2}})[-/.](\d{{1,2}}){_HORA}$")
_COMPACTA = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_NN_NN_A = re.compile(rf"^(\d{{1,2}})[-/.](\d{{1,2}})[-/.](\d{{4}}|\d{{2}}){_HORA}$")
_TEXTUAL = re.compile(r"^(\d{1,2})[\s\-/.]*(?:de\s+)?([a-z]+)\.?[\s\-/.]*(?:de(?:l)?\s+)?(\d{4})$")

MESES = {
    "enero": 1, "ene": 1, "january": 1, "jan": 1,
    "febrero": 2, "feb": 2, "february": 2,
    "marzo": 3, "mar": 3, "march": 3,
    "abril": 4, "abr": 4, "april": 4, "apr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6, "june": 6,
    "julio": 7, "jul": 7, "july": 7,
    "agosto": 8, "ago": 8, "august": 8, "aug": 8,
    "septiembre": 9, "setiembre": 9, "sep": 9, "sept": 9, "set": 9, "september": 9,
    "octubre": 10, "oct": 10, "october": 10,
    "noviembre": 11, "nov": 11, "november": 11,
    "diciembre": 12, "dic": 12, "december": 12, "dec": 12,
}


def _construir(anio: int, mes: int, dia: int, original: str) -> date:
    try:
        return date(anio, mes, dia)
    except ValueError as exc:
        raise FechaInvalida(f"'{original}' no es una fecha existente") from exc


def interpretar_fecha(valor: object) -> tuple[date, str | None, str | None]:
    if valor is None:
        raise FechaInvalida("fecha nula")
    if not isinstance(valor, str):
        raise FechaInvalida(f"se esperaba texto AAAA-MM-DD y se recibió {type(valor).__name__}")
    texto = valor.strip()
    if not texto:
        raise FechaInvalida("fecha vacía")

    if m := _ISO.match(texto):
        fecha = _construir(int(m[1]), int(m[2]), int(m[3]), valor)
        if texto != valor:
            return fecha, REPARAR, f"'{valor}' se interpretó como '{fecha.isoformat()}'"
        return fecha, None, None

    if m := _AMD.match(texto) or _COMPACTA.match(texto):
        fecha = _construir(int(m[1]), int(m[2]), int(m[3]), valor)
        return fecha, REPARAR, f"'{valor}' se interpretó como '{fecha.isoformat()}'"

    if m := _NN_NN_A.match(texto):
        a, b, anio_txt = int(m[1]), int(m[2]), m[3]
        accion, notas = REPARAR, []
        anio = int(anio_txt)
        if len(anio_txt) == 2:
            anio += 2000
            accion = ADVERTIR
            notas.append(f"año de dos dígitos, se asumió {anio}")
        if a > 12 and b > 12:
            raise FechaInvalida(f"'{valor}' no tiene un mes válido")
        if a > 12 or a == b:          # solo puede ser DD/MM
            dia, mes = a, b
        elif b > 12:                  # solo puede ser MM/DD
            mes, dia = a, b
            notas.append("orden MM/DD")
        else:                         # ambiguo: se asume el uso local DD/MM
            dia, mes = a, b
            accion = ADVERTIR
            notas.append("día y mes ambiguos, se asumió DD/MM")
        fecha = _construir(anio, mes, dia, valor)
        detalle = f"'{valor}' se interpretó como '{fecha.isoformat()}'"
        if notas:
            detalle += " (" + "; ".join(notas) + ")"
        return fecha, accion, detalle

    if m := _TEXTUAL.match(quitar_tildes(texto).casefold()):
        mes = MESES.get(m[2])
        if mes is None:
            raise FechaInvalida(f"'{valor}': mes '{m[2]}' no reconocido")
        fecha = _construir(int(m[3]), mes, int(m[1]), valor)
        return fecha, REPARAR, f"'{valor}' se interpretó como '{fecha.isoformat()}'"

    raise FechaInvalida(f"'{valor}' no tiene un formato de fecha reconocible (se espera AAAA-MM-DD)")
