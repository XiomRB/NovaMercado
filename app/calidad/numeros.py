"""Interpretación general de valores numéricos recibidos como JSON.

Regla de separadores (convención guatemalteca, igual a la de JSON):
coma = separador de miles, punto = separador decimal. Solo se reinterpreta un
separador cuando la convención local es imposible:

* ``"1,234"``      -> 1234   (coma seguida de grupos de 3 dígitos: miles)
* ``"812,0"``      -> 812.0  (coma seguida de 1, 2 o 4+ dígitos: no puede ser
  separador de miles, solo puede ser coma decimal)
* ``"1.234.567"``  -> 1234567 (varios puntos: solo pueden ser miles)
* ``"1.234,5"``    -> 1234.5 (formato europeo completo)
* ``"1 234"``      -> 1234   (espacio como separador de miles)

Cualquier otra cosa (texto libre, booleanos, NaN, infinitos) es inválida.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class NumeroInvalido(ValueError):
    pass


@dataclass(frozen=True)
class NumeroInterpretado:
    valor: float
    original: object
    formato: str | None  # descripción del cambio de formato; None si llegó como número JSON

    @property
    def es_entero(self) -> bool:
        return float(self.valor).is_integer()


_SIGNO = r"[+-]?"
_ENTERO = re.compile(rf"^{_SIGNO}\d+$")
_DECIMAL_PUNTO = re.compile(rf"^{_SIGNO}\d*\.\d+$")
_MILES_COMA = re.compile(rf"^{_SIGNO}\d{{1,3}}(,\d{{3}})+$")
_MILES_COMA_DEC_PUNTO = re.compile(rf"^{_SIGNO}\d{{1,3}}(,\d{{3}})+\.\d+$")
_MILES_PUNTO = re.compile(rf"^{_SIGNO}\d{{1,3}}(\.\d{{3}}){{2,}}$")
_MILES_PUNTO_DEC_COMA = re.compile(rf"^{_SIGNO}\d{{1,3}}(\.\d{{3}})+,\d+$")
_DECIMAL_COMA = re.compile(rf"^{_SIGNO}\d+,\d+$")
_MILES_ESPACIO = re.compile(rf"^{_SIGNO}\d{{1,3}}( \d{{3}})+$")


def _desde_texto(texto: str) -> tuple[float, str]:
    t = texto.strip().replace(" ", " ")
    if t != texto:
        nota_espacios = " (con espacios alrededor)"
    else:
        nota_espacios = ""
    if _ENTERO.match(t) or _DECIMAL_PUNTO.match(t):
        return float(t), "número recibido como texto" + nota_espacios
    if _MILES_COMA_DEC_PUNTO.match(t):
        return float(t.replace(",", "")), "separador de miles ',' eliminado"
    if _MILES_COMA.match(t):
        return float(t.replace(",", "")), "separador de miles ',' eliminado"
    if _MILES_PUNTO_DEC_COMA.match(t):
        return float(t.replace(".", "").replace(",", ".")), "formato '1.234,5' convertido"
    if _MILES_PUNTO.match(t):
        return float(t.replace(".", "")), "separador de miles '.' eliminado"
    if _DECIMAL_COMA.match(t):
        return float(t.replace(",", ".")), "coma decimal convertida a punto"
    if _MILES_ESPACIO.match(t):
        return float(t.replace(" ", "")), "separador de miles ' ' eliminado"
    raise NumeroInvalido(f"'{texto}' no es un número reconocible")


def interpretar_numero(valor: object) -> NumeroInterpretado:
    """Convierte ``valor`` a número o lanza :class:`NumeroInvalido`."""
    if valor is None:
        raise NumeroInvalido("valor nulo")
    if isinstance(valor, bool):  # bool es subclase de int en Python
        raise NumeroInvalido(f"valor booleano {valor!r} no es una cantidad")
    if isinstance(valor, int):
        return NumeroInterpretado(float(valor), valor, None)
    if isinstance(valor, float):
        if not math.isfinite(valor):
            raise NumeroInvalido(f"valor no finito {valor!r}")
        return NumeroInterpretado(valor, valor, None)
    if isinstance(valor, str):
        numero, formato = _desde_texto(valor)
        if not math.isfinite(numero):
            raise NumeroInvalido(f"'{valor}' no es finito")
        return NumeroInterpretado(numero, valor, formato)
    raise NumeroInvalido(f"tipo {type(valor).__name__} no es numérico")


def redondear(valor: float) -> int:
    """Redondeo aritmético (0.5 hacia arriba), no el bancario de ``round``."""
    return int(math.floor(valor + 0.5))


def describir(valor: object) -> str:
    return f"'{valor}'" if isinstance(valor, str) else repr(valor)
