"""Normalización de texto y coincidencia aproximada, sin dependencias externas."""

from __future__ import annotations

import difflib
import re
import unicodedata

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NO_ALFANUM = re.compile(r"[^a-z0-9]+")

# Palabras de enlace que no distinguen familias ("Frutas y verduras" ~ frutas_verduras).
PALABRAS_VACIAS = frozenset({"y", "e", "de", "del", "la", "las", "el", "los", "para", "con"})


def quitar_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def normalizar_clave(texto: str) -> str:
    """'  Frutas y Verduras ' -> 'frutas_y_verduras'; 'tiendaId' -> 'tienda_id'."""
    texto = _CAMEL.sub("_", texto.strip())
    texto = quitar_tildes(texto).casefold()
    return _NO_ALFANUM.sub("_", texto).strip("_")


def sin_palabras_vacias(clave: str) -> str:
    return "_".join(p for p in clave.split("_") if p and p not in PALABRAS_VACIAS)


def mejor_coincidencia(
    clave: str, candidatos: dict[str, str], umbral: float = 0.8, margen: float = 0.08
) -> str | None:
    """Devuelve el destino del candidato más parecido si la coincidencia es clara.

    ``candidatos`` mapea clave normalizada -> destino (varias claves pueden apuntar
    al mismo destino). Se exige similitud >= ``umbral`` y que el mejor destino
    supere al segundo destino distinto por al menos ``margen``; si no, la
    coincidencia es ambigua y se devuelve ``None``.
    """
    mejor_por_destino: dict[str, float] = {}
    for candidato, destino in candidatos.items():
        r = difflib.SequenceMatcher(None, clave, candidato).ratio()
        if r > mejor_por_destino.get(destino, 0.0):
            mejor_por_destino[destino] = r
    if not mejor_por_destino:
        return None
    ordenados = sorted(mejor_por_destino.items(), key=lambda kv: kv[1], reverse=True)
    destino, puntaje = ordenados[0]
    segundo = ordenados[1][1] if len(ordenados) > 1 else 0.0
    if puntaje >= umbral and puntaje - segundo >= margen:
        return destino
    return None
