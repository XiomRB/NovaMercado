"""Configuración del servicio a partir de variables de entorno (ver ``.env.example``)."""

from __future__ import annotations

import os


class ConfiguracionInvalida(RuntimeError):
    pass


def url_base_datos(variable: str = "DATABASE_URL") -> str:
    """URL SQLAlchemy de la base del producto, p. ej.
    ``postgresql+psycopg2://usuario:clave@host:5432/novamercado``.

    No hay valor por defecto: las credenciales nunca se escriben en el código.
    """
    url = os.environ.get(variable, "").strip()
    if not url:
        raise ConfiguracionInvalida(f"falta la variable de entorno {variable}")
    if not url.startswith("postgresql"):
        raise ConfiguracionInvalida(f"{variable} debe ser una URL de PostgreSQL")
    return url


def segundos(variable: str, por_defecto: float) -> float:
    valor = os.environ.get(variable)
    if valor is None or valor.strip() == "":
        return por_defecto
    try:
        n = float(valor)
    except ValueError as exc:
        raise ConfiguracionInvalida(f"{variable} debe ser un número de segundos") from exc
    if n <= 0:
        raise ConfiguracionInvalida(f"{variable} debe ser positivo")
    return n
