"""Configuración del servicio a partir de variables de entorno (ver ``.env.example``)."""

from __future__ import annotations

import os
from pathlib import Path


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


COMPONENTES = ("api", "base_datos", "modelo", "mlflow", "airflow")
# Etapa 4: MLflow y Airflow todavía no existen (DECISIONES.md §13, D-B). El
# Compose del producto (etapa 6) exigirá los cinco componentes.
COMPONENTES_REQUERIDOS_POR_DEFECTO = ("api", "base_datos", "modelo")


def ruta_modelo() -> Path:
    return Path(os.environ.get("MODELO_RUTA", "modelos/modelo_local.joblib"))


def componentes_requeridos() -> tuple[str, ...]:
    valor = os.environ.get("COMPONENTES_REQUERIDOS", "").strip()
    if not valor:
        return COMPONENTES_REQUERIDOS_POR_DEFECTO
    nombres = tuple(dict.fromkeys(n.strip() for n in valor.split(",") if n.strip()))
    desconocidos = set(nombres) - set(COMPONENTES)
    if desconocidos:
        raise ConfiguracionInvalida(f"COMPONENTES_REQUERIDOS tiene componentes desconocidos: {sorted(desconocidos)}")
    return tuple(dict.fromkeys(("api", *nombres)))
