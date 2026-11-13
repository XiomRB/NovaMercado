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
# Etapa 5: con MODELO_ORIGEN=mlflow se exige además "mlflow" (DECISIONES.md §14).
# Airflow se agrega en la etapa 7; el Compose del producto (etapa 6) exigirá los cinco.
COMPONENTES_REQUERIDOS_POR_DEFECTO = ("api", "base_datos", "modelo")


def ruta_modelo() -> Path:
    return Path(os.environ.get("MODELO_RUTA", "modelos/modelo_local.joblib"))


def componentes_requeridos() -> tuple[str, ...]:
    valor = os.environ.get("COMPONENTES_REQUERIDOS", "").strip()
    if not valor:
        if os.environ.get("MODELO_ORIGEN", "mlflow").strip().lower() in ("", "mlflow"):
            return (*COMPONENTES_REQUERIDOS_POR_DEFECTO, "mlflow")
        return COMPONENTES_REQUERIDOS_POR_DEFECTO
    nombres = tuple(dict.fromkeys(n.strip() for n in valor.split(",") if n.strip()))
    desconocidos = set(nombres) - set(COMPONENTES)
    if desconocidos:
        raise ConfiguracionInvalida(f"COMPONENTES_REQUERIDOS tiene componentes desconocidos: {sorted(desconocidos)}")
    return tuple(dict.fromkeys(("api", *nombres)))


# ------------------------------------------------------------------ MLflow (etapa 5)
ORIGENES_MODELO = ("mlflow", "archivo")


def modelo_origen() -> str:
    """``mlflow`` (producto) o ``archivo`` (desarrollo y pruebas, etapa 4)."""
    origen = os.environ.get("MODELO_ORIGEN", "mlflow").strip().lower() or "mlflow"
    if origen not in ORIGENES_MODELO:
        raise ConfiguracionInvalida(f"MODELO_ORIGEN debe ser uno de {ORIGENES_MODELO}")
    return origen


def mlflow_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise ConfiguracionInvalida("falta la variable de entorno MLFLOW_TRACKING_URI")
    return uri.rstrip("/")


def mlflow_modelo_nombre() -> str:
    return os.environ.get("MLFLOW_MODELO_NOMBRE", "novamercado-ventas").strip() or "novamercado-ventas"


def mlflow_alias() -> str:
    return os.environ.get("MLFLOW_ALIAS", "vigente").strip() or "vigente"


def ruta_cache_modelos() -> Path:
    return Path(os.environ.get("MODELO_CACHE_DIR", "modelos/cache"))


def configurar_cliente_mlflow() -> None:
    """Límites del cliente HTTP de MLflow. Sus valores por defecto (120 s y 7
    reintentos con espera creciente) pueden bloquear minutos; se usan 5 s y 0
    reintentos salvo que el entorno indique otra cosa."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
