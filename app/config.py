"""Configuración del servicio a partir de variables de entorno (ver ``.env.example``)."""

from __future__ import annotations

import os
from pathlib import Path


class ConfiguracionInvalida(RuntimeError):
    pass


def url_base_datos(variable: str = "DATABASE_URL") -> str:
    """URL SQLAlchemy de la base del producto, p. ej.
    ``postgresql+psycopg2://usuario:clave@host:5432/novamercado``.

    * Si ``DATABASE_URL`` está definida, se usa tal cual (equipo de desarrollo).
    * Si no, se construye **por partes** (contenedores, etapa 6) desde
      ``POSTGRES_USER``, ``POSTGRES_PASSWORD``, ``POSTGRES_DB``, ``DB_HOST`` y
      ``DB_PORT`` con ``URL.create``, que escapa los caracteres especiales de la
      clave. Así el ``DATABASE_URL`` del ``.env`` (``localhost:5433``) nunca se
      filtra a los contenedores.

    No hay valor por defecto: las credenciales nunca se escriben en el código.
    """
    url = os.environ.get(variable, "").strip()
    if not url and variable == "DATABASE_URL":
        url = _url_por_partes()
    if not url:
        raise ConfiguracionInvalida(f"falta la variable de entorno {variable}")
    if not url.startswith("postgresql"):
        raise ConfiguracionInvalida(f"{variable} debe ser una URL de PostgreSQL")
    return url


PARTES_URL = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "DB_HOST")


def _url_por_partes() -> str:
    valores = {v: os.environ.get(v, "") for v in PARTES_URL}
    if not any(valores.values()):
        return ""
    faltan = [v for v, x in valores.items() if not x.strip()]
    if faltan:
        raise ConfiguracionInvalida(f"sin DATABASE_URL, faltan las variables {faltan}")
    puerto = os.environ.get("DB_PORT", "5432").strip() or "5432"
    if not puerto.isdigit():
        raise ConfiguracionInvalida("DB_PORT debe ser un número de puerto")
    from sqlalchemy.engine import URL
    return URL.create("postgresql+psycopg2", username=valores["POSTGRES_USER"],
                      password=valores["POSTGRES_PASSWORD"], host=valores["DB_HOST"].strip(),
                      port=int(puerto), database=valores["POSTGRES_DB"].strip()).render_as_string(hide_password=False)


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
        base = COMPONENTES_REQUERIDOS_POR_DEFECTO
        if os.environ.get("MODELO_ORIGEN", "mlflow").strip().lower() in ("", "mlflow"):
            base = (*base, "mlflow")
        if orquestador_configurado() == "airflow":           # etapa 7
            base = (*base, "airflow")
        return base
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


# ------------------------------------------------------------ reentrenamiento (etapa 7)
def parametros_rol(usuario: str, variable_clave: str) -> dict:
    """Parámetros psycopg2 para un rol de la etapa 7 (``nm_cierre``): host, puerto y base
    de la URL del producto, con el usuario indicado y la clave de ``variable_clave``."""
    from sqlalchemy.engine import make_url
    clave = os.environ.get(variable_clave, "")
    if not clave:
        raise ConfiguracionInvalida(f"falta la variable de entorno {variable_clave}")
    u = make_url(url_base_datos())
    return {"host": u.host, "port": u.port or 5432, "dbname": u.database, "user": usuario, "password": clave}


ORQUESTADORES = ("ninguno", "airflow")


def orquestador_configurado() -> str:
    """``ORQUESTADOR=airflow`` (producto, etapa 7) o ``ninguno`` (por defecto: la API
    responde 503 a ``POST /admin/reentrenar``)."""
    valor = os.environ.get("ORQUESTADOR", "ninguno").strip().lower() or "ninguno"
    if valor not in ORQUESTADORES:
        raise ConfiguracionInvalida(f"ORQUESTADOR debe ser uno de {ORQUESTADORES}")
    return valor


def config_airflow():
    """URL de la API REST del webserver y el usuario de servicio de la API."""
    from app.orquestador_airflow import ConfigAirflow
    faltan = [v for v in ("AIRFLOW_URL", "AIRFLOW_API_USUARIO", "AIRFLOW_API_CLAVE") if not os.environ.get(v)]
    if faltan:
        raise ConfiguracionInvalida(f"con ORQUESTADOR=airflow faltan las variables {faltan}")
    return ConfigAirflow(os.environ["AIRFLOW_URL"].strip().rstrip("/"), os.environ["AIRFLOW_API_USUARIO"],
                         os.environ["AIRFLOW_API_CLAVE"])
