"""Motor de base de datos, unidad de trabajo y manejo de errores de conexión."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, OperationalError

from app.config import segundos, url_base_datos


class BaseDatosNoDisponible(RuntimeError):
    """No se pudo conectar o la conexión se perdió. La API responderá 503."""


def crear_motor(url: str | None = None, timeout_conexion: float | None = None) -> Engine:
    """Motor con ``pool_pre_ping`` (descarta conexiones rotas tras un reinicio de la
    base) y ``connect_timeout`` (falla rápido si la base no responde)."""
    timeout = timeout_conexion or segundos("DB_TIMEOUT_CONEXION", 5)
    return create_engine(
        url or url_base_datos(),
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": max(int(timeout), 1)},
        executemany_mode="values_plus_batch",
    )


def _es_error_de_conexion(exc: DBAPIError) -> bool:
    return isinstance(exc, OperationalError) or bool(getattr(exc, "connection_invalidated", False))


@contextmanager
def transaccion(motor: Engine) -> Iterator[Connection]:
    """Unidad de trabajo: todo lo hecho con la conexión se confirma junto o se revierte.

    Los errores de conexión (servidor caído, conexión cortada) se convierten en
    :class:`BaseDatosNoDisponible`; los errores de integridad se propagan.
    """
    try:
        with motor.begin() as conn:
            yield conn
    except DBAPIError as exc:
        if _es_error_de_conexion(exc):
            raise BaseDatosNoDisponible(f"base de datos no disponible: {exc.orig}") from exc
        raise


def verificar_conexion(motor: Engine) -> bool:
    """``True`` si la base responde a ``SELECT 1``. Para ``GET /health``."""
    try:
        with motor.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except DBAPIError:
        return False
