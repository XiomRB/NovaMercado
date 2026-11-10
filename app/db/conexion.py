"""Motor de base de datos, unidad de trabajo, límites de tiempo y errores de conexión.

Límites (DECISIONES.md §13, "Límite de las consultas"):

* ``connect_timeout`` (``DB_TIMEOUT_CONEXION``, 5 s): establecer una conexión.
* ``statement_timeout`` del **servidor** (``DB_TIMEOUT_CONSULTA``, 5 s): PostgreSQL
  cancela la consulta lenta (incluida la espera de locks); la conexión sigue
  siendo usable. ``idle_in_transaction_session_timeout`` libera transacciones
  abandonadas por un cliente que dejó de responder.
* Vigilante del **cliente** (límite de consulta + ``DB_MARGEN_CLIENTE``, 2 s): si
  el servidor no responde (congelado, red cortada), no puede cancelar nada; el
  vigilante cierra el socket desde el cliente para que psycopg2 salga del
  bloqueo con un error y la conexión se descarte. Cubre ``execute``,
  ``executemany``, ``commit`` y ``rollback``.

Ambos casos se convierten en :class:`BaseDatosNoDisponible` (503 en la API).
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
import psycopg2.extensions
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, OperationalError

from app.config import segundos, url_base_datos


class BaseDatosNoDisponible(RuntimeError):
    """No se pudo conectar o la conexión se perdió. La API responderá 503."""


class _Vigilante:
    """Corta la conexión si una operación supera ``limite`` segundos sin respuesta."""

    def __init__(self, conexion, limite: float):
        self.conexion = conexion
        self.limite = limite
        self.excedido = False
        self._terminado = False
        self._candado = threading.Lock()
        self._temporizador = threading.Timer(limite, self._cortar)
        self._temporizador.daemon = True

    def _cortar(self) -> None:
        with self._candado:
            if self._terminado:
                return
            self.excedido = True
            try:
                # Se cierra el socket (no el descriptor de psycopg2): el recv bloqueado
                # en el otro hilo vuelve con error y la conexión queda marcada como rota.
                with socket.socket(fileno=os.dup(self.conexion.fileno())) as s:
                    s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def __enter__(self):
        self._temporizador.start()
        return self

    def __exit__(self, tipo, exc, tb):
        with self._candado:
            self._terminado = True
            self._temporizador.cancel()
        if self.excedido and isinstance(exc, psycopg2.Error):
            raise psycopg2.OperationalError(
                f"la base de datos no respondió en {self.limite:g} s; se cortó la conexión") from exc
        return False


def _fabrica_conexion(limite: float):
    """Clase de conexión psycopg2 cuyas operaciones de red están vigiladas."""

    class CursorVigilado(psycopg2.extensions.cursor):
        def execute(self, query, vars=None):
            with _Vigilante(self.connection, limite):
                return super().execute(query, vars)

        def executemany(self, query, vars_list):
            with _Vigilante(self.connection, limite):
                return super().executemany(query, vars_list)

    class ConexionVigilada(psycopg2.extensions.connection):
        limite_cliente = limite

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cursor_factory = CursorVigilado

        def commit(self):
            with _Vigilante(self, limite):
                return super().commit()

        def rollback(self):
            with _Vigilante(self, limite):
                return super().rollback()

    return ConexionVigilada


def crear_motor(url: str | None = None, timeout_conexion: float | None = None,
                timeout_consulta: float | None = None, margen_cliente: float | None = None) -> Engine:
    """Motor con ``pool_pre_ping`` (descarta conexiones rotas tras un reinicio de la
    base), ``connect_timeout`` y límite de tiempo por consulta en servidor y cliente.

    Los procesos por lotes (carga de datos, entrenamiento) pasan un
    ``timeout_consulta`` mayor que el de la API.
    """
    t_conexion = timeout_conexion or segundos("DB_TIMEOUT_CONEXION", 5)
    t_consulta = timeout_consulta or segundos("DB_TIMEOUT_CONSULTA", 5)
    margen = margen_cliente or segundos("DB_MARGEN_CLIENTE", 2)
    ms = int(t_consulta * 1000)
    opciones = (f"-c statement_timeout={ms} "
                f"-c idle_in_transaction_session_timeout={max(6 * ms, 30_000)}")
    return create_engine(
        url or url_base_datos(),
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": max(int(t_conexion), 1), "options": opciones,
                      "connection_factory": _fabrica_conexion(t_consulta + margen)},
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
