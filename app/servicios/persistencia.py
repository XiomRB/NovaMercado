"""Interfaz de persistencia que usan los servicios de la API.

Los servicios no conocen SQL ni SQLAlchemy: reciben una :class:`Persistencia`.
En producción es :class:`PersistenciaPostgres`; las pruebas de contrato usan
una implementación en memoria con la misma interfaz (``tests/api/memoria.py``).
Una :class:`Unidad` es una transacción: lo que se lee y escribe dentro de
``with persistencia.unidad() as u:`` se confirma junto o se revierte.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import date
from typing import Protocol

import pandas as pd
from sqlalchemy.engine import Connection, Engine

from app.calidad.catalogo import Catalogo
from app.calidad.resultado import RegistroCalidad
from app.db import repositorio as repo
from app.db.conexion import transaccion, verificar_conexion
from app.schemas.contrato import SolicitudObservacion, SolicitudPrediccion


class Unidad(Protocol):
    def ventas_previas(self, tienda_id: int, fecha: date, dias: int) -> pd.DataFrame: ...

    def registrar_prediccion(self, *, payload: object, solicitud: SolicitudPrediccion, registro: RegistroCalidad,
                             predicciones: Sequence[repo.PrediccionFamilia], nombre_modelo: str,
                             version_modelo: str) -> repo.SolicitudRegistrada: ...

    def registrar_rechazo_prediccion(self, *, payload: object, detalle: str) -> repo.SolicitudRegistrada: ...

    def registrar_observacion(self, *, payload: object, solicitud: SolicitudObservacion,
                              registro: RegistroCalidad) -> repo.ResultadoObservacion: ...

    def registrar_rechazo_observacion(self, *, payload: object, detalle: str) -> int: ...


class Persistencia(Protocol):
    def catalogo(self) -> Catalogo: ...

    def unidad(self) -> AbstractContextManager[Unidad]: ...

    def disponible(self) -> bool: ...


class _UnidadPostgres:
    def __init__(self, conn: Connection):
        self.conn = conn

    def ventas_previas(self, tienda_id, fecha, dias):
        return repo.leer_ventas_previas(self.conn, tienda_id, fecha, dias)

    def registrar_prediccion(self, **kw):
        return repo.registrar_prediccion(self.conn, **kw)

    def registrar_rechazo_prediccion(self, **kw):
        return repo.registrar_rechazo_prediccion(self.conn, **kw)

    def registrar_observacion(self, **kw):
        return repo.registrar_observacion(self.conn, **kw)

    def registrar_rechazo_observacion(self, **kw):
        return repo.registrar_rechazo_observacion(self.conn, **kw)


class PersistenciaPostgres:
    """Persistencia sobre la base del producto. El catálogo se lee una vez y se
    guarda en memoria: las tablas de catálogo solo cambian con una carga nueva."""

    def __init__(self, motor: Engine):
        self.motor = motor
        self._catalogo: Catalogo | None = None
        self._candado = threading.Lock()

    def catalogo(self) -> Catalogo:
        if self._catalogo is None:
            with self._candado:
                if self._catalogo is None:
                    with transaccion(self.motor) as conn:
                        self._catalogo = repo.cargar_catalogo(conn)
        return self._catalogo

    @contextmanager
    def unidad(self) -> Iterator[Unidad]:
        with transaccion(self.motor) as conn:
            yield _UnidadPostgres(conn)

    def disponible(self) -> bool:
        return verificar_conexion(self.motor)
