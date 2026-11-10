"""Errores de conexión con un servidor real: cortes y reconexión."""

import pytest
from sqlalchemy import text

from app.db.conexion import BaseDatosNoDisponible, transaccion, verificar_conexion
from tests.integracion.conftest import Servidor


def _cortar_conexiones(nombre_bd):
    s = Servidor()
    with s.admin.connect() as c:
        n = c.execute(text("SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                           "WHERE datname = :d AND pid <> pg_backend_pid()"), {"d": nombre_bd}).scalar()
    s.admin.dispose()
    return n


def test_conexion_cortada_a_mitad_de_transaccion(motor):
    with pytest.raises(BaseDatosNoDisponible):
        with transaccion(motor) as c:
            c.execute(text("SELECT 1"))
            _cortar_conexiones(motor.url.database)
            c.execute(text("SELECT 1"))


def test_el_pool_se_recupera_tras_un_corte(motor):
    assert verificar_conexion(motor)
    with transaccion(motor) as c:            # deja una conexión en el pool
        c.execute(text("SELECT 1"))
    assert _cortar_conexiones(motor.url.database) >= 1
    with transaccion(motor) as c:            # pool_pre_ping descarta la rota y reconecta
        assert c.execute(text("SELECT 1")).scalar() == 1
    assert verificar_conexion(motor)
