"""Configuración y errores de conexión que no requieren un servidor."""

import time

import pytest

from app.config import ConfiguracionInvalida, segundos, url_base_datos
from app.db.conexion import BaseDatosNoDisponible, crear_motor, transaccion, verificar_conexion
from app.db.repositorio import json_seguro

# Puerto 1 de localhost: nada escucha ahí, la conexión se rechaza de inmediato.
SIN_SERVIDOR = "postgresql+psycopg2://usuario@127.0.0.1:1/inexistente"


def test_url_obligatoria(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ConfiguracionInvalida, match="DATABASE_URL"):
        url_base_datos()
    monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")
    with pytest.raises(ConfiguracionInvalida, match="PostgreSQL"):
        url_base_datos()
    monkeypatch.setenv("DATABASE_URL", " postgresql+psycopg2://u@h/db ")
    assert url_base_datos() == "postgresql+psycopg2://u@h/db"


def test_segundos(monkeypatch):
    monkeypatch.delenv("X_TIMEOUT", raising=False)
    assert segundos("X_TIMEOUT", 5) == 5
    monkeypatch.setenv("X_TIMEOUT", "2.5")
    assert segundos("X_TIMEOUT", 5) == 2.5
    for malo in ("abc", "0", "-1"):
        monkeypatch.setenv("X_TIMEOUT", malo)
        with pytest.raises(ConfiguracionInvalida):
            segundos("X_TIMEOUT", 5)


def test_servidor_inaccesible_se_informa_rapido():
    motor = crear_motor(SIN_SERVIDOR, timeout_conexion=2)
    inicio = time.monotonic()
    assert verificar_conexion(motor) is False
    with pytest.raises(BaseDatosNoDisponible, match="no disponible"):
        with transaccion(motor) as c:
            c.exec_driver_sql("SELECT 1")
    assert time.monotonic() - inicio < 10
    motor.dispose()


def test_json_seguro():
    assert json_seguro({"a": float("nan"), "b": [1, float("inf")], 3: None}) == {
        "a": "nan", "b": [1, "inf"], "3": None}
    assert json_seguro([{"x": (1, 2)}]) == [{"x": [1, 2]}]
    assert json_seguro("texto") == "texto"
