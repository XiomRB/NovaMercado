"""Infraestructura de las pruebas de integración con PostgreSQL.

Requieren ``TEST_DATABASE_URL`` apuntando a un servidor PostgreSQL desechable
con permiso para crear bases (la CI levanta uno; localmente:
``scripts/pruebas_integracion_local.sh``). Sin esa variable se omiten con un
motivo explícito: nunca se usa la base real del producto.

Cada prueba recibe una base **nueva**, clonada de una plantilla ya migrada
(``CREATE DATABASE ... TEMPLATE``), así que las pruebas son independientes y
pueden comprobar que los triggers impiden borrar o truncar.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.db.conexion import crear_motor

RAIZ = Path(__file__).resolve().parents[2]
URL = os.environ.get("TEST_DATABASE_URL", "").strip()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if Path(str(item.fspath)).is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.integracion)
            if not URL:
                item.add_marker(pytest.mark.skip(reason="requiere TEST_DATABASE_URL (PostgreSQL de pruebas)"))


def url_para(nombre: str) -> str:
    return make_url(URL).set(database=nombre).render_as_string(hide_password=False)


def config_alembic(url: str) -> Config:
    cfg = Config(str(RAIZ / "alembic.ini"))
    cfg.set_main_option("script_location", str(RAIZ / "migraciones"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["configurar_logging"] = False
    return cfg


class Servidor:
    def __init__(self):
        self.admin = create_engine(URL, isolation_level="AUTOCOMMIT", future=True)
        self.creadas: list[str] = []

    def crear(self, plantilla: str | None = None) -> str:
        nombre = f"nm_test_{uuid.uuid4().hex[:12]}"
        sufijo = f" TEMPLATE {plantilla}" if plantilla else ""
        with self.admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{nombre}"{sufijo}'))
        self.creadas.append(nombre)
        return nombre

    def borrar(self, nombre: str) -> None:
        with self.admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{nombre}" WITH (FORCE)'))
        if nombre in self.creadas:
            self.creadas.remove(nombre)

    def cerrar(self) -> None:
        for n in list(self.creadas):
            self.borrar(n)
        self.admin.dispose()


@pytest.fixture(scope="session")
def servidor():
    s = Servidor()
    yield s
    s.cerrar()


@pytest.fixture(scope="session")
def plantilla(servidor):
    nombre = servidor.crear()
    command.upgrade(config_alembic(url_para(nombre)), "head")
    return nombre


@pytest.fixture
def url_bd(servidor, plantilla):
    nombre = servidor.crear(plantilla)
    yield url_para(nombre)
    servidor.borrar(nombre)


@pytest.fixture
def motor(url_bd):
    m = crear_motor(url_bd, timeout_conexion=5)
    yield m
    m.dispose()


@pytest.fixture
def url_bd_vacia(servidor):
    nombre = servidor.crear()
    yield url_para(nombre)
    servidor.borrar(nombre)
