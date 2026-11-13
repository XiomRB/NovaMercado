"""Concesión de entrenamiento con PostgreSQL y procesos reales (plan §4): competencia entre
procesos, liberación ante errores, SIGKILL del titular, pérdida de la sesión
(``pg_terminate_backend``), sin reconexión silenciosa, reinicio real de PostgreSQL y
transacciones condicionadas contra un titular antiguo."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.modelo.bloqueo import (
    CLAVE_BLOQUEO,
    BloqueoOcupado,
    BloqueoYaTomado,
    ConcesionEntrenamiento,
    ConcesionPerdida,
)
from tests.integracion.apoyo_mlflow import url_de

RAIZ = Path(__file__).resolve().parents[2]


def lanzar(accion, **args):
    return subprocess.Popen([sys.executable, "-m", "tests.integracion.procesos", accion, json.dumps(args)],
                            cwd=RAIZ, env={**os.environ, "PYTHONPATH": str(RAIZ)}, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def primera_linea(p, timeout=60):
    t = time.monotonic()
    linea = p.stdout.readline()
    assert linea, f"el proceso no respondió en {time.monotonic() - t:.0f} s: {p.stderr.read()[-1500:]}"
    return json.loads(linea)


def intentar(url, **kw):
    p = lanzar("intentar", url=url, **kw)
    salida, err = p.communicate(timeout=120)
    return p.returncode, json.loads(salida.strip().splitlines()[-1]), err


def advisory_tomado(motor) -> bool:
    with motor.connect() as c:
        return bool(c.execute(text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
                                   "AND objsubid = 1 AND classid = 0 AND objid = :k"), {"k": CLAVE_BLOQUEO}).scalar())


def test_dos_procesos_compiten_y_uno_espera(motor):
    url = url_de(motor)
    with ConcesionEntrenamiento(url, modo="prueba") as c:
        codigo, salida, _ = intentar(url, espera=1)
        assert codigo == 5 and "otro entrenamiento" in salida["ocupado"]
        epoca = c.epoca
    codigo, salida, _ = intentar(url, espera=1)            # liberada: el otro proceso la toma
    assert codigo == 0 and salida["epoca"] == epoca + 1


def test_competencia_con_espera_se_serializa(motor):
    url = url_de(motor)
    a = lanzar("intentar", url=url, espera=30, mantener=3)
    b = lanzar("intentar", url=url, espera=30, mantener=3)
    sa, sb = a.communicate(timeout=120)[0], b.communicate(timeout=120)[0]
    epocas = sorted(json.loads(s.strip().splitlines()[-1])["epoca"] for s in (sa, sb))
    assert a.returncode == b.returncode == 0 and epocas[1] == epocas[0] + 1


def test_se_libera_ante_una_excepcion(motor):
    url = url_de(motor)
    with pytest.raises(RuntimeError, match="fallo inyectado"):
        with ConcesionEntrenamiento(url, modo="prueba"):
            raise RuntimeError("fallo inyectado durante el entrenamiento")
    assert not advisory_tomado(motor)
    codigo, _, _ = intentar(url)                           # disponible de inmediato
    assert codigo == 0


def test_sin_reentrada_en_el_mismo_proceso(motor):
    url = url_de(motor)
    with ConcesionEntrenamiento(url, modo="prueba"):
        t = time.monotonic()
        with pytest.raises(BloqueoYaTomado):
            ConcesionEntrenamiento(url, espera=30).adquirir()
        assert time.monotonic() - t < 1                    # no se queda esperando su propio lock


def test_sigkill_del_titular_libera_el_lock_y_la_concesion_vence(motor):
    url = url_de(motor)
    p = lanzar("tomar", url=url)
    epoca = primera_linea(p)["epoca"]
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=30)
    time.sleep(1)
    assert not advisory_tomado(motor)                      # el servidor cerró la sesión
    codigo, salida, _ = intentar(url, vencimiento=4)       # el latido todavía no venció
    assert codigo == 5 and "no venció" in salida["ocupado"]
    codigo, salida, _ = intentar(url, vencimiento=4, espera=15)
    assert codigo == 0 and salida["epoca"] == epoca + 1


def test_sesion_terminada_se_detecta_sin_reconexion(motor):
    url = url_de(motor)
    c = ConcesionEntrenamiento(url, modo="prueba", vencimiento=4)
    c.adquirir()
    try:
        assert c.sondear()
        with motor.connect() as adm:
            adm.execute(text("SELECT pg_terminate_backend(:p)"), {"p": c._pid})
        assert c.sondear() is False and c.perdida
        with pytest.raises(ConcesionPerdida):
            c.verificar()
        with pytest.raises(ConcesionPerdida):              # tampoco puede escribir decisiones
            c.insertar_decision(nombre="m", version="1", run_id="r", ruta="x", sha256="a" * 64, origen="p")
        assert c.sondear() is False                        # no reabre una sesión nueva
        codigo, salida, _ = intentar(url, vencimiento=4)
        assert codigo == 5                                 # la concesión persiste hasta vencer
    finally:
        c.liberar()


def test_titular_antiguo_no_puede_escribir_tras_una_epoca_nueva(motor):
    url = url_de(motor)
    a = ConcesionEntrenamiento(url, modo="prueba", vencimiento=2)
    a.adquirir()
    try:
        d = a.insertar_decision(nombre="m", version="1", run_id="r", ruta="x", sha256="a" * 64, origen="p")
        with motor.connect() as adm:                       # A pierde la sesión (p. ej. partición)
            adm.execute(text("SELECT pg_terminate_backend(:p)"), {"p": a._pid})
        codigo, salida, _ = intentar(url, vencimiento=2, espera=10, mantener=0)
        assert codigo == 0 and salida["epoca"] == a.epoca + 1     # B tomó una época nueva
        with pytest.raises(ConcesionPerdida):
            a.insertar_resolucion(d, "confirmada", "tardía")
    finally:
        a.liberar()
    with motor.connect() as c:
        assert c.execute(text("SELECT count(*) FROM resoluciones_promocion")).scalar() == 0


# ------------------------------------------------------------- reinicio real
@pytest.fixture
def postgres_reiniciable(tmp_path):
    """PostgreSQL desechable en Docker que la prueba puede reiniciar."""
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("requiere Docker para reiniciar un PostgreSQL de prueba")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        puerto = s.getsockname()[1]
    nombre = f"nm-pg-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--name", nombre, "-e", "POSTGRES_PASSWORD=prueba", "-p",
                    f"127.0.0.1:{puerto}:5432", "postgres:16.14-alpine"], check=True, capture_output=True)
    url = f"postgresql+psycopg2://postgres:prueba@127.0.0.1:{puerto}/postgres"
    try:
        _esperar_pg(url)
        from alembic import command

        from tests.integracion.conftest import config_alembic
        command.upgrade(config_alembic(url), "head")
        yield nombre, url
    finally:
        subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)


def _esperar_pg(url, limite=90):
    t = time.monotonic()
    while True:
        try:
            m = create_engine(url)
            with m.connect() as c:
                c.execute(text("SELECT 1"))
            m.dispose()
            return
        except Exception:  # noqa: BLE001
            if time.monotonic() - t > limite:
                raise
            time.sleep(1)


def test_reinicio_de_postgres_se_detecta_y_la_concesion_persiste(postgres_reiniciable):
    nombre, url = postgres_reiniciable
    c = ConcesionEntrenamiento(url, modo="prueba", vencimiento=5)
    c.adquirir()
    try:
        assert c.sondear()
        subprocess.run(["docker", "restart", nombre], check=True, capture_output=True, timeout=120)
        _esperar_pg(url)
        assert c.sondear() is False                        # sesión nueva/servidor reiniciado = pérdida
        c.liberar()                                        # el supervisor termina (código 6)
        t = time.monotonic()
        with pytest.raises(BloqueoOcupado, match="no venció"):     # el último latido persistió
            ConcesionEntrenamiento(url, vencimiento=60).adquirir()
        nueva = ConcesionEntrenamiento(url, vencimiento=5, espera=30, reintento=0.5)
        nueva.adquirir()
        assert nueva.epoca == c.epoca + 1 and time.monotonic() - t < 30
        nueva.liberar()
    finally:
        c.liberar()
