"""Límite de tiempo de las consultas: cancelación en el servidor y corte en el cliente.

* Servidor lento (``pg_sleep``): ``statement_timeout`` cancela la consulta.
* Servidor que no responde (proxy congelado): el vigilante del cliente corta la conexión.
En ambos casos: BaseDatosNoDisponible → 503, sin registros parciales, y la API y
``/health`` siguen respondiendo.
"""

import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url

from app.db import esquema as e
from app.db.conexion import BaseDatosNoDisponible, crear_motor, transaccion
from app.main import create_app
from app.modelo.entrenamiento import empaquetar
from ml.pronosticador import Pronosticador
from tests.api.conftest import CONFIG_RAPIDA, DATOS, FECHA, ProveedorMemoria
from tests.integracion.proxy import ProxyCongelable
from tests.integracion.test_api_postgres import _deps, ej
from tests.ml.sinteticos import escribir_carpeta

# Si el límite dejara de funcionar, estas pruebas se quedarían esperando: pytest-timeout
# las hace fallar en 60 s en lugar de bloquear la CI.
pytestmark = pytest.mark.timeout(60, method="thread")

TABLAS = (e.solicitudes_prediccion, e.solicitud_familias, e.predicciones, e.calidad_observaciones)


@pytest.fixture(scope="module")
def artefacto():
    h = DATOS.historico
    p = Pronosticador.entrenar(h, h, DATOS.tiendas, DATOS.feriados, DATOS.costos,
                               list(DATOS.familias["familia_id"]), pd.Timestamp(FECHA), CONFIG_RAPIDA)
    return empaquetar(p)


@pytest.fixture
def bd(motor, tmp_path):
    from app.db.carga import cargar_datos_publicados
    escribir_carpeta(DATOS, tmp_path)
    cargar_datos_publicados(motor, tmp_path)
    return motor


@pytest.fixture
def proxy(url_bd):
    u = make_url(url_bd)
    p = ProxyCongelable(u.host, u.port or 5432)
    yield p
    p.cerrar()


def motor_limitado(url, proxy=None, consulta=1.0, margen=1.0, conexion=2.0):
    if proxy is not None:
        url = make_url(url).set(host="127.0.0.1", port=proxy.puerto).render_as_string(hide_password=False)
    return crear_motor(url, timeout_conexion=conexion, timeout_consulta=consulta, margen_cliente=margen)


def contar(motor):
    with motor.connect() as c:
        return {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in TABLAS}


def dormir_al_insertar_predicciones(motor, segundos):
    with transaccion(motor) as c:
        c.execute(text(f"""
            CREATE FUNCTION dormir_prueba() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_sleep({segundos}); RETURN NEW; END $$;
            CREATE TRIGGER trg_dormir_prueba BEFORE INSERT ON predicciones
                FOR EACH STATEMENT EXECUTE FUNCTION dormir_prueba();"""))


def quitar_dormir(motor):
    with transaccion(motor) as c:
        c.execute(text("DROP TRIGGER trg_dormir_prueba ON predicciones; DROP FUNCTION dormir_prueba();"))


# ------------------------------------------------------------------ nivel conexión
def test_consulta_lenta_se_cancela_en_el_servidor(url_bd):
    m = motor_limitado(url_bd, consulta=1)
    t = time.monotonic()
    with pytest.raises(BaseDatosNoDisponible, match="statement timeout"):
        with transaccion(m) as c:
            c.execute(text("SELECT pg_sleep(10)"))
    assert time.monotonic() - t < 2.5
    with transaccion(m) as c:                                   # la conexión sigue usable
        assert c.execute(text("SELECT 1")).scalar() == 1
    m.dispose()


def test_servidor_que_no_responde_se_corta_en_el_cliente(url_bd, proxy):
    m = motor_limitado(url_bd, proxy, consulta=1, margen=1, conexion=2)
    with transaccion(m) as c:
        assert c.execute(text("SELECT 1")).scalar() == 1
    proxy.congelar()
    t = time.monotonic()
    with pytest.raises(BaseDatosNoDisponible):
        with transaccion(m) as c:
            c.execute(text("SELECT 1"))
    # vigilante (1 + 1 s) + reintento de conexión (2 s); nunca indefinido
    assert time.monotonic() - t < 6
    proxy.descongelar()
    with transaccion(m) as c:                                   # se recupera sin reiniciar
        assert c.execute(text("SELECT 1")).scalar() == 1
    m.dispose()


def test_corte_a_mitad_de_transaccion_con_el_servidor_congelado(url_bd, proxy):
    """Ya se enviaron sentencias; el servidor deja de responder antes de terminar."""
    m = motor_limitado(url_bd, proxy, consulta=1, margen=1, conexion=2)
    with transaccion(m) as c:
        c.execute(text("CREATE TABLE prueba_corte (x int)"))
    t = time.monotonic()
    with pytest.raises(BaseDatosNoDisponible):
        with transaccion(m) as c:
            c.execute(text("INSERT INTO prueba_corte VALUES (1)"))
            proxy.congelar()
            c.execute(text("INSERT INTO prueba_corte VALUES (2)"))
    assert time.monotonic() - t < 6
    proxy.descongelar()
    proxy.cortar_conexiones()                                   # el servidor ve el corte y revierte
    directo = crear_motor(url_bd)
    with directo.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prueba_corte")).scalar() == 0
    directo.dispose()
    m.dispose()


# ------------------------------------------------------------------ nivel API
def test_api_consulta_lenta_503_sin_registros_parciales(bd, url_bd, artefacto):
    """El INSERT de predicciones tarda más que el límite: se cancela a mitad de la
    transacción (la solicitud y sus familias ya se habían insertado) y se revierte todo."""
    m = motor_limitado(url_bd, consulta=1)
    d = _deps(m, ProveedorMemoria(artefacto))
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 200
        antes = contar(bd)
        dormir_al_insertar_predicciones(bd, 5)
        t = time.monotonic()
        r = c.post("/v1/prediccion", json=ej("prediccion_valida.json"))
        assert r.status_code == 503 and r.json() == {"detail": "la base de datos no está disponible"}
        assert time.monotonic() - t < 3
        assert contar(bd) == antes                               # ni solicitud, ni familias, ni calidad
        assert c.get("/health").status_code == 200                # la API sigue respondiendo
        d.monitor.ejecutar_ciclo()
        assert c.get("/health").json()["estado"] == "ok"          # la base está viva
        quitar_dormir(bd)
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 200
        assert contar(bd)["solicitudes_prediccion"] == antes["solicitudes_prediccion"] + 1
    d.monitor.detener()
    m.dispose()


def test_api_servidor_congelado_503_health_y_recuperacion(bd, url_bd, proxy, artefacto):
    m = motor_limitado(url_bd, proxy, consulta=1, margen=1, conexion=2)
    d = _deps(m, ProveedorMemoria(artefacto))
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 200
        antes = contar(bd)
        proxy.congelar()
        t = time.monotonic()
        r = c.post("/v1/prediccion", json=ej("prediccion_valida.json"))
        assert r.status_code == 503 and time.monotonic() - t < 10    # no queda esperando
        t = time.monotonic()
        assert c.get("/health").status_code == 200 and time.monotonic() - t < 0.5
        d.monitor.ejecutar_ciclo()                                     # acotado por el límite global
        h = c.get("/health").json()
        assert h["estado"] == "degradado" and h["motivo"] == "base de datos no disponible"
        assert c.get("/modelo").status_code == 200                     # el resto de la API responde
        proxy.descongelar()
        proxy.cortar_conexiones()
        assert contar(bd) == antes                                      # nada parcial confirmado
        d.monitor.ejecutar_ciclo()
        assert c.get("/health").json()["estado"] == "ok"
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 200
    d.monitor.detener()
    m.dispose()


def test_api_congelada_a_mitad_del_registro_no_confirma(bd, url_bd, proxy, artefacto):
    """El servidor queda sin responder mientras ejecuta el INSERT de predicciones."""
    m = motor_limitado(url_bd, proxy, consulta=3, margen=1, conexion=2)
    d = _deps(m, ProveedorMemoria(artefacto))
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 200
        antes = contar(bd)
        dormir_al_insertar_predicciones(bd, 1.5)          # < 3 s: el servidor la completaría
        threading.Timer(0.7, proxy.congelar).start()      # pero la respuesta nunca vuelve
        t = time.monotonic()
        r = c.post("/v1/prediccion", json=ej("prediccion_valida.json"))
        assert r.status_code == 503 and time.monotonic() - t < 12
        assert contar(bd) == antes                         # la transacción no se confirmó
        proxy.descongelar()
        proxy.cortar_conexiones()
        time.sleep(0.3)
        assert contar(bd) == antes                         # y el servidor la revirtió
        quitar_dormir(bd)
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 200
    d.monitor.detener()
    m.dispose()


def test_monitor_con_base_congelada_respeta_su_limite(bd, url_bd, proxy, artefacto):
    m = motor_limitado(url_bd, proxy, consulta=1, margen=1, conexion=2)
    d = _deps(m, ProveedorMemoria(artefacto))
    proxy.congelar()
    t = time.monotonic()
    d.monitor.ejecutar_ciclo()
    assert time.monotonic() - t < d.monitor.limite + 0.5
    assert d.monitor.diagnostico().componentes["base_datos"] == "falla"
    proxy.descongelar()
    d.monitor.detener()
    m.dispose()
