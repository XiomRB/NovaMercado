"""API con PostgreSQL como fuente de verdad del vigente y MLflow real (etapas 5 y 6):
/modelo, recarga sin reinicio, MLflow caído, caché, E-ANT, alias desincronizado y
concurrencia durante promociones."""

import json
import shutil
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import esquema as e
from app.dependencias import Dependencias, construir_monitor
from app.main import create_app
from app.modelo.artefacto import sha256_archivo
from app.modelo.cache import CacheModelos
from app.modelo.gestor import GestorVigente
from app.modelo.proveedor_mlflow import ProveedorMLflow
from app.orquestador import OrquestadorNoConfigurado
from app.servicios.persistencia import PersistenciaPostgres
from ml.pronosticador import ConfigEntrenamiento
from tests.conftest import FIXTURES
from tests.integracion.apoyo_mlflow import (
    alias_actual,
    cargar,
    cliente,
    promover_existente,
    pronosticador_de,
    registrar,
    tags_version,
    vigente_pg,
)

REQ = ("api", "base_datos", "modelo", "mlflow")
PRED = json.loads((FIXTURES / "ejemplos" / "prediccion_valida.json").read_text(encoding="utf-8"))


class BaseCaida:
    """Lector del vigente que simula PostgreSQL inaccesible para el proveedor."""

    def __call__(self):
        raise ConnectionError("PostgreSQL no responde (simulado)")


@pytest.fixture
def bd(motor, tmp_path):
    cargar(motor, tmp_path / "raw")
    return motor


def armar(motor, uri, nombre, cache_dir, intervalo=15, lector=None):
    prov = ProveedorMLflow(uri, nombre, "vigente", CacheModelos(cache_dir), motor=motor, lector_vigente=lector)
    gestor = GestorVigente(prov, intervalo=intervalo)
    gestor.refrescar()
    persistencia = PersistenciaPostgres(motor)
    orq = OrquestadorNoConfigurado()
    monitor = construir_monitor(persistencia, gestor, orq, requeridos=REQ, intervalo=1, limite=3)
    monitor.ejecutar_ciclo()
    return Dependencias(persistencia, gestor, monitor, orq)


def cliente_http(d):
    return TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False)


def test_modelo_y_prediccion_con_la_version_confirmada(bd, servidor_mlflow, nombre_modelo, tmp_path):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache")
    with cliente_http(d) as c:
        assert c.get("/modelo").json() == {"nombre": nombre_modelo, "alias": "vigente", "version": "1",
                                           "es_vigente": True, "estado_vigencia": "vigente",
                                           "version_vigente": "1"}
        r = c.post("/v1/prediccion", json=PRED)
        assert r.status_code == 200 and r.json()["version_modelo"] == "1" and r.json()["modelo_es_vigente"]
        h = c.get("/health").json()
        assert h["estado"] == "ok" and h["componentes"]["mlflow"] == "ok"
    with bd.connect() as conn:                                    # trazabilidad desde PostgreSQL
        s = conn.execute(select(e.solicitudes_prediccion)).one()
    assert (s.nombre_modelo, s.version_modelo) == (nombre_modelo, "1")
    mv = cliente(uri).get_model_version(s.nombre_modelo, s.version_modelo)
    assert tags_version(uri, nombre_modelo, "1")["huella_datos"] == cliente(uri).get_run(mv.run_id).data.tags[
        "huella_datos"]
    assert CacheModelos(tmp_path / "cache").versiones() == ["1"]
    meta = json.loads((tmp_path / "cache" / "1" / "metadatos.json").read_text())
    v = vigente_pg(bd, nombre_modelo)
    assert (meta["decision_id"], meta["secuencia"]) == (v.decision_id, v.secuencia)
    d.monitor.detener()


def test_promocion_se_detecta_en_menos_de_30_s(bd, servidor_mlflow, nombre_modelo, tmp_path):
    """Con el intervalo por defecto (15 s) y el gestor en segundo plano, sin reiniciar."""
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache", intervalo=15)
    d.gestor.iniciar()
    try:
        r2 = registrar(bd, uri, nombre_modelo, promover="siempre")
        t = time.monotonic()
        while d.gestor.actual().version != r2.version:
            assert time.monotonic() - t < 30, "la promoción no se detectó en 30 s"
            time.sleep(0.2)
    finally:
        d.gestor.detener()
        d.monitor.detener()


def test_mlflow_caido_la_api_sigue_prediciendo(bd, servidor_mlflow_propio, nombre_modelo, tmp_path):
    srv = servidor_mlflow_propio
    registrar(bd, srv.uri, nombre_modelo)
    d = armar(bd, srv.uri, nombre_modelo, tmp_path / "cache")
    with cliente_http(d) as c:
        srv.detener()
        d.gestor.refrescar()
        d.monitor.ejecutar_ciclo()
        for _ in range(5):
            r = c.post("/v1/prediccion", json=PRED)
            assert r.status_code == 200 and r.json()["version_modelo"] == "1"
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["estado"] == "degradado"
        assert h.json()["motivo"] == "MLflow no disponible" and h.json()["componentes"]["modelo"] == "ok"
        srv.iniciar()
        d.gestor.refrescar()
        d.monitor.ejecutar_ciclo()
        assert c.get("/health").json()["estado"] == "ok"
        assert c.get("/modelo").json()["version"] == "1"
    d.monitor.detener()


def _dos_versiones_en_cache(bd, srv, nombre, cache_dir):
    from tests.integracion.apoyo_mlflow import CONFIG
    registrar(bd, srv.uri, nombre)
    armar(bd, srv.uri, nombre, cache_dir).monitor.detener()          # cachea v1
    # v2 con otra configuración: el entrenamiento es determinista y con la misma
    # configuración los dos artefactos serían idénticos byte a byte
    otra = ConfigEntrenamiento(**{**CONFIG.__dict__, "params_hgb": dict(max_iter=15)})
    registrar(bd, srv.uri, nombre, promover="siempre", config=otra)
    armar(bd, srv.uri, nombre, cache_dir).monitor.detener()          # cachea v2
    assert CacheModelos(cache_dir).versiones() == ["2", "1"]


def test_reinicio_sin_mlflow_usa_la_entrada_del_vigente(bd, servidor_mlflow_propio, nombre_modelo, tmp_path):
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    _dos_versiones_en_cache(bd, srv, nombre_modelo, cache)
    srv.detener()
    d = armar(bd, srv.uri, nombre_modelo, cache)                      # "reinicio de la API" sin MLflow
    assert d.gestor.servido().modelo.version == "2" and d.gestor.servido().es_vigente
    with cliente_http(d) as c:
        assert c.post("/v1/prediccion", json=PRED).json()["version_modelo"] == "2"
        assert c.get("/health").json()["motivo"] == "MLflow no disponible"
    d.monitor.detener()
    # la entrada del vigente se corrompe: con PostgreSQL disponible NO se usa otra versión
    (cache / "2" / "modelo.joblib").write_bytes(b"corrupto")
    d = armar(bd, srv.uri, nombre_modelo, cache)
    with cliente_http(d) as c:
        assert c.post("/v1/prediccion", json=PRED).status_code == 503
        assert c.get("/health").status_code == 503
    d.monitor.detener()


def test_base_inaccesible_al_arrancar_sirve_la_cache_sin_verificar(bd, servidor_mlflow_propio, nombre_modelo,
                                                                   tmp_path):
    """PostgreSQL no respondió ninguna consulta desde el arranque: se sirve la entrada
    confirmada de mayor secuencia en estado sin_verificar (revisión del PR #6)."""
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    _dos_versiones_en_cache(bd, srv, nombre_modelo, cache)
    srv.detener()
    caida = BaseCaida()
    d = armar(bd, srv.uri, nombre_modelo, cache, lector=caida)        # PostgreSQL y MLflow caídos
    d.monitor.verificaciones["base_datos"] = lambda: False             # la API tampoco alcanza la base
    d.monitor.ejecutar_ciclo()
    with cliente_http(d) as c:
        m = c.get("/modelo").json()
        assert m == {"nombre": nombre_modelo, "alias": "vigente", "version": "2", "es_vigente": False,
                     "estado_vigencia": "sin_verificar", "motivo": m["motivo"]}
        assert "version_vigente" not in m and "no se pudo verificar el vigente actual" in m["motivo"]
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["estado"] == "degradado"
        assert h.json()["componentes"]["modelo"] == "sin_verificar"
        assert "base de datos no disponible" in h.json()["motivo"]
        assert "no se pudo verificar la vigencia del modelo" in h.json()["motivo"]
        assert c.get("/health").json()["componentes"]["base_datos"] == "falla"
    # la base vuelve: misma decisión que la caché → vigente, sin descargar (MLflow sigue caído)
    d.gestor.proveedor._lector = None
    d.monitor.verificaciones["base_datos"] = d.persistencia.disponible
    d.gestor.refrescar()
    d.monitor.ejecutar_ciclo()
    s = d.gestor.servido()
    assert (s.modelo.version, s.estado_vigencia, s.es_vigente, s.version_vigente) == ("2", "vigente", True, "2")
    with cliente_http(d) as c:
        r = c.post("/v1/prediccion", json=PRED)
        assert r.status_code == 200 and r.json()["version_modelo"] == "2" and r.json()["modelo_es_vigente"]
        assert c.get("/health").json()["componentes"]["modelo"] == "ok"
    d.monitor.detener()


def test_sin_verificar_responde_predicciones_marcadas(bd, servidor_mlflow_propio, nombre_modelo, tmp_path):
    """Las predicciones necesitan la base (se registran): con la base solo inaccesible para
    el proveedor, responden con la versión de la caché marcada como no vigente."""
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    _dos_versiones_en_cache(bd, srv, nombre_modelo, cache)
    srv.detener()
    d = armar(bd, srv.uri, nombre_modelo, cache, lector=BaseCaida())
    with cliente_http(d) as c:
        r = c.post("/v1/prediccion", json=PRED)
        assert r.status_code == 200 and r.json()["version_modelo"] == "2" and r.json()["modelo_es_vigente"] is False
    (cache / "2" / "metadatos.json").write_text("{}")                 # la de mayor secuencia se corrompe
    d2 = armar(bd, srv.uri, nombre_modelo, cache, lector=BaseCaida())
    assert d2.gestor.servido().modelo.version == "1" and d2.gestor.servido().estado_vigencia == "sin_verificar"
    d.monitor.detener()
    d2.monitor.detener()


@pytest.mark.parametrize("al_volver", ["otro_vigente", "sin_vigente"])
def test_sin_verificar_transiciones_al_volver_la_base(bd, servidor_mlflow, nombre_modelo, tmp_path, al_volver):
    uri, cache = servidor_mlflow.uri, tmp_path / "cache"
    from tests.integracion.apoyo_mlflow import CONFIG
    registrar(bd, uri, nombre_modelo)
    armar(bd, uri, nombre_modelo, cache).monitor.detener()            # cachea v1 confirmada
    d = armar(bd, uri, nombre_modelo, cache, lector=BaseCaida())
    assert d.gestor.servido().estado_vigencia == "sin_verificar"
    if al_volver == "otro_vigente":
        otra = ConfigEntrenamiento(**{**CONFIG.__dict__, "params_hgb": dict(max_iter=15)})
        registrar(bd, uri, nombre_modelo, promover="siempre", config=otra)          # v2 confirmada
        d.gestor.proveedor._lector = None
        assert d.gestor.refrescar() is True
        s = d.gestor.servido()
        assert (s.modelo.version, s.estado_vigencia, s.version_vigente) == ("2", "vigente", "2")
    else:
        d.gestor.proveedor._lector = lambda: None                    # la base responde: no hay vigente
        d.gestor.refrescar()
        assert not d.gestor.cargado()
        with cliente_http(d) as c:
            assert c.post("/v1/prediccion", json=PRED).status_code == 503
    d.monitor.detener()


def test_sin_vigente_en_postgres_no_se_usa_la_cache(bd, servidor_mlflow, nombre_modelo, tmp_path):
    """PostgreSQL disponible que responde "no hay vigente": ni la caché ni un modelo cargado
    prevalecen (plan §7.2)."""
    uri, cache = servidor_mlflow.uri, tmp_path / "cache"
    registrar(bd, uri, nombre_modelo)
    d = armar(bd, uri, nombre_modelo, cache)
    assert d.gestor.cargado() and CacheModelos(cache).versiones() == ["1"]
    d.gestor.proveedor._lector = lambda: None                         # la base responde: cero filas
    d.gestor.refrescar()
    d.monitor.ejecutar_ciclo()
    with cliente_http(d) as c:
        assert c.post("/v1/prediccion", json=PRED).status_code == 503
        assert c.get("/health").status_code == 503
    d.monitor.detener()
    d2 = armar(bd, uri, nombre_modelo, cache, lector=lambda: None)    # arranque con la caché llena
    assert not d2.gestor.cargado()
    d2.monitor.detener()


def test_cache_corrupta_no_reemplaza_un_modelo_cargado(bd, servidor_mlflow_propio, nombre_modelo, tmp_path):
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    registrar(bd, srv.uri, nombre_modelo)
    d = armar(bd, srv.uri, nombre_modelo, cache)
    srv.detener()
    shutil.rmtree(cache / "1")
    (cache / "1").mkdir()
    (cache / "1" / "modelo.joblib").write_bytes(b"basura")
    for _ in range(3):
        d.gestor.refrescar()
    assert d.gestor.actual().version == "1"
    with cliente_http(d) as c:
        assert c.post("/v1/prediccion", json=PRED).status_code == 200
    d.monitor.detener()


def test_manipulacion_coherente_sin_base_ni_mlflow_se_corrige_al_volver(bd, servidor_mlflow_propio,
                                                                        nombre_modelo, tmp_path):
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    _dos_versiones_en_cache(bd, srv, nombre_modelo, cache)
    # se reemplaza la versión 2 de la caché por el artefacto de la 1, con metadatos coherentes
    shutil.copyfile(cache / "1" / "modelo.joblib", cache / "2" / "modelo.joblib")
    m = json.loads((cache / "2" / "metadatos.json").read_text())
    m["sha256_artefacto"] = sha256_archivo(cache / "2" / "modelo.joblib")
    (cache / "2" / "metadatos.json").write_text(json.dumps(m))
    srv.detener()
    d = armar(bd, srv.uri, nombre_modelo, cache, lector=BaseCaida())
    assert d.gestor.actual().version == "2"                          # limitación documentada: se acepta
    assert d.gestor.servido().estado_vigencia == "sin_verificar"
    # con PostgreSQL disponible, la entrada manipulada NO coincide con el SHA de la decisión
    d.gestor.proveedor._lector = None
    d.gestor.refrescar()
    assert d.gestor.servido().estado_vigencia == "anterior"          # la vigente no se pudo activar
    srv.iniciar()
    assert d.gestor.refrescar() is True                              # vuelve MLflow: se descarga la real
    assert d.gestor.servido().es_vigente
    real = pronosticador_de(srv.uri, nombre_modelo, "2")
    from tests.integracion.apoyo_mlflow import objetivos
    obj = objetivos()
    vacias = pd.DataFrame(columns=["fecha", "tienda_id", "familia_id", "ventas"])
    pd.testing.assert_frame_equal(d.gestor.actual().pronosticador.predecir(obj, vacias), real.predecir(obj, vacias))
    assert sha256_archivo(cache / "2" / "modelo.joblib") == vigente_pg(bd, nombre_modelo).sha256_artefacto
    d.monitor.detener()


def test_alias_movido_a_mano_no_cambia_el_modelo_y_se_informa(bd, servidor_mlflow, nombre_modelo, tmp_path):
    """El alias de MLflow no decide nada: se sirve PostgreSQL y una diferencia que persiste
    dos refrescos degrada /health."""
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    r2 = registrar(bd, uri, nombre_modelo, promover="nunca")
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache")
    cliente(uri).set_registered_model_alias(nombre_modelo, "vigente", r2.version)   # alguien lo mueve a mano
    d.gestor.refrescar()
    d.monitor.ejecutar_ciclo()
    with cliente_http(d) as c:
        assert c.get("/health").json()["estado"] == "ok"             # un solo refresco: no degrada
        d.gestor.refrescar()
        d.monitor.ejecutar_ciclo()
        assert c.get("/modelo").json()["version"] == "1"
        assert c.post("/v1/prediccion", json=PRED).json()["version_modelo"] == "1"
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["estado"] == "degradado"
        assert "el alias de MLflow no coincide con la promoción confirmada" in h.json()["motivo"]
    d.monitor.detener()


def test_e_ant_vigente_no_obtenible_se_sirve_el_anterior_marcado(bd, servidor_mlflow_propio, nombre_modelo,
                                                                 tmp_path):
    srv = servidor_mlflow_propio
    registrar(bd, srv.uri, nombre_modelo)
    d = armar(bd, srv.uri, nombre_modelo, tmp_path / "cache")
    registrar(bd, srv.uri, nombre_modelo, promover="siempre")        # v2 confirmada en PostgreSQL
    srv.detener()                                                    # ... y MLflow cae antes de descargarla
    d.gestor.refrescar()
    d.monitor.ejecutar_ciclo()
    with cliente_http(d) as c:
        r = c.post("/v1/prediccion", json=PRED)
        assert r.status_code == 200 and r.json()["version_modelo"] == "1" and r.json()["modelo_es_vigente"] is False
        m = c.get("/modelo").json()
        assert (m["version"], m["es_vigente"], m["estado_vigencia"], m["version_vigente"]) == ("1", False,
                                                                                            "anterior", "2")
        h = c.get("/health").json()
        assert h["estado"] == "degradado" and "se sirve la versión 1; la vigente 2" in h["motivo"]
        srv.iniciar()
        assert d.gestor.refrescar() is True                          # primer refresco con B obtenible
        m = c.get("/modelo").json()
        assert (m["version"], m["es_vigente"], m["estado_vigencia"]) == ("2", True, "vigente")
    d.monitor.detener()


def test_concurrencia_durante_promociones(bd, servidor_mlflow, nombre_modelo, tmp_path):
    uri = servidor_mlflow.uri
    from tests.integracion.apoyo_mlflow import CONFIG
    registrar(bd, uri, nombre_modelo)
    otra = ConfigEntrenamiento(**{**CONFIG.__dict__, "params_hgb": dict(max_iter=15)})
    assert registrar(bd, uri, nombre_modelo, promover="nunca", config=otra).valido   # v2: otro modelo válido
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache")
    esperado = {}
    for v in ("1", "2"):                                               # respuesta esperada de cada versión
        promover_existente(bd, uri, nombre_modelo, v)
        d.gestor.refrescar()
        with cliente_http(d) as c:
            esperado[v] = c.post("/v1/prediccion", json=PRED).json()["predicciones"]
    assert esperado["1"] != esperado["2"]          # sin esto no se podría detectar una mezcla de versiones
    errores, versiones, detener = [], [], threading.Event()

    def pedir():
        with cliente_http(d) as c:
            while not detener.is_set():
                r = c.post("/v1/prediccion", json=PRED)
                if r.status_code != 200:
                    errores.append(r.status_code)
                    continue
                v = r.json()["version_modelo"]
                versiones.append(v)
                if r.json()["predicciones"] != esperado[v]:
                    errores.append(f"predicciones incoherentes con la versión {v}")

    hilos = [threading.Thread(target=pedir) for _ in range(8)]
    for h in hilos:
        h.start()
    for i in range(10):
        promover_existente(bd, uri, nombre_modelo, "1" if i % 2 else "2")
        d.gestor.refrescar()
        time.sleep(0.2)
    detener.set()
    for h in hilos:
        h.join()
    assert errores == [] and set(versiones) == {"1", "2"} and len(versiones) > 20
    assert alias_actual(uri, nombre_modelo) == "1" and vigente_pg(bd, nombre_modelo).version == "1"
    d.monitor.detener()


def test_descargas_usan_el_servidor_indicado_sin_estado_global(bd, servidor_mlflow, nombre_modelo, tmp_path):
    """Regresión: MLflow 2.22 ignora tracking_uri con artifact_uri="runs:/…". En un proceso
    nuevo, sin MLFLOW_TRACKING_URI ni URI global, todo debe ir al servidor indicado."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo)
    t = tags_version(servidor_mlflow.uri, nombre_modelo, r.version)
    url = bd.url.render_as_string(hide_password=False)
    codigo = f"""
import mlflow
from mlflow import MlflowClient
from app.db.conexion import crear_motor
from app.modelo.cache import CacheModelos
from app.modelo.proveedor_mlflow import ProveedorMLflow
from app.modelo.registro import verificar_t3_t4b
mlflow.set_tracking_uri("file:///nonexistent-global")       # una URI global distinta y vacía
p = ProveedorMLflow({servidor_mlflow.uri!r}, {nombre_modelo!r}, "vigente", CacheModelos({str(tmp_path / 'c')!r}),
                    motor=crear_motor({url!r}))
o = p.obtener(p.consultar_vigente())
assert o.modelo.version == "1" and not o.desde_cache
c = MlflowClient(tracking_uri={servidor_mlflow.uri!r}, registry_uri={servidor_mlflow.uri!r})
assert verificar_t3_t4b(c, {servidor_mlflow.uri!r}, {nombre_modelo!r}, "1", {r.run_id!r}, {t['ruta_artefacto']!r},
                        {t['sha256_artefacto']!r}) == (None, None)
print("ok")
"""
    raiz = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_TRACKING")}
    p = subprocess.run([sys.executable, "-c", codigo], cwd=tmp_path, env={**env, "PYTHONPATH": str(raiz)},
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-2000:]
    assert not (tmp_path / "mlruns").exists()
