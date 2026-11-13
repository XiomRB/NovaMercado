"""API con el proveedor MLflow real: /modelo, recarga por alias, MLflow caído, caché y concurrencia."""

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
from app.modelo.gestor import GestorModelo
from app.modelo.proveedor_mlflow import ProveedorMLflow
from app.orquestador import OrquestadorNoConfigurado
from app.servicios.persistencia import PersistenciaPostgres
from ml.pronosticador import ConfigEntrenamiento
from tests.conftest import FIXTURES
from tests.integracion.apoyo_mlflow import (
    alias_actual,
    cargar,
    cliente,
    pronosticador_de,
    registrar,
    tags_version,
)

REQ = ("api", "base_datos", "modelo", "mlflow")
PRED = json.loads((FIXTURES / "ejemplos" / "prediccion_valida.json").read_text(encoding="utf-8"))


@pytest.fixture
def bd(motor, tmp_path):
    cargar(motor, tmp_path / "raw")
    return motor


def armar(motor, uri, nombre, cache_dir, intervalo=15):
    prov = ProveedorMLflow(uri, nombre, "vigente", CacheModelos(cache_dir))
    gestor = GestorModelo(prov, intervalo=intervalo)
    gestor.refrescar()
    persistencia = PersistenciaPostgres(motor)
    orq = OrquestadorNoConfigurado()
    monitor = construir_monitor(persistencia, gestor, orq, requeridos=REQ, intervalo=1, limite=3)
    monitor.ejecutar_ciclo()
    return Dependencias(persistencia, gestor, monitor, orq)


def cliente_http(d):
    return TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False)


def test_modelo_y_prediccion_con_la_version_registrada(bd, servidor_mlflow, nombre_modelo, tmp_path):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache")
    with cliente_http(d) as c:
        assert c.get("/modelo").json() == {"nombre": nombre_modelo, "alias": "vigente", "version": "1"}
        r = c.post("/v1/prediccion", json=PRED)
        assert r.status_code == 200 and r.json()["version_modelo"] == "1"
        h = c.get("/health").json()
        assert h["estado"] == "ok" and h["componentes"]["mlflow"] == "ok"
    with bd.connect() as conn:                                    # trazabilidad desde PostgreSQL
        s = conn.execute(select(e.solicitudes_prediccion)).one()
    assert (s.nombre_modelo, s.version_modelo) == (nombre_modelo, "1")
    mv = cliente(uri).get_model_version(s.nombre_modelo, s.version_modelo)
    assert tags_version(uri, nombre_modelo, "1")["huella_datos"] == cliente(uri).get_run(mv.run_id).data.tags[
        "huella_datos"]
    assert CacheModelos(tmp_path / "cache").versiones() == ["1"]
    d.monitor.detener()


def test_cambio_de_alias_se_detecta_en_menos_de_30_s(bd, servidor_mlflow, nombre_modelo, tmp_path):
    """Con el intervalo por defecto (15 s) y el gestor en segundo plano, sin reiniciar."""
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache", intervalo=15)
    d.gestor.iniciar()
    try:
        r2 = registrar(bd, uri, nombre_modelo, promover="siempre")
        t = time.monotonic()
        while d.gestor.actual().version != r2.version:
            assert time.monotonic() - t < 30, "el cambio de alias no se detectó en 30 s"
            time.sleep(0.2)
        assert time.monotonic() - t <= 30
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
    registrar(bd, srv.uri, nombre)
    armar(bd, srv.uri, nombre, cache_dir).monitor.detener()          # cachea v1
    registrar(bd, srv.uri, nombre, promover="siempre")
    armar(bd, srv.uri, nombre, cache_dir).monitor.detener()          # cachea v2
    assert CacheModelos(cache_dir).versiones() == ["2", "1"]


def test_arranque_sin_mlflow_usa_la_cache_y_la_version_anterior_valida(bd, servidor_mlflow_propio,
                                                                     nombre_modelo, tmp_path):
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    _dos_versiones_en_cache(bd, srv, nombre_modelo, cache)
    srv.detener()
    d = armar(bd, srv.uri, nombre_modelo, cache)                      # "reinicio de la API" sin MLflow
    assert d.gestor.actual().version == "2"
    with cliente_http(d) as c:
        assert c.post("/v1/prediccion", json=PRED).json()["version_modelo"] == "2"
        assert c.get("/health").json()["motivo"] == "MLflow no disponible"
    d.monitor.detener()
    (cache / "2" / "modelo.joblib").write_bytes(b"corrupto")         # la más reciente se corrompe
    d = armar(bd, srv.uri, nombre_modelo, cache)
    with cliente_http(d) as c:
        assert c.get("/modelo").json()["version"] == "1"              # se recupera la anterior válida
        assert c.post("/v1/prediccion", json=PRED).json()["version_modelo"] == "1"
    d.monitor.detener()
    (cache / "1" / "metadatos.json").write_text("{}")
    d = armar(bd, srv.uri, nombre_modelo, cache)                      # ninguna válida
    with cliente_http(d) as c:
        assert c.post("/v1/prediccion", json=PRED).status_code == 503
        assert c.get("/health").status_code == 503
    d.monitor.detener()


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


def test_manipulacion_coherente_sin_mlflow_se_corrige_al_volver(bd, servidor_mlflow_propio, nombre_modelo,
                                                                tmp_path):
    srv, cache = servidor_mlflow_propio, tmp_path / "cache"
    _dos_versiones_en_cache(bd, srv, nombre_modelo, cache)
    # se reemplaza la versión 2 de la caché por el artefacto de la 1, con metadatos coherentes
    shutil.copyfile(cache / "1" / "modelo.joblib", cache / "2" / "modelo.joblib")
    m = json.loads((cache / "2" / "metadatos.json").read_text())
    m["sha256_artefacto"] = sha256_archivo(cache / "2" / "modelo.joblib")
    (cache / "2" / "metadatos.json").write_text(json.dumps(m))
    srv.detener()
    d = armar(bd, srv.uri, nombre_modelo, cache)
    assert d.gestor.actual().version == "2"                          # limitación documentada: se acepta
    srv.iniciar()
    assert d.gestor.refrescar() is True                              # vuelve MLflow: SHA contra el tag
    tag = tags_version(srv.uri, nombre_modelo, "2")["sha256_artefacto"]
    assert sha256_archivo(cache / "2" / "modelo.joblib") == tag      # la caché quedó corregida
    real = pronosticador_de(srv.uri, nombre_modelo, "2")
    from tests.integracion.apoyo_mlflow import objetivos
    obj = objetivos()
    vacias = pd.DataFrame(columns=["fecha", "tienda_id", "familia_id", "ventas"])
    pd.testing.assert_frame_equal(d.gestor.actual().pronosticador.predecir(obj, vacias), real.predecir(obj, vacias))
    d.monitor.detener()


def test_alias_a_una_version_invalida_conserva_el_modelo(bd, servidor_mlflow, nombre_modelo, tmp_path):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    r2 = registrar(bd, uri, nombre_modelo, promover="nunca")
    cliente(uri).set_model_version_tag(nombre_modelo, r2.version, "estado", "invalido")
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache")
    cliente(uri).set_registered_model_alias(nombre_modelo, "vigente", r2.version)   # alguien lo mueve a mano
    d.gestor.refrescar()
    d.monitor.ejecutar_ciclo()
    with cliente_http(d) as c:
        assert c.get("/modelo").json()["version"] == "1"
        assert c.post("/v1/prediccion", json=PRED).json()["version_modelo"] == "1"
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["estado"] == "degradado"
        assert h.json()["motivo"] == "no se pudo actualizar el modelo vigente"
        assert h.json()["componentes"]["modelo"] == "desactualizado"
        cliente(uri).delete_registered_model_alias(nombre_modelo, "vigente")       # alias inexistente
        d.gestor.refrescar()
        assert c.get("/modelo").json()["version"] == "1" and not d.gestor.actualizado()
    d.monitor.detener()


def test_concurrencia_durante_cambios_de_alias(bd, servidor_mlflow, nombre_modelo, tmp_path):
    uri = servidor_mlflow.uri
    from tests.integracion.apoyo_mlflow import CONFIG
    registrar(bd, uri, nombre_modelo)
    otra = ConfigEntrenamiento(**{**CONFIG.__dict__, "params_hgb": dict(max_iter=15)})
    assert registrar(bd, uri, nombre_modelo, promover="nunca", config=otra).valido   # v2: otro modelo válido
    d = armar(bd, uri, nombre_modelo, tmp_path / "cache")
    esperado = {}
    for v in ("1", "2"):                                               # respuesta esperada de cada versión
        cliente(uri).set_registered_model_alias(nombre_modelo, "vigente", v)
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
        cliente(uri).set_registered_model_alias(nombre_modelo, "vigente", "1" if i % 2 else "2")
        d.gestor.refrescar()
        time.sleep(0.2)
    detener.set()
    for h in hilos:
        h.join()
    assert errores == [] and set(versiones) == {"1", "2"} and len(versiones) > 20
    assert alias_actual(uri, nombre_modelo) == "1"
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
    codigo = f"""
import mlflow
from app.modelo.cache import CacheModelos
from app.modelo.proveedor_mlflow import ProveedorMLflow
from app.modelo.registro import verificar_t4b
mlflow.set_tracking_uri("file:///nonexistent-global")       # una URI global distinta y vacía
p = ProveedorMLflow({servidor_mlflow.uri!r}, {nombre_modelo!r}, "vigente", CacheModelos({str(tmp_path / 'c')!r}))
p.huella()
assert p.obtener().version == "1"
assert verificar_t4b({servidor_mlflow.uri!r}, {r.run_id!r}, {t['ruta_artefacto']!r}, {t['sha256_artefacto']!r}) is None
print("ok")
"""
    raiz = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_TRACKING")}
    p = subprocess.run([sys.executable, "-c", codigo], cwd=tmp_path, env={**env, "PYTHONPATH": str(raiz)},
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-2000:]
    assert not (tmp_path / "mlruns").exists()
