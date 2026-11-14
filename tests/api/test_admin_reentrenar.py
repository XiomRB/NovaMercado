"""``/admin/reentrenar`` con un orquestador simulado y ``OrquestadorAirflow`` frente a un
servidor HTTP simulado de Airflow (etapa 7, plan v3–v12 §4). Sin PostgreSQL: lo que usa
la base está en ``tests/integracion/test_orquestador_airflow.py``."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.orquestador import OrquestadorNoDisponible
from app.orquestador_airflow import (
    DAG_ID,
    ClaveInvalida,
    ClienteAirflow,
    ConfigAirflow,
    OrquestadorAirflow,
    huella,
    validar_clave,
)
from app.schemas.contrato import EstadoReentrenamiento


class OrquestadorFalso:
    def __init__(self):
        self.llamadas = []
        self.estados = {}
        self.error = None

    def iniciar(self, clave=None, cuerpo=b""):
        self.llamadas.append((clave, cuerpo))
        if self.error:
            raise self.error
        return "nm-1"

    def estado(self, id_):
        return self.estados.get(id_)

    def disponible(self):
        return True


@pytest.fixture
def orq(deps):
    o = OrquestadorFalso()
    deps.orquestador = o
    return o


def test_post_202_pasa_la_clave_y_el_cuerpo(cliente, orq):
    r = cliente.post("/admin/reentrenar", headers={"Idempotency-Key": "abc-1"})
    assert r.status_code == 202 and r.json() == {"id": "nm-1"}
    assert orq.llamadas == [("abc-1", b"")]


@pytest.mark.parametrize("error, codigo", [(ClaveInvalida("x"), 422), (OrquestadorNoDisponible("y"), 503)])
def test_post_errores(cliente, orq, error, codigo):
    orq.error = error
    r = cliente.post("/admin/reentrenar")
    assert r.status_code == codigo and set(r.json()) == {"detail"}


def test_get_estados(cliente, orq):
    orq.estados = {"a": EstadoReentrenamiento(estado="en_curso"),
                   "b": EstadoReentrenamiento(estado="exitoso", promovido=True, version="4", motivo="m")}
    assert cliente.get("/admin/reentrenar/a").json() == {"estado": "en_curso"}
    assert cliente.get("/admin/reentrenar/b").json() == {"estado": "exitoso", "promovido": True, "version": "4",
                                                         "motivo": "m"}
    assert cliente.get("/admin/reentrenar/c").status_code == 404


@pytest.mark.parametrize("clave", ["", "a" * 129, "con espacio", "ñ", "a/b", "a.b"])
def test_claves_invalidas(clave):
    with pytest.raises(ClaveInvalida):
        validar_clave(clave)


def test_claves_validas_y_huella():
    assert validar_clave("A" * 128) == "A" * 128 and validar_clave(None) is None
    assert huella(b"") != huella(b"{}") and len(huella(b"")) == 64


# --------------------------------------------------------- Airflow simulado
class AirflowSimulado:
    """Responde según ``self.post`` / ``self.get`` (código, cuerpo, demora) y registra
    las solicitudes con su autenticación."""

    def __init__(self):
        self.post = (200, None, 0)
        self.get = [(404, {}, 0)]
        self.salud = {"metadatabase": {"status": "healthy"}, "scheduler": {"status": "healthy"}}
        self.recibidas = []
        sim = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _responder(self, codigo, cuerpo, demora):
                time.sleep(demora)
                datos = json.dumps(cuerpo).encode()
                self.send_response(codigo)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                cuerpo = json.loads(self.rfile.read(n))
                sim.recibidas.append(("POST", self.path, cuerpo, self.headers.get("Authorization")))
                codigo, resp, demora = sim.post
                if resp is None:
                    resp = {"dag_run_id": cuerpo["dag_run_id"], "dag_id": DAG_ID, "state": "queued"}
                self._responder(codigo, resp, demora)

            def do_GET(self):
                sim.recibidas.append(("GET", self.path, None, self.headers.get("Authorization")))
                if self.path == "/health":
                    return self._responder(200, sim.salud, 0)
                codigo, resp, demora = sim.get.pop(0) if len(sim.get) > 1 else sim.get[0]
                self._responder(codigo, resp, demora)

        self.servidor = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.servidor.server_address[1]}"
        threading.Thread(target=self.servidor.serve_forever, daemon=True).start()

    def cerrar(self):
        self.servidor.shutdown()
        self.servidor.server_close()


@pytest.fixture
def airflow():
    a = AirflowSimulado()
    yield a
    a.cerrar()


def orquestador(url, timeout=1.0):
    cfg = ConfigAirflow(url, "api", "clave-api", timeout=timeout)
    return OrquestadorAirflow(None, cfg, nombre_modelo="m", parametros_cierre=dict, espera_confirmacion=0.05)


def test_200_coherente(airflow):
    o = orquestador(airflow.url)
    assert o._enviar("nm-x") is True
    metodo, ruta, cuerpo, auth = airflow.recibidas[0]
    assert (metodo, ruta, cuerpo) == ("POST", f"/api/v1/dags/{DAG_ID}/dagRuns", {"dag_run_id": "nm-x", "conf": {}})
    assert auth.startswith("Basic ")


@pytest.mark.parametrize("cuerpo", [{"dag_run_id": "otro", "dag_id": DAG_ID}, {"dag_run_id": "nm-x", "dag_id": "x"},
                                    {}])
def test_200_incoherente_se_confirma(airflow, cuerpo):
    airflow.post = (200, cuerpo, 0)
    airflow.get = [(404, {}, 0)]
    assert orquestador(airflow.url)._enviar("nm-x") is False
    airflow.get = [(200, {"dag_run_id": "nm-x", "dag_id": DAG_ID}, 0)]
    assert orquestador(airflow.url)._enviar("nm-x") is True


@pytest.mark.parametrize("get, aceptada", [
    ([(200, {"dag_run_id": "nm-x", "dag_id": DAG_ID}, 0)], True),
    ([(200, {"dag_run_id": "nm-x", "dag_id": "otro"}, 0)], False),
    ([(404, {}, 0)], False),
    ([(500, {}, 0), (500, {}, 0), (500, {}, 0)], False),                     # sin confirmación: pendiente
    ([(500, {}, 0), (200, {"dag_run_id": "nm-x", "dag_id": DAG_ID}, 0)], True),
])
def test_409_solo_con_confirmacion(airflow, get, aceptada):
    airflow.post = (409, {"detail": "ya existe"}, 0)
    airflow.get = list(get)
    assert orquestador(airflow.url)._enviar("nm-x") is aceptada


def test_respuesta_perdida_se_confirma(airflow):
    airflow.post = (200, None, 2.0)                                            # más que el timeout del cliente
    airflow.get = [(200, {"dag_run_id": "nm-x", "dag_id": DAG_ID}, 0)]
    t = time.monotonic()
    assert orquestador(airflow.url, timeout=0.5)._enviar("nm-x") is True
    assert time.monotonic() - t < 2.0


@pytest.mark.parametrize("codigo", [400, 401, 403, 404, 500, 503])
def test_otros_codigos_no_aceptan(airflow, codigo):
    airflow.post = (codigo, {}, 0)
    assert orquestador(airflow.url)._enviar("nm-x") is False


def test_airflow_caido():
    o = orquestador("http://127.0.0.1:1", timeout=0.3)
    assert o._enviar("nm-x") is False and o.disponible() is False


def test_salud(airflow):
    o = orquestador(airflow.url)
    assert o.disponible() is True
    airflow.salud = {"metadatabase": {"status": "healthy"}, "scheduler": {"status": "unhealthy"}}
    assert o.disponible() is False


def test_id_con_caracteres_especiales_se_codifica(airflow):
    airflow.get = [(200, {"dag_run_id": "a:b+c", "dag_id": DAG_ID}, 0)]
    ClienteAirflow(ConfigAirflow(airflow.url, "u", "c")).leer_ejecucion("a:b+c")
    assert airflow.recibidas[-1][1] == f"/api/v1/dags/{DAG_ID}/dagRuns/a%3Ab%2Bc"
