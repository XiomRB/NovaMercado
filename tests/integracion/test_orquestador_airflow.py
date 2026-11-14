"""``OrquestadorAirflow`` con PostgreSQL real y un Airflow simulado (etapa 7, plan v3–v12
§4): la solicitud se registra antes de llamar a Airflow, idempotencia con
``Idempotency-Key``, estado leído solo de PostgreSQL y recuperación (plazo de cola, G2 y
cierre tras el veredicto) con la condición evaluada bajo el bloqueo."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db import cierre
from app.db.conexion import BaseDatosNoDisponible, crear_motor
from app.db.roles import configurar_roles
from app.orquestador import OrquestadorNoDisponible
from app.orquestador_airflow import DAG_ID, ClaveInvalida, ConfigAirflow, OrquestadorAirflow
from tests.api.test_admin_reentrenar import AirflowSimulado

CLAVE = "clave-cierre-prueba"


@pytest.fixture
def airflow():
    a = AirflowSimulado()
    yield a
    a.cerrar()


@pytest.fixture
def orq(motor, url_bd, airflow):
    configurar_roles(motor, CLAVE, "clave-vigilante-prueba")
    params = {**cierre.parametros_desde_url(url_bd), "user": "nm_cierre", "password": CLAVE}
    return OrquestadorAirflow(motor, ConfigAirflow(airflow.url, "api", "c", timeout=1.0), nombre_modelo="m",
                              parametros_cierre=lambda: params, espera_confirmacion=0.05)


def filas(motor, tabla, id_=None):
    with motor.connect() as c:
        q = f"SELECT * FROM {tabla}" + (" WHERE id = :i" if id_ else "")
        return c.execute(text(q), {"i": id_}).mappings().all()


def posts(airflow):
    return [r for r in airflow.recibidas if r[0] == "POST"]


def test_registra_antes_de_llamar_y_acepta(orq, motor, airflow):
    id_ = orq.iniciar()
    assert id_.startswith("nm-") and len(id_) == 39
    s = filas(motor, "solicitudes_reentrenamiento", id_)[0]
    assert s["origen"] == "api" and s["clave_idempotencia"] is None
    assert posts(airflow)[0][2]["dag_run_id"] == id_
    assert orq.estado(id_).como_respuesta() == {"estado": "en_curso"}


def test_sin_base_no_llama_a_airflow(airflow):
    m = crear_motor("postgresql+psycopg2://u:c@127.0.0.1:1/x", timeout_conexion=1)
    o = OrquestadorAirflow(m, ConfigAirflow(airflow.url, "a", "b"), nombre_modelo="m", parametros_cierre=dict)
    with pytest.raises(OrquestadorNoDisponible, match="registrar"):
        o.iniciar()
    assert posts(airflow) == []
    with pytest.raises(BaseDatosNoDisponible):
        o.estado("nm-x")


def test_airflow_rechaza_la_solicitud_queda_pendiente(orq, motor, airflow):
    airflow.post = (500, {}, 0)
    with pytest.raises(OrquestadorNoDisponible, match="pendiente"):
        orq.iniciar(clave="k1")
    (s,) = filas(motor, "solicitudes_reentrenamiento")
    assert filas(motor, "resultados_reentrenamiento") == []
    airflow.post = (200, None, 0)                            # reintento con la misma clave: mismo id
    assert orq.iniciar(clave="k1") == s["id"]
    assert [p[2]["dag_run_id"] for p in posts(airflow)] == [s["id"], s["id"]]


def test_idempotencia(orq, motor, airflow):
    a = orq.iniciar(clave="k-1")
    assert orq.iniciar(clave="k-1") == a                    # pendiente: se reenvía (Airflow: 200)
    assert len(posts(airflow)) == 2
    with motor.begin() as c:                                 # la tarea ya empezó: no se reenvía
        c.execute(text("INSERT INTO tareas_reentrenamiento VALUES (:i, clock_timestamp())"), {"i": a})
    assert orq.iniciar(clave="k-1") == a and len(posts(airflow)) == 2
    with pytest.raises(ClaveInvalida, match="otra solicitud"):
        orq.iniciar(clave="k-1", cuerpo=b'{"x": 1}')
    with pytest.raises(ClaveInvalida):
        orq.iniciar(clave="no válida")
    b = orq.iniciar(clave="k-2")
    assert b != a and orq.iniciar() not in (a, b)
    assert len(filas(motor, "solicitudes_reentrenamiento")) == 3


def test_clave_de_una_solicitud_terminada_responde_el_mismo_id(orq, motor, airflow):
    a = orq.iniciar(clave="k")
    with motor.begin() as c:
        c.execute(text("INSERT INTO resultados_reentrenamiento (id, estado, promovido, version, motivo, "
                       "escrito_por) VALUES (:i, 'exitoso', true, '4', 'ok', 'reentrenar')"), {"i": a})
    assert orq.iniciar(clave="k") == a and len(posts(airflow)) == 1
    assert orq.estado(a).como_respuesta() == {"estado": "exitoso", "promovido": True, "version": "4", "motivo": "ok"}


def test_estado_desconocido_y_version_sin_vigente(orq, motor):
    assert orq.estado("nm-no-existe") is None and orq.estado("x" * 300) is None
    a = orq.iniciar()
    with motor.begin() as c:
        c.execute(text("INSERT INTO resultados_reentrenamiento (id, estado, promovido, version, motivo, "
                       "escrito_por) VALUES (:i, 'fallido', false, NULL, 'm', 'dag')"), {"i": a})
    assert orq.estado(a).version == ""


# ---------------------------------------------------------------- recuperación
def solicitud(motor, *, hace_min=0, tarea_hace_min=None, veredicto=False):
    import uuid
    id_ = f"nm-{uuid.uuid4()}"
    with motor.begin() as c:
        c.execute(text("INSERT INTO solicitudes_reentrenamiento (id, origen, solicitado_en) VALUES "
                       "(:i, 'api', clock_timestamp() - make_interval(mins => :m))"), {"i": id_, "m": hace_min})
        if tarea_hace_min is not None:
            c.execute(text("INSERT INTO tareas_reentrenamiento VALUES (:i, clock_timestamp() - "
                           "make_interval(secs => :s))"), {"i": id_, "s": tarea_hace_min * 60})
        if veredicto:
            c.execute(text("INSERT INTO veredictos_reentrenamiento (id, veredicto) VALUES (:i, 'sin_resultado')"),
                      {"i": id_})
    return id_


@pytest.mark.parametrize("kw, cerrado, fragmento", [
    ({"hace_min": 29}, False, None),
    ({"hace_min": 31}, True, "plazo de cola"),
    ({"hace_min": 31, "tarea_hace_min": 5}, False, None),       # ya empezó: rige G2, no la cola
    ({"hace_min": 40, "tarea_hace_min": 9.9}, False, None),
    ({"hace_min": 40, "tarea_hace_min": 10.05}, True, "se interrumpió"),
    ({"tarea_hace_min": 1, "veredicto": True}, True, "veredicto"),
])
def test_recuperacion(orq, motor, kw, cerrado, fragmento):
    id_ = solicitud(motor, **kw)
    e = orq.estado(id_)                                          # G2-lectura: el GET recupera antes de leer
    assert (e.estado == "fallido") is cerrado
    if cerrado:
        assert fragmento in e.motivo and filas(motor, "resultados_reentrenamiento", id_)[0]["escrito_por"] == "api"


def test_recuperacion_periodica_no_toca_resultados(orq, motor):
    vencida = solicitud(motor, hace_min=31)
    terminada = solicitud(motor, tarea_hace_min=20)
    with motor.begin() as c:
        c.execute(text("INSERT INTO resultados_reentrenamiento (id, estado, promovido, motivo, escrito_por) "
                       "VALUES (:i, 'exitoso', false, 'ok', 'reentrenar')"), {"i": terminada})
    assert orq.recuperar() == [(vencida, "escrito")]
    assert filas(motor, "resultados_reentrenamiento", terminada)[0]["estado"] == "exitoso"


def test_condicion_se_evalua_bajo_el_bloqueo(orq, motor, url_bd):
    """La recuperación decidió 'cola vencida' pero la tarea empezó antes de escribir: la
    sentencia condicional no escribe."""
    id_ = solicitud(motor, hace_min=31)
    with motor.begin() as c:
        c.execute(text("INSERT INTO tareas_reentrenamiento VALUES (:i, clock_timestamp())"), {"i": id_})
    r = cierre.cerrar(orq.parametros_cierre(), id_, estado="fallido", motivo="x", escrito_por="api",
                      nombre_modelo="m", condicion="cola_vencida")
    assert r.codigo == "no_escrito" and filas(motor, "resultados_reentrenamiento", id_) == []


def test_hilo_de_recuperacion(orq, motor):
    id_ = solicitud(motor, hace_min=45)
    orq.intervalo = 0.2
    orq.iniciar_hilos()
    try:
        import time
        for _ in range(100):
            if filas(motor, "resultados_reentrenamiento", id_):
                break
            time.sleep(0.1)
    finally:
        orq.detener_hilos()
    assert filas(motor, "resultados_reentrenamiento", id_)[0]["estado"] == "fallido"


def test_dag_id_en_la_ruta(orq, airflow):
    orq.iniciar()
    assert posts(airflow)[0][1] == f"/api/v1/dags/{DAG_ID}/dagRuns"
