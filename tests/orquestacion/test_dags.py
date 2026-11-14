"""DAG de la etapa 7 con Airflow 2.10.5 real (job "Airflow" de la CI; se omiten si Airflow
no está instalado). No importan código del proyecto (Airflow trae SQLAlchemy 1.4).

* Estructura: sin errores de importación, cadena, reglas de disparo, reintentos, límites
  de tiempo y opciones del ``BashOperator``.
* Tareas SQL contra PostgreSQL real (si ``NM_PRUEBA_DB`` está definida: la CI migra esa
  base y crea los roles con el entorno de la aplicación antes de este paso): t0
  conservado, ``skipped`` con la solicitud cerrada, ``registrar_resultado`` nunca escribe
  ``exitoso`` y el vigilante decide por lo persistido.

Ejecución:  python -m pytest tests/orquestacion --confcutdir=tests/orquestacion -q
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")

from airflow.exceptions import AirflowException, AirflowFailException, AirflowSkipException  # noqa: E402
from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[2] / "dags"


@pytest.fixture(scope="module")
def bolsa():
    import sys
    if str(DAGS) not in sys.path:          # Airflow agrega DAGS_FOLDER a sys.path (en la imagen, /opt/novamercado/dags)
        sys.path.insert(0, str(DAGS))
    b = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert b.import_errors == {}
    return b


def test_dags_cargan(bolsa):
    assert set(bolsa.dag_ids) == {"novamercado_reentrenamiento", "novamercado_reconciliacion"}


def test_reentrenamiento_estructura(bolsa):
    d = bolsa.dags["novamercado_reentrenamiento"]
    assert d.schedule_interval is None and d.catchup is False and d.max_active_runs == 1
    t = {x.task_id: x for x in d.tasks}
    assert list(t) == ["marcar_inicio", "reentrenar", "registrar_resultado", "vigilante"]
    assert t["marcar_inicio"].downstream_task_ids == {"reentrenar"}
    assert t["reentrenar"].downstream_task_ids == {"registrar_resultado"}
    assert t["registrar_resultado"].downstream_task_ids == {"vigilante"}
    assert t["vigilante"].downstream_task_ids == set()
    assert [t[x].trigger_rule for x in ("reentrenar", "registrar_resultado", "vigilante")] == \
        ["all_success", "all_done", "all_done"]
    assert {x: t[x].retries for x in t} == {"marcar_inicio": 0, "reentrenar": 0, "registrar_resultado": 0,
                                           "vigilante": 3}
    assert t["vigilante"].retry_delay == timedelta(seconds=30)
    assert t["reentrenar"].execution_timeout == timedelta(minutes=8, seconds=30)
    assert t["registrar_resultado"].execution_timeout == timedelta(seconds=20)
    assert t["vigilante"].execution_timeout == timedelta(seconds=40)
    r = t["reentrenar"]
    assert r.skip_on_exit_code in (None, [])                         # ningún código es "saltada"
    assert r.do_xcom_push is False and r.append_env is True
    assert r.env == {"NM_EJECUCION": "{{ run_id }}"} and "{{" not in r.bash_command   # run_id solo por el entorno
    assert r.bash_command.startswith("/opt/novamercado/docker/ejecutar reentrenar ")
    for x in t.values():                                              # credenciales nunca en plantillas
        for campo in getattr(x, "template_fields", ()):
            assert "PASSWORD" not in str(getattr(x, campo, ""))


def test_reconciliacion_estructura(bolsa):
    d = bolsa.dags["novamercado_reconciliacion"]
    assert d.schedule_interval == timedelta(minutes=5) and d.max_active_runs == 1 and d.catchup is False
    (t,) = d.tasks
    assert t.skip_on_exit_code == [5] and t.bash_command == "/opt/novamercado/docker/ejecutar reconciliacion"


# ------------------------------------------------------------- tareas SQL reales
BD = os.environ.get("NM_PRUEBA_DB")
requiere_bd = pytest.mark.skipif(not BD, reason="requiere NM_PRUEBA_DB (base migrada con roles)")


class TI:
    def __init__(self, state=None):
        self.state = state
        self.xcom = {}

    def xcom_push(self, key, value):
        self.xcom[key] = value


class DagRun:
    def __init__(self, estado_reentrenar):
        self.ti = TI(estado_reentrenar)

    def get_task_instance(self, task_id):
        assert task_id == "reentrenar"
        return self.ti


def ctx(id_, estado="success"):
    return {"run_id": id_, "ti": TI(), "dag_run": DagRun(estado)}


@pytest.fixture
def modulo(monkeypatch):
    if not BD:
        pytest.skip("requiere NM_PRUEBA_DB")
    import sys
    sys.path.insert(0, str(DAGS))
    import novamercado_reentrenamiento as m
    return m


def conexion():
    import psycopg2
    c = psycopg2.connect(host=os.environ["DB_HOST"], port=int(os.environ.get("DB_PORT", "5432")),
                         dbname=os.environ["POSTGRES_DB"], user=os.environ["POSTGRES_USER"],
                         password=os.environ.get("POSTGRES_PASSWORD", ""))
    c.autocommit = True
    return c


def sql(q, args=()):
    with conexion() as c, c.cursor() as cur:
        cur.execute(q, args)
        return cur.fetchall() if cur.description else None


@requiere_bd
def test_marcar_inicio_conserva_t0_y_crea_la_solicitud(modulo):
    id_ = f"manual__{uuid.uuid4().hex}"
    t0 = modulo.tarea_marcar_inicio(**ctx(id_))
    assert modulo.tarea_marcar_inicio(**ctx(id_)) == t0                  # reintento: el primer t0
    assert sql("SELECT origen FROM solicitudes_reentrenamiento WHERE id = %s", (id_,)) == [("airflow",)]


@requiere_bd
def test_marcar_inicio_skipped_si_ya_esta_cerrada(modulo):
    id_ = f"nm-{uuid.uuid4()}"
    sql("INSERT INTO solicitudes_reentrenamiento (id, origen) VALUES (%s, 'api')", (id_,))
    sql("INSERT INTO resultados_reentrenamiento (id, estado, promovido, motivo, escrito_por) "
        "VALUES (%s, 'fallido', false, 'cola', 'api')", (id_,))
    with pytest.raises(AirflowSkipException):
        modulo.tarea_marcar_inicio(**ctx(id_))
    assert sql("SELECT count(*) FROM tareas_reentrenamiento WHERE id = %s", (id_,)) == [(0,)]


@requiere_bd
def test_registrar_resultado_y_vigilante(modulo):
    id_ = f"nm-{uuid.uuid4()}"
    sql("INSERT INTO solicitudes_reentrenamiento (id, origen) VALUES (%s, 'api')", (id_,))
    c = ctx(id_, "failed")
    assert modulo.tarea_registrar_resultado(**c) == "escrito"
    assert isinstance(c["ti"].xcom["inicio_efectivo"], int)
    assert sql("SELECT estado, escrito_por FROM resultados_reentrenamiento WHERE id = %s", (id_,)) == \
        [("fallido", "registrar_resultado")]
    assert modulo.tarea_registrar_resultado(**ctx(id_)) == "ya_registrado"
    with pytest.raises(AirflowFailException, match="fallido"):
        modulo.tarea_vigilante(**ctx(id_, "failed"))


@requiere_bd
def test_vigilante_exitoso_solo_con_reentrenar_success(modulo):
    id_ = f"nm-{uuid.uuid4()}"
    sql("INSERT INTO solicitudes_reentrenamiento (id, origen) VALUES (%s, 'api')", (id_,))
    sql("INSERT INTO resultados_reentrenamiento (id, estado, promovido, motivo, escrito_por) "
        "VALUES (%s, 'exitoso', false, 'ok', 'reentrenar')", (id_,))
    assert modulo.tarea_vigilante(**ctx(id_, "success")) == "exitoso"
    with pytest.raises(AirflowFailException):
        modulo.tarea_vigilante(**ctx(id_, "failed"))


@requiere_bd
def test_vigilante_ausencia_falla_sin_reintento(modulo):
    id_ = f"nm-{uuid.uuid4()}"
    sql("INSERT INTO solicitudes_reentrenamiento (id, origen) VALUES (%s, 'api')", (id_,))
    with pytest.raises(AirflowFailException, match="veredicto"):
        modulo.tarea_vigilante(**ctx(id_, "failed"))
    assert sql("SELECT veredicto FROM veredictos_reentrenamiento WHERE id = %s", (id_,)) == [("sin_resultado",)]


@requiere_bd
def test_vigilante_indeterminado_reintenta(modulo, monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")                                  # PostgreSQL inalcanzable
    with pytest.raises(AirflowException) as e:
        modulo.tarea_vigilante(**ctx("nm-x"))
    assert not isinstance(e.value, AirflowFailException)
