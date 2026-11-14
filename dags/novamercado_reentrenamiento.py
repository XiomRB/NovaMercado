"""DAG ``novamercado_reentrenamiento`` (etapa 7, DECISIONES.md §16, plan v12).

Cadena: ``marcar_inicio >> reentrenar >> registrar_resultado >> vigilante``.

* ``marcar_inicio`` (SQL, entorno de Airflow): bloquea la solicitud (la crea con
  ``origen=airflow`` si la ejecución se lanzó desde la interfaz); si ya tiene resultado o
  veredicto → ``skipped``; si ya tiene inicio conserva el primer t0; si no, registra
  t0 = ``clock_timestamp()``.
* ``reentrenar``: lanzador → arranque verificado → ``python -m app.modelo.reentrenar``.
  ``success`` solo si el resultado de negocio es ``exitoso`` (códigos internos 0 y 3).
  ``execution_timeout`` 8:30 (SIGTERM; SIGKILL a los 15 s de
  ``KILLED_TASK_CLEANUP_TIME``).
* ``registrar_resultado`` (``all_done``; SQL con ``nm_cierre``): si no hay resultado,
  escribe ``fallido`` (nunca ``exitoso``). Presupuesto 12 s (8 s de escritura + verificación).
* ``vigilante`` (``all_done``; SQL con ``nm_vigilante``): decide por el resultado
  persistido; sin resultado, bajo el bloqueo de la solicitud registra el veredicto de
  ausencia. Indeterminado → excepción y reintento (3 reintentos, 30 s). El DAG termina
  ``success`` solo si el resultado persistido es ``exitoso`` y ``reentrenar`` terminó bien.

Ninguna tarea importa código del proyecto: el SQL está en ``nm_cierre.py`` (copia
idéntica de ``app/db/cierre.py``, verificada por una prueba y por la procedencia). Las
credenciales salen del entorno del scheduler, nunca de campos con plantilla.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import nm_cierre
from airflow import DAG
from airflow.exceptions import AirflowException, AirflowFailException, AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

log = logging.getLogger("novamercado.dag")

DAG_ID = "novamercado_reentrenamiento"
LANZADOR = "/opt/novamercado/docker/ejecutar"
MODELO = os.environ.get("MLFLOW_MODELO_NOMBRE", "novamercado-ventas")
TIMEOUT_REENTRENAR = timedelta(minutes=8, seconds=30)
TIMEOUT_REGISTRAR = timedelta(seconds=20)
TIMEOUT_VIGILANTE = timedelta(seconds=40)
REINTENTOS_VIGILANTE = 3
ESPERA_REINTENTO_VIGILANTE = timedelta(seconds=30)


def _params(usuario_var: str | None, clave_var: str, usuario: str | None = None) -> dict:
    return {"host": os.environ["DB_HOST"], "port": int(os.environ.get("DB_PORT", "5432")),
            "dbname": os.environ["POSTGRES_DB"], "user": usuario or os.environ[usuario_var],
            "password": os.environ[clave_var]}


def tarea_marcar_inicio(**ctx):
    id_ = ctx["run_id"]
    r = nm_cierre.marcar_inicio(_params("POSTGRES_USER", "POSTGRES_PASSWORD"), id_)
    if r.codigo == "cerrada":
        raise AirflowSkipException(f"la solicitud {id_} ya está cerrada: no se entrena")
    log.info("t0 de %s: %s (%s)", id_, r.t0, r.codigo)
    return r.t0.isoformat()


def tarea_registrar_resultado(**ctx):
    inicio_efectivo = time.time_ns()                     # primera instrucción (presupuesto G2)
    log.info("registrar_resultado: inicio_efectivo=%s", inicio_efectivo)
    ctx["ti"].xcom_push(key="inicio_efectivo", value=inicio_efectivo)
    id_ = ctx["run_id"]
    previa = ctx["dag_run"].get_task_instance("reentrenar")
    estado = getattr(previa, "state", None) or "desconocido"
    r = nm_cierre.cerrar(_params(None, "NM_CIERRE_PASSWORD", usuario=nm_cierre.ROL_CIERRE), id_, estado="fallido",
                         motivo=f"la tarea reentrenar terminó sin registrar su resultado (estado: {estado})",
                         escrito_por="registrar_resultado", nombre_modelo=MODELO)
    log.info("registrar_resultado de %s: %s (persistido: %s) %s", id_, r.codigo, r.estado_persistido, r.detalle)
    if r.codigo == "indeterminado":
        raise AirflowException("no se pudo confirmar el resultado: lo decide el vigilante")
    return r.codigo


def tarea_vigilante(**ctx):
    id_ = ctx["run_id"]
    d = nm_cierre.vigilar(_params(None, "NM_VIGILANTE_PASSWORD", usuario="nm_vigilante"), id_)
    log.info("vigilante de %s: %s %s %s fases=%s", id_, d.tipo, d.estado, d.detalle, d.fases)
    if d.tipo == "indeterminado":
        raise AirflowException(f"resultado indeterminado ({d.detalle}): se reintenta")
    if d.tipo == "ausencia":
        raise AirflowFailException("la ejecución terminó sin resultado: veredicto de ausencia registrado")
    previa = ctx["dag_run"].get_task_instance("reentrenar")
    if d.estado == "exitoso" and getattr(previa, "state", None) == "success":
        return "exitoso"
    raise AirflowFailException(f"resultado {d.estado} (reentrenar: {getattr(previa, 'state', None)})")


with DAG(
    dag_id=DAG_ID,
    description="Reentrenamiento de NovaMercado solicitado por POST /admin/reentrenar",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "owner": "novamercado"},
    tags=["novamercado"],
) as dag:
    marcar_inicio = PythonOperator(task_id="marcar_inicio", python_callable=tarea_marcar_inicio,
                                   execution_timeout=timedelta(seconds=60))
    reentrenar = BashOperator(
        task_id="reentrenar",
        bash_command=f'{LANZADOR} reentrenar --ejecucion "$NM_EJECUCION"',
        env={"NM_EJECUCION": "{{ run_id }}"},
        append_env=True,
        skip_on_exit_code=None,
        do_xcom_push=False,
        execution_timeout=TIMEOUT_REENTRENAR,
    )
    registrar_resultado = PythonOperator(task_id="registrar_resultado", python_callable=tarea_registrar_resultado,
                                         trigger_rule="all_done", execution_timeout=TIMEOUT_REGISTRAR)
    vigilante = PythonOperator(task_id="vigilante", python_callable=tarea_vigilante, trigger_rule="all_done",
                               retries=REINTENTOS_VIGILANTE, retry_delay=ESPERA_REINTENTO_VIGILANTE,
                               execution_timeout=TIMEOUT_VIGILANTE)
    marcar_inicio >> reentrenar >> registrar_resultado >> vigilante
