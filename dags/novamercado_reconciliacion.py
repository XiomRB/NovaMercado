"""DAG ``novamercado_reconciliacion`` (etapa 7, E7-8): cada 5 min, la reconciliación de la
etapa 6 (decisiones huérfanas y alias de MLflow = vigente de PostgreSQL). Toma la
concesión sin esperar: si otro proceso la tiene (código 5), la tarea queda ``skipped``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="novamercado_reconciliacion",
    description="Reconciliación periódica del alias de MLflow con el vigente de PostgreSQL",
    schedule=timedelta(minutes=5),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "owner": "novamercado"},
    tags=["novamercado"],
) as dag:
    BashOperator(task_id="reconciliar", bash_command="/opt/novamercado/docker/ejecutar reconciliacion",
                 append_env=True, skip_on_exit_code=5, do_xcom_push=False,
                 execution_timeout=timedelta(minutes=4))
