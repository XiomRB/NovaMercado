"""Acceso a artefactos de MLflow sin depender del estado global del cliente.

En MLflow 2.22, ``mlflow.artifacts.download_artifacts`` resuelve los artefactos
``mlflow-artifacts:/…`` (servidor con ``--serve-artifacts``) usando la URI de
tracking **global** (``mlflow.set_tracking_uri`` o ``MLFLOW_TRACKING_URI``),
aunque se le pase ``tracking_uri``. Aquí la URI de artefactos de la ejecución se
resuelve explícitamente contra el ``tracking_uri`` indicado. Toda descarga del
proyecto pasa por esta función (DECISIONES.md §14).
"""

from __future__ import annotations

from pathlib import Path


def descargar_de_ejecucion(tracking_uri: str, run_id: str, ruta: str, destino: str | None = None) -> str:
    import tempfile

    from mlflow import MlflowClient
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository

    raiz = MlflowClient(tracking_uri=tracking_uri).get_run(run_id).info.artifact_uri
    if raiz.startswith("mlflow-artifacts:"):
        raiz = MlflowArtifactsRepository.resolve_uri(raiz, tracking_uri)   # http(s)://<servidor indicado>/…
    destino = destino or tempfile.mkdtemp(prefix="nm_artefacto_")
    Path(destino).mkdir(parents=True, exist_ok=True)      # el repositorio exige que exista
    return get_artifact_repository(raiz).download_artifacts(ruta, dst_path=destino)
