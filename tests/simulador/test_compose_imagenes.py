"""I1 e I2: ``docker compose config`` con el Compose real. Necesitan el binario de Docker
Compose (no un servidor): corren en el job "Persistencia con Docker Compose" de la CI
(``-m compose``) y se excluyen del job sin Docker."""

from __future__ import annotations

import json
import subprocess

import pytest

from simulador import entorno as ent
from simulador import imagenes

pytestmark = pytest.mark.compose


def _config(env_file) -> dict:
    r = subprocess.run(["docker", "compose", "-f", str(ent.RAIZ / "docker-compose.yml"), "--project-directory",
                        str(ent.RAIZ), "--env-file", str(env_file), "config", "--format", "json"],
                       capture_output=True, text=True, env=ent.entorno_minimo(), check=True)
    return json.loads(r.stdout)["services"]


def test_i1_compose_normal_conserva_las_etiquetas():
    servicios = _config(ent.RAIZ / ".env.example")
    for tipo, nombres in imagenes.SERVICIOS.items():
        for s in nombres:
            assert servicios[s]["image"] == imagenes.PROTEGIDAS[tipo], s
    assert servicios["db"]["image"] == "postgres:16.14-alpine"
    assert servicios["mlflow"]["image"] == "ghcr.io/mlflow/mlflow:v2.22.1"


def test_i2_simulador_usa_sus_etiquetas(tmp_path):
    commit = "f" * 40
    ent.escribir_entorno(tmp_path / "e.env", tmp_path, dict(ent.PUERTOS_POR_DEFECTO), imagenes.etiquetas(commit))
    servicios = _config(tmp_path / "e.env")
    normal = _config(ent.RAIZ / ".env.example")
    propias = {s for ss in imagenes.SERVICIOS.values() for s in ss}
    for tipo, nombres in imagenes.SERVICIOS.items():
        for s in nombres:
            assert servicios[s]["image"] == imagenes.etiquetas(commit)[tipo], s
    assert not any(v.get("image", "").endswith(":local") for v in servicios.values())
    assert {s: v.get("image") for s, v in servicios.items() if s not in propias} == \
        {s: v.get("image") for s, v in normal.items() if s not in propias}
