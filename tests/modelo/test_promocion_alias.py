"""Promoción y ciclo sin servidores: resolución de un fallo al escribir el alias, códigos de
salida de los comandos, criterio del respaldo baseline, carrera de PDEATHSIG del hijo y URL
de la base por partes. Las pruebas con servidores reales están en ``tests/integracion``."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

import app.modelo.ciclo_entrenamiento as ciclo
import app.modelo.entrenar_mlflow as cli
import app.modelo.procedencia as pr
from app.modelo.bloqueo import BloqueoOcupado, ConcesionPerdida
from app.modelo.promocion import PromocionIndeterminada, resolver_asignacion_alias
from app.modelo.registro import ResultadoRegistro, califica_para_respaldo

RAIZ = Path(__file__).resolve().parents[2]


class ClienteConsulta:
    def __init__(self, respuesta):
        self.respuesta = respuesta                  # versión observada, None o una excepción

    def get_model_version_by_alias(self, nombre, alias):
        if isinstance(self.respuesta, BaseException):
            raise self.respuesta
        if self.respuesta is None:
            raise MlflowException("Registered model alias vigente not found.", error_code=INVALID_PARAMETER_VALUE)
        return SimpleNamespace(version=self.respuesta)


@pytest.mark.parametrize("respuesta, esperado", [
    ("2", ("aplicada", "2")),                       # el servidor aplicó el cambio; se perdió la respuesta
    ("1", ("no_aplicada", "1")),                    # sigue en la versión anterior
    ("7", ("no_aplicada", "7")),                    # otra versión (otro proceso): no es nuestra promoción
    (None, ("no_aplicada", None)),                  # no había alias y sigue sin haberlo
    (MlflowException("no existe", error_code=RESOURCE_DOES_NOT_EXIST), ("no_aplicada", None)),
    (ConnectionError("servidor caído"), ("indeterminada", None)),
    (TimeoutError("sin respuesta"), ("indeterminada", None)),
    (MlflowException("error interno"), ("indeterminada", None)),
])
def test_resolver_asignacion_alias(respuesta, esperado):
    assert resolver_asignacion_alias(ClienteConsulta(respuesta), "m", "vigente", "2") == esperado


def test_borrado_del_alias_se_resuelve_igual():
    assert resolver_asignacion_alias(ClienteConsulta(None), "m", "vigente", None) == ("aplicada", None)


def test_promocion_indeterminada_informa_que_postgres_no_la_sirve():
    msg = str(PromocionIndeterminada("novamercado-ventas", "vigente", "2", TimeoutError()))
    assert "indeterminado" in msg and "no servirá esta versión" in msg and "no está garantizado" in msg


# --------------------------------------------------------------- códigos de salida
@pytest.fixture
def cli_falso(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                     # ejecutar() cambia al directorio raíz; se restaura
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-falso:5000")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u@127.0.0.1:1/x")
    monkeypatch.setattr(pr, "verificar_procedencia", lambda: pr.Procedencia("a" * 40, "git", tmp_path))

    def preparar(efecto):
        def falso(**k):
            if isinstance(efecto, BaseException):
                raise efecto
            return efecto
        monkeypatch.setattr(ciclo, "ejecutar_entrenamiento", falso)
    return preparar


@pytest.mark.parametrize("efecto, codigo", [
    (ciclo.ResultadoCiclo(0, "ok", "1", True), 0),
    (ciclo.ResultadoCiclo(3, "no cumple C1", None, False), 3),
    (PromocionIndeterminada("m", "vigente", "2"), 4),
    (BloqueoOcupado("otro"), 5),
    (ConcesionPerdida("perdida"), 6),
    (RuntimeError("MLflow caído"), 1),
])
def test_codigos_de_salida(cli_falso, efecto, codigo, capsys):
    cli_falso(efecto)
    assert cli.main(["--promover", "siempre"]) == codigo


def test_procedencia_invalida_sale_con_2_sin_entrenar(monkeypatch, tmp_path, capsys):
    llamado = []
    monkeypatch.setattr(ciclo, "ejecutar_entrenamiento", lambda **k: llamado.append(1))

    def invalida():
        raise pr.ProcedenciaInvalida("No se puede entrenar: cambios sin confirmar")
    monkeypatch.setattr(pr, "verificar_procedencia", invalida)
    assert cli.main([]) == 2 and llamado == []
    assert "cambios sin confirmar" in capsys.readouterr().err


# --------------------------------------------------------- criterio del respaldo
def resultado(**cond):
    base = {k: True for k in ("T1", "T2", "T3", "D1", "T4a", "T4b", "C1", "C2", "C3")}
    return ResultadoRegistro("run", "2", False, "m", {**base, **cond})


@pytest.mark.parametrize("cond, califica", [
    ({"C1": False}, True), ({"C2": False, "C3": False}, True), ({"C1": False, "C2": False, "C3": False}, True),
    ({}, False),                                                    # válido: no hace falta respaldo
    ({"C1": False, "T3": False}, False), ({"C2": False, "T4a": False}, False), ({"C3": False, "T4b": False}, False),
    ({"C1": False, "T2": False}, False), ({"C1": False, "D1": False}, False), ({"C1": False, "T1": False}, False),
    ({"T3": False}, False), ({"D1": False}, False),
])
def test_califica_para_respaldo_solo_si_fallan_exclusivamente_c1_c3(cond, califica):
    assert califica_para_respaldo(resultado(**cond)) is califica


def test_sin_version_no_califica():
    r = resultado(C1=False)
    r.version = None
    assert not califica_para_respaldo(r)


# ------------------------------------------------------------- PDEATHSIG del hijo
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG es de Linux")
def test_hijo_sale_si_el_supervisor_ya_no_es_su_padre():
    """Carrera: el padre murió antes del prctl. El hijo lo detecta comparando getppid()
    después de configurar PDEATHSIG y sale con 6 sin tocar la base ni MLflow."""
    env = {**os.environ, "NM_PID_SUPERVISOR": "1", "PYTHONPATH": str(RAIZ),
           "NM_URL_BD_ENTRENAMIENTO": "postgresql+psycopg2://u@127.0.0.1:1/x"}
    p = subprocess.run([sys.executable, "-m", "app.modelo.trabajo_entrenamiento", "/no/existe", "/no/existe"],
                       env=env, cwd=RAIZ, capture_output=True, text=True, timeout=60)
    assert p.returncode == 6 and "el supervisor ya no existe" in p.stderr


def test_hijo_sin_pid_del_supervisor_sale():
    env = {k: v for k, v in os.environ.items() if k != "NM_PID_SUPERVISOR"}
    p = subprocess.run([sys.executable, "-m", "app.modelo.trabajo_entrenamiento", "x", "y"],
                       env={**env, "PYTHONPATH": str(RAIZ)}, cwd=RAIZ, capture_output=True, text=True, timeout=60)
    assert p.returncode == 6


# ------------------------------------------------------------- URL por partes
def test_url_por_partes_escapa_la_clave(monkeypatch):
    from sqlalchemy.engine import make_url

    from app.config import url_base_datos
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for k, v in {"POSTGRES_USER": "nova", "POSTGRES_PASSWORD": "c/l@v:e%#?", "POSTGRES_DB": "novamercado",
                 "DB_HOST": "db", "DB_PORT": "5432"}.items():
        monkeypatch.setenv(k, v)
    u = make_url(url_base_datos())
    assert (u.username, u.password, u.host, u.port, u.database) == ("nova", "c/l@v:e%#?", "db", 5432, "novamercado")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://otro@h:1/x")      # DATABASE_URL tiene prioridad
    assert make_url(url_base_datos()).host == "h"


def test_url_por_partes_incompleta(monkeypatch):
    from app.config import ConfiguracionInvalida, url_base_datos
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for k in ("POSTGRES_PASSWORD", "POSTGRES_DB", "DB_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("POSTGRES_USER", "nova")
    with pytest.raises(ConfiguracionInvalida, match="faltan"):
        url_base_datos()


@pytest.mark.parametrize("resultado, codigo", [("indeterminada", 4), ("no_aplicada", 1)])
def test_reconciliacion_incompleta_no_entrena_y_distingue_el_resultado(cli_falso, resultado, codigo, capsys):
    from app.modelo.promocion import ReconciliacionIncompleta
    cli_falso(ReconciliacionIncompleta("m", "vigente", "1", resultado, "2"))
    assert cli.main([]) == codigo
    err = capsys.readouterr().err
    assert "Reconciliación incompleta" in err and "No se entrena" in err
    assert ("no está garantizado" in err) == (resultado == "indeterminada")
