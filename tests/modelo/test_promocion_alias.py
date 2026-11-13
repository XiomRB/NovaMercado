"""Resolución de un fallo de ``set_registered_model_alias`` (sin servidores).

El servidor pudo aplicar el cambio aunque el cliente reciba un error, así que tras
el fallo se consulta el alias: aplicada / no aplicada / indeterminada. Las pruebas
con un servidor MLflow real están en ``tests/integracion/test_mlflow_registro.py``.
"""

from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

import app.modelo.entrenar_mlflow as cli
import app.modelo.registro as reg
from app.modelo.git_estado import EstadoGit
from app.modelo.registro import PromocionIndeterminada, resolver_asignacion_alias


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


def test_promocion_indeterminada_no_garantiza_el_alias_anterior():
    exc = PromocionIndeterminada("novamercado-ventas", "vigente", "2", "1", TimeoutError())
    msg = str(exc)
    assert "indeterminado" in msg and "versión 2 (válida)" in msg and "versión 1" in msg
    assert "no está garantizado" in msg


def test_cli_termina_con_codigo_4_si_la_promocion_es_indeterminada(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)                     # main() cambia al directorio raíz; se restaura
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-falso:5000")
    monkeypatch.setattr(cli, "verificar_arbol_limpio", lambda: EstadoGit(tmp_path, "abc123"))
    import app.db.conexion as conexion
    monkeypatch.setattr(conexion, "crear_motor", lambda **k: SimpleNamespace(dispose=lambda: None))

    def indeterminada(*a, **k):
        raise PromocionIndeterminada("novamercado-ventas", "vigente", "2", "1", TimeoutError())
    monkeypatch.setattr(reg, "entrenar_y_registrar", indeterminada)
    assert cli.main(["--promover", "siempre"]) == 4
    assert "indeterminado" in capsys.readouterr().err
