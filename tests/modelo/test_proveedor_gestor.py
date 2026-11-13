"""ProveedorMLflow (cliente simulado) y GestorModelo: errores, respaldo y actualización fallida."""

import os
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE

from app.config import ConfiguracionInvalida, configurar_cliente_mlflow, modelo_origen
from app.modelo.artefacto import sha256_bytes
from app.modelo.cache import CacheModelos
from app.modelo.gestor import GestorModelo
from app.modelo.proveedor_mlflow import ProveedorMLflow
from app.modelo.proveedores import OrigenNoDisponible, VersionNoUtilizable


class ClienteFalso:
    def __init__(self):
        self.version = None          # SimpleNamespace(version, run_id, tags) o una excepción
        self.llamadas = 0

    def get_model_version_by_alias(self, nombre, alias):
        self.llamadas += 1
        if isinstance(self.version, Exception):
            raise self.version
        if self.version is None:
            raise MlflowException("Registered model alias vigente not found.", error_code=INVALID_PARAMETER_VALUE)
        return self.version


def version(v, sha, estado="valido"):
    return SimpleNamespace(version=str(v), run_id=f"run{v}",
                           tags={"sha256_artefacto": sha, "ruta_artefacto": "modelo/artifacts/modelo.joblib",
                                 "estado": estado})


@pytest.fixture
def entorno(tmp_path, artefacto):
    cache = CacheModelos(tmp_path / "cache")
    cliente = ClienteFalso()
    prov = ProveedorMLflow("http://mlflow-falso:5000", "novamercado-ventas", "vigente", cache, cliente=cliente)
    sha = sha256_bytes(artefacto.artefacto)
    ruta = tmp_path / "modelo.joblib"
    ruta.write_bytes(artefacto.artefacto)

    def descargar(v, metadatos):            # simula la descarga de MLflow con validación real
        from app.modelo.artefacto import validar_archivo
        a = validar_archivo(ruta, v.sha256)
        cache.guardar(ruta, metadatos)
        return a
    prov._descargar_y_guardar = descargar
    return SimpleNamespace(prov=prov, cliente=cliente, cache=cache, sha=sha, ruta=ruta)


def test_alias_inexistente_es_version_no_utilizable(entorno):
    with pytest.raises(VersionNoUtilizable, match="no apunta a ninguna versión"):
        entorno.prov.huella()


@pytest.mark.parametrize("error", [MlflowException("API request failed", error_code=INTERNAL_ERROR),
                                   ConnectionError("rechazada")])
def test_mlflow_caido_es_origen_no_disponible(entorno, error):
    entorno.cliente.version = error
    with pytest.raises(OrigenNoDisponible):
        entorno.prov.huella()


def test_carga_y_guarda_en_cache(entorno):
    entorno.cliente.version = version(1, entorno.sha)
    assert entorno.prov.huella() == f"mlflow:1:{entorno.sha}"
    m = entorno.prov.obtener()
    assert (m.nombre, m.alias, m.version) == ("novamercado-ventas", "vigente", "1")
    assert entorno.cache.versiones() == ["1"]


@pytest.mark.parametrize("estado, sha", [("invalido", None), ("pendiente", None), ("valido", "")])
def test_version_no_valida_o_sin_tags(entorno, estado, sha):
    entorno.cliente.version = version(1, entorno.sha if sha is None else sha, estado)
    entorno.prov.huella()
    with pytest.raises(VersionNoUtilizable):
        entorno.prov.obtener()


def test_sha_del_registry_distinto_al_artefacto(entorno):
    entorno.cliente.version = version(1, "0" * 64)
    entorno.prov.huella()
    with pytest.raises(VersionNoUtilizable):
        entorno.prov.obtener()


# ----------------------------------------------------------------------- gestor
def test_sin_mlflow_y_sin_modelo_usa_la_cache(entorno):
    entorno.cliente.version = version(1, entorno.sha)
    g = GestorModelo(entorno.prov)
    assert g.refrescar() and g.actual().version == "1"
    g2 = GestorModelo(entorno.prov)                         # "reinicio de la API"
    entorno.cliente.version = ConnectionError("mlflow caído")
    assert g2.refrescar() and g2.actual().version == "1"    # desde la caché validada


def test_sin_mlflow_con_modelo_no_toca_la_cache(entorno, monkeypatch):
    entorno.cliente.version = version(1, entorno.sha)
    g = GestorModelo(entorno.prov)
    g.refrescar()
    entorno.cliente.version = ConnectionError("mlflow caído")
    monkeypatch.setattr(entorno.prov, "respaldo", lambda: pytest.fail("no debe leer la caché"))
    assert g.refrescar() is False and g.actual().version == "1" and g.actualizado()


def test_sin_mlflow_y_cache_corrupta_no_hay_modelo(entorno):
    entorno.cliente.version = version(1, entorno.sha)
    GestorModelo(entorno.prov).refrescar()
    (entorno.cache.dir / "1" / "modelo.joblib").write_bytes(b"corrupto")
    entorno.cliente.version = ConnectionError("mlflow caído")
    g = GestorModelo(entorno.prov)
    assert g.refrescar() is False and not g.cargado()


def test_alias_invalido_conserva_el_modelo_y_marca_desactualizado(entorno):
    entorno.cliente.version = version(1, entorno.sha)
    g = GestorModelo(entorno.prov)
    g.refrescar()
    entorno.cliente.version = version(2, entorno.sha, estado="invalido")
    assert g.refrescar() is False and g.actual().version == "1"
    assert not g.actualizado() and "no está marcada como válida" in g.actualizacion_fallida
    entorno.cliente.version = None                               # alias borrado
    assert g.refrescar() is False and not g.actualizado()
    entorno.cliente.version = version(1, entorno.sha)            # vuelve a una versión válida
    g.refrescar()
    assert g.actualizado() and g.actual().version == "1"


def test_mlflow_caido_no_marca_actualizacion_fallida(entorno):
    entorno.cliente.version = version(1, entorno.sha)
    g = GestorModelo(entorno.prov)
    g.refrescar()
    entorno.cliente.version = MlflowException("API request failed", error_code=INTERNAL_ERROR)
    g.refrescar()
    assert g.actualizado()            # la caída de MLflow se informa como componente, no como actualización


# ---------------------------------------------------------------- configuración
def test_cliente_mlflow_con_limites(monkeypatch):
    for v in ("MLFLOW_HTTP_REQUEST_TIMEOUT", "MLFLOW_HTTP_REQUEST_MAX_RETRIES"):
        monkeypatch.delenv(v, raising=False)
    configurar_cliente_mlflow()
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "5"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "9")
    configurar_cliente_mlflow()
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "9"     # respeta el entorno


def test_modelo_origen(monkeypatch):
    monkeypatch.delenv("MODELO_ORIGEN", raising=False)
    assert modelo_origen() == "mlflow"
    monkeypatch.setenv("MODELO_ORIGEN", "archivo")
    assert modelo_origen() == "archivo"
    monkeypatch.setenv("MODELO_ORIGEN", "s3")
    with pytest.raises(ConfiguracionInvalida):
        modelo_origen()
