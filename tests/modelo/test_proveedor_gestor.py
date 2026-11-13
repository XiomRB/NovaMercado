"""ProveedorMLflow de la etapa 6 (sin servidores): vigente desde PostgreSQL (tres
resultados), artefacto desde MLflow verificado contra el SHA de PostgreSQL, caché por
decisión y secuencia, y configuración del cliente."""

import os
import shutil
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE

import app.modelo.mlflow_util as mu
from app.config import ConfiguracionInvalida, configurar_cliente_mlflow, modelo_origen
from app.db.promociones import Vigente
from app.modelo.artefacto import ArtefactoInvalido, sha256_bytes
from app.modelo.cache import CacheModelos
from app.modelo.proveedor_mlflow import ConsultaIndeterminada, ProveedorMLflow
from app.modelo.proveedores import ModeloNoDisponible, OrigenNoDisponible, VersionNoUtilizable


class ClienteFalso:
    def __init__(self):
        self.alias = None            # versión o excepción
        self.version = None          # SimpleNamespace(run_id, tags) o excepción
        self.consultas = 0

    def get_model_version_by_alias(self, nombre, alias):
        if isinstance(self.alias, Exception):
            raise self.alias
        if self.alias is None:
            raise MlflowException("Registered model alias vigente not found.", error_code=INVALID_PARAMETER_VALUE)
        return SimpleNamespace(version=self.alias)

    def get_model_version(self, nombre, version):
        self.consultas += 1
        if isinstance(self.version, Exception):
            raise self.version
        return self.version


@pytest.fixture
def entorno(tmp_path, artefacto, monkeypatch):
    cache = CacheModelos(tmp_path / "cache")
    cliente = ClienteFalso()
    sha = sha256_bytes(artefacto.artefacto)
    origen = tmp_path / "origen" / "modelo.joblib"
    origen.parent.mkdir()
    origen.write_bytes(artefacto.artefacto)

    def descargar(uri, run_id, ruta, destino):          # simula la descarga desde MLflow
        shutil.copyfile(origen, f"{destino}/modelo.joblib")
        return f"{destino}/modelo.joblib"
    monkeypatch.setattr(mu, "descargar_de_ejecucion", descargar)
    lector = SimpleNamespace(valor=None)
    prov = ProveedorMLflow("http://mlflow-falso:5000", "novamercado-ventas", "vigente", cache, cliente=cliente,
                           lector_vigente=lambda: lector.valor() if callable(lector.valor) else lector.valor)
    return SimpleNamespace(prov=prov, cliente=cliente, cache=cache, sha=sha, origen=origen, lector=lector)


def vig(sha, decision=1, version="1", secuencia=1):
    return Vigente(decision, secuencia, version, f"run{version}", "modelo/artifacts/modelo.joblib", sha, 1)


def mv(sha, version="1"):
    return SimpleNamespace(run_id=f"run{version}", tags={"sha256_artefacto": sha})


# ------------------------------------------------------------- consulta del vigente
def test_tres_resultados_de_la_consulta(entorno):
    entorno.lector.valor = None
    assert entorno.prov.consultar_vigente() is None                       # sin_vigente
    entorno.lector.valor = vig(entorno.sha)
    assert entorno.prov.consultar_vigente().version == "1"               # vigente(V)

    def caida():
        raise RuntimeError('relation "resoluciones_promocion" does not exist')
    entorno.lector.valor = caida
    with pytest.raises(ConsultaIndeterminada):                           # nunca "sin modelo"
        entorno.prov.consultar_vigente()


def test_sin_motor_ni_lector_es_indeterminado(tmp_path):
    p = ProveedorMLflow("http://x:1", "m", "vigente", CacheModelos(tmp_path), cliente=ClienteFalso())
    with pytest.raises(ConsultaIndeterminada):
        p.consultar_vigente()


def test_alias_de_mlflow(entorno):
    assert entorno.prov.alias_mlflow() is None
    entorno.cliente.alias = "3"
    assert entorno.prov.alias_mlflow() == "3"
    for err in (MlflowException("caído", error_code=INTERNAL_ERROR), ConnectionError("rechazada")):
        entorno.cliente.alias = err
        with pytest.raises(OrigenNoDisponible):
            entorno.prov.alias_mlflow()


# ----------------------------------------------------------------- obtención
def test_descarga_verifica_y_se_guarda_tras_el_intercambio(entorno):
    entorno.cliente.version = mv(entorno.sha)
    o = entorno.prov.obtener(vig(entorno.sha))
    assert not o.desde_cache and o.modelo.version == "1" and entorno.cache.versiones() == []
    entorno.prov.guardar_en_cache(o)                                     # después del intercambio
    o.limpiar()
    assert entorno.cache.versiones() == ["1"]
    m = entorno.cache.validar("1").metadatos
    assert (m["decision_id"], m["secuencia"]) == (1, 1)


def test_cache_solo_para_la_misma_decision(entorno):
    entorno.cliente.version = mv(entorno.sha)
    o = entorno.prov.obtener(vig(entorno.sha))
    entorno.prov.guardar_en_cache(o)
    antes = entorno.cliente.consultas
    assert entorno.prov.obtener(vig(entorno.sha)).desde_cache                       # misma decisión
    assert entorno.cliente.consultas == antes
    assert not entorno.prov.obtener(vig(entorno.sha, decision=2, secuencia=2)).desde_cache   # otra decisión


def test_mlflow_caido_es_origen_no_disponible(entorno):
    entorno.cliente.version = ConnectionError("mlflow caído")
    with pytest.raises(OrigenNoDisponible):
        entorno.prov.obtener(vig(entorno.sha))


def test_tag_de_mlflow_distinto_del_sha_de_postgres(entorno):
    entorno.cliente.version = mv("0" * 64)
    with pytest.raises(VersionNoUtilizable, match="no coincide con la promoción confirmada"):
        entorno.prov.obtener(vig(entorno.sha))


def test_artefacto_descargado_distinto_del_sha_de_postgres(entorno):
    entorno.cliente.version = mv(entorno.sha)
    entorno.origen.write_bytes(entorno.origen.read_bytes() + b"x")
    with pytest.raises(ArtefactoInvalido, match="SHA-256"):
        entorno.prov.obtener(vig(entorno.sha))


def test_respaldo_sin_base_por_secuencia(entorno):
    with pytest.raises(ModeloNoDisponible, match="caché"):
        entorno.prov.respaldo_sin_base()
    entorno.cliente.version = mv(entorno.sha)
    entorno.prov.guardar_en_cache(entorno.prov.obtener(vig(entorno.sha)))
    modelo, meta = entorno.prov.respaldo_sin_base()
    assert modelo.version == "1" and meta["secuencia"] == 1


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


@pytest.mark.parametrize("dano", ["sin_decision", "decision_cero", "secuencia_texto", "sha_metadatos", "artefacto"])
def test_cache_no_confirmada_o_con_sha_incorrecto_nunca_se_sirve(entorno, dano):
    """``sin_verificar`` solo puede usar entradas confirmadas en PostgreSQL en su momento,
    con decision_id, secuencia y SHA verificados."""
    import json
    entorno.cliente.version = mv(entorno.sha)
    entorno.prov.guardar_en_cache(entorno.prov.obtener(vig(entorno.sha)))
    meta = entorno.cache.dir / "1" / "metadatos.json"
    m = json.loads(meta.read_text())
    if dano == "sin_decision":
        del m["decision_id"]
    elif dano == "decision_cero":
        m["decision_id"] = 0
    elif dano == "secuencia_texto":
        m["secuencia"] = "1"
    elif dano == "sha_metadatos":
        m["sha256_artefacto"] = "0" * 64
    meta.write_text(json.dumps(m))
    if dano == "artefacto":
        f = entorno.cache.dir / "1" / "modelo.joblib"
        f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(ModeloNoDisponible):
        entorno.prov.respaldo_sin_base()
    from app.modelo.gestor import GestorVigente
    entorno.lector.valor = lambda: (_ for _ in ()).throw(ConnectionError("PostgreSQL caído"))
    g = GestorVigente(entorno.prov)
    assert g.refrescar() is False and not g.cargado()            # nunca se sirve
