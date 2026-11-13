"""Proveedor del modelo vigente desde el MLflow Model Registry (DECISIONES.md §14).

* ``huella()``: versión a la que apunta el alias (``mlflow:<versión>:<sha>``).
  - MLflow no responde → :class:`OrigenNoDisponible` (el gestor conserva el
    modelo cargado o, si no hay ninguno, recurre a ``respaldo()``).
  - El alias o el modelo no existen → :class:`VersionNoUtilizable`.
* ``obtener()``: descarga **solo** ``modelo.joblib`` desde
  ``runs:/<run_id>/<ruta_artefacto>`` (o lo toma de la caché si su SHA
  recalculado coincide con el tag), verifica el SHA contra el tag del Registry
  **antes** de deserializar, valida formato, bibliotecas y humo, y lo guarda en
  la caché.
* ``respaldo()``: primera versión válida de la caché (sin MLflow).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass

import requests

from app.config import configurar_cliente_mlflow
from app.modelo.cache import CacheModelos
from app.modelo.proveedores import ModeloCargado, OrigenNoDisponible, VersionNoUtilizable

logger = logging.getLogger(__name__)

CODIGOS_SIN_VERSION = {"INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"}
ESTADO_VALIDO = "valido"


@dataclass(frozen=True)
class VersionRegistrada:
    version: str
    run_id: str
    sha256: str | None
    ruta: str | None
    estado: str | None


class ProveedorMLflow:
    def __init__(self, tracking_uri: str, nombre: str, alias: str, cache: CacheModelos, cliente=None):
        configurar_cliente_mlflow()
        self.tracking_uri = tracking_uri.rstrip("/")
        self.nombre = nombre
        self.alias = alias
        self.cache = cache
        if cliente is None:
            from mlflow import MlflowClient
            cliente = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        self.cliente = cliente
        self._ultima: VersionRegistrada | None = None

    # ---------------------------------------------------------------- consulta
    def _version_del_alias(self) -> VersionRegistrada:
        from mlflow.exceptions import MlflowException
        try:
            mv = self.cliente.get_model_version_by_alias(self.nombre, self.alias)
        except MlflowException as exc:
            if getattr(exc, "error_code", None) in CODIGOS_SIN_VERSION:
                raise VersionNoUtilizable(f"el alias '{self.alias}' no apunta a ninguna versión "
                                          f"del modelo '{self.nombre}'") from exc
            raise OrigenNoDisponible("MLflow no disponible") from exc
        except Exception as exc:  # noqa: BLE001 - red, DNS, timeouts
            raise OrigenNoDisponible("MLflow no disponible") from exc
        t = dict(mv.tags or {})
        return VersionRegistrada(str(mv.version), mv.run_id, t.get("sha256_artefacto"),
                                 t.get("ruta_artefacto"), t.get("estado"))

    def huella(self) -> str | None:
        v = self._version_del_alias()
        self._ultima = v
        return f"mlflow:{v.version}:{v.sha256}"

    def disponible(self) -> bool:
        """Para el monitor de salud: ``GET /health`` del servidor de MLflow."""
        try:
            r = requests.get(f"{self.tracking_uri}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------ carga
    def obtener(self) -> ModeloCargado:
        v = self._ultima or self._version_del_alias()
        if v.estado != ESTADO_VALIDO:
            raise VersionNoUtilizable(f"la versión {v.version} no está marcada como válida")
        if not v.sha256 or not v.ruta:
            raise VersionNoUtilizable(f"la versión {v.version} no tiene sha256_artefacto o ruta_artefacto")
        metadatos = {"nombre": self.nombre, "alias": self.alias, "version": v.version, "run_id": v.run_id,
                     "ruta_artefacto": v.ruta, "sha256_artefacto": v.sha256}
        en_cache = self.cache.archivo_si_coincide(v.version, v.sha256)
        if en_cache is not None:
            entrada = self.cache.validar(v.version)      # SHA recalculado = tag del Registry
            if entrada.metadatos["sha256_artefacto"] != v.sha256:
                raise VersionNoUtilizable("los metadatos de la caché no coinciden con el Registry")
            artefacto = entrada.artefacto
        else:
            artefacto = self._descargar_y_guardar(v, metadatos)
        return ModeloCargado(artefacto["pronosticador"], self.nombre, self.alias, v.version)

    def _descargar_y_guardar(self, v: VersionRegistrada, metadatos: dict) -> dict:
        import mlflow.artifacts

        from app.modelo.artefacto import validar_archivo
        with tempfile.TemporaryDirectory(prefix="nm_descarga_") as tmp:
            try:
                ruta = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{v.run_id}/{v.ruta}", dst_path=tmp, tracking_uri=self.tracking_uri)
            except Exception as exc:  # noqa: BLE001
                raise OrigenNoDisponible("no se pudo descargar el artefacto desde MLflow") from exc
            artefacto = validar_archivo(ruta, v.sha256)  # SHA contra el tag ANTES de deserializar
            self.cache.guardar(ruta, metadatos)
        return artefacto

    def respaldo(self) -> tuple[ModeloCargado, str]:
        """Modelo y huella desde la caché, para operar sin MLflow."""
        e = self.cache.primera_valida()
        m = e.metadatos
        modelo = ModeloCargado(e.artefacto["pronosticador"], m["nombre"], m["alias"], str(m["version"]))
        return modelo, f"cache:{m['version']}:{m['sha256_artefacto']}"
