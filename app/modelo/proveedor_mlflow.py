"""Proveedor del modelo vigente (etapas 5 y 6, DECISIONES.md §14–§15).

Desde la etapa 6 **PostgreSQL es la fuente de verdad del vigente** (E6-9): la decisión
con resolución ``confirmada`` y mayor ``secuencia``. MLflow aporta el artefacto (y el
alias, que solo se compara para informar una desincronización).

* :meth:`ProveedorMLflow.consultar_vigente` → ``Vigente`` | ``None``
  (``sin_vigente``: consulta exitosa con cero filas) | :class:`ConsultaIndeterminada`
  (cualquier error: conexión, timeout, tabla inexistente, permiso…; nunca equivale a
  "sin modelo").
* :meth:`ProveedorMLflow.obtener` descarga **solo** ``modelo.joblib`` de
  ``runs:/<run_id>/<ruta>`` guardados en PostgreSQL (o lo toma de la caché si la
  entrada es de esa misma decisión y SHA), verifica el SHA de PostgreSQL **antes** de
  deserializar y aplica T3 (formato, bibliotecas, humo). Si el tag de MLflow
  ``sha256_artefacto`` difiere del SHA de PostgreSQL, se rechaza.
* La activación (segunda lectura, intercambio, estados de vigencia) la hace
  ``GestorVigente`` (``app.modelo.gestor``).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from app.config import configurar_cliente_mlflow
from app.db.promociones import Vigente, leer_vigente
from app.modelo.cache import CacheModelos
from app.modelo.proveedores import ModeloCargado, OrigenNoDisponible, VersionNoUtilizable

logger = logging.getLogger(__name__)


class ConsultaIndeterminada(OrigenNoDisponible):
    """No se pudo consultar el vigente en PostgreSQL (no significa "sin modelo")."""


@dataclass
class Obtenido:
    """Artefacto validado listo para activar; ``ruta`` se guarda en la caché tras el
    intercambio y luego se limpia."""
    modelo: ModeloCargado
    vigente: Vigente
    ruta: Path | None
    desde_cache: bool
    _tmp: str | None = None

    def limpiar(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


class ProveedorMLflow:
    def __init__(self, tracking_uri: str, nombre: str, alias: str, cache: CacheModelos, motor=None,
                 cliente=None, lector_vigente=None):
        configurar_cliente_mlflow()
        self.tracking_uri = tracking_uri.rstrip("/")
        self.nombre = nombre
        self.alias = alias
        self.cache = cache
        self.motor = motor
        if cliente is None:
            from mlflow import MlflowClient
            cliente = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        self.cliente = cliente
        self._lector = lector_vigente

    # ---------------------------------------------------------------- PostgreSQL
    def consultar_vigente(self) -> Vigente | None:
        try:
            if self._lector is not None:
                return self._lector()
            with self.motor.connect() as conn:
                return leer_vigente(conn, self.nombre)
        except Exception as exc:  # noqa: BLE001 - cualquier error = indeterminado
            raise ConsultaIndeterminada(f"no se pudo consultar el vigente en PostgreSQL ({type(exc).__name__})") \
                from exc

    # -------------------------------------------------------------------- MLflow
    def alias_mlflow(self) -> str | None:
        from mlflow.exceptions import MlflowException
        try:
            return str(self.cliente.get_model_version_by_alias(self.nombre, self.alias).version)
        except MlflowException as exc:
            if getattr(exc, "error_code", None) in ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"):
                return None
            raise OrigenNoDisponible("MLflow no disponible") from exc
        except Exception as exc:  # noqa: BLE001
            raise OrigenNoDisponible("MLflow no disponible") from exc

    def disponible(self) -> bool:
        """Para el monitor de salud: ``GET /health`` del servidor de MLflow."""
        try:
            r = requests.get(f"{self.tracking_uri}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _modelo(self, artefacto: dict, v: Vigente) -> ModeloCargado:
        return ModeloCargado(artefacto["pronosticador"], self.nombre, self.alias, v.version)

    def obtener(self, v: Vigente) -> Obtenido:
        """Artefacto de la decisión ``v``, validado (SHA de PostgreSQL antes de
        deserializar + T3). :class:`OrigenNoDisponible` si no se puede obtener;
        :class:`VersionNoUtilizable` si no supera la validación."""
        entrada = self.cache.entrada_para(v.decision_id, v.version, v.sha256_artefacto)
        if entrada is not None:
            return Obtenido(self._modelo(entrada.artefacto, v), v, None, True)
        from app.modelo.artefacto import validar_archivo
        from app.modelo.mlflow_util import descargar_de_ejecucion
        try:
            mv = self.cliente.get_model_version(self.nombre, v.version)
        except Exception as exc:  # noqa: BLE001
            raise OrigenNoDisponible("MLflow no disponible") from exc
        tags = dict(mv.tags or {})
        if mv.run_id != v.run_id or tags.get("sha256_artefacto") != v.sha256_artefacto:
            raise VersionNoUtilizable(f"la versión {v.version} de MLflow no coincide con la promoción confirmada "
                                      "en PostgreSQL (run_id o SHA-256)")
        tmp = tempfile.mkdtemp(prefix="nm_descarga_")
        try:
            try:
                ruta = descargar_de_ejecucion(self.tracking_uri, v.run_id, v.ruta_artefacto, tmp)
            except Exception as exc:  # noqa: BLE001
                raise OrigenNoDisponible("no se pudo descargar el artefacto desde MLflow") from exc
            artefacto = validar_archivo(ruta, v.sha256_artefacto)   # SHA de PostgreSQL ANTES de deserializar + T3
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return Obtenido(self._modelo(artefacto, v), v, Path(ruta), False, tmp)

    def guardar_en_cache(self, o: Obtenido) -> None:
        """Después del intercambio: solo versiones confirmadas."""
        if o.desde_cache or o.ruta is None:
            return
        v = o.vigente
        try:
            self.cache.guardar(o.ruta, {"nombre": self.nombre, "alias": self.alias, "version": v.version,
                                        "run_id": v.run_id, "ruta_artefacto": v.ruta_artefacto,
                                        "sha256_artefacto": v.sha256_artefacto, "decision_id": v.decision_id,
                                        "secuencia": v.secuencia})
        except Exception as exc:  # noqa: BLE001 - la caché es un respaldo
            logger.warning("no se pudo guardar la versión %s en la caché: %s", v.version, exc)

    def respaldo_sin_base(self) -> tuple[ModeloCargado, dict]:
        """Solo con PostgreSQL inaccesible y sin modelo cargado: entrada de mayor secuencia."""
        e = self.cache.mejor_por_secuencia()
        m = e.metadatos
        return ModeloCargado(e.artefacto["pronosticador"], m["nombre"], m["alias"], str(m["version"])), m
