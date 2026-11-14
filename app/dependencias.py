"""Dependencias de la API y su construcción desde variables de entorno.

Las pruebas construyen :class:`Dependencias` con implementaciones de prueba
(persistencia en memoria, modelo sintético) y no inician hilos.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    componentes_requeridos,
    mlflow_alias,
    mlflow_modelo_nombre,
    mlflow_tracking_uri,
    modelo_origen,
    ruta_cache_modelos,
    ruta_modelo,
    segundos,
)
from app.modelo.gestor import GestorModelo
from app.modelo.proveedores import ProveedorArchivo
from app.orquestador import Orquestador, OrquestadorNoConfigurado
from app.salud import MonitorSalud
from app.servicios.persistencia import Persistencia


@dataclass
class Dependencias:
    persistencia: Persistencia
    gestor: GestorModelo
    monitor: MonitorSalud
    orquestador: Orquestador
    motor: object | None = None      # Engine en producción, para cerrarlo al apagar

    def iniciar(self) -> None:
        self.gestor.iniciar()
        self.monitor.iniciar()
        getattr(self.orquestador, "iniciar_hilos", lambda: None)()     # recuperación (etapa 7)

    def detener(self) -> None:
        getattr(self.orquestador, "detener_hilos", lambda: None)()
        self.monitor.detener()
        self.gestor.detener()
        if self.motor is not None:
            self.motor.dispose()


def construir_monitor(persistencia: Persistencia, gestor: GestorModelo, orquestador: Orquestador,
                      requeridos=None, intervalo: float | None = None, limite: float | None = None) -> MonitorSalud:
    verificaciones = {"base_datos": persistencia.disponible}
    # MLflow: si el proveedor del modelo sabe verificarlo (ProveedorMLflow.disponible).
    disponible_mlflow = getattr(getattr(gestor, "proveedor", None), "disponible", None)
    if callable(disponible_mlflow):
        verificaciones["mlflow"] = disponible_mlflow
    # Etapa 7: "airflow" (orquestador.disponible).
    if not isinstance(orquestador, OrquestadorNoConfigurado):
        verificaciones["airflow"] = orquestador.disponible
    return MonitorSalud(
        verificaciones=verificaciones,
        requeridos=requeridos if requeridos is not None else componentes_requeridos(),
        modelo_cargado=gestor.cargado,
        modelo_actualizado=getattr(gestor, "actualizado", lambda: True),
        motivos_modelo=getattr(gestor, "motivos_degradacion", lambda: []),
        vigencia_modelo=getattr(gestor, "vigencia", lambda: None),
        intervalo=intervalo if intervalo is not None else segundos("SALUD_INTERVALO", 10),
        limite=limite if limite is not None else segundos("SALUD_LIMITE", 3),
    )


def construir_proveedor(motor=None):
    """``MODELO_ORIGEN=mlflow`` (producto): vigente según PostgreSQL (``motor``),
    artefacto desde MLflow y caché local. ``MODELO_ORIGEN=archivo``: artefacto local de
    la etapa 4 (desarrollo y pruebas)."""
    if modelo_origen() == "archivo":
        return ProveedorArchivo(ruta_modelo())
    from app.modelo.cache import CacheModelos
    from app.modelo.proveedor_mlflow import ProveedorMLflow
    return ProveedorMLflow(mlflow_tracking_uri(), mlflow_modelo_nombre(), mlflow_alias(),
                           CacheModelos(ruta_cache_modelos()), motor=motor)


def construir_gestor(proveedor, intervalo: float):
    from app.modelo.gestor import GestorVigente
    from app.modelo.proveedor_mlflow import ProveedorMLflow
    if isinstance(proveedor, ProveedorMLflow):
        return GestorVigente(proveedor, intervalo=intervalo)
    return GestorModelo(proveedor, intervalo=intervalo)


def construir_orquestador(motor):
    """``ORQUESTADOR=airflow`` (producto, etapa 7) u orquestador no configurado."""
    from app.config import config_airflow, orquestador_configurado
    if orquestador_configurado() != "airflow":
        return OrquestadorNoConfigurado()
    from app.config import parametros_rol
    from app.orquestador_airflow import OrquestadorAirflow
    return OrquestadorAirflow(motor, config_airflow(), nombre_modelo=mlflow_modelo_nombre(),
                              parametros_cierre=lambda: parametros_rol("nm_cierre", "NM_CIERRE_PASSWORD"))


def desde_entorno() -> Dependencias:
    from app.db.conexion import crear_motor
    from app.servicios.persistencia import PersistenciaPostgres

    motor = crear_motor()
    persistencia = PersistenciaPostgres(motor)
    gestor = construir_gestor(construir_proveedor(motor), segundos("MODELO_INTERVALO_REFRESCO", 15))
    orquestador = construir_orquestador(motor)
    monitor = construir_monitor(persistencia, gestor, orquestador)
    return Dependencias(persistencia, gestor, monitor, orquestador, motor)
