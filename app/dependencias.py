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

    def detener(self) -> None:
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
        intervalo=intervalo if intervalo is not None else segundos("SALUD_INTERVALO", 10),
        limite=limite if limite is not None else segundos("SALUD_LIMITE", 3),
    )


def construir_proveedor():
    """``MODELO_ORIGEN=mlflow`` (producto): Model Registry con caché local.
    ``MODELO_ORIGEN=archivo``: artefacto local de la etapa 4 (desarrollo y pruebas)."""
    if modelo_origen() == "archivo":
        return ProveedorArchivo(ruta_modelo())
    from app.modelo.cache import CacheModelos
    from app.modelo.proveedor_mlflow import ProveedorMLflow
    return ProveedorMLflow(mlflow_tracking_uri(), mlflow_modelo_nombre(), mlflow_alias(),
                           CacheModelos(ruta_cache_modelos()))


def desde_entorno() -> Dependencias:
    from app.db.conexion import crear_motor
    from app.servicios.persistencia import PersistenciaPostgres

    motor = crear_motor()
    persistencia = PersistenciaPostgres(motor)
    gestor = GestorModelo(construir_proveedor(), intervalo=segundos("MODELO_INTERVALO_REFRESCO", 15))
    orquestador = OrquestadorNoConfigurado()
    monitor = construir_monitor(persistencia, gestor, orquestador)
    return Dependencias(persistencia, gestor, monitor, orquestador, motor)
