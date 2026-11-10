"""Orquestación del reentrenamiento (contrato ``/admin/reentrenar``).

Etapa 4: no hay orquestador (Airflow llega en la etapa 7). ``iniciar`` no puede
lanzar nada (la API responde 503) y ningún identificador existe (404).
"""

from __future__ import annotations

from typing import Protocol

from app.schemas.contrato import EstadoReentrenamiento


class OrquestadorNoDisponible(RuntimeError):
    """No se puede iniciar un reentrenamiento. La API responde 503."""


class Orquestador(Protocol):
    def iniciar(self) -> str:
        """Lanza una ejecución y devuelve su identificador."""

    def estado(self, id_ejecucion: str) -> EstadoReentrenamiento | None:
        """Estado de la ejecución, o ``None`` si el identificador no existe."""

    def disponible(self) -> bool: ...


class OrquestadorNoConfigurado:
    motivo = "el reentrenamiento no está disponible: el orquestador (Airflow) no está configurado"

    def iniciar(self) -> str:
        raise OrquestadorNoDisponible(self.motivo)

    def estado(self, id_ejecucion: str) -> EstadoReentrenamiento | None:
        return None

    def disponible(self) -> bool:
        return False
