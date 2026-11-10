"""Registro de lo que la capa de calidad hizo con una solicitud.

Cada problema detectado termina en una de tres acciones:

* ``reparar``  – la corrección es inequívoca; se informa en ``calidad``.
* ``advertir`` – se responde, pero usando un supuesto que se informa.
* rechazo      – se lanza :class:`SolicitudRechazada` (la API responde 422).
"""

from __future__ import annotations

from dataclasses import dataclass, field

REPARAR = "reparar"
ADVERTIR = "advertir"

ESTADO_OK = "ok"
ESTADO_CORREGIDO = "corregido"
ESTADO_ADVERTENCIA = "advertencia"


class SolicitudRechazada(Exception):
    """El dato no permite una respuesta confiable. Se traduce a HTTP 422."""

    def __init__(self, campo: str | None, motivo: str):
        self.campo = campo
        self.motivo = motivo
        super().__init__(self.detail)

    @property
    def detail(self) -> str:
        return f"{self.campo}: {self.motivo}" if self.campo else self.motivo


@dataclass(frozen=True)
class Observacion:
    campo: str
    accion: str
    detalle: str

    def como_dict(self) -> dict[str, str]:
        return {"campo": self.campo, "accion": self.accion, "detalle": self.detalle}


@dataclass
class RegistroCalidad:
    observaciones: list[Observacion] = field(default_factory=list)

    def reparar(self, campo: str, detalle: str) -> None:
        self.observaciones.append(Observacion(campo, REPARAR, detalle))

    def advertir(self, campo: str, detalle: str) -> None:
        self.observaciones.append(Observacion(campo, ADVERTIR, detalle))

    def registrar(self, campo: str, accion: str | None, detalle: str | None) -> None:
        """Registra el resultado de un intérprete (``accion`` None = sin cambios)."""
        if accion is None:
            return
        if accion not in (REPARAR, ADVERTIR):
            raise ValueError(f"acción desconocida: {accion}")
        self.observaciones.append(Observacion(campo, accion, detalle or ""))

    @property
    def estado(self) -> str:
        if any(o.accion == ADVERTIR for o in self.observaciones):
            return ESTADO_ADVERTENCIA
        if self.observaciones:
            return ESTADO_CORREGIDO
        return ESTADO_OK

    def como_dict(self) -> dict:
        return {
            "estado": self.estado,
            "observaciones": [o.como_dict() for o in self.observaciones],
        }
