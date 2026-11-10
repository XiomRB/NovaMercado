"""Modelos Pydantic del contrato del servicio (Anexo).

Las solicitudes se validan en modo estricto *después* de la capa de calidad
(``app.calidad``): la limpieza convierte lo recibido a los tipos del contrato y
estos modelos garantizan que el resultado lo cumple. Las respuestas también se
validan, para que ningún endpoint pueda devolver algo fuera del contrato.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EstadoCalidad = Literal["ok", "corregido", "advertencia"]
AccionCalidad = Literal["reparar", "advertir"]


class _Estricto(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


# ----------------------------------------------------------------- calidad
class ObservacionCalidad(_Estricto):
    campo: str
    accion: AccionCalidad
    detalle: str


class Calidad(_Estricto):
    estado: EstadoCalidad
    observaciones: list[ObservacionCalidad] = Field(default_factory=list)

    @model_validator(mode="after")
    def _estado_coherente(self) -> "Calidad":
        acciones = {o.accion for o in self.observaciones}
        esperado = "advertencia" if "advertir" in acciones else ("corregido" if acciones else "ok")
        if self.estado != esperado:
            raise ValueError(f"estado '{self.estado}' incoherente con las observaciones (debe ser '{esperado}')")
        return self


# -------------------------------------------------------------- predicción
class FamiliaSolicitada(_Estricto):
    familia_id: str = Field(min_length=1)
    en_promocion: int = Field(ge=0)


class SolicitudPrediccion(_Estricto):
    tienda_id: int = Field(gt=0)
    fecha: date
    familias: list[FamiliaSolicitada] = Field(min_length=1)

    @model_validator(mode="after")
    def _familias_unicas(self) -> "SolicitudPrediccion":
        ids = [f.familia_id for f in self.familias]
        if len(ids) != len(set(ids)):
            raise ValueError("familias repetidas")
        return self


class PrediccionFamilia(_Estricto):
    familia_id: str
    venta_esperada: float = Field(ge=0)
    cantidad_sugerida: int = Field(ge=0)


class RespuestaPrediccion(_Estricto):
    tienda_id: int
    fecha: date
    version_modelo: str
    predicciones: list[PrediccionFamilia]
    calidad: Calidad


# ------------------------------------------------------------ observaciones
class VentaFamilia(_Estricto):
    familia_id: str = Field(min_length=1)
    ventas: int = Field(ge=0)


class SolicitudObservacion(_Estricto):
    tienda_id: int = Field(gt=0)
    fecha: date
    ventas: list[VentaFamilia] = Field(min_length=1)

    @model_validator(mode="after")
    def _familias_unicas(self) -> "SolicitudObservacion":
        ids = [v.familia_id for v in self.ventas]
        if len(ids) != len(set(ids)):
            raise ValueError("familias repetidas")
        return self


class RespuestaObservacion(_Estricto):
    registros_guardados: int = Field(ge=0)
    calidad: Calidad


# ---------------------------------------------------------- administración
class ReentrenamientoAceptado(_Estricto):
    id: str


class EstadoReentrenamiento(_Estricto):
    estado: Literal["en_curso", "exitoso", "fallido"]
    promovido: bool | None = None
    version: str | None = None
    motivo: str | None = None

    @model_validator(mode="after")
    def _campos_al_terminar(self) -> "EstadoReentrenamiento":
        terminado = self.estado != "en_curso"
        if terminado and (self.promovido is None or self.version is None or self.motivo is None):
            raise ValueError("una ejecución terminada debe incluir promovido, version y motivo")
        if not terminado and any(v is not None for v in (self.promovido, self.version, self.motivo)):
            raise ValueError("promovido, version y motivo solo se incluyen al terminar")
        return self

    def como_respuesta(self) -> dict:
        """Mientras está en curso solo se envía ``estado``; al terminar, los cuatro campos."""
        if self.estado == "en_curso":
            return {"estado": self.estado}
        return self.model_dump()


class ModeloVigente(_Estricto):
    nombre: str
    alias: str
    version: str


class EstadoSalud(BaseModel):
    """``ok`` o ``degradado`` (con ``motivo``). Admite campos extra (componentes)."""

    model_config = ConfigDict(extra="allow")
    estado: Literal["ok", "degradado"]
    motivo: str | None = None

    @model_validator(mode="after")
    def _motivo_si_degradado(self) -> "EstadoSalud":
        if self.estado == "degradado" and not self.motivo:
            raise ValueError("el estado degradado requiere motivo")
        return self


class ErrorContrato(BaseModel):
    detail: str
