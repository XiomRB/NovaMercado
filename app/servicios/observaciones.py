"""Servicio de ``POST /v1/observaciones``: limpieza y registro idempotente."""

from __future__ import annotations

from app.calidad import SolicitudRechazada, limpiar_solicitud_observacion
from app.schemas.contrato import RespuestaObservacion
from app.servicios.persistencia import Persistencia
from app.servicios.rechazos import registrar_rechazo


def observar(payload: object, persistencia: Persistencia) -> dict:
    catalogo = persistencia.catalogo()
    try:
        solicitud, registro = limpiar_solicitud_observacion(payload, catalogo)
    except SolicitudRechazada as exc:
        registrar_rechazo(persistencia, "observacion", payload, exc.detail)
        raise
    with persistencia.unidad() as u:
        r = u.registrar_observacion(payload=payload, solicitud=solicitud, registro=registro)
    return RespuestaObservacion.model_validate(
        {"registros_guardados": r.registros_guardados, "calidad": registro.como_dict()}).model_dump(mode="json")
