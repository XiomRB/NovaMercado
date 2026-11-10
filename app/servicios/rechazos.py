"""Registro de solicitudes rechazadas (422) con mejor esfuerzo (D-E).

Si la base no está disponible, la respuesta sigue siendo 422 y el fallo queda
en el log **sin el contenido de la solicitud** (puede traer datos de terceros).
"""

from __future__ import annotations

import logging

from app.servicios.persistencia import Persistencia

logger = logging.getLogger(__name__)


def registrar_rechazo(persistencia: Persistencia, tipo: str, payload: object, detalle: str) -> bool:
    try:
        with persistencia.unidad() as u:
            if tipo == "prediccion":
                u.registrar_rechazo_prediccion(payload=payload, detalle=detalle)
            else:
                u.registrar_rechazo_observacion(payload=payload, detalle=detalle)
        return True
    except Exception as exc:  # noqa: BLE001 - el rechazo se responde igual aunque no se pueda registrar
        logger.error("no se pudo registrar una solicitud rechazada de %s: %s", tipo, type(exc).__name__)
        return False
