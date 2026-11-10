"""Servicio de ``POST /v1/prediccion`` (sin dependencias de HTTP).

Flujo: modelo vigente → limpieza (con los máximos de promoción del modelo) →
en una transacción: ventas reales **anteriores a F**, advertencia de
antigüedad, predicción y registro. Errores:

* ``SolicitudRechazada`` → se registra el rechazo (mejor esfuerzo) y se propaga (422).
* ``ModeloNoDisponible`` / ``BaseDatosNoDisponible`` → se propagan (503). Nunca
  se entrega una predicción sin registrarla (D-C).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.calidad import SolicitudRechazada, limpiar_solicitud_prediccion
from app.calidad.antiguedad import advertir_antiguedad
from app.db.repositorio import PrediccionFamilia
from app.modelo.gestor import GestorModelo
from app.schemas.contrato import RespuestaPrediccion
from app.servicios.persistencia import Persistencia
from app.servicios.rechazos import registrar_rechazo

logger = logging.getLogger(__name__)

DIAS_HISTORIA = 60   # ventana de ventas previas; cubre rezagos (28) y medias móviles (28)


def predecir(payload: object, persistencia: Persistencia, gestor: GestorModelo) -> dict:
    modelo = gestor.actual()
    catalogo = persistencia.catalogo()
    try:
        solicitud, registro = limpiar_solicitud_prediccion(
            payload, catalogo, maximos_promocion=modelo.pronosticador.maximos_promocion)
    except SolicitudRechazada as exc:
        registrar_rechazo(persistencia, "prediccion", payload, exc.detail)
        raise

    fecha = solicitud.fecha
    familias = [f.familia_id for f in solicitud.familias]
    with persistencia.unidad() as u:
        ventas = u.ventas_previas(solicitud.tienda_id, fecha, DIAS_HISTORIA)
        ventas = ventas[pd.to_datetime(ventas["fecha"]) < pd.Timestamp(fecha)]   # defensa: nunca F ni posteriores
        ultimas = {f: d.date() for f, d in pd.to_datetime(ventas["fecha"]).groupby(ventas["familia_id"]).max().items()}
        advertir_antiguedad(fecha, ultimas, familias, registro)

        objetivos = pd.DataFrame({"fecha": pd.Timestamp(fecha), "tienda_id": solicitud.tienda_id,
                                  "familia_id": familias,
                                  "en_promocion": [f.en_promocion for f in solicitud.familias]})
        pred = modelo.pronosticador.predecir(objetivos, ventas).set_index("familia_id")
        predicciones = [PrediccionFamilia(f, round(float(pred.at[f, "venta_esperada"]), 2),
                                          int(pred.at[f, "cantidad_sugerida"])) for f in familias]
        if not all(np.isfinite(p.venta_esperada) for p in predicciones):
            raise RuntimeError("el modelo produjo una predicción no finita")

        respuesta = RespuestaPrediccion.model_validate({
            "tienda_id": solicitud.tienda_id,
            "fecha": fecha,
            "version_modelo": modelo.version,
            "predicciones": [{"familia_id": p.familia_id, "venta_esperada": p.venta_esperada,
                              "cantidad_sugerida": p.cantidad_sugerida} for p in predicciones],
            "calidad": registro.como_dict(),
        })
        u.registrar_prediccion(payload=payload, solicitud=solicitud, registro=registro, predicciones=predicciones,
                               nombre_modelo=modelo.nombre, version_modelo=modelo.version)
    return respuesta.model_dump(mode="json")

