"""Persistencia en memoria con la interfaz de ``app.servicios.persistencia`` (solo pruebas).

Reproduce las reglas que importan al contrato: ventas previas estrictamente
anteriores a F con prioridad del histórico, registro de solicitudes y rechazos,
y observaciones idempotentes (nuevos + modificados). La implementación real se
prueba contra PostgreSQL en ``tests/integracion/``.
"""

from __future__ import annotations

import copy
import itertools
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from app.calidad.catalogo import Catalogo, Tienda
from app.db.conexion import BaseDatosNoDisponible
from app.db.repositorio import ResultadoObservacion, SolicitudRegistrada
from ml.datos import CLAVE, consolidar_ventas


class PersistenciaMemoria:
    def __init__(self, datos):
        self.historico = datos.historico[CLAVE + ["ventas", "en_promocion"]].copy()
        self._catalogo = Catalogo(
            {int(t.tienda_id): Tienda(int(t.tienda_id), t.ciudad, t.tipo, t.fecha_apertura.date())
             for t in datos.tiendas.itertuples()},
            dict(zip(datos.familias["familia_id"], datos.familias["nombre"])))
        self.observaciones: dict[tuple, int] = {}
        self.solicitudes: list[dict] = []
        self.rechazos_prediccion: list[dict] = []
        self.envios: list[dict] = []
        self.rechazos_observacion: list[dict] = []
        self.caida = False
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _verificar(self):
        if self.caida:
            raise BaseDatosNoDisponible("base de datos no disponible (simulado)")

    def catalogo(self):
        self._verificar()
        return self._catalogo

    def disponible(self):
        return not self.caida

    @contextmanager
    def unidad(self):
        self._verificar()
        with self._lock:
            u = _Unidad(self)
            yield u
            u.confirmar()


class _Unidad:
    def __init__(self, p: PersistenciaMemoria):
        self.p = p
        self.pendientes: list = []

    def confirmar(self):
        for accion in self.pendientes:
            accion()

    def ventas_previas(self, tienda_id, fecha, dias):
        f = pd.Timestamp(fecha)
        h = self.p.historico
        h = h[(h["tienda_id"] == tienda_id) & (h["fecha"] < f)][CLAVE + ["ventas"]]
        o = pd.DataFrame([(pd.Timestamp(k[0]), k[1], k[2], v) for k, v in self.p.observaciones.items()
                          if k[1] == tienda_id and pd.Timestamp(k[0]) < f], columns=CLAVE + ["ventas"])
        return consolidar_ventas(h, o)[CLAVE + ["ventas"]]

    def registrar_prediccion(self, *, payload, solicitud, registro, predicciones, nombre_modelo, version_modelo):
        sid = next(self.p._ids)
        fila = {"solicitud_id": sid, "registrado_en": datetime.now(timezone.utc), "payload": copy.deepcopy(payload),
                "solicitud": solicitud, "estado": registro.estado,
                "observaciones": [o.como_dict() for o in registro.observaciones],
                "supuestas": set(registro.promociones_supuestas), "predicciones": list(predicciones),
                "nombre_modelo": nombre_modelo, "version_modelo": version_modelo}
        self.pendientes.append(lambda: self.p.solicitudes.append(fila))
        return SolicitudRegistrada(sid, fila["registrado_en"])

    def registrar_rechazo_prediccion(self, *, payload, detalle):
        sid = next(self.p._ids)
        self.pendientes.append(lambda: self.p.rechazos_prediccion.append(
            {"solicitud_id": sid, "payload": copy.deepcopy(payload), "detalle": detalle}))
        return SolicitudRegistrada(sid, datetime.now(timezone.utc))

    def registrar_observacion(self, *, payload, solicitud, registro):
        nuevos = modificados = 0
        cambios = {}
        for v in solicitud.ventas:
            k = (solicitud.fecha, solicitud.tienda_id, v.familia_id)
            if k not in self.p.observaciones:
                nuevos += 1
            elif self.p.observaciones[k] != v.ventas:
                modificados += 1
            else:
                continue
            cambios[k] = v.ventas
        eid = next(self.p._ids)

        def aplicar():
            self.p.observaciones.update(cambios)
            self.p.envios.append({"envio_id": eid, "nuevos": nuevos, "modificados": modificados,
                                  "estado": registro.estado})
        self.pendientes.append(aplicar)
        return ResultadoObservacion(eid, nuevos, modificados)

    def registrar_rechazo_observacion(self, *, payload, detalle):
        eid = next(self.p._ids)
        self.pendientes.append(lambda: self.p.rechazos_observacion.append(
            {"envio_id": eid, "payload": copy.deepcopy(payload), "detalle": detalle}))
        return eid


__all__ = ["PersistenciaMemoria"]
