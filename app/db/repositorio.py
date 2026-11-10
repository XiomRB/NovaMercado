"""Operaciones de la base del producto. Diseño: DECISIONES.md §11.

Todas reciben una ``Connection`` abierta dentro de :func:`app.db.conexion.transaccion`;
quien llama define la unidad de trabajo (normalmente, una solicitud HTTP).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.engine import Connection

from app.calidad.catalogo import Catalogo, Tienda
from app.calidad.resultado import RegistroCalidad
from app.db import esquema as e
from app.schemas.contrato import SolicitudObservacion, SolicitudPrediccion
from ml.datos import CLAVE, COLUMNAS_SOLICITUDES, DatosBase


# ------------------------------------------------------------------ utilidades
def json_seguro(payload: object) -> object:
    """Copia serializable a JSONB de lo recibido (NaN/Infinito no existen en JSON)."""
    def limpiar(v):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        if isinstance(v, dict):
            return {str(k): limpiar(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [limpiar(x) for x in v]
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        return str(v)
    return json.loads(json.dumps(limpiar(payload), allow_nan=False))


def _insertar_calidad(conn: Connection, registro: RegistroCalidad, *, solicitud_id: int | None = None,
                      envio_id: int | None = None) -> None:
    filas = [{"solicitud_id": solicitud_id, "envio_id": envio_id, "orden": i, "campo": o.campo,
              "accion": o.accion, "detalle": o.detalle} for i, o in enumerate(registro.observaciones)]
    if filas:
        conn.execute(insert(e.calidad_observaciones), filas)


# ------------------------------------------------------------------ predicciones
@dataclass(frozen=True)
class PrediccionFamilia:
    familia_id: str
    venta_esperada: float
    cantidad_sugerida: int


@dataclass(frozen=True)
class SolicitudRegistrada:
    solicitud_id: int
    registrado_en: datetime


def registrar_prediccion(conn: Connection, *, payload: object, solicitud: SolicitudPrediccion,
                         registro: RegistroCalidad, predicciones: Sequence[PrediccionFamilia],
                         nombre_modelo: str, version_modelo: str) -> SolicitudRegistrada:
    """Guarda solicitud, familias (promoción real o supuesta), predicciones y calidad.

    Exige una predicción por cada familia de la solicitud: el contrato no permite
    omitir familias. Nunca actualiza registros previos (tablas de solo inserción).
    """
    solicitadas = [f.familia_id for f in solicitud.familias]
    predichas = [p.familia_id for p in predicciones]
    if sorted(solicitadas) != sorted(predichas) or len(set(predichas)) != len(predichas):
        raise ValueError("las predicciones deben cubrir exactamente las familias de la solicitud")
    fila = conn.execute(
        insert(e.solicitudes_prediccion).values(
            resultado="respondida", payload=json_seguro(payload), tienda_id=solicitud.tienda_id,
            fecha=solicitud.fecha, nombre_modelo=nombre_modelo, version_modelo=str(version_modelo),
            estado_calidad=registro.estado,
        ).returning(e.solicitudes_prediccion.c.solicitud_id, e.solicitudes_prediccion.c.registrado_en)
    ).one()
    sid = fila.solicitud_id
    conn.execute(insert(e.solicitud_familias), [
        {"solicitud_id": sid, "familia_id": f.familia_id, "en_promocion": f.en_promocion,
         "promocion_supuesta": f.familia_id in registro.promociones_supuestas}
        for f in solicitud.familias])
    conn.execute(insert(e.predicciones), [
        {"solicitud_id": sid, "familia_id": p.familia_id, "venta_esperada": float(p.venta_esperada),
         "cantidad_sugerida": int(p.cantidad_sugerida)} for p in predicciones])
    _insertar_calidad(conn, registro, solicitud_id=sid)
    return SolicitudRegistrada(sid, fila.registrado_en)


def registrar_rechazo_prediccion(conn: Connection, *, payload: object, detalle: str) -> SolicitudRegistrada:
    fila = conn.execute(
        insert(e.solicitudes_prediccion).values(resultado="rechazada", payload=json_seguro(payload),
                                                detalle_rechazo=detalle)
        .returning(e.solicitudes_prediccion.c.solicitud_id, e.solicitudes_prediccion.c.registrado_en)
    ).one()
    return SolicitudRegistrada(fila.solicitud_id, fila.registrado_en)


# ---------------------------------------------------------------- observaciones
@dataclass(frozen=True)
class ResultadoObservacion:
    envio_id: int
    nuevos: int
    modificados: int

    @property
    def registros_guardados(self) -> int:
        """Lo que informa el contrato: registros nuevos o modificados por el envío."""
        return self.nuevos + self.modificados


def _bloquear_tienda_fecha(conn: Connection, tienda_id: int, fecha: date) -> None:
    """Serializa los envíos concurrentes de la misma tienda y fecha (lock de transacción)."""
    conn.execute(text("SELECT pg_advisory_xact_lock(:a, :b)"),
                 {"a": int(tienda_id), "b": (fecha - date(2000, 1, 1)).days})


def registrar_observacion(conn: Connection, *, payload: object, solicitud: SolicitudObservacion,
                          registro: RegistroCalidad) -> ResultadoObservacion:
    """Registro idempotente de ventas reales.

    Clave nueva -> se inserta; mismo valor -> nada; valor distinto -> se actualiza
    y el valor anterior queda en ``observaciones_historial``. El envío siempre se
    registra (aunque no cambie nada) con sus conteos.
    """
    _bloquear_tienda_fecha(conn, solicitud.tienda_id, solicitud.fecha)
    o = e.observaciones
    actuales = dict(conn.execute(
        select(o.c.familia_id, o.c.ventas).where(o.c.tienda_id == solicitud.tienda_id,
                                                 o.c.fecha == solicitud.fecha).with_for_update()
    ).all())
    nuevas = [v for v in solicitud.ventas if v.familia_id not in actuales]
    cambiadas = [v for v in solicitud.ventas if v.familia_id in actuales and actuales[v.familia_id] != v.ventas]

    envio_id = conn.execute(
        insert(e.envios_observacion).values(
            resultado="aceptado", payload=json_seguro(payload), tienda_id=solicitud.tienda_id,
            fecha=solicitud.fecha, estado_calidad=registro.estado,
            registros_nuevos=len(nuevas), registros_modificados=len(cambiadas),
        ).returning(e.envios_observacion.c.envio_id)
    ).scalar_one()
    if nuevas:
        conn.execute(insert(o), [{"fecha": solicitud.fecha, "tienda_id": solicitud.tienda_id,
                                  "familia_id": v.familia_id, "ventas": v.ventas,
                                  "envio_id_creacion": envio_id} for v in nuevas])
    for v in cambiadas:
        conn.execute(update(o).where(o.c.fecha == solicitud.fecha, o.c.tienda_id == solicitud.tienda_id,
                                     o.c.familia_id == v.familia_id)
                     .values(ventas=v.ventas, envio_id_modificacion=envio_id, modificado_en=func.now()))
    if cambiadas:
        conn.execute(insert(e.observaciones_historial), [
            {"fecha": solicitud.fecha, "tienda_id": solicitud.tienda_id, "familia_id": v.familia_id,
             "ventas_anterior": actuales[v.familia_id], "ventas_nueva": v.ventas, "envio_id": envio_id}
            for v in cambiadas])
    _insertar_calidad(conn, registro, envio_id=envio_id)
    return ResultadoObservacion(envio_id, len(nuevas), len(cambiadas))


def registrar_rechazo_observacion(conn: Connection, *, payload: object, detalle: str) -> int:
    return conn.execute(
        insert(e.envios_observacion).values(resultado="rechazado", payload=json_seguro(payload),
                                            detalle_rechazo=detalle)
        .returning(e.envios_observacion.c.envio_id)
    ).scalar_one()


# ------------------------------------------------------------------ consultas
def cargar_catalogo(conn: Connection) -> Catalogo:
    t = e.tiendas
    tiendas = {r.tienda_id: Tienda(r.tienda_id, r.ciudad, r.tipo, r.fecha_apertura)
               for r in conn.execute(select(t.c.tienda_id, t.c.ciudad, t.c.tipo, t.c.fecha_apertura))}
    familias = dict(conn.execute(select(e.familias.c.familia_id, e.familias.c.nombre)
                                 .order_by(e.familias.c.familia_id)).all())
    return Catalogo(tiendas, familias)


def _df(conn: Connection, consulta, fechas: Iterable[str] = ()) -> pd.DataFrame:
    resultado = conn.execute(consulta)
    # str(): SQLAlchemy entrega nombres como quoted_name, que scikit-learn rechaza como nombres de variables
    df = pd.DataFrame(resultado.all(), columns=[str(k) for k in resultado.keys()])
    for c in fechas:
        df[c] = pd.to_datetime(df[c])
    return df


def leer_datos_base(conn: Connection) -> DatosBase:
    """Los datos publicados, con el mismo formato que :func:`ml.datos.cargar_raw`."""
    vh = e.ventas_historicas
    historico = _df(conn, select(vh.c.fecha, vh.c.tienda_id, vh.c.familia_id, vh.c.ventas, vh.c.en_promocion)
                    .order_by(vh.c.fecha, vh.c.tienda_id, vh.c.familia_id), ["fecha"])
    tiendas = _df(conn, select(e.tiendas).order_by(e.tiendas.c.tienda_id), ["fecha_apertura"])
    familias = _df(conn, select(e.familias).order_by(e.familias.c.familia_id))
    costos = _df(conn, select(e.costos).order_by(e.costos.c.familia_id))
    costos[["costo_faltante", "costo_sobrante"]] = costos[["costo_faltante", "costo_sobrante"]].astype(float)
    feriados = _df(conn, select(e.feriados.c.fecha, e.feriados.c.nombre, e.feriados.c.tipo, e.feriados.c.ambito)
                   .order_by(e.feriados.c.fecha, e.feriados.c.nombre), ["fecha"])
    return DatosBase(historico, tiendas, familias, costos, feriados)


def leer_observaciones(conn: Connection) -> pd.DataFrame:
    """Ventas reales registradas (valor vigente): fecha, tienda_id, familia_id, ventas."""
    o = e.observaciones
    return _df(conn, select(o.c.fecha, o.c.tienda_id, o.c.familia_id, o.c.ventas)
               .order_by(o.c.fecha, o.c.tienda_id, o.c.familia_id), ["fecha"])


def leer_promociones_solicitudes(conn: Connection) -> pd.DataFrame:
    """Una fila por familia de cada solicitud **respondida**, con
    :data:`ml.datos.COLUMNAS_SOLICITUDES` (entrada de ``consolidar_promociones``)."""
    s, f = e.solicitudes_prediccion, e.solicitud_familias
    consulta = (select(s.c.fecha, s.c.tienda_id, f.c.familia_id, f.c.en_promocion, f.c.promocion_supuesta,
                       s.c.registrado_en, s.c.solicitud_id)
                .join(f, f.c.solicitud_id == s.c.solicitud_id)
                .where(s.c.resultado == "respondida")
                .order_by(s.c.registrado_en, s.c.solicitud_id, f.c.familia_id))
    df = _df(conn, consulta, ["fecha"])
    return df[COLUMNAS_SOLICITUDES]


_VENTAS_PREVIAS = """
(SELECT fecha, tienda_id, familia_id, ventas FROM {tabla}
  WHERE tienda_id = :tienda AND fecha >= :desde AND fecha < :fecha)
UNION
(SELECT DISTINCT ON (familia_id) fecha, tienda_id, familia_id, ventas FROM {tabla}
  WHERE tienda_id = :tienda AND fecha < :fecha
  ORDER BY familia_id, fecha DESC)
"""


def leer_ventas_previas(conn: Connection, tienda_id: int, fecha: date, dias: int = 60) -> pd.DataFrame:
    """Ventas reales de la tienda **anteriores a** ``fecha`` (nunca de ``fecha`` ni posteriores).

    Incluye los ``dias`` previos y, por familia, la última venta conocida aunque
    sea más antigua: así ``ultimo_valor`` y ``dias_desde_ultimo`` coinciden con
    los que se calcularían con todo el histórico. Histórico y observaciones se
    consolidan con :func:`ml.datos.consolidar_ventas` (manda el histórico).
    """
    from datetime import timedelta

    from ml.datos import consolidar_ventas
    params = {"tienda": int(tienda_id), "fecha": fecha, "desde": fecha - timedelta(days=dias)}
    hist = _df(conn, text(_VENTAS_PREVIAS.format(tabla="ventas_historicas")).bindparams(**params), ["fecha"])
    obs = _df(conn, text(_VENTAS_PREVIAS.format(tabla="observaciones")).bindparams(**params), ["fecha"])
    if hist.empty:
        hist = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             [("fecha", "datetime64[ns]"), ("tienda_id", int), ("familia_id", str), ("ventas", int)]})
    return consolidar_ventas(hist, obs)[CLAVE + ["ventas"]]


def leer_predicciones_reales(conn: Connection) -> pd.DataFrame:
    """Vista ``v_predicciones_reales``: cada predicción con la venta real conocida."""
    return _df(conn, text("SELECT * FROM v_predicciones_reales ORDER BY solicitud_id, familia_id"), ["fecha"])


__all__ = ["CLAVE", "PrediccionFamilia", "ResultadoObservacion", "SolicitudRegistrada", "cargar_catalogo",
           "json_seguro", "leer_datos_base", "leer_observaciones", "leer_predicciones_reales", "leer_ventas_previas",
           "leer_promociones_solicitudes", "registrar_observacion", "registrar_prediccion",
           "registrar_rechazo_observacion", "registrar_rechazo_prediccion"]
