"""Solicitudes y predicciones: inserción, consulta, promociones, inmutabilidad e integridad."""

import pandas as pd
import pytest
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from app.calidad import limpiar_solicitud_prediccion
from app.db import esquema as e
from app.db.conexion import transaccion
from app.db.repositorio import (
    PrediccionFamilia,
    cargar_catalogo,
    leer_promociones_solicitudes,
    registrar_prediccion,
    registrar_rechazo_prediccion,
)
from ml.datos import COLUMNAS_SOLICITUDES, consolidar_promociones
from tests.integracion.apoyo import FECHA, cargar_sinteticos, payload_prediccion, predecir_y_registrar


@pytest.fixture
def bd(motor, tmp_path):
    cargar_sinteticos(motor, tmp_path)
    return motor


def contar(motor, tabla):
    with motor.connect() as c:
        return c.execute(select(func.count()).select_from(tabla)).scalar()


def test_registrar_y_consultar(bd):
    r, solicitud, registro = predecir_y_registrar(bd, payload_prediccion(), version="3")
    with bd.connect() as c:
        s = c.execute(select(e.solicitudes_prediccion).where(
            e.solicitudes_prediccion.c.solicitud_id == r.solicitud_id)).one()
        fams = c.execute(select(e.solicitud_familias).order_by(e.solicitud_familias.c.familia_id)).all()
        preds = c.execute(select(e.predicciones).order_by(e.predicciones.c.familia_id)).all()
    assert s.resultado == "respondida" and s.version_modelo == "3" and s.nombre_modelo == "novamercado-ventas"
    assert s.tienda_id == 1 and str(s.fecha) == FECHA and s.estado_calidad == "ok"
    assert s.payload == payload_prediccion() and s.registrado_en is not None
    assert [(f.familia_id, f.en_promocion, f.promocion_supuesta) for f in fams] == [
        ("fam_a", 2, False), ("fam_b", 0, False), ("fam_c", 1, False)]
    assert [p.familia_id for p in preds] == ["fam_a", "fam_b", "fam_c"]


def test_promociones_supuestas_y_calidad_se_guardan(bd):
    p = payload_prediccion(quitar=("fam_c",))
    p["familias"][0]["en_promocion"] = "abc"       # fam_a -> 0 supuesto
    p["fecha"] = "20/06/2025"                       # reparar
    r, _, registro = predecir_y_registrar(bd, p)
    with bd.connect() as c:
        sup = dict(c.execute(select(e.solicitud_familias.c.familia_id, e.solicitud_familias.c.promocion_supuesta)
                             .where(e.solicitud_familias.c.solicitud_id == r.solicitud_id)).all())
        cal = c.execute(select(e.calidad_observaciones.c.orden, e.calidad_observaciones.c.campo,
                               e.calidad_observaciones.c.accion)
                        .where(e.calidad_observaciones.c.solicitud_id == r.solicitud_id)
                        .order_by(e.calidad_observaciones.c.orden)).all()
        estado = c.execute(select(e.solicitudes_prediccion.c.estado_calidad)).scalar()
    assert sup == {"fam_a": True, "fam_b": False, "fam_c": True}
    assert [(o.campo, o.accion) for o in cal] == [(o.campo, o.accion) for o in registro.observaciones]
    assert [o.orden for o in cal] == list(range(len(cal))) and estado == "advertencia"


def test_rechazo_se_guarda_con_payload_y_motivo(bd):
    with transaccion(bd) as c:
        r = registrar_rechazo_prediccion(c, payload={"tienda": "xyz", "valor": float("nan")},
                                         detalle="tienda_id: falta el campo obligatorio")
    with bd.connect() as c:
        s = c.execute(select(e.solicitudes_prediccion)).one()
    assert s.solicitud_id == r.solicitud_id and s.resultado == "rechazada"
    assert s.detalle_rechazo == "tienda_id: falta el campo obligatorio"
    assert s.payload == {"tienda": "xyz", "valor": "nan"} and s.tienda_id is None and s.version_modelo is None


def test_varias_solicitudes_del_mismo_dia_no_se_sobrescriben(bd):
    a, _, _ = predecir_y_registrar(bd, payload_prediccion(promos={"fam_a": 1, "fam_b": 0, "fam_c": 0}), "1")
    b, _, _ = predecir_y_registrar(bd, payload_prediccion(promos={"fam_a": 5, "fam_b": 0, "fam_c": 0}), "2")
    assert b.solicitud_id > a.solicitud_id and b.registrado_en >= a.registrado_en
    assert contar(bd, e.solicitudes_prediccion) == 2 and contar(bd, e.predicciones) == 6
    with transaccion(bd) as c:
        promos = leer_promociones_solicitudes(c)
    assert list(promos.columns) == COLUMNAS_SOLICITUDES and len(promos) == 6
    consolidado = consolidar_promociones(pd.DataFrame(columns=["fecha", "tienda_id", "familia_id", "en_promocion"])
                                         .astype({"tienda_id": int, "en_promocion": int}), promos)
    assert consolidado.set_index("familia_id").loc["fam_a", "en_promocion"] == 5   # la última


def test_rechazadas_no_alimentan_las_promociones(bd):
    with transaccion(bd) as c:
        registrar_rechazo_prediccion(c, payload={}, detalle="x")
    predecir_y_registrar(bd, payload_prediccion())
    with transaccion(bd) as c:
        assert len(leer_promociones_solicitudes(c)) == 3


@pytest.mark.parametrize("sentencia", [
    "UPDATE solicitudes_prediccion SET version_modelo = '9'",
    "DELETE FROM solicitudes_prediccion",
    "TRUNCATE solicitudes_prediccion CASCADE",
    "UPDATE predicciones SET cantidad_sugerida = 0",
    "DELETE FROM predicciones",
    "UPDATE solicitud_familias SET promocion_supuesta = true",
    "DELETE FROM calidad_observaciones",
    "UPDATE ventas_historicas SET ventas = 0",
    "DELETE FROM ventas_historicas",
])
def test_registros_historicos_no_se_sobrescriben(bd, sentencia):
    p = payload_prediccion()
    p["fecha"] = "20/06/2025"                 # genera una fila de calidad (reparar)
    predecir_y_registrar(bd, p)
    assert contar(bd, e.calidad_observaciones) == 1
    with pytest.raises(IntegrityError, match="no admite"):
        with transaccion(bd) as c:
            c.execute(text(sentencia))
    assert contar(bd, e.solicitudes_prediccion) == 1 and contar(bd, e.predicciones) == 3


def test_check_respondida_requiere_modelo(bd):
    with pytest.raises(IntegrityError, match="coherencia_resultado"):
        with transaccion(bd) as c:
            c.execute(insert(e.solicitudes_prediccion).values(resultado="respondida", payload={}, tienda_id=1,
                                                              fecha=FECHA, estado_calidad="ok"))


def test_prediccion_solo_para_familias_solicitadas(bd):
    r, _, _ = predecir_y_registrar(bd, payload_prediccion())
    with pytest.raises(IntegrityError, match="fk_predicciones_solicitud_id"):
        with transaccion(bd) as c:
            c.execute(insert(e.predicciones).values(solicitud_id=r.solicitud_id + 100, familia_id="fam_a",
                                                    venta_esperada=1.0, cantidad_sugerida=1))


def _limpia(bd, payload):
    with transaccion(bd) as c:
        return limpiar_solicitud_prediccion(payload, cargar_catalogo(c))


def test_predicciones_incompletas_no_se_registran(bd):
    solicitud, registro = _limpia(bd, payload_prediccion())
    with pytest.raises(ValueError, match="exactamente las familias"):
        with transaccion(bd) as c:
            registrar_prediccion(c, payload={}, solicitud=solicitud, registro=registro,
                                 predicciones=[PrediccionFamilia("fam_a", 1.0, 1)], nombre_modelo="m",
                                 version_modelo="1")
    assert contar(bd, e.solicitudes_prediccion) == 0


def test_transaccion_atomica_si_falla_a_mitad(bd):
    """Una cantidad negativa viola un CHECK al insertar predicciones: no queda nada de la solicitud."""
    solicitud, registro = _limpia(bd, payload_prediccion())
    preds = [PrediccionFamilia(f.familia_id, 1.0, 1) for f in solicitud.familias]
    preds[-1] = PrediccionFamilia(preds[-1].familia_id, 1.0, -1)
    with pytest.raises(IntegrityError, match="cantidad_no_negativa"):
        with transaccion(bd) as c:
            registrar_prediccion(c, payload={}, solicitud=solicitud, registro=registro, predicciones=preds,
                                 nombre_modelo="m", version_modelo="1")
    for t in (e.solicitudes_prediccion, e.solicitud_familias, e.predicciones, e.calidad_observaciones):
        assert contar(bd, t) == 0


def test_tienda_inexistente_viola_fk(bd):
    with pytest.raises(IntegrityError, match="fk_solicitudes_prediccion_tienda_id"):
        with transaccion(bd) as c:
            c.execute(insert(e.solicitudes_prediccion).values(
                resultado="respondida", payload={}, tienda_id=99, fecha=FECHA, nombre_modelo="m",
                version_modelo="1", estado_calidad="ok"))


def test_update_de_observaciones_permitido_no_de_envios(bd):
    """Las observaciones admiten corrección; los envíos son de solo inserción."""
    from tests.integracion.apoyo import observar, payload_observacion
    observar(bd, payload_observacion())
    with transaccion(bd) as c:
        c.execute(update(e.observaciones).values(ventas=1))
    with pytest.raises(IntegrityError, match="no admite"):
        with transaccion(bd) as c:
            c.execute(update(e.envios_observacion).values(registros_nuevos=0))


# ------------------------------------------- verificación: rechazos sin tienda ni fecha
@pytest.mark.parametrize("payload", [
    {"fecha": "2025-06-20", "familias": [{"familia_id": "fam_a", "en_promocion": 1}]},   # sin tienda
    {"tienda_id": "T5-2", "fecha": "2025-06-20", "familias": []},                      # tienda ambigua
    {"tienda_id": 1, "fecha": "31/02/2025", "familias": []},                           # fecha inexistente
    ["no", "es", "objeto"],
])
def test_rechazo_sin_tienda_ni_fecha_determinables(bd, payload):
    from app.calidad import SolicitudRechazada
    with pytest.raises(SolicitudRechazada) as exc:
        _limpia(bd, payload)
    with transaccion(bd) as c:
        r = registrar_rechazo_prediccion(c, payload=payload, detalle=exc.value.detail)
    with bd.connect() as c:
        s = c.execute(select(e.solicitudes_prediccion)).one()
    assert s.solicitud_id == r.solicitud_id and s.resultado == "rechazada" and s.detalle_rechazo == exc.value.detail
    assert s.tienda_id is None and s.fecha is None and s.version_modelo is None and s.estado_calidad is None
    assert s.payload == payload
    for t in (e.solicitud_familias, e.predicciones, e.calidad_observaciones):
        assert contar(bd, t) == 0


def test_una_solicitud_rechazada_no_admite_familias_ni_predicciones(bd):
    with transaccion(bd) as c:
        r = registrar_rechazo_prediccion(c, payload={}, detalle="x")
    with pytest.raises(IntegrityError, match="no fue respondida"):
        with transaccion(bd) as c:
            c.execute(insert(e.solicitud_familias).values(solicitud_id=r.solicitud_id, familia_id="fam_a",
                                                          en_promocion=0, promocion_supuesta=False))
    assert contar(bd, e.solicitud_familias) == 0 and contar(bd, e.predicciones) == 0
