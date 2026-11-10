"""Observaciones: idempotencia, modificación con historial, concurrencia y relaciones."""

import threading

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db import esquema as e
from app.db.conexion import transaccion
from app.db.repositorio import (
    leer_datos_base,
    leer_observaciones,
    leer_predicciones_reales,
    registrar_rechazo_observacion,
)
from ml.datos import consolidar_ventas
from tests.integracion.apoyo import (
    FECHA,
    cargar_sinteticos,
    observar,
    payload_observacion,
    payload_prediccion,
    predecir_y_registrar,
)


@pytest.fixture
def bd(motor, tmp_path):
    cargar_sinteticos(motor, tmp_path)
    return motor


def contar(motor, tabla):
    with motor.connect() as c:
        return c.execute(select(func.count()).select_from(tabla)).scalar()


def test_nueva_observacion(bd):
    r = observar(bd, payload_observacion())
    assert (r.nuevos, r.modificados, r.registros_guardados) == (3, 0, 3)
    with bd.connect() as c:
        filas = c.execute(select(e.observaciones).order_by(e.observaciones.c.familia_id)).all()
        envio = c.execute(select(e.envios_observacion)).one()
    assert [(f.familia_id, f.ventas, f.envio_id_creacion) for f in filas] == [
        ("fam_a", 300, r.envio_id), ("fam_b", 150, r.envio_id), ("fam_c", 60, r.envio_id)]
    assert all(f.envio_id_modificacion is None and f.modificado_en is None for f in filas)
    assert envio.resultado == "aceptado" and (envio.registros_nuevos, envio.registros_modificados) == (3, 0)


def test_reenvio_identico_es_idempotente(bd):
    observar(bd, payload_observacion())
    r = observar(bd, payload_observacion())
    assert r.registros_guardados == 0
    assert contar(bd, e.observaciones) == 3 and contar(bd, e.observaciones_historial) == 0
    assert contar(bd, e.envios_observacion) == 2     # el envío repetido queda registrado
    with bd.connect() as c:
        ultimo = c.execute(select(e.envios_observacion).where(e.envios_observacion.c.envio_id == r.envio_id)).one()
    assert (ultimo.registros_nuevos, ultimo.registros_modificados) == (0, 0)


def test_reenvio_con_formato_distinto_pero_mismo_valor_es_idempotente(bd):
    observar(bd, payload_observacion(ventas={"fam_a": 1300, "fam_b": 150, "fam_c": 60}))
    p = payload_observacion()
    p["ventas"] = [{"familia_id": "FAM_A", "ventas": "1,300"}, {"familia_id": "fam_b", "ventas": 150.0},
                   {"familia_id": "fam_c", "ventas": "60"}]
    assert observar(bd, p).registros_guardados == 0


def test_valor_modificado_se_actualiza_y_se_cuenta(bd):
    primero = observar(bd, payload_observacion())
    r = observar(bd, payload_observacion(ventas={"fam_a": 300, "fam_b": 175, "fam_c": 60}))
    assert (r.nuevos, r.modificados, r.registros_guardados) == (0, 1, 1)
    with bd.connect() as c:
        fila = c.execute(select(e.observaciones).where(e.observaciones.c.familia_id == "fam_b")).one()
        hist = c.execute(select(e.observaciones_historial)).one()
    assert fila.ventas == 175 and fila.envio_id_creacion == primero.envio_id
    assert fila.envio_id_modificacion == r.envio_id and fila.modificado_en is not None
    assert (hist.familia_id, hist.ventas_anterior, hist.ventas_nueva, hist.envio_id) == ("fam_b", 150, 175, r.envio_id)


def test_varias_modificaciones_dejan_todo_el_historial(bd):
    observar(bd, payload_observacion())
    for v in (151, 152, 150):
        observar(bd, payload_observacion(ventas={"fam_a": 300, "fam_b": v, "fam_c": 60}))
    with bd.connect() as c:
        hist = c.execute(select(e.observaciones_historial.c.ventas_anterior, e.observaciones_historial.c.ventas_nueva)
                         .order_by(e.observaciones_historial.c.historial_id)).all()
    assert [tuple(h) for h in hist] == [(150, 151), (151, 152), (152, 150)]


def test_envio_parcial_y_luego_completo(bd):
    r1 = observar(bd, payload_observacion(ventas={"fam_a": 300}))
    r2 = observar(bd, payload_observacion(ventas={"fam_a": 300, "fam_b": 150, "fam_c": 60}))
    assert (r1.nuevos, r2.nuevos, r2.modificados) == (1, 2, 0)


def test_calidad_del_envio_se_guarda(bd):
    p = payload_observacion()
    p["ventas"].append({"familia_id": "fam_x", "ventas": 1})            # familia desconocida -> advertir
    p["ventas"][0]["ventas"] = "1,300"                                  # reparar
    r = observar(bd, p)
    with bd.connect() as c:
        cal = c.execute(select(e.calidad_observaciones.c.accion).where(
            e.calidad_observaciones.c.envio_id == r.envio_id).order_by(e.calidad_observaciones.c.orden)).all()
        estado = c.execute(select(e.envios_observacion.c.estado_calidad)).scalar()
    assert [x.accion for x in cal] == ["reparar", "advertir"] and estado == "advertencia"


def test_rechazo_de_observacion(bd):
    with transaccion(bd) as c:
        envio_id = registrar_rechazo_observacion(c, payload={"ventas": []}, detalle="ventas: la lista está vacía")
    with bd.connect() as c:
        envio = c.execute(select(e.envios_observacion)).one()
    assert envio.envio_id == envio_id and envio.resultado == "rechazado" and envio.registros_nuevos is None
    assert contar(bd, e.observaciones) == 0


@pytest.mark.parametrize("sentencia", ["DELETE FROM observaciones", "TRUNCATE observaciones CASCADE",
                                       "UPDATE observaciones_historial SET ventas_nueva = 0",
                                       "DELETE FROM envios_observacion"])
def test_observaciones_no_se_borran(bd, sentencia):
    observar(bd, payload_observacion())
    observar(bd, payload_observacion(ventas={"fam_a": 1, "fam_b": 150, "fam_c": 60}))
    from sqlalchemy import text
    with pytest.raises(IntegrityError, match="no admite"):
        with transaccion(bd) as c:
            c.execute(text(sentencia))
    assert contar(bd, e.observaciones) == 3


def test_envios_concurrentes_del_mismo_dia(bd):
    """Dos envíos simultáneos con el mismo contenido: exactamente uno registra las 3 ventas."""
    resultados, errores = [], []
    barrera = threading.Barrier(4)

    def enviar():
        try:
            barrera.wait()
            resultados.append(observar(bd, payload_observacion()).registros_guardados)
        except Exception as exc:  # noqa: BLE001 - se informa en la aserción
            errores.append(exc)

    hilos = [threading.Thread(target=enviar) for _ in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert errores == []
    assert sorted(resultados) == [0, 0, 0, 3]
    assert contar(bd, e.observaciones) == 3 and contar(bd, e.envios_observacion) == 4


def test_observacion_de_fecha_del_historico_se_guarda_y_el_historico_manda(bd):
    from tests.integracion.apoyo import DATOS
    fila = DATOS.historico.iloc[0]
    fecha = fila.fecha.date().isoformat()
    observar(bd, payload_observacion(tienda=int(fila.tienda_id), fecha=fecha,
                                     ventas={fila.familia_id: int(fila.ventas) + 5}))
    with transaccion(bd) as c:
        historico = leer_datos_base(c).historico
        obs = leer_observaciones(c)
    assert len(obs) == 1                                       # se guardó sin tocar el histórico
    v = consolidar_ventas(historico, obs)
    d = v[v["discrepancia"]]
    assert len(d) == 1 and d.iloc[0]["ventas"] == fila.ventas and d.iloc[0]["ventas_observacion"] == fila.ventas + 5


def test_relacion_prediccion_y_venta_real(bd):
    r, _, _ = predecir_y_registrar(bd, payload_prediccion(), version="2")
    predecir_y_registrar(bd, payload_prediccion(tienda=2))          # sin observación
    observar(bd, payload_observacion())
    with transaccion(bd) as c:
        v = leer_predicciones_reales(c)
    t1 = v[v["tienda_id"] == 1].set_index("familia_id")
    assert t1.loc["fam_b", "ventas_reales"] == 150 and t1.loc["fam_b", "ventas_observadas"] == 150
    assert (t1["solicitud_id"] == r.solicitud_id).all() and (t1["version_modelo"] == "2").all()
    assert v[v["tienda_id"] == 2]["ventas_reales"].isna().all()
    assert str(v["fecha"].iloc[0].date()) == FECHA
