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


# ------------------------------------------- verificación: historial completo y atómico
def test_historial_conserva_anterior_nuevo_envio_y_fecha_de_registro(bd):
    observar(bd, payload_observacion())
    r = observar(bd, payload_observacion(ventas={"fam_a": 300, "fam_b": 175, "fam_c": 60}))
    with bd.connect() as c:
        hist = c.execute(select(e.observaciones_historial)).one()
        envio = c.execute(select(e.envios_observacion).where(e.envios_observacion.c.envio_id == r.envio_id)).one()
        obs = c.execute(select(e.observaciones).where(e.observaciones.c.familia_id == "fam_b")).one()
    assert (hist.ventas_anterior, hist.ventas_nueva, hist.envio_id) == (150, 175, r.envio_id)
    assert hist.registrado_en is not None
    # misma transacción: el historial, el envío y la modificación comparten el instante de registro
    assert hist.registrado_en == envio.registrado_en == obs.modificado_en


def test_modificacion_e_historial_son_atomicos(bd, monkeypatch):
    """Si falla algo después del UPDATE y del historial, no queda ni la modificación ni el historial."""
    import app.db.repositorio as repo
    observar(bd, payload_observacion())

    def falla(*a, **k):
        raise RuntimeError("fallo simulado después de actualizar y registrar el historial")

    monkeypatch.setattr(repo, "_insertar_calidad", falla)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        observar(bd, payload_observacion(ventas={"fam_a": 300, "fam_b": 999, "fam_c": 60}))
    with bd.connect() as c:
        fam_b = c.execute(select(e.observaciones.c.ventas).where(e.observaciones.c.familia_id == "fam_b")).scalar()
    assert fam_b == 150
    assert contar(bd, e.observaciones_historial) == 0 and contar(bd, e.envios_observacion) == 1


@pytest.mark.parametrize("sentencia", [
    "INSERT INTO observaciones (fecha, tienda_id, familia_id, ventas, envio_id_creacion) "
    "VALUES ('2025-06-21', 1, 'fam_a', 5, :rechazado)",
    "UPDATE observaciones SET ventas = 1, envio_id_modificacion = :rechazado, modificado_en = now()",
    "INSERT INTO observaciones_historial (fecha, tienda_id, familia_id, ventas_anterior, ventas_nueva, envio_id) "
    "VALUES ('2025-06-20', 1, 'fam_a', 300, 1, :rechazado)",
])
def test_un_envio_rechazado_no_crea_ni_modifica_observaciones(bd, sentencia):
    from sqlalchemy import text
    observar(bd, payload_observacion())
    with transaccion(bd) as c:
        rechazado = registrar_rechazo_observacion(c, payload={}, detalle="x")
    with pytest.raises(IntegrityError, match="no fue aceptado"):
        with transaccion(bd) as c:
            c.execute(text(sentencia), {"rechazado": rechazado})


# ------------------------------------------- verificación: consolidación con la base
def test_modificar_observaciones_no_altera_el_historico(bd):
    from tests.integracion.apoyo import DATOS
    with transaccion(bd) as c:
        antes = leer_datos_base(c).historico
    fila = DATOS.historico.iloc[0]
    fecha = fila.fecha.date().isoformat()
    for delta in (5, 9, 0):            # crear, modificar dos veces (la última igual al histórico)
        observar(bd, payload_observacion(tienda=int(fila.tienda_id), fecha=fecha,
                                         ventas={fila.familia_id: int(fila.ventas) + delta}))
    with transaccion(bd) as c:
        despues = leer_datos_base(c).historico
        obs = leer_observaciones(c)
    import pandas as pd
    pd.testing.assert_frame_equal(antes, despues)            # el histórico no cambió
    assert contar(bd, e.observaciones_historial) == 2
    v = consolidar_ventas(despues, obs)
    clave = (v["tienda_id"] == fila.tienda_id) & (v["familia_id"] == fila.familia_id) & (v["fecha"] == fila.fecha)
    assert v.loc[clave, "ventas"].item() == fila.ventas and not v.loc[clave, "discrepancia"].item()


def test_observaciones_sin_historico_se_conservan_y_se_pueden_usar(bd):
    from ml.calendario import Calendario
    from ml.pronosticador import preparar
    observar(bd, payload_observacion())                       # 2025-06-20: no hay histórico
    with transaccion(bd) as c:
        base = leer_datos_base(c)
        obs = leer_observaciones(c)
    v = consolidar_ventas(base.historico, obs)
    nuevas = v[v["origen"] == "observacion"]
    assert len(nuevas) == 3 and set(nuevas["ventas"]) == {300, 150, 60}
    X, _, sin_promo = preparar(v, base.historico, base.tiendas, Calendario(base.feriados),
                               list(base.familias["familia_id"]))
    assert sin_promo == 3 and len(X) == len(base.historico)  # sin promoción: rezagos sí, ejemplo no


def test_consolidacion_desde_la_base_es_determinista(bd):
    observar(bd, payload_observacion())
    observar(bd, payload_observacion(tienda=2))
    resultados = []
    for _ in range(3):
        with transaccion(bd) as c:
            resultados.append(consolidar_ventas(leer_datos_base(c).historico, leer_observaciones(c)))
    import pandas as pd
    for r in resultados[1:]:
        pd.testing.assert_frame_equal(resultados[0], r)
