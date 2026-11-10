"""API completa contra PostgreSQL real (modelo sintético pequeño, sin MLflow)."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import esquema as e
from app.db.carga import cargar_datos_publicados
from app.db.conexion import crear_motor, transaccion
from app.db.repositorio import leer_predicciones_reales, leer_ventas_previas
from app.dependencias import Dependencias, construir_monitor
from app.main import create_app
from app.modelo.entrenamiento import empaquetar, entrenar_desde_bd
from app.modelo.entrenar_local import guardar_atomico
from app.modelo.gestor import GestorModelo
from app.modelo.proveedores import ProveedorArchivo
from app.orquestador import OrquestadorNoConfigurado
from app.servicios.persistencia import PersistenciaPostgres
from ml.pronosticador import ConfigEntrenamiento, Pronosticador
from tests.api.conftest import CONFIG_RAPIDA, DATOS, FECHA, ProveedorMemoria
from tests.conftest import FIXTURES
from tests.ml.sinteticos import escribir_carpeta

EJ = FIXTURES / "ejemplos"


@pytest.fixture(scope="module")
def artefacto():
    h = DATOS.historico
    p = Pronosticador.entrenar(h, h, DATOS.tiendas, DATOS.feriados, DATOS.costos,
                               list(DATOS.familias["familia_id"]), pd.Timestamp(FECHA), CONFIG_RAPIDA)
    return empaquetar(p)


@pytest.fixture
def bd(motor, tmp_path):
    escribir_carpeta(DATOS, tmp_path)
    cargar_datos_publicados(motor, tmp_path)
    return motor


def _deps(motor, proveedor):
    persistencia = PersistenciaPostgres(motor)
    gestor = GestorModelo(proveedor)
    gestor.refrescar()
    orq = OrquestadorNoConfigurado()
    monitor = construir_monitor(persistencia, gestor, orq, requeridos=("api", "base_datos", "modelo"),
                                intervalo=1, limite=2)
    monitor.ejecutar_ciclo()
    return Dependencias(persistencia, gestor, monitor, orq)


@pytest.fixture
def cliente(bd, artefacto):
    d = _deps(bd, ProveedorMemoria(artefacto))
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        yield c
    d.monitor.detener()


def ej(nombre):
    return json.loads((EJ / nombre).read_text(encoding="utf-8"))


def contar(motor, tabla):
    with motor.connect() as c:
        return c.execute(select(func.count()).select_from(tabla)).scalar()


def test_prediccion_queda_registrada_con_version_y_calidad(cliente, bd, artefacto):
    p = ej("errores/E05_faltan_familias.json")
    r = cliente.post("/v1/prediccion", json=p)
    assert r.status_code == 200 and r.json()["calidad"]["estado"] == "advertencia"
    with bd.connect() as c:
        s = c.execute(select(e.solicitudes_prediccion)).one()
        sup = dict(c.execute(select(e.solicitud_familias.c.familia_id,
                                    e.solicitud_familias.c.promocion_supuesta)).all())
        preds = {x.familia_id: x for x in c.execute(select(e.predicciones))}
        n_cal = c.execute(select(func.count()).select_from(e.calidad_observaciones)).scalar()
    assert (s.resultado, s.version_modelo, s.nombre_modelo, s.estado_calidad) == (
        "respondida", artefacto.version, "novamercado-ventas", "advertencia")
    assert s.payload == p and {f for f, v in sup.items() if v} == {"bebe", "mascotas"}
    for x in r.json()["predicciones"]:
        assert preds[x["familia_id"]].cantidad_sugerida == x["cantidad_sugerida"]
        assert preds[x["familia_id"]].venta_esperada == x["venta_esperada"]
    assert n_cal == len(r.json()["calidad"]["observaciones"])


def test_observaciones_idempotentes_y_relacion_con_la_prediccion(cliente, bd):
    pred = cliente.post("/v1/prediccion", json=ej("prediccion_valida.json")).json()
    obs = ej("observacion_valida.json")
    assert cliente.post("/v1/observaciones", json=obs).json()["registros_guardados"] == 12
    assert cliente.post("/v1/observaciones", json=obs).json()["registros_guardados"] == 0
    obs["ventas"][3]["ventas"] += 7
    assert cliente.post("/v1/observaciones", json=obs).json()["registros_guardados"] == 1
    assert contar(bd, e.observaciones_historial) == 1 and contar(bd, e.envios_observacion) == 3
    with transaccion(bd) as c:
        v = leer_predicciones_reales(c).set_index("familia_id")
    reales = {x["familia_id"]: x["ventas"] for x in obs["ventas"]}
    for x in pred["predicciones"]:
        assert v.at[x["familia_id"], "ventas_reales"] == reales[x["familia_id"]]


def test_rechazos_quedan_registrados(cliente, bd):
    assert cliente.post("/v1/prediccion", json=ej("errores/E06_familia_repetida.json")).status_code == 422
    assert cliente.post("/v1/prediccion", content=b"{roto").status_code == 422
    assert cliente.post("/v1/observaciones", json={"tienda_id": 5}).status_code == 422
    with bd.connect() as c:
        rech = c.execute(select(e.solicitudes_prediccion.c.resultado, e.solicitudes_prediccion.c.payload)
                         .order_by(e.solicitudes_prediccion.c.solicitud_id)).all()
        env = c.execute(select(e.envios_observacion.c.resultado)).scalars().all()
    assert [x.resultado for x in rech] == ["rechazada", "rechazada"]
    assert rech[1].payload == {"cuerpo_no_json": "{roto"} and env == ["rechazado"]
    assert contar(bd, e.predicciones) == 0


def test_sin_fuga_temporal_con_la_base_real(cliente, bd, artefacto):
    """La base tiene ventas de F y posteriores: la API predice F solo con lo anterior."""
    F = "2026-04-01"
    p = ej("prediccion_valida.json")
    p["fecha"] = F
    antes = cliente.post("/v1/prediccion", json=p).json()
    obs = {"tienda_id": 5, "fecha": F, "ventas": [{"familia_id": "abarrotes", "ventas": 99999}]}
    cliente.post("/v1/observaciones", json=obs)
    assert cliente.post("/v1/prediccion", json=p).json() == antes
    # mismo resultado que el modelo con el histórico truncado en F
    h = DATOS.historico
    objetivos = pd.DataFrame({"fecha": pd.Timestamp(F), "tienda_id": 5,
                              "familia_id": [f["familia_id"] for f in p["familias"]],
                              "en_promocion": [f["en_promocion"] for f in p["familias"]]})
    directo = artefacto.pronosticador.predecir(objetivos, h[h["fecha"] < pd.Timestamp(F)]).set_index("familia_id")
    for x in antes["predicciones"]:
        assert x["cantidad_sugerida"] == directo.at[x["familia_id"], "cantidad_sugerida"]
        assert x["venta_esperada"] == round(directo.at[x["familia_id"], "venta_esperada"], 2)


def test_ventas_previas_estrictamente_anteriores(bd):
    with transaccion(bd) as c:
        v = leer_ventas_previas(c, 5, pd.Timestamp("2026-04-01").date(), dias=60)
    assert v["fecha"].max() < pd.Timestamp("2026-04-01")
    assert v["fecha"].min() >= pd.Timestamp("2026-04-01") - pd.Timedelta(days=60)
    with transaccion(bd) as c:                       # muy lejos: incluye la última venta de cada familia
        lejos = leer_ventas_previas(c, 5, pd.Timestamp("2027-01-01").date(), dias=60)
    assert len(lejos) == 12 and (lejos["fecha"] == DATOS.historico["fecha"].max()).all()


def test_base_caida_503_health_degradado_y_rechazo_422(artefacto):
    motor = crear_motor("postgresql+psycopg2://usuario@127.0.0.1:1/inexistente", timeout_conexion=2)
    d = _deps(motor, ProveedorMemoria(artefacto))
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 503
        assert c.post("/v1/observaciones", json=ej("observacion_valida.json")).status_code == 503
        assert c.post("/v1/prediccion", content=b"no json").status_code == 422
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["estado"] == "degradado"
        assert h.json()["motivo"] == "base de datos no disponible"
    d.monitor.detener()
    motor.dispose()


def test_entrenar_local_desde_la_base_sin_promociones_supuestas(cliente, bd, tmp_path):
    p = ej("prediccion_valida.json")                       # 2026-04-14, día sin ventas en el histórico
    p["familias"] = [f for f in p["familias"] if f["familia_id"] != "bebe"]      # bebe: supuesta
    assert cliente.post("/v1/prediccion", json=p).status_code == 200
    assert cliente.post("/v1/observaciones", json=ej("observacion_valida.json")).status_code == 200
    r = entrenar_desde_bd(bd, ConfigEntrenamiento(dias_calibracion=28, dias_minimos_historia=14,
                                                  params_hgb=dict(max_iter=20)))
    info = r.pronosticador.info
    assert info["corte"] == pd.Timestamp("2026-04-15")                     # día siguiente a la última venta
    assert info["filas_sin_promocion"] == 1                                # solo bebe (promoción supuesta)
    assert info["fecha_max_entrenamiento"] == pd.Timestamp("2026-04-14")   # las demás sí se entrenan
    ruta = tmp_path / "modelos" / "modelo_local.joblib"
    guardar_atomico(r.artefacto, ruta)
    m = ProveedorArchivo(ruta).obtener()
    assert m.version == r.version and m.version.startswith("local-20260415-")


def test_la_api_carga_el_artefacto_nuevo_sin_reiniciar(bd, artefacto, tmp_path):
    ruta = tmp_path / "modelo.joblib"
    d = _deps(bd, ProveedorArchivo(ruta))
    with TestClient(create_app(d, iniciar_hilos=False), raise_server_exceptions=False) as c:
        assert c.post("/v1/prediccion", json=ej("prediccion_valida.json")).status_code == 503
        guardar_atomico(artefacto.artefacto, ruta)
        d.gestor.refrescar()                               # lo hace el hilo del gestor cada 30 s
        r = c.post("/v1/prediccion", json=ej("prediccion_valida.json"))
        assert r.status_code == 200 and r.json()["version_modelo"] == artefacto.version
        assert c.get("/modelo").json()["version"] == artefacto.version
    d.monitor.detener()


def test_columnas_leidas_de_la_base_son_str(bd):
    """Regresión: SQLAlchemy entrega quoted_name y scikit-learn rechaza nombres mixtos."""
    from app.db.repositorio import leer_datos_base
    with transaccion(bd) as c:
        base = leer_datos_base(c)
        previas = leer_ventas_previas(c, 5, pd.Timestamp(FECHA).date())
    for df in (base.historico, base.tiendas, base.costos, base.feriados, previas):
        assert all(type(col) is str for col in df.columns)
