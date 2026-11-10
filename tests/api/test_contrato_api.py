"""Contrato HTTP de los seis endpoints (sin PostgreSQL ni MLflow)."""

import copy
import json
import logging
import random

import pandas as pd
import pytest

from app.schemas.contrato import RespuestaObservacion, RespuestaPrediccion
from tests.api.conftest import DATOS, FECHA
from tests.conftest import FIXTURES
from tests.ejemplos.test_ejemplos_catalogo import ESPERADO
from tests.unit.test_robustez import _mutar

EJEMPLOS = FIXTURES / "ejemplos"


def ejemplo(nombre):
    return json.loads((EJEMPLOS / nombre).read_text(encoding="utf-8"))


def valida():
    return ejemplo("prediccion_valida.json")


# ------------------------------------------------------------------ predicción
def test_prediccion_valida_cumple_el_contrato(cliente, deps, artefacto):
    r = cliente.post("/v1/prediccion", json=valida())
    assert r.status_code == 200
    cuerpo = r.json()
    RespuestaPrediccion.model_validate_json(r.text)
    assert list(cuerpo) == ["tienda_id", "fecha", "version_modelo", "predicciones", "calidad"]
    assert cuerpo["tienda_id"] == 5 and cuerpo["fecha"] == FECHA
    assert cuerpo["version_modelo"] == artefacto.version and isinstance(cuerpo["version_modelo"], str)
    assert [p["familia_id"] for p in cuerpo["predicciones"]] == [f["familia_id"] for f in valida()["familias"]]
    for p in cuerpo["predicciones"]:
        assert set(p) == {"familia_id", "venta_esperada", "cantidad_sugerida"}
        assert isinstance(p["cantidad_sugerida"], int) and p["cantidad_sugerida"] >= 0
        assert isinstance(p["venta_esperada"], float) and p["venta_esperada"] >= 0
        assert round(p["venta_esperada"], 2) == p["venta_esperada"]
    assert cuerpo["calidad"] == {"estado": "ok", "observaciones": []}
    guardada = deps.persistencia.solicitudes[-1]
    assert guardada["version_modelo"] == artefacto.version and guardada["payload"] == valida()


@pytest.mark.parametrize("archivo", sorted(k for k, v in ESPERADO.items() if v[0] == "p"))
def test_ejemplos_de_prediccion(cliente, deps, archivo):
    _, estado, acciones = ESPERADO[archivo]
    r = cliente.post("/v1/prediccion", json=ejemplo(archivo))
    if estado == "rechazo":
        assert r.status_code == 422 and set(r.json()) == {"detail"}
        assert "repetida con valores distintos" in r.json()["detail"]
        assert deps.persistencia.rechazos_prediccion[-1]["detalle"] == r.json()["detail"]
        assert deps.persistencia.solicitudes == []
        return
    assert r.status_code == 200
    calidad = r.json()["calidad"]
    assert calidad["estado"] == estado
    assert [(o["campo"], o["accion"]) for o in calidad["observaciones"]] == acciones
    assert len(r.json()["predicciones"]) == 12


@pytest.mark.parametrize("cuerpo, fragmento", [
    (b"{no es json", "no es JSON"), (b"", "no es JSON"), (b"[1, 2]", "objeto JSON"),
    (b'"texto"', "objeto JSON"), (b"null", "objeto JSON"),
])
def test_cuerpos_invalidos_422_y_registrados(cliente, deps, cuerpo, fragmento):
    r = cliente.post("/v1/prediccion", content=cuerpo, headers={"content-type": "application/json"})
    assert r.status_code == 422 and fragmento in r.json()["detail"]
    assert len(deps.persistencia.rechazos_prediccion) == 1


def test_cuerpo_no_json_se_guarda_truncado(cliente, deps):
    cliente.post("/v1/prediccion", content=b"x" * 5000)
    assert deps.persistencia.rechazos_prediccion[0]["payload"] == {"cuerpo_no_json": "x" * 2000}


def test_promociones_supuestas_y_atipicas(cliente, deps, artefacto):
    p = valida()
    maximos = artefacto.pronosticador.maximos_promocion
    p["familias"] = [f for f in p["familias"] if f["familia_id"] != "bebe"]       # supuesta (D1)
    p["familias"][0]["en_promocion"] = maximos["abarrotes"] + 20                  # atípica
    r = cliente.post("/v1/prediccion", json=p)
    assert r.status_code == 200
    obs = r.json()["calidad"]["observaciones"]
    assert any(o["campo"] == "familias[0].en_promocion" and "máximo" in o["detalle"] for o in obs)
    assert any(o["campo"] == "familias" and "'bebe'" in o["detalle"] for o in obs)
    guardada = deps.persistencia.solicitudes[-1]
    assert guardada["supuestas"] == {"bebe"}
    promos = {f.familia_id: f.en_promocion for f in guardada["solicitud"].familias}
    assert promos["abarrotes"] == maximos["abarrotes"] + 20                       # se conserva


@pytest.mark.parametrize("fecha, fragmento", [
    ("2026-04-17", "anterior al día previo"),        # faltan 3 días de observaciones
    ("2026-06-30", "no hay ventas reales en los 28 días previos"),
])
def test_advertencia_de_antiguedad(cliente, fecha, fragmento):
    p = valida()
    p["fecha"] = fecha
    r = cliente.post("/v1/prediccion", json=p)
    assert r.status_code == 200 and r.json()["calidad"]["estado"] == "advertencia"
    fechas = [o for o in r.json()["calidad"]["observaciones"] if o["campo"] == "fecha"]
    assert len(fechas) == 1 and fragmento in fechas[0]["detalle"]


def test_sin_ventas_previas_se_advierte(cliente):
    p = valida()
    p.update(tienda_id=10, fecha=DATOS.historico["fecha"].min().date().isoformat())
    r = cliente.post("/v1/prediccion", json=p)
    assert r.status_code == 200
    assert "no hay ventas reales anteriores" in r.json()["calidad"]["observaciones"][-1]["detalle"]


def test_dia_siguiente_al_ultimo_dato_no_advierte(cliente):
    assert cliente.post("/v1/prediccion", json=valida()).json()["calidad"]["estado"] == "ok"


def test_sin_fuga_temporal_en_la_api(artefacto):
    """Predecir F con datos posteriores a F en la base da lo mismo que sin ellos."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.api.conftest import construir
    from tests.api.memoria import PersistenciaMemoria
    F = "2026-04-01"
    p = valida()
    p["fecha"] = F
    completa = construir(artefacto)                                      # incluye ventas de F y posteriores
    truncada = PersistenciaMemoria(DATOS)
    truncada.historico = truncada.historico[truncada.historico["fecha"] < pd.Timestamp(F)]
    solo_pasado = construir(artefacto, persistencia=truncada)
    respuestas = []
    for d in (completa, solo_pasado):
        with TestClient(create_app(d, iniciar_hilos=False)) as c:
            respuestas.append(c.post("/v1/prediccion", json=p).json())
            c.post("/v1/observaciones", json={"tienda_id": 5, "fecha": F,
                                              "ventas": [{"familia_id": "abarrotes", "ventas": 99999}]})
            respuestas.append(c.post("/v1/prediccion", json=p).json())   # una observación de F no influye
    assert all(r == respuestas[0] for r in respuestas)


def test_solicitudes_repetidas_mismo_resultado_y_dos_registros(cliente, deps):
    a = cliente.post("/v1/prediccion", json=valida()).json()
    b = cliente.post("/v1/prediccion", json=valida()).json()
    assert a == b and len(deps.persistencia.solicitudes) == 2


def test_fuzzing_http_nunca_500(cliente):
    base = valida()
    for semilla in range(300):
        cuerpo = _mutar(copy.deepcopy(base), random.Random(semilla))
        r = cliente.post("/v1/prediccion", content=json.dumps(cuerpo, allow_nan=True),
                         headers={"content-type": "application/json"})
        assert r.status_code in (200, 422), (semilla, r.status_code, r.text)
        if r.status_code == 200:
            RespuestaPrediccion.model_validate_json(r.text)
        else:
            assert isinstance(r.json()["detail"], str)


# --------------------------------------------------------------- observaciones
def test_observaciones_idempotentes(cliente, deps):
    obs = ejemplo("observacion_valida.json")
    r1 = cliente.post("/v1/observaciones", json=obs)
    assert r1.status_code == 200 and r1.json() == {"registros_guardados": 12,
                                                   "calidad": {"estado": "ok", "observaciones": []}}
    assert cliente.post("/v1/observaciones", json=ejemplo("errores/E09_envio_repetido.json")).json()[
        "registros_guardados"] == 0
    obs["ventas"][0]["ventas"] += 1
    assert cliente.post("/v1/observaciones", json=obs).json()["registros_guardados"] == 1
    assert len(deps.persistencia.envios) == 3


@pytest.mark.parametrize("archivo", sorted(k for k, v in ESPERADO.items() if v[0] == "o"))
def test_ejemplos_de_observacion(cliente, archivo):
    _, estado, acciones = ESPERADO[archivo]
    r = cliente.post("/v1/observaciones", json=ejemplo(archivo))
    assert r.status_code == 200
    RespuestaObservacion.model_validate_json(r.text)
    assert r.json()["calidad"]["estado"] == estado
    assert [(o["campo"], o["accion"]) for o in r.json()["calidad"]["observaciones"]] == acciones


def test_observacion_rechazada(cliente, deps):
    r = cliente.post("/v1/observaciones", json={"tienda_id": 5, "fecha": FECHA, "ventas": []})
    assert r.status_code == 422 and r.json() == {"detail": "ventas: la lista está vacía"}
    assert deps.persistencia.rechazos_observacion[0]["detalle"] == r.json()["detail"]


# ------------------------------------------------------- modelo, salud y admin
def test_modelo_vigente(cliente, artefacto):
    r = cliente.get("/modelo")
    assert r.status_code == 200
    assert r.json() == {"nombre": "novamercado-ventas", "alias": "local", "version": artefacto.version}


def test_health_ok(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "ok" and "motivo" not in r.json()
    assert r.json()["componentes"] == {"api": "ok", "modelo": "ok", "base_datos": "ok",
                                       "mlflow": "no_configurado", "airflow": "no_configurado"}


def test_sin_modelo_503_y_health_503(cliente_sin_modelo):
    c, d = cliente_sin_modelo
    r = c.post("/v1/prediccion", json=valida())
    assert r.status_code == 503 and "no hay un modelo disponible" in r.json()["detail"]
    assert d.persistencia.solicitudes == []
    assert c.get("/modelo").status_code == 503
    h = c.get("/health")
    assert h.status_code == 503 and h.json()["estado"] == "degradado" and "modelo" in h.json()["motivo"]
    assert c.post("/v1/observaciones", json=ejemplo("observacion_valida.json")).status_code == 200


def test_base_caida(cliente, deps, caplog):
    deps.persistencia.caida = True
    r = cliente.post("/v1/prediccion", json=valida())
    assert r.status_code == 503 and r.json() == {"detail": "la base de datos no está disponible"}
    assert cliente.post("/v1/observaciones", json=ejemplo("observacion_valida.json")).status_code == 503
    deps.monitor.ejecutar_ciclo()
    h = cliente.get("/health")
    assert h.status_code == 200 and h.json()["estado"] == "degradado"
    assert h.json()["motivo"] == "base de datos no disponible"
    deps.persistencia.caida = False
    deps.monitor.ejecutar_ciclo()
    assert cliente.get("/health").json()["estado"] == "ok"


def test_rechazo_con_base_caida_sigue_siendo_422_sin_exponer_datos(cliente, deps, caplog):
    deps.persistencia._catalogo_previo = deps.persistencia.catalogo()          # catálogo ya en memoria
    deps.persistencia.catalogo = lambda: deps.persistencia._catalogo_previo
    deps.persistencia.caida = True
    secreto = "DATO-SENSIBLE-123"
    with caplog.at_level(logging.ERROR):
        r = cliente.post("/v1/prediccion", json={"tienda_id": secreto})
        r2 = cliente.post("/v1/observaciones", content=secreto.encode())
    assert r.status_code == 422 and r2.status_code == 422
    assert "no se pudo registrar una solicitud rechazada" in caplog.text
    assert secreto not in caplog.text


def test_requerir_componente_sin_verificacion_degrada(artefacto):
    from tests.api.conftest import construir
    d = construir(artefacto, requeridos=("api", "base_datos", "modelo", "mlflow"))
    diag = d.monitor.diagnostico()
    assert diag.estado == "degradado" and diag.motivo == "MLflow no disponible" and diag.codigo_http == 200


def test_admin_sin_orquestador(cliente):
    r = cliente.post("/admin/reentrenar")
    assert r.status_code == 503 and "Airflow" in r.json()["detail"]
    r = cliente.get("/admin/reentrenar/ejecucion-x")
    assert r.status_code == 404 and set(r.json()) == {"detail"}


def test_error_inesperado_500_generico(cliente, monkeypatch):
    import app.main as main

    def explota(*a, **k):
        raise RuntimeError("detalle interno que no debe salir")

    monkeypatch.setattr(main, "predecir", explota)
    r = cliente.post("/v1/prediccion", json=valida())
    assert r.status_code == 500 and r.json() == {"detail": "error interno del servicio"}


def test_ruta_inexistente_404(cliente):
    r = cliente.get("/no-existe")
    assert r.status_code == 404 and set(r.json()) == {"detail"}
