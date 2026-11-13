from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas.contrato import (
    Calidad,
    EstadoReentrenamiento,
    EstadoSalud,
    ModeloVigente,
    RespuestaObservacion,
    RespuestaPrediccion,
    SolicitudPrediccion,
)

OBS_REPARAR = {"campo": "fecha", "accion": "reparar", "detalle": "x"}
OBS_ADVERTIR = {"campo": "familias", "accion": "advertir", "detalle": "y"}


@pytest.mark.parametrize("estado, obs", [
    ("ok", []),
    ("corregido", [OBS_REPARAR]),
    ("advertencia", [OBS_ADVERTIR]),
    ("advertencia", [OBS_REPARAR, OBS_ADVERTIR]),
])
def test_calidad_coherente(estado, obs):
    Calidad.model_validate({"estado": estado, "observaciones": obs})


@pytest.mark.parametrize("estado, obs", [
    ("ok", [OBS_REPARAR]),
    ("corregido", []),
    ("corregido", [OBS_ADVERTIR]),
    ("advertencia", [OBS_REPARAR]),
    ("bien", []),
])
def test_calidad_incoherente(estado, obs):
    with pytest.raises(ValidationError):
        Calidad.model_validate({"estado": estado, "observaciones": obs})


def test_accion_invalida():
    with pytest.raises(ValidationError):
        Calidad.model_validate({"estado": "corregido",
                                "observaciones": [{"campo": "a", "accion": "rechazar", "detalle": "b"}]})


def _respuesta(**cambios):
    base = {
        "tienda_id": 5, "fecha": date(2025, 11, 17), "version_modelo": "3",
        "predicciones": [{"familia_id": "abarrotes", "venta_esperada": 612.4, "cantidad_sugerida": 640}],
        "calidad": {"estado": "ok", "observaciones": []},
    }
    base.update(cambios)
    return base


def test_respuesta_prediccion_serializa_como_el_contrato():
    r = RespuestaPrediccion.model_validate(_respuesta())
    js = r.model_dump(mode="json")
    assert js["fecha"] == "2025-11-17"
    assert js["version_modelo"] == "3"
    # campos del Anexo más el agregado de la etapa 6 (se puede agregar, nunca omitir)
    assert set(js) == {"tienda_id", "fecha", "version_modelo", "predicciones", "calidad", "modelo_es_vigente"}
    assert js["modelo_es_vigente"] is True
    assert set(js["predicciones"][0]) == {"familia_id", "venta_esperada", "cantidad_sugerida"}


@pytest.mark.parametrize("cambio", [
    {"version_modelo": 3},
    {"predicciones": [{"familia_id": "a", "venta_esperada": 1.0, "cantidad_sugerida": -1}]},
    {"predicciones": [{"familia_id": "a", "venta_esperada": 1.0, "cantidad_sugerida": 1.5}]},
    {"predicciones": [{"familia_id": "a", "venta_esperada": -1.0, "cantidad_sugerida": 1}]},
])
def test_respuesta_prediccion_fuera_de_contrato(cambio):
    with pytest.raises(ValidationError):
        RespuestaPrediccion.model_validate(_respuesta(**cambio))


def test_solicitud_estricta_no_acepta_texto():
    """Pydantic valida el contrato *después* de la limpieza: no hace conversiones implícitas."""
    with pytest.raises(ValidationError):
        SolicitudPrediccion.model_validate({"tienda_id": "5", "fecha": date(2025, 1, 1),
                                            "familias": [{"familia_id": "a", "en_promocion": 0}]})


def test_respuesta_observacion():
    RespuestaObservacion.model_validate({"registros_guardados": 0, "calidad": {"estado": "ok", "observaciones": []}})
    with pytest.raises(ValidationError):
        RespuestaObservacion.model_validate({"registros_guardados": -1, "calidad": {"estado": "ok"}})


def test_estado_reentrenamiento():
    assert EstadoReentrenamiento(estado="en_curso").como_respuesta() == {"estado": "en_curso"}
    fin = EstadoReentrenamiento(estado="exitoso", promovido=True, version="4", motivo="m")
    assert fin.como_respuesta() == {"estado": "exitoso", "promovido": True, "version": "4", "motivo": "m"}
    with pytest.raises(ValidationError):
        EstadoReentrenamiento(estado="fallido")
    with pytest.raises(ValidationError):
        EstadoReentrenamiento(estado="en_curso", promovido=False)
    with pytest.raises(ValidationError):
        EstadoReentrenamiento(estado="pendiente")


def test_modelo_y_salud():
    ModeloVigente(nombre="ventas", alias="vigente", version="3")
    with pytest.raises(ValidationError):
        ModeloVigente(nombre="ventas", alias="vigente", version=3)
    EstadoSalud(estado="ok")
    EstadoSalud(estado="degradado", motivo="MLflow no disponible", componentes={"mlflow": False})
    with pytest.raises(ValidationError):
        EstadoSalud(estado="degradado")


def test_modelo_vigente_estados_de_vigencia_coherentes():
    from app.schemas.contrato import ModeloVigente
    base = {"nombre": "m", "alias": "vigente", "version": "1"}
    assert ModeloVigente(**base).model_dump(exclude_none=True) == {**base, "es_vigente": True,
                                                                   "estado_vigencia": "vigente"}
    a = ModeloVigente(**base, es_vigente=False, estado_vigencia="actualizando", version_vigente="2")
    assert a.version == "1" and a.version_vigente == "2"
    for malo in ({"es_vigente": True, "estado_vigencia": "anterior"},
                 {"es_vigente": False, "estado_vigencia": "vigente"},
                 {"es_vigente": False, "estado_vigencia": "otro"}):
        with pytest.raises(ValueError):
            ModeloVigente(**base, **malo)
