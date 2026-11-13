"""Monitor de salud: verificaciones concurrentes, límite global y tiempos de transición."""

import threading
import time
from pathlib import Path

import pytest

from app.config import ConfiguracionInvalida, componentes_requeridos
from app.salud import MonitorSalud

REQ = ("api", "base_datos", "modelo")


def monitor(verificaciones, modelo=True, requeridos=REQ, limite=1.0):
    return MonitorSalud(verificaciones=verificaciones, requeridos=requeridos, modelo_cargado=lambda: modelo,
                        intervalo=0.2, limite=limite)


def test_verificaciones_en_paralelo_con_limite_global():
    lento = lambda: time.sleep(0.4) or True                 # noqa: E731
    m = monitor({"base_datos": lento, "mlflow": lento, "airflow": lento}, limite=1.0)
    t = time.monotonic()
    res = m.ejecutar_ciclo()
    assert time.monotonic() - t < 0.9                         # 3 × 0.4 s en serie serían 1.2 s
    assert res == {"base_datos": "ok", "mlflow": "ok", "airflow": "ok"}
    m.detener()


def test_verificacion_colgada_cuenta_como_falla_y_no_se_relanza():
    soltar = threading.Event()
    llamadas = []

    def colgada():
        llamadas.append(1)
        soltar.wait(10)
        return True

    m = monitor({"base_datos": colgada}, limite=0.3)
    t = time.monotonic()
    assert m.ejecutar_ciclo() == {"base_datos": "falla"}
    assert time.monotonic() - t < 1.0                         # el límite global corta la espera
    assert m.ejecutar_ciclo() == {"base_datos": "falla"}      # sigue colgada: no se lanza otra
    assert len(llamadas) == 1
    soltar.set()
    time.sleep(0.1)
    assert m.ejecutar_ciclo() == {"base_datos": "ok"}         # se recupera
    m.detener()


def test_excepcion_en_verificacion_es_falla():
    def explota():
        raise ConnectionError("x")
    m = monitor({"base_datos": explota})
    assert m.ejecutar_ciclo() == {"base_datos": "falla"}
    m.detener()


def test_diagnostico_ok_degradado_y_503():
    estado = {"bd": True}
    m = monitor({"base_datos": lambda: estado["bd"]})
    assert m.diagnostico().motivo == "base_datos: verificación en curso"    # antes del primer ciclo
    m.ejecutar_ciclo()
    d = m.diagnostico()
    assert (d.estado, d.motivo, d.codigo_http) == ("ok", None, 200)
    assert d.componentes == {"api": "ok", "modelo": "ok", "base_datos": "ok",
                             "mlflow": "no_configurado", "airflow": "no_configurado"}
    estado["bd"] = False
    m.ejecutar_ciclo()
    d = m.diagnostico()
    assert (d.estado, d.motivo, d.codigo_http) == ("degradado", "base de datos no disponible", 200)
    m.detener()


def test_sin_modelo_responde_503():
    m = monitor({"base_datos": lambda: True}, modelo=False)
    m.ejecutar_ciclo()
    d = m.diagnostico()
    assert d.codigo_http == 503 and d.estado == "degradado" and "modelo" in d.motivo
    m.detener()


def test_varios_problemas_en_el_motivo():
    m = monitor({"base_datos": lambda: False}, modelo=False, requeridos=("api", "base_datos", "modelo", "airflow"))
    m.ejecutar_ciclo()
    assert m.diagnostico().motivo == "base de datos no disponible; no hay un modelo cargado; Airflow no disponible"
    m.detener()


def test_hilo_en_segundo_plano_refleja_caida_y_recuperacion():
    estado = {"bd": True}
    m = monitor({"base_datos": lambda: estado["bd"]})
    m.iniciar()
    try:
        assert m.diagnostico().estado == "ok"
        estado["bd"] = False
        _esperar(lambda: m.diagnostico().estado == "degradado", 2)
        estado["bd"] = True
        _esperar(lambda: m.diagnostico().estado == "ok", 2)
    finally:
        m.detener()


def _esperar(condicion, segundos):
    fin = time.monotonic() + segundos
    while time.monotonic() < fin:
        if condicion():
            return
        time.sleep(0.05)
    pytest.fail("la condición no se cumplió a tiempo")


def test_tiempos_por_defecto_cumplen_el_contrato(monkeypatch):
    from app.dependencias import construir_monitor
    from app.orquestador import OrquestadorNoConfigurado
    for v in ("SALUD_INTERVALO", "SALUD_LIMITE"):
        monkeypatch.delenv(v, raising=False)

    class P:
        disponible = staticmethod(lambda: True)

    class G:
        cargado = staticmethod(lambda: True)

    m = construir_monitor(P(), G(), OrquestadorNoConfigurado(), requeridos=REQ)
    # peor caso de transición = intervalo + límite; contrato: < 60 s (degradar) y < 120 s (volver)
    assert m.intervalo + m.limite < 60
    m.detener()


def test_componentes_requeridos(monkeypatch):
    monkeypatch.delenv("COMPONENTES_REQUERIDOS", raising=False)
    monkeypatch.setenv("MODELO_ORIGEN", "archivo")
    assert componentes_requeridos() == ("api", "base_datos", "modelo")
    monkeypatch.setenv("MODELO_ORIGEN", "mlflow")                     # producto: también MLflow (M10)
    assert componentes_requeridos() == ("api", "base_datos", "modelo", "mlflow")
    monkeypatch.delenv("MODELO_ORIGEN")
    assert componentes_requeridos() == ("api", "base_datos", "modelo", "mlflow")
    monkeypatch.setenv("COMPONENTES_REQUERIDOS", "base_datos, modelo,mlflow,airflow")
    assert componentes_requeridos() == ("api", "base_datos", "modelo", "mlflow", "airflow")
    monkeypatch.setenv("COMPONENTES_REQUERIDOS", "base_datos,redis")
    with pytest.raises(ConfiguracionInvalida, match="redis"):
        componentes_requeridos()


def _variables_env_example() -> dict[str, str]:
    raiz = Path(__file__).resolve().parents[2]
    variables = {}
    for linea in (raiz / ".env.example").read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            variables[clave.strip()] = valor.strip()
    return variables


@pytest.mark.parametrize("origen, requeridos", [
    ("mlflow", ("api", "base_datos", "modelo", "mlflow")),
    ("archivo", ("api", "base_datos", "modelo")),
])
def test_env_example_deja_que_modelo_origen_decida_si_mlflow_es_requerido(monkeypatch, origen, requeridos):
    variables = _variables_env_example()
    assert "COMPONENTES_REQUERIDOS" not in variables       # definirla anularía la selección automática
    assert variables["MODELO_ORIGEN"] == "mlflow"
    monkeypatch.delenv("COMPONENTES_REQUERIDOS", raising=False)
    for clave, valor in variables.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.setenv("MODELO_ORIGEN", origen)
    assert componentes_requeridos() == requeridos


def test_modelo_desactualizado_degrada_sin_503():
    estado = {"actualizado": False}
    m = MonitorSalud(verificaciones={"base_datos": lambda: True}, requeridos=REQ, modelo_cargado=lambda: True,
                     modelo_actualizado=lambda: estado["actualizado"], intervalo=0.2, limite=1)
    m.ejecutar_ciclo()
    d = m.diagnostico()
    assert (d.estado, d.codigo_http) == ("degradado", 200)
    assert d.motivo == "no se pudo actualizar el modelo vigente" and d.componentes["modelo"] == "desactualizado"
    estado["actualizado"] = True
    assert m.diagnostico().estado == "ok"
    m.detener()
