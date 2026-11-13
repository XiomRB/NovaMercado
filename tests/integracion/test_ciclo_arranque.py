"""Supervisor + proceso hijo y arranque (plan §4.2, §4.3 y §6) con PostgreSQL, MLflow y
procesos reales: primer modelo, reinicio sin entrenar, respaldo baseline solo por C1–C3,
arranques concurrentes, suspensión del supervisor o de ambos, y PDEATHSIG.

Las pruebas que inyectan fallos dentro del hijo (T3/T4a/T4b, excepción) sustituyen el
lanzamiento del proceso por la ejecución en el proceso de la prueba (``lanzar_hijo``
reemplazado), porque monkeypatch no cruza procesos; el resto del supervisor es el real."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from sqlalchemy import text

import app.modelo.ciclo_entrenamiento as ciclo
import app.modelo.registro as reg
from app.modelo.bloqueo import ConcesionPerdida
from tests.integracion.apoyo_mlflow import CONFIG, EVALUACION, cargar, cliente, tags_version, url_de, vigente_pg

RAIZ = Path(__file__).resolve().parents[2]
MALA = {**asdict(CONFIG), "params_hgb": {"max_iter": 1}}            # HGB de 1 iteración: no supera C1–C3
LENTA = {**asdict(CONFIG), "params_hgb": {"max_iter": 3000, "early_stopping": False}}


@pytest.fixture
def bd(motor, tmp_path):
    cargar(motor, tmp_path / "raw")
    return motor


def ejecutar(bd, uri, nombre, modo="arranque", config=None, **kw):
    return ciclo.ejecutar_entrenamiento(url_bd=url_de(bd), tracking_uri=uri, nombre=nombre, alias="vigente",
                                        modo=modo, git_commit="commit-de-prueba",
                                        config=config or asdict(CONFIG), evaluacion=EVALUACION.como_dict(), **kw)


def ejecuciones(uri):
    c = cliente(uri)
    exp = c.get_experiment_by_name(reg.EXPERIMENTO)
    return [] if exp is None else c.search_runs([exp.experiment_id])


def ejecuciones_de(uri, nombre):
    return [r for r in ejecuciones(uri) if r.data.tags.get("version_registrada")
            and cliente(uri).get_model_version(nombre, r.data.tags["version_registrada"]).run_id == r.info.run_id
            and any(v.run_id == r.info.run_id for v in cliente(uri).search_model_versions(f"name='{nombre}'"))]


# ------------------------------------------------------------- arranque normal
def test_arranque_desde_cero_y_reinicio_sin_repetir(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = ejecutar(bd, uri, nombre_modelo)
    assert r.codigo == 0 and r.promovido and r.version_vigente == "1"
    assert vigente_pg(bd, nombre_modelo).version == "1"
    n = len(cliente(uri).search_model_versions(f"name='{nombre_modelo}'"))
    t = time.monotonic()
    r2 = ejecutar(bd, uri, nombre_modelo)                           # cada reinicio del producto
    assert r2.codigo == 0 and not r2.promovido and "ya existe" in r2.mensaje and r2.version_vigente == "1"
    assert time.monotonic() - t < 30
    assert len(cliente(uri).search_model_versions(f"name='{nombre_modelo}'")) == n   # no entrenó


def test_respaldo_baseline_si_el_candidato_falla_solo_c1_c3(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = ejecutar(bd, uri, nombre_modelo, config=MALA)
    assert r.codigo == 0 and r.promovido and [d["criterio"] for d in r.detalles] == ["candidato", "respaldo"]
    cand, resp = r.detalles
    assert not cand["autorizada"] and all(cand["condiciones"][k] for k in reg.TECNICOS)
    assert not all(cand["condiciones"][k] for k in reg.COMERCIALES)
    assert resp["promovido"] and resp["condiciones"] == {"T1": True, "T2": True, "D1": True, "R1": True,
                                                         "T3": True, "T4a": True, "T4b": True}
    v = vigente_pg(bd, nombre_modelo)
    assert v.version == resp["version"] and v.run_id == resp["run_id"]
    tc, tr = tags_version(uri, nombre_modelo, cand["version"]), tags_version(uri, nombre_modelo, resp["version"])
    assert tc["estado"] == "invalido" and tr["estado"] == "valido"
    assert tr["tipo"] == "respaldo_baseline" and tr["run_candidato"] == cand["run_id"]
    assert tr["promovido_por"] == "arranque_respaldo:si_no_hay"
    # misma copia inmutable de datos: los archivos son idénticos (la huella del manifiesto
    # difiere porque registra la configuración, que aquí tiene modelo=baseline_dow4)
    from app.modelo.mlflow_util import descargar_de_ejecucion
    mc, mr = (json.loads(Path(descargar_de_ejecucion(uri, x["run_id"], "datos/manifiesto.json")).read_text())
              for x in (cand, resp))
    assert mc["archivos"] == mr["archivos"] and mc["corte"] == mr["corte"]
    assert mr["config"]["modelo"] == "baseline_dow4" and mr["modelos_evaluados"] == ["baseline_dow4"]
    run = cliente(uri).get_run(resp["run_id"])
    assert run.data.tags["modelo"] == "baseline_dow4" and run.data.tags["origen"] == "arranque_respaldo"


def _hijo_en_proceso(monkeypatch, bd):
    """Sustituye el proceso hijo por la ejecución en el proceso de la prueba."""
    def lanzar(c, url_bd, entrada, **kw):
        comprobar = c.verificar
        comunes = dict(tracking_uri=entrada["tracking_uri"], nombre=entrada["nombre"],
                       git_commit=entrada["git_commit"], epoca=c.epoca, comprobar=comprobar,
                       etiquetas=entrada.get("etiquetas") or {})
        if entrada["tarea"] == "candidato":
            return reg.registrar_version(bd, origen=entrada["origen"],
                                         config=reg.ConfigEntrenamiento(**entrada["config"]),
                                         evaluacion=reg.ConfigEvaluacion(**entrada["evaluacion"]), **comunes)
        return reg.registrar_respaldo(run_candidato=entrada["run_candidato"], **comunes)
    monkeypatch.setattr(ciclo, "lanzar_hijo", lanzar)


@pytest.mark.parametrize("falla", ["T3", "T4b", "T4a"])
def test_falla_tecnica_no_activa_el_respaldo(bd, servidor_mlflow, nombre_modelo, monkeypatch, falla):
    uri = servidor_mlflow.uri
    _hijo_en_proceso(monkeypatch, bd)
    if falla == "T4a":
        monkeypatch.setattr(reg, "verificar_t4a", lambda *a, **k: ["costo distinto (inyectado)"])
    else:
        monkeypatch.setattr(reg, "verificar_t3_t4b", lambda *a, **k: ("humo (inyectado)", None) if falla == "T3"
                            else (None, "SHA (inyectado)"))
    r = ejecutar(bd, uri, nombre_modelo, config=MALA)              # además falla C1–C3
    assert r.codigo == 3 and not r.promovido and [d["criterio"] for d in r.detalles] == ["candidato"]
    assert vigente_pg(bd, nombre_modelo) is None
    assert len(cliente(uri).search_model_versions(f"name='{nombre_modelo}'")) == 1   # sin ejecución de respaldo


def test_d1_no_activa_el_respaldo(motor, tmp_path, servidor_mlflow, nombre_modelo, monkeypatch):
    import pandas as pd

    from ml.datos import DatosBase
    from tests.integracion.apoyo_mlflow import DATOS
    corto = DATOS.historico[DATOS.historico["fecha"] >= DATOS.historico["fecha"].max() - pd.Timedelta(days=40)]
    cargar(motor, tmp_path / "raw", DatosBase(corto, DATOS.tiendas, DATOS.familias, DATOS.costos, DATOS.feriados))
    r = ejecutar(motor, servidor_mlflow.uri, nombre_modelo)
    assert r.codigo == 3 and len(r.detalles) == 1 and vigente_pg(motor, nombre_modelo) is None


def test_excepcion_o_mlflow_caido_no_activa_el_respaldo(bd, servidor_mlflow, nombre_modelo, monkeypatch):
    _hijo_en_proceso(monkeypatch, bd)

    def caido(*a, **k):
        raise ConnectionError("MLflow no responde (inyectado)")
    monkeypatch.setattr(reg, "evaluar", caido)
    with pytest.raises(ConnectionError):
        ejecutar(bd, servidor_mlflow.uri, nombre_modelo, config=MALA)
    assert vigente_pg(bd, nombre_modelo) is None
    with bd.connect() as c:
        assert c.execute(text("SELECT count(*) FROM decisiones_promocion")).scalar() == 0
        assert c.execute(text("SELECT latido FROM bloqueo_entrenamiento")).scalar() is None   # liberada


def test_el_respaldo_tambien_falla_no_hay_vigente(bd, servidor_mlflow, nombre_modelo, monkeypatch):
    uri = servidor_mlflow.uri
    _hijo_en_proceso(monkeypatch, bd)
    real = reg.verificar_t3_t4b
    llamadas = []

    def t3(*a, **k):
        llamadas.append(1)
        return ("humo del baseline (inyectado)", None) if len(llamadas) > 1 else real(*a, **k)
    monkeypatch.setattr(reg, "verificar_t3_t4b", t3)
    r = ejecutar(bd, uri, nombre_modelo, config=MALA)
    assert r.codigo == 3 and not r.promovido and [d["criterio"] for d in r.detalles] == ["candidato", "respaldo"]
    assert vigente_pg(bd, nombre_modelo) is None and r.version_vigente is None
    for v in cliente(uri).search_model_versions(f"name='{nombre_modelo}'"):
        assert dict(v.tags)["estado"] == "invalido"
    monkeypatch.undo()                                              # sin fallos inyectados (hijo real)
    assert ejecutar(bd, uri, nombre_modelo).codigo == 0             # el siguiente arranque reintenta


# --------------------------------------------------------------- procesos reales
def supervisor(bd, uri, nombre, modo="arranque", config=None):
    args = dict(url_bd=url_de(bd), tracking_uri=uri, nombre=nombre, alias="vigente", modo=modo,
                git_commit="commit-de-prueba", config=config or asdict(CONFIG), evaluacion=EVALUACION.como_dict())
    return subprocess.Popen([sys.executable, "-m", "tests.integracion.procesos", "supervisor", json.dumps(args)],
                            cwd=RAIZ, env={**os.environ, "PYTHONPATH": str(RAIZ)}, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def hijos(pid):
    try:
        return [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []


def vivo(pid):
    try:
        estado = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return estado != "Z"
    except OSError:
        return False


def esperar_hijo(p, limite=120):
    t = time.monotonic()
    while time.monotonic() - t < limite:
        h = hijos(p.pid)
        if h:
            return h[0]
        assert p.poll() is None, p.stderr.read()[-2000:]
        time.sleep(0.2)
    raise AssertionError("el supervisor no lanzó el hijo")


def test_dos_arranques_concurrentes_entrena_uno_solo(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    a, b = supervisor(bd, uri, nombre_modelo), supervisor(bd, uri, nombre_modelo)
    salidas = [p.communicate(timeout=600) for p in (a, b)]
    assert a.returncode == b.returncode == 0, [s[1][-1500:] for s in salidas]
    resultados = [json.loads(s[0].split("RESULTADO ", 1)[1]) for s in salidas]
    assert sorted(r["promovido"] for r in resultados) == [False, True]
    assert any("ya existe" in r["mensaje"] for r in resultados)
    assert len(cliente(uri).search_model_versions(f"name='{nombre_modelo}'")) == 1   # un solo entrenamiento
    assert vigente_pg(bd, nombre_modelo).version == "1"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc y PDEATHSIG")
def test_sigkill_del_supervisor_mata_al_hijo(bd, servidor_mlflow, nombre_modelo):
    p = supervisor(bd, servidor_mlflow.uri, nombre_modelo, modo="manual", config=LENTA)
    h = esperar_hijo(p)
    time.sleep(2)
    p.send_signal(signal.SIGKILL)
    p.wait(30)
    t = time.monotonic()
    while vivo(h):
        assert time.monotonic() - t < 5, "el hijo sobrevivió al supervisor"
        time.sleep(0.1)
    assert vigente_pg(bd, nombre_modelo) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc y señales")
def test_supervisor_suspendido_el_hijo_se_detiene_solo(bd, servidor_mlflow, nombre_modelo):
    """§4.3: con el supervisor suspendido (sin latidos), el vigilante propio del hijo lo
    termina en ≤ 42 s. Al reanudarse, el supervisor detecta la pérdida y sale con 6."""
    p = supervisor(bd, servidor_mlflow.uri, nombre_modelo, modo="manual", config=LENTA)
    h = esperar_hijo(p)
    time.sleep(3)
    p.send_signal(signal.SIGSTOP)
    t = time.monotonic()
    try:
        while vivo(h):
            assert time.monotonic() - t < 50, "el hijo siguió trabajando sin latidos"
            time.sleep(0.5)
        detenido_en = time.monotonic() - t
        assert 25 < detenido_en <= 45, detenido_en                   # latido > 30 s (+ sondeo y límite)
    finally:
        p.send_signal(signal.SIGCONT)
    _, err = p.communicate(timeout=120)
    assert p.returncode == 6, err[-2000:]
    assert vigente_pg(bd, nombre_modelo) is None
    with bd.connect() as c:
        assert c.execute(text("SELECT count(*) FROM resoluciones_promocion")).scalar() == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc y señales")
def test_suspension_de_supervisor_e_hijo_mas_alla_del_vencimiento(bd, servidor_mlflow, nombre_modelo):
    """Fuera del modelo cubierto en tiempo: con ambos suspendidos, la sesión del supervisor
    sigue viva (el núcleo responde los keepalives) y su advisory lock impide que otro
    proceso empiece. Al reanudarse, ambos detectan la pérdida (latido vencido) y salen sin
    autorizar nada (S1–S3)."""
    from tests.integracion.test_bloqueo import intentar
    p = supervisor(bd, servidor_mlflow.uri, nombre_modelo, modo="manual", config=LENTA)
    h = esperar_hijo(p)
    time.sleep(3)
    os.kill(h, signal.SIGSTOP)
    p.send_signal(signal.SIGSTOP)
    try:
        time.sleep(62)
        codigo, salida, _ = intentar(url_de(bd), espera=1)
        assert codigo == 5                                          # nadie más puede empezar
    finally:
        p.send_signal(signal.SIGCONT)
        os.kill(h, signal.SIGCONT)
    _, err = p.communicate(timeout=120)
    assert p.returncode == 6, err[-2000:]
    t = time.monotonic()
    while vivo(h):
        assert time.monotonic() - t < 15
        time.sleep(0.2)
    assert vigente_pg(bd, nombre_modelo) is None
    codigo, _, _ = intentar(url_de(bd), espera=5)                   # ahora sí: la concesión venció
    assert codigo == 0


def test_lanzar_hijo_sale_con_concesion_perdida_si_el_hijo_lo_detecta(bd, servidor_mlflow, nombre_modelo):
    """Si la concesión desaparece mientras el hijo trabaja (p. ej. la sesión del supervisor
    se termina), el supervisor detiene al hijo y lanza ConcesionPerdida."""
    import threading
    resultado = {}

    def correr():
        try:
            ejecutar(bd, servidor_mlflow.uri, nombre_modelo, modo="manual", config=LENTA)
        except ConcesionPerdida as exc:
            resultado["perdida"] = exc
    hilo = threading.Thread(target=correr)
    hilo.start()
    t = time.monotonic()
    while True:                                                     # espera a que el hijo esté trabajando
        with bd.connect() as c:
            pid = c.execute(text("SELECT pid FROM pg_stat_activity WHERE application_name = "
                                 "'novamercado-entrenamiento' AND pid <> pg_backend_pid() ORDER BY backend_start "
                                 "LIMIT 1")).scalar()
            hijo_activo = c.execute(text("SELECT count(*) FROM pg_stat_activity WHERE application_name = "
                                         "'novamercado-entrenamiento'")).scalar() >= 3
        if pid and hijo_activo:
            break
        assert time.monotonic() - t < 120
        time.sleep(0.5)
    with bd.connect() as c:
        c.execute(text("SELECT pg_terminate_backend(:p)"), {"p": pid})   # la sesión del supervisor
    hilo.join(timeout=60)
    assert "perdida" in resultado and vigente_pg(bd, nombre_modelo) is None
