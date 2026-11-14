"""Modo ``reentrenamiento`` del supervisor y CLI ``reentrenar`` con PostgreSQL, MLflow y
procesos reales (etapa 7, DECISIONES.md §16): reclamación atómica, R2 contra el vigente
de PostgreSQL, promoción ``si_mejora`` sin reiniciar nada, excepción del respaldo
baseline (E7-6b), plazo del hijo, ventana de promoción, SIGTERM y escritura del
resultado con el rol ``nm_cierre``.

El vigente se entrena con el histórico sin sus últimos 14 días; luego esos días "llegan"
(se insertan en ``ventas_historicas``, tabla de solo inserción) y el reentrenamiento los
usa: así R2 tiene una ventana W de 14 días posteriores a ``corte_v``."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

import app.modelo.ciclo_entrenamiento as ciclo
from app.db.roles import configurar_roles
from app.modelo.bloqueo import ConcesionEntrenamiento, ReclamacionRechazada
from ml.datos import DatosBase
from tests.integracion.apoyo_mlflow import CONFIG, DATOS, EVALUACION, cargar, cliente, tags_version, url_de, vigente_pg

RAIZ = Path(__file__).resolve().parents[2]
CLAVE_CIERRE = "clave-cierre-prueba"
NUEVOS = 14
FIN_VIGENTE = DATOS.historico["fecha"].max() - pd.Timedelta(days=NUEVOS)
MALA = {**asdict(CONFIG), "params_hgb": {"max_iter": 1}}


@pytest.fixture
def bd(motor, tmp_path):
    h = DATOS.historico
    cargar(motor, tmp_path / "raw", DatosBase(h[h["fecha"] <= FIN_VIGENTE], DATOS.tiendas, DATOS.familias,
                                               DATOS.costos, DATOS.feriados))
    configurar_roles(motor, CLAVE_CIERRE, "clave-vigilante-prueba")
    return motor


def llegan_los_datos_nuevos(motor):
    nuevos = DATOS.historico[DATOS.historico["fecha"] > FIN_VIGENTE]
    with motor.begin() as c:
        carga = c.execute(text("SELECT carga_id FROM cargas_datos WHERE archivo = 'historico_ventas.csv'")).scalar()
        c.execute(text("INSERT INTO ventas_historicas (fecha, tienda_id, familia_id, ventas, en_promocion, carga_id) "
                       "VALUES (:f, :t, :fa, :v, :p, :c)"),
                  [{"f": r.fecha.date(), "t": int(r.tienda_id), "fa": r.familia_id, "v": int(r.ventas),
                    "p": int(r.en_promocion), "c": carga} for r in nuevos.itertuples()])


def solicitud(motor, iniciar=True):
    id_ = f"nm-{uuid.uuid4()}"
    with motor.begin() as c:
        c.execute(text("INSERT INTO solicitudes_reentrenamiento (id, origen) VALUES (:i, 'api')"), {"i": id_})
        if iniciar:
            c.execute(text("INSERT INTO tareas_reentrenamiento (id, iniciada_en) VALUES (:i, clock_timestamp())"),
                      {"i": id_})
    return id_


def plazos(id_, hijo=480, promocion=435, reconciliacion=485):
    t0 = time.monotonic()
    return ciclo.PlazosReentrenamiento(id_, t0 + hijo, t0 + promocion, t0 + reconciliacion)


def ejecutar(bd, uri, nombre, modo, config=None, **kw):
    return ciclo.ejecutar_entrenamiento(url_bd=url_de(bd), tracking_uri=uri, nombre=nombre, alias="vigente",
                                        modo=modo, git_commit="commit-de-prueba",
                                        config=config or asdict(CONFIG), evaluacion=EVALUACION.como_dict(), **kw)


def reclamacion(motor, id_):
    with motor.connect() as c:
        return c.execute(text("SELECT epoca FROM reclamaciones_reentrenamiento WHERE id = :i"), {"i": id_}).scalar()


def concesion_libre(bd):
    with ConcesionEntrenamiento(url_de(bd), modo="prueba", espera=0):
        return True


# ------------------------------------------------------------- R2 y si_mejora
def test_reentrenamiento_con_vigente_normal_decide_por_r2(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    assert ejecutar(bd, uri, nombre_modelo, "arranque").version_vigente == "1"
    v1 = vigente_pg(bd, nombre_modelo)
    llegan_los_datos_nuevos(bd)
    id_ = solicitud(bd)
    r = ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(id_))
    d = r.detalles[0]
    assert r.codigo == 0 and d["autorizada"]
    assert d["r2"]["decision_id"] == v1.decision_id and d["r2"]["dias"] == NUEVOS
    assert d["r2"]["resultado"] in (True, False) and d["condiciones"]["R2"] == d["r2"]["resultado"]
    assert r.promovido is d["r2"]["resultado"]                       # vigente normal: solo R2 true
    assert "Costo en validación" in r.mensaje and f"({NUEVOS} días" in r.mensaje
    t = tags_version(uri, nombre_modelo, d["version"])
    assert t["estado"] == "valido" and json.loads(t["r2"]) == d["r2"]["resultado"]
    assert t["reentrenamiento"] == id_ and t["sha256_r2"] == d["r2"]["sha256_r2"]
    # artefacto de R2 y manifiesto con el campeón
    from app.modelo.copia_datos import sha256
    from app.modelo.mlflow_util import descargar_de_ejecucion
    crudo = Path(descargar_de_ejecucion(uri, d["run_id"], "r2/resultados_por_bloque.csv")).read_bytes()
    assert sha256(crudo) == t["sha256_r2"]
    man = json.loads(Path(descargar_de_ejecucion(uri, d["run_id"], "datos/manifiesto.json")).read_text())
    assert man["campeon"]["decision_id"] == v1.decision_id and man["campeon"]["sha256_artefacto"] == \
        v1.sha256_artefacto
    assert man["campeon"]["corte_v"] == (FIN_VIGENTE + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    actual = vigente_pg(bd, nombre_modelo)
    assert (actual.version == d["version"]) is r.promovido and reclamacion(bd, id_) is not None
    assert concesion_libre(bd)


def test_r2_reproducido_desde_mlflow(bd, servidor_mlflow, nombre_modelo):
    """La tabla R2 registrada se reproduce desde la copia y el campeón descargados."""
    uri = servidor_mlflow.uri
    ejecutar(bd, uri, nombre_modelo, "arranque")
    llegan_los_datos_nuevos(bd)
    r = ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(solicitud(bd)))
    from app.modelo.reproducir import reproducir_r2
    difs = reproducir_r2(uri, nombre_modelo, r.detalles[0]["version"])
    assert difs == []


def test_sin_dias_nuevos_no_reemplaza(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    ejecutar(bd, uri, nombre_modelo, "arranque")
    r = ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(solicitud(bd)))
    d = r.detalles[0]
    assert r.codigo == 0 and d["autorizada"] and not r.promovido and r.version_vigente == "1"
    assert d["r2"]["resultado"] == "no_evaluable" and "insuficientes" in r.mensaje


def test_vigente_respaldo_baseline_se_reemplaza_con_c1_c3(bd, servidor_mlflow, nombre_modelo):
    """E7-6b: el vigente es el respaldo baseline (origen arranque_respaldo en PostgreSQL);
    un candidato que supera T1–T4b y C1–C3 lo reemplaza aunque R2 no se cumpla."""
    uri = servidor_mlflow.uri
    a = ejecutar(bd, uri, nombre_modelo, "arranque", config=MALA)
    assert [x["criterio"] for x in a.detalles] == ["candidato", "respaldo"]
    respaldo = vigente_pg(bd, nombre_modelo)
    r = ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(solicitud(bd)))
    d = r.detalles[0]
    assert d["autorizada"] and r.promovido and "respaldo baseline" in r.mensaje
    assert d["r2"]["decision_id"] == respaldo.decision_id
    assert vigente_pg(bd, nombre_modelo).version == d["version"] != respaldo.version


def test_falla_en_el_hijo_no_cambia_el_vigente(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    ejecutar(bd, uri, nombre_modelo, "arranque")
    antes = vigente_pg(bd, nombre_modelo)
    llegan_los_datos_nuevos(bd)
    with pytest.raises(ciclo.ErrorHijo):
        ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(solicitud(bd)),
                 config={**asdict(CONFIG), "modelo": "no_existe"})
    assert vigente_pg(bd, nombre_modelo) == antes and concesion_libre(bd)


# ------------------------------------------------------------------ reclamación
def test_reclamacion_atomica(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    id_ = solicitud(bd)
    ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(id_))
    n = len(cliente(uri).search_model_versions(f"name='{nombre_modelo}'"))
    with pytest.raises(ReclamacionRechazada, match="ya fue reclamada"):         # segundo intento de la tarea
        ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(id_))
    sin_inicio = solicitud(bd, iniciar=False)
    with pytest.raises(ReclamacionRechazada, match="no tiene inicio"):
        ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(sin_inicio))
    assert len(cliente(uri).search_model_versions(f"name='{nombre_modelo}'")) == n   # nunca entrenó
    assert concesion_libre(bd)


@pytest.mark.parametrize("tabla", ["resultados", "veredictos"])
def test_no_reclama_si_ya_hay_resultado_o_veredicto(bd, servidor_mlflow, nombre_modelo, tabla):
    id_ = solicitud(bd)
    with bd.begin() as c:
        if tabla == "resultados":
            c.execute(text("INSERT INTO resultados_reentrenamiento (id, estado, promovido, motivo, escrito_por) "
                           "VALUES (:i, 'fallido', false, 'x', 'api')"), {"i": id_})
        else:
            c.execute(text("INSERT INTO veredictos_reentrenamiento (id, veredicto) VALUES (:i, 'sin_resultado')"),
                      {"i": id_})
    with pytest.raises(ReclamacionRechazada):
        ejecutar(bd, servidor_mlflow.uri, nombre_modelo, "reentrenamiento", plazos=plazos(id_))
    assert reclamacion(bd, id_) is None


# ---------------------------------------------------------------------- plazos
def test_plazo_del_hijo(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    ejecutar(bd, uri, nombre_modelo, "arranque")
    antes = vigente_pg(bd, nombre_modelo)
    t = time.monotonic()
    with pytest.raises(ciclo.PlazoExcedido):
        ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(solicitud(bd), hijo=3))
    assert time.monotonic() - t < 3 + ciclo.GRACIA_TERMINACION_S + 8
    assert vigente_pg(bd, nombre_modelo) == antes and concesion_libre(bd)


def test_sin_tiempo_para_promover(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    a = ejecutar(bd, uri, nombre_modelo, "arranque", config=MALA)        # respaldo: un candidato bueno promovería
    antes = vigente_pg(bd, nombre_modelo)
    r = ejecutar(bd, uri, nombre_modelo, "reentrenamiento", plazos=plazos(solicitud(bd), promocion=0))
    d = r.detalles[0]
    assert a.promovido and r.codigo == 0 and not r.promovido and "sin tiempo para promover" in r.mensaje
    assert vigente_pg(bd, nombre_modelo) == antes
    assert tags_version(uri, nombre_modelo, d["version"])["estado"] == "pendiente"   # nunca autorizada


# ------------------------------------------------------------------------ CLI
def entorno_cli(bd, uri):
    from sqlalchemy.engine import make_url
    u = make_url(url_de(bd))
    return {**os.environ, "DATABASE_URL": url_de(bd), "MLFLOW_TRACKING_URI": uri,
            "NM_CIERRE_PASSWORD": CLAVE_CIERRE, "PYTHONPATH": str(RAIZ), "DB_HOST": u.host}


def resultado(bd, id_):
    with bd.connect() as c:
        return c.execute(text("SELECT * FROM resultados_reentrenamiento WHERE id = :i"), {"i": id_}).mappings().first()


def cli(bd, uri, nombre, id_, extra=None, esperar=True):
    args = {"id": id_, "nombre": nombre, **(extra or {})}
    p = subprocess.Popen([sys.executable, "-m", "tests.integracion.procesos", "reentrenar", json.dumps(args)],
                         env={**entorno_cli(bd, uri), "MLFLOW_MODELO_NOMBRE": nombre}, cwd=RAIZ,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if esperar:
        out, err = p.communicate(timeout=600)
        return p.returncode, out, err
    return p


def test_cli_escribe_exitoso_con_nm_cierre(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    ejecutar(bd, uri, nombre_modelo, "arranque")
    llegan_los_datos_nuevos(bd)
    id_ = solicitud(bd)
    codigo, out, err = cli(bd, uri, nombre_modelo, id_)
    assert codigo == 0, err[-3000:]
    f = resultado(bd, id_)
    assert f["estado"] == "exitoso" and f["escrito_por"] == "reentrenar"
    v = vigente_pg(bd, nombre_modelo)
    assert f["version"] == v.version
    assert f["promovido"] == (v.version != "1")                     # promovido = confirmada en su época
    codigo, _, err = cli(bd, uri, nombre_modelo, id_)                 # reintento de la tarea: código 7
    assert codigo == 7 and "ya tiene resultado" in err


def test_cli_fallo_escribe_fallido(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    id_ = solicitud(bd)
    codigo, _, err = cli(bd, uri, nombre_modelo, id_, {"config": {**asdict(CONFIG), "modelo": "no_existe"}})
    assert codigo == 1, err[-3000:]
    f = resultado(bd, id_)
    assert f["estado"] == "fallido" and not f["promovido"] and "ErrorHijo" in f["motivo"]


def test_cli_sigterm_detiene_y_escribe_fallido(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    id_ = solicitud(bd)
    lenta = {**asdict(CONFIG), "params_hgb": {"max_iter": 3000, "early_stopping": False}}
    p = cli(bd, uri, nombre_modelo, id_, {"config": lenta}, esperar=False)
    for _ in range(600):
        if reclamacion(bd, id_) is not None:
            break
        time.sleep(0.1)
    time.sleep(3)                                                      # el hijo ya entrena
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 143, err[-3000:]
    f = resultado(bd, id_)
    assert f["estado"] == "fallido" and "tiempo límite" in f["motivo"]
    assert concesion_libre(bd)
