"""Registro en MLflow: criterio, validaciones T4a/T4b, alias, fallos, reproducción y trazabilidad."""

import gzip
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

import app.modelo.registro as reg
from app.db.conexion import transaccion
from app.modelo.reproducir import reproducir
from tests.integracion.apoyo_mlflow import (
    alias_actual,
    cargar,
    cliente,
    registrar,
    tags_version,
)


@pytest.fixture
def bd(motor, tmp_path):
    cargar(motor, tmp_path / "raw")
    return motor


class ConstanteMala:
    """Candidato degradado: predice la media global; no supera al baseline."""

    def ajustar(self, X, y):
        self.media = float(np.mean(y))
        return self

    def predecir(self, X):
        return np.full(len(X), self.media)


@pytest.fixture
def candidato_malo(monkeypatch):
    import ml.pronosticador as pr
    real = pr.crear_modelo
    monkeypatch.setattr(pr, "crear_modelo", lambda n, *a, **k: ConstanteMala() if n == "hgb_relativo"
                        else real(n, *a, **k))


# --------------------------------------------------------------------- válido
def test_candidato_valido_se_registra_y_recibe_el_alias(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = registrar(bd, uri, nombre_modelo)
    assert r.valido and r.promovido and r.version == "1" and alias_actual(uri, nombre_modelo) == "1"
    t = tags_version(uri, nombre_modelo, "1")
    assert t["estado"] == "valido" and t["promovido_por"] == "si_no_hay"
    assert t["ruta_artefacto"] == "modelo/artifacts/modelo.joblib" and len(t["sha256_artefacto"]) == 64
    run = cliente(uri).get_run(r.run_id)
    assert run.info.status == "FINISHED" and run.data.tags["git_limpio"] == "true"
    assert run.data.tags["huella_datos"] == t["huella_datos"] and run.data.tags["version_registrada"] == "1"
    m = run.data.metrics
    assert m["mejora_pct"] > 5 and m["cortes_ganados"] == 3 and "segundos_t4a" in m
    arte = {a.path for a in cliente(uri).list_artifacts(r.run_id)}
    assert {"datos", "backtest", "modelo", "criterio.json"} <= arte
    criterio = json.loads(Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{r.run_id}/criterio.json", tracking_uri=uri)).read_text())
    assert criterio["condiciones"] == {"T2": True, "D1": True, "C1": True, "C2": True, "C3": True,
                                       "T4a": True, "T4b": True}


def test_si_no_hay_no_mueve_un_alias_existente(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    r2 = registrar(bd, uri, nombre_modelo)
    assert r2.valido and not r2.promovido and r2.alias_anterior == "1" and alias_actual(uri, nombre_modelo) == "1"


def test_siempre_mueve_el_alias_con_un_candidato_valido(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    r2 = registrar(bd, uri, nombre_modelo, promover="siempre")
    assert r2.promovido and alias_actual(uri, nombre_modelo) == "2"
    t = tags_version(uri, nombre_modelo, "2")
    assert t["promovido_por"] == "manual" and t["alias_anterior"] == "1"


def test_nunca_solo_registra(bd, servidor_mlflow, nombre_modelo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo, promover="nunca")
    assert r.valido and not r.promovido and alias_actual(servidor_mlflow.uri, nombre_modelo) is None


# ------------------------------------------------------------------- no válidos
@pytest.mark.parametrize("promover", ["si_no_hay", "siempre"])
def test_candidato_que_no_supera_el_costo_no_se_promueve(bd, servidor_mlflow, nombre_modelo, promover,
                                                         monkeypatch):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)                                # v1 válida y vigente
    import ml.pronosticador as pr
    real = pr.crear_modelo
    monkeypatch.setattr(pr, "crear_modelo", lambda n, *a, **k: ConstanteMala() if n == "hgb_relativo"
                        else real(n, *a, **k))
    r = registrar(bd, uri, nombre_modelo, promover=promover)
    assert not r.valido and not r.promovido and "C1" in r.motivo
    assert alias_actual(uri, nombre_modelo) == "1"
    assert tags_version(uri, nombre_modelo, r.version)["estado"] == "invalido"


def test_candidato_malo_sin_alias_previo_tampoco(bd, servidor_mlflow, nombre_modelo, candidato_malo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo)
    assert not r.valido and alias_actual(servidor_mlflow.uri, nombre_modelo) is None


def test_datos_insuficientes_no_registran(motor, tmp_path, servidor_mlflow, nombre_modelo):
    from tests.integracion.apoyo_mlflow import DATOS
    corto = DATOS.historico[DATOS.historico["fecha"] >= DATOS.historico["fecha"].max() - pd.Timedelta(days=40)]
    from ml.datos import DatosBase
    cargar(motor, tmp_path / "raw", DatosBase(corto, DATOS.tiendas, DATOS.familias, DATOS.costos, DATOS.feriados))
    r = registrar(motor, servidor_mlflow.uri, nombre_modelo)
    assert not r.valido and r.version is None and "D1" in r.motivo
    assert cliente(servidor_mlflow.uri).get_run(r.run_id).data.tags["valido"] == "false"


@pytest.mark.parametrize("paso", ["evaluar", "log_model", "register_model", "t4b", "t4a"])
def test_fallo_en_cualquier_paso_no_mueve_el_alias(bd, servidor_mlflow, nombre_modelo, monkeypatch, paso):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)                                       # v1 vigente

    def explota(*a, **k):
        raise RuntimeError(f"fallo inyectado en {paso}")
    if paso == "evaluar":
        monkeypatch.setattr(reg, "evaluar", explota)
    elif paso == "log_model":
        monkeypatch.setattr(mlflow.pyfunc, "log_model", explota)
    elif paso == "register_model":
        monkeypatch.setattr(mlflow, "register_model", explota)
    elif paso == "t4b":
        monkeypatch.setattr(reg, "verificar_t4b", lambda *a, **k: "SHA del artefacto distinto (inyectado)")
    else:
        monkeypatch.setattr(reg, "verificar_t4a", lambda *a, **k: ["costo distinto (inyectado)"])
    if paso in ("t4b", "t4a"):
        r = registrar(bd, uri, nombre_modelo, promover="siempre")
        assert not r.valido and not r.promovido and paso.upper().replace("T4", "T4") in r.motivo.upper()
        assert tags_version(uri, nombre_modelo, r.version)["estado"] == "invalido"
    else:
        with pytest.raises(RuntimeError, match="fallo inyectado"):
            registrar(bd, uri, nombre_modelo, promover="siempre")
        runs = cliente(uri).search_runs([cliente(uri).get_experiment_by_name(reg.EXPERIMENTO).experiment_id],
                                        order_by=["start_time DESC"], max_results=1)
        assert runs[0].info.status == "FAILED"
    assert alias_actual(uri, nombre_modelo) == "1"
    versiones = cliente(uri).search_model_versions(f"name='{nombre_modelo}'")
    for v in versiones:
        if v.version != "1":
            assert dict(v.tags).get("estado") != "valido"


# ----------------------------------------------------------- T4a, T4b y reproducción
def _ruta_local_artefactos(servidor, run_id, sub):
    base = Path(servidor.carpeta) / "artefactos"
    encontrados = [p for p in base.rglob(f"{run_id}/artifacts/{sub}")]
    assert encontrados, f"no se encontró {sub} del run {run_id}"
    return encontrados[0]


def test_reproduccion_tras_modificar_la_base(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = registrar(bd, uri, nombre_modelo)
    with transaccion(bd) as c:                       # la base cambia después del entrenamiento
        c.execute(text("INSERT INTO envios_observacion (resultado, payload, tienda_id, fecha, estado_calidad, "
                       "registros_nuevos, registros_modificados) "
                       "VALUES ('aceptado', '{}', 5, '2026-04-14', 'ok', 1, 0)"))
        c.execute(text("INSERT INTO observaciones (fecha, tienda_id, familia_id, ventas, envio_id_creacion) "
                       "VALUES ('2026-04-14', 5, 'abarrotes', 999, 1)"))
    res = reproducir(r.run_id, uri)
    assert res.estado == "identica" and res.backtest_igual and res.sha_reproducido == res.sha_registrado
    # entrenar con la base actual daría otra huella de datos
    from app.modelo.copia_datos import desde_bd, escribir_copia
    _, huella_actual = escribir_copia(desde_bd(bd), Path(servidor_mlflow.carpeta) / "tmp_actual")
    assert huella_actual != tags_version(uri, nombre_modelo, r.version)["huella_datos"]


def test_reproduccion_detecta_una_copia_alterada(bd, servidor_mlflow, nombre_modelo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo, promover="nunca")
    f = _ruta_local_artefactos(servidor_mlflow, r.run_id, "datos/ventas_consolidadas.csv.gz")
    crudo = gzip.decompress(f.read_bytes()).replace(b",1,", b",2,", 1)
    f.write_bytes(gzip.compress(crudo, mtime=0))
    res = reproducir(r.run_id, servidor_mlflow.uri)
    assert res.estado == "no_reproduce" and "copia o backtest inválidos" in res.diferencias[0]


def test_t4a_detecta_resultados_registrados_alterados(bd, servidor_mlflow, nombre_modelo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo, promover="nunca")
    t = tags_version(servidor_mlflow.uri, nombre_modelo, r.version)
    f = _ruta_local_artefactos(servidor_mlflow, r.run_id, "backtest/resultados_por_corte.csv")
    f.write_bytes(f.read_bytes().replace(b"hgb_relativo", b"hgb_relativo", 1) + b"\n")
    difs = reg.verificar_t4a(servidor_mlflow.uri, r.run_id, t["huella_datos"], t["sha256_backtest"],
                             reg.ConfigEvaluacion(cortes=3, horizonte=14), reg.ConfigEntrenamiento(
                                 dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=40)))
    assert difs and "SHA-256" in difs[0]


def test_t4b_detecta_un_artefacto_registrado_alterado(bd, servidor_mlflow, nombre_modelo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo, promover="nunca")
    t = tags_version(servidor_mlflow.uri, nombre_modelo, r.version)
    assert reg.verificar_t4b(servidor_mlflow.uri, r.run_id, t["ruta_artefacto"], t["sha256_artefacto"]) is None
    f = _ruta_local_artefactos(servidor_mlflow, r.run_id, t["ruta_artefacto"])
    f.write_bytes(f.read_bytes() + b"x")
    assert "SHA-256" in reg.verificar_t4b(servidor_mlflow.uri, r.run_id, t["ruta_artefacto"], t["sha256_artefacto"])


def test_trazabilidad_prediccion_version_ejecucion_datos(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = registrar(bd, uri, nombre_modelo)
    mv = cliente(uri).get_model_version(nombre_modelo, r.version)
    run = cliente(uri).get_run(mv.run_id)
    t = dict(mv.tags)
    assert run.info.run_id == r.run_id
    assert run.data.tags["huella_datos"] == t["huella_datos"]
    assert run.data.tags["sha256_backtest"] == t["sha256_backtest"]
    manifiesto = json.loads(Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{r.run_id}/datos/manifiesto.json", tracking_uri=uri)).read_text())
    assert manifiesto["git_commit"] == "commit-de-prueba" and manifiesto["corte"] == t["corte"]
    assert manifiesto["hilos_entrenamiento"] == reg.hilos_entrenamiento()
    assert set(manifiesto["entorno"]["librerias"]) >= {"scikit-learn", "numpy", "pandas", "scipy", "joblib", "mlflow"}
