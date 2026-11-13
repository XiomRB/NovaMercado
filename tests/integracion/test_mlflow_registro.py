"""Registro en MLflow y autorización (etapas 5 y 6): criterio, T1–T4b (T3 separado), estados
de las versiones, PostgreSQL como fuente de verdad, fallos, reproducción y trazabilidad.

``registrar`` (apoyo_mlflow) recorre el mismo camino que el supervisor —concesión real,
``registrar_version`` (hijo: solo ``pendiente``/``invalido``) y ``autorizar_y_promover``—
dentro del proceso de la prueba para poder inyectar fallos."""

import gzip
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sqlalchemy import text

import app.modelo.registro as reg
from app.db.conexion import transaccion
from app.modelo.bloqueo import ConcesionEntrenamiento
from app.modelo.mlflow_util import descargar_de_ejecucion
from app.modelo.promocion import ErrorPromocion, PromocionIndeterminada, reconciliar
from app.modelo.reproducir import reproducir
from tests.integracion.apoyo_mlflow import (
    CONFIG,
    EVALUACION,
    alias_actual,
    cargar,
    cliente,
    registrar,
    tags_version,
    url_de,
    vigente_pg,
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


def _resoluciones(motor):
    with motor.connect() as c:
        return c.execute(text("SELECT d.version, r.resultado, r.secuencia FROM resoluciones_promocion r "
                              "JOIN decisiones_promocion d USING (decision_id) ORDER BY r.secuencia")).all()


def _ejecuciones(uri):
    c = cliente(uri)
    return c.search_runs([c.get_experiment_by_name(reg.EXPERIMENTO).experiment_id], order_by=["start_time DESC"])


# --------------------------------------------------------------------- válido
def test_candidato_valido_se_registra_y_se_promueve(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = registrar(bd, uri, nombre_modelo)
    assert r.valido and r.promovido and r.version == "1" and alias_actual(uri, nombre_modelo) == "1"
    v = vigente_pg(bd, nombre_modelo)                               # fuente de verdad: PostgreSQL
    assert v.version == "1" and v.secuencia == 1 and v.run_id == r.run_id
    t = tags_version(uri, nombre_modelo, "1")
    assert t["estado"] == "valido" and t["promovido_por"] == "manual:si_no_hay"
    assert t["decision_id"] == str(v.decision_id) and t["verificacion"] == "superada"
    assert t["sha256_artefacto"] == v.sha256_artefacto and t["ruta_artefacto"] == "modelo/artifacts/modelo.joblib"
    assert int(t["epoca_bloqueo"]) == int(t["epoca_promocion"]) >= 1
    run = cliente(uri).get_run(r.run_id)
    assert run.info.status == "FINISHED" and run.data.tags["git_limpio"] == "true"
    assert run.data.tags["huella_datos"] == t["huella_datos"] and run.data.tags["version_registrada"] == "1"
    m = run.data.metrics
    assert m["mejora_pct"] > 5 and m["cortes_ganados"] == 3 and "segundos_t4a" in m
    arte = {a.path for a in cliente(uri).list_artifacts(r.run_id)}
    assert {"datos", "backtest", "modelo", "criterio.json"} <= arte
    criterio = json.loads(Path(descargar_de_ejecucion(uri, r.run_id, "criterio.json")).read_text())
    # T3 se registra por separado de T4b (etapa 6)
    assert criterio["condiciones"] == {"T1": True, "T2": True, "T3": True, "D1": True, "C1": True, "C2": True,
                                       "C3": True, "T4a": True, "T4b": True}
    assert json.loads(t["condiciones"]) == criterio["condiciones"]


def test_si_no_hay_no_promueve_si_postgres_ya_tiene_vigente(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    r2 = registrar(bd, uri, nombre_modelo)
    assert r2.valido and not r2.promovido and alias_actual(uri, nombre_modelo) == "1"
    assert vigente_pg(bd, nombre_modelo).version == "1"
    t = tags_version(uri, nombre_modelo, "2")
    assert t["estado"] == "valido" and "decision_id" not in t        # validada, sin decisión de promoción


def test_siempre_promueve_un_candidato_valido(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    r2 = registrar(bd, uri, nombre_modelo, promover="siempre")
    assert r2.promovido and alias_actual(uri, nombre_modelo) == "2"
    v = vigente_pg(bd, nombre_modelo)
    assert v.version == "2" and v.secuencia == 2
    t = tags_version(uri, nombre_modelo, "2")
    assert t["promovido_por"] == "manual:siempre" and t["alias_anterior"] == "1"


def test_nunca_solo_registra(bd, servidor_mlflow, nombre_modelo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo, promover="nunca")
    assert r.valido and not r.promovido and alias_actual(servidor_mlflow.uri, nombre_modelo) is None
    assert vigente_pg(bd, nombre_modelo) is None


def test_el_hijo_nunca_escribe_valido_ni_el_alias(bd, servidor_mlflow, nombre_modelo, monkeypatch):
    """El registro (trabajo del hijo) solo deja pendiente/invalido."""
    uri = servidor_mlflow.uri
    escritos = []
    real_tag = MlflowClient.set_model_version_tag
    real_alias = MlflowClient.set_registered_model_alias

    def tag(self, nombre, version, clave, valor):
        escritos.append((clave, valor))
        return real_tag(self, nombre, version, clave, valor)

    def alias(self, *a, **k):
        escritos.append(("ALIAS", a))
        return real_alias(self, *a, **k)
    with ConcesionEntrenamiento(url_de(bd), modo="prueba") as c, monkeypatch.context() as mp:
        mp.setattr(MlflowClient, "set_model_version_tag", tag)
        mp.setattr(MlflowClient, "set_registered_model_alias", alias)
        r = reg.registrar_version(bd, tracking_uri=uri, nombre=nombre_modelo, git_commit="x", epoca=c.epoca,
                                  comprobar=c.verificar, config=CONFIG, evaluacion=EVALUACION)
    assert r.verificacion and ("estado", "pendiente") in escritos
    assert ("estado", "valido") not in escritos and not any(k == "ALIAS" for k, _ in escritos)
    assert tags_version(uri, nombre_modelo, r.version)["estado"] == "pendiente"


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
    assert alias_actual(uri, nombre_modelo) == "1" and vigente_pg(bd, nombre_modelo).version == "1"
    assert tags_version(uri, nombre_modelo, r.version)["estado"] == "invalido"
    assert reg.califica_para_respaldo(r.r)                           # falló solo C1–C3


def test_candidato_malo_sin_vigente_previo_tampoco(bd, servidor_mlflow, nombre_modelo, candidato_malo):
    r = registrar(bd, servidor_mlflow.uri, nombre_modelo)
    assert not r.valido and alias_actual(servidor_mlflow.uri, nombre_modelo) is None
    assert vigente_pg(bd, nombre_modelo) is None


def test_datos_insuficientes_no_registran(motor, tmp_path, servidor_mlflow, nombre_modelo):
    from tests.integracion.apoyo_mlflow import DATOS
    corto = DATOS.historico[DATOS.historico["fecha"] >= DATOS.historico["fecha"].max() - pd.Timedelta(days=40)]
    from ml.datos import DatosBase
    cargar(motor, tmp_path / "raw", DatosBase(corto, DATOS.tiendas, DATOS.familias, DATOS.costos, DATOS.feriados))
    r = registrar(motor, servidor_mlflow.uri, nombre_modelo)
    assert not r.valido and r.version is None and "D1" in r.motivo
    assert cliente(servidor_mlflow.uri).get_run(r.run_id).data.tags["verificacion"] == "fallida"
    assert not reg.califica_para_respaldo(r.r)


@pytest.mark.parametrize("paso", ["evaluar", "log_model", "register_model"])
def test_excepcion_en_un_paso_no_promueve(bd, servidor_mlflow, nombre_modelo, monkeypatch, paso):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)                                       # v1 vigente

    def explota(*a, **k):
        raise RuntimeError(f"fallo inyectado en {paso}")
    objetivo = {"evaluar": (reg, "evaluar"), "log_model": (mlflow.pyfunc, "log_model"),
                "register_model": (mlflow, "register_model")}[paso]
    monkeypatch.setattr(*objetivo, explota)
    with pytest.raises(RuntimeError, match="fallo inyectado"):
        registrar(bd, uri, nombre_modelo, promover="siempre")
    assert _ejecuciones(uri)[0].info.status == "FAILED"
    assert alias_actual(uri, nombre_modelo) == "1" and vigente_pg(bd, nombre_modelo).version == "1"
    for v in cliente(uri).search_model_versions(f"name='{nombre_modelo}'"):
        if v.version != "1":
            assert dict(v.tags).get("estado") != "valido"


@pytest.mark.parametrize("falla", ["T3", "T4b", "T4a"])
def test_validacion_tecnica_fallida_no_promueve(bd, servidor_mlflow, nombre_modelo, monkeypatch, falla):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    if falla == "T3":                    # el artefacto descargado no pasa formato/bibliotecas/humo
        monkeypatch.setattr(reg, "verificar_t3_t4b", lambda *a, **k: ("prueba de humo fallida (inyectado)", None))
    elif falla == "T4b":
        monkeypatch.setattr(reg, "verificar_t3_t4b", lambda *a, **k: (None, "SHA distinto (inyectado)"))
    else:
        monkeypatch.setattr(reg, "verificar_t4a", lambda *a, **k: ["costo distinto (inyectado)"])
    r = registrar(bd, uri, nombre_modelo, promover="siempre")
    assert not r.valido and not r.promovido and f"{falla}:" in r.motivo
    assert r.condiciones[falla] is False and all(r.condiciones[k] for k in ("C1", "C2", "C3"))
    assert tags_version(uri, nombre_modelo, r.version)["estado"] == "invalido"
    assert vigente_pg(bd, nombre_modelo).version == "1" and alias_actual(uri, nombre_modelo) == "1"
    assert not reg.califica_para_respaldo(r.r)                       # nunca activa el respaldo


# ------------------------------------------------ fallos alrededor de la autorización
@pytest.mark.parametrize("paso", ["metrica_final", "criterio_final", "tags_finales", "cerrar_ejecucion"])
def test_fallo_al_cerrar_el_registro_del_hijo(bd, servidor_mlflow, nombre_modelo, monkeypatch, paso):
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    antes = len(_ejecuciones(uri))

    def falla():
        raise RuntimeError(f"fallo inyectado en {paso}")
    with monkeypatch.context() as mp, pytest.raises(RuntimeError, match="fallo inyectado"):
        if paso == "metrica_final":
            real = mlflow.log_metric
            mp.setattr(mlflow, "log_metric",
                       lambda k, *a, **kw: falla() if k == "segundos_total" else real(k, *a, **kw))
        elif paso == "criterio_final":
            real_d = mlflow.log_dict
            mp.setattr(mlflow, "log_dict", lambda d, f, *a, **kw: falla() if "T3" in d.get("condiciones", {})
                       else real_d(d, f, *a, **kw))
        elif paso == "tags_finales":
            real_t = mlflow.set_tags
            mp.setattr(mlflow, "set_tags",
                       lambda t, *a, **kw: falla() if "verificacion" in t else real_t(t, *a, **kw))
        else:
            real_e = mlflow.end_run
            mp.setattr(mlflow, "end_run",
                       lambda status="FINISHED": falla() if status == "FINISHED" else real_e(status))
        registrar(bd, uri, nombre_modelo, promover="siempre")
    ejecuciones = _ejecuciones(uri)
    assert len(ejecuciones) == antes + 1                    # el manejo del error no abre otra ejecución
    assert ejecuciones[0].info.status == "FAILED" and mlflow.active_run() is None
    assert tags_version(uri, nombre_modelo, "2")["estado"] == "invalido"
    assert vigente_pg(bd, nombre_modelo).version == "1" and alias_actual(uri, nombre_modelo) == "1"
    assert [r.version for r in _resoluciones(bd)] == ["1"]           # no hubo decisión para la v2


@pytest.mark.parametrize("clave", ["estado", "promovido_por"])
def test_fallo_en_el_reflejo_anula_la_decision(bd, servidor_mlflow, nombre_modelo, monkeypatch, clave):
    """Fallo en las escrituras del supervisor entre la decisión y el alias."""
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    real = MlflowClient.set_model_version_tag

    def tag(self, nombre, version, k, v):
        if k == clave and (clave != "estado" or v == "valido"):
            raise RuntimeError("fallo inyectado en el reflejo")
        return real(self, nombre, version, k, v)
    with monkeypatch.context() as mp, pytest.raises(RuntimeError, match="fallo inyectado"):
        mp.setattr(MlflowClient, "set_model_version_tag", tag)
        registrar(bd, uri, nombre_modelo, promover="siempre")
    assert [(r.version, r.resultado) for r in _resoluciones(bd)] == [("1", "confirmada"), ("2", "anulada")]
    assert vigente_pg(bd, nombre_modelo).version == "1" and alias_actual(uri, nombre_modelo) == "1"
    assert tags_version(uri, nombre_modelo, "2")["estado"] == "invalido"


@pytest.mark.parametrize("aplicada, consulta", [(False, True), (True, True), (False, False), (True, False)],
                         ids=["rechazada", "aplicada_sin_respuesta", "rechazada_sin_consulta", "aplicada_sin_consulta"])
def test_fallo_al_escribir_el_alias(bd, servidor_mlflow, nombre_modelo, monkeypatch, aplicada, consulta):
    """El servidor pudo aplicar el alias aunque el cliente reciba un error: se consulta.
    Solo una aplicación confirmada lleva a la resolución ``confirmada`` en PostgreSQL."""
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    real_alias = MlflowClient.set_registered_model_alias
    real_consulta = MlflowClient.get_model_version_by_alias
    tras_fallo = []

    def asignar(self, *a, **k):
        if aplicada:
            real_alias(self, *a, **k)
        tras_fallo.append(True)
        raise MlflowException("fallo inyectado: la solicitud del alias venció sin respuesta")

    def consultar(self, *a, **k):
        if tras_fallo and not consulta:
            raise ConnectionError("fallo inyectado en la consulta del alias")
        return real_consulta(self, *a, **k)
    with monkeypatch.context() as mp:
        mp.setattr(MlflowClient, "set_registered_model_alias", asignar)
        mp.setattr(MlflowClient, "get_model_version_by_alias", consultar)
        if not consulta:
            with pytest.raises(PromocionIndeterminada):
                registrar(bd, uri, nombre_modelo, promover="siempre")
        elif aplicada:
            r = registrar(bd, uri, nombre_modelo, promover="siempre")
        else:
            with pytest.raises(ErrorPromocion):
                registrar(bd, uri, nombre_modelo, promover="siempre")
    if aplicada and consulta:                     # la promoción sí se realizó: se reconoce
        assert r.promovido and vigente_pg(bd, nombre_modelo).version == "2"
        assert alias_actual(uri, nombre_modelo) == "2"
        return
    # en cualquier otro caso la decisión se anula: PostgreSQL sigue en la v1
    assert vigente_pg(bd, nombre_modelo).version == "1"
    assert [(r.version, r.resultado) for r in _resoluciones(bd)] == [("1", "confirmada"), ("2", "anulada")]
    assert tags_version(uri, nombre_modelo, "2")["estado"] == "invalido"
    assert alias_actual(uri, nombre_modelo) == ("2" if aplicada else "1")   # MLflow puede quedar desfasado
    with ConcesionEntrenamiento(url_de(bd), modo="prueba") as c:            # la reconciliación lo corrige
        reconciliar(c, cliente(uri), nombre_modelo, "vigente")
    assert alias_actual(uri, nombre_modelo) == "1"


def test_orden_decision_alias_confirmacion(bd, servidor_mlflow, nombre_modelo, monkeypatch):
    """El alias es la última escritura en MLflow; la confirmación en PostgreSQL va después."""
    uri = servidor_mlflow.uri
    registrar(bd, uri, nombre_modelo)
    llamadas = []

    def espiar(mp, dueno, metodo, etiqueta=None):
        real = getattr(dueno, metodo)

        def envoltura(*a, **k):
            llamadas.append(etiqueta or metodo)
            return real(*a, **k)
        mp.setattr(dueno, metodo, envoltura)
    with monkeypatch.context() as mp:
        for m in ("set_model_version_tag", "set_registered_model_alias", "set_tag", "set_terminated",
                  "log_metric", "log_dict", "delete_model_version_tag"):
            espiar(mp, MlflowClient, m)
        for m in ("log_metric", "log_dict", "set_tags", "set_tag", "end_run"):
            espiar(mp, mlflow, m)
        espiar(mp, ConcesionEntrenamiento, "insertar_decision", "PG:decision")
        espiar(mp, ConcesionEntrenamiento, "insertar_resolucion", "PG:resolucion")
        r = registrar(bd, uri, nombre_modelo, promover="siempre")
    assert r.promovido and alias_actual(uri, nombre_modelo) == "2"
    i_dec, i_alias, i_conf = (llamadas.index("PG:decision"), llamadas.index("set_registered_model_alias"),
                              llamadas.index("PG:resolucion"))
    assert i_dec < i_alias < i_conf and llamadas[-1] == "PG:resolucion"
    assert llamadas.index("end_run") < i_dec                        # el hijo ya cerró su ejecución
    assert [x for x in llamadas[i_alias + 1:] if not x.startswith("PG:")] == []


# ----------------------------------------------------------- T4a, T3/T4b y reproducción
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
    f.write_bytes(f.read_bytes() + b"\n")
    difs = reg.verificar_t4a(servidor_mlflow.uri, r.run_id, t["huella_datos"], t["sha256_backtest"],
                             reg.ConfigEvaluacion(cortes=3, horizonte=14), reg.ConfigEntrenamiento(
                                 dias_calibracion=28, dias_minimos_historia=14, params_hgb=dict(max_iter=40)))
    assert difs and "SHA-256" in difs[0]


def test_t3_y_t4b_detectan_un_artefacto_registrado_alterado(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = registrar(bd, uri, nombre_modelo, promover="nunca")
    t = tags_version(uri, nombre_modelo, r.version)
    args = (cliente(uri), uri, nombre_modelo, r.version, r.run_id, t["ruta_artefacto"], t["sha256_artefacto"])
    assert reg.verificar_t3_t4b(*args) == (None, None)
    f = _ruta_local_artefactos(servidor_mlflow, r.run_id, t["ruta_artefacto"])
    f.write_bytes(f.read_bytes() + b"x")
    t3, t4b = reg.verificar_t3_t4b(*args)
    assert "SHA-256" in t3 and "SHA-256" in t4b
    # T4b también compara la identidad registrada (tag) con lo entrenado
    _, t4b = reg.verificar_t3_t4b(*args[:-1], "0" * 64)
    assert t4b and "sha256_artefacto" in t4b


def test_trazabilidad_prediccion_version_ejecucion_datos(bd, servidor_mlflow, nombre_modelo):
    uri = servidor_mlflow.uri
    r = registrar(bd, uri, nombre_modelo)
    mv = cliente(uri).get_model_version(nombre_modelo, r.version)
    run = cliente(uri).get_run(mv.run_id)
    t = dict(mv.tags)
    assert run.info.run_id == r.run_id
    assert run.data.tags["huella_datos"] == t["huella_datos"]
    assert run.data.tags["sha256_backtest"] == t["sha256_backtest"]
    manifiesto = json.loads(Path(descargar_de_ejecucion(uri, r.run_id, "datos/manifiesto.json")).read_text())
    assert manifiesto["git_commit"] == "commit-de-prueba" and manifiesto["corte"] == t["corte"]
    assert manifiesto["hilos_entrenamiento"] == reg.hilos_entrenamiento()
    assert manifiesto["modelos_evaluados"] == ["baseline_dow4", "hgb_relativo"]
    assert set(manifiesto["entorno"]["librerias"]) >= {"scikit-learn", "numpy", "pandas", "scipy", "joblib", "mlflow"}
