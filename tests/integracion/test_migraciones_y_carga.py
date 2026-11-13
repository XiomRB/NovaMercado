"""Migraciones reproducibles y carga idempotente de los datos publicados."""

import pandas as pd
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError

from app.calidad.catalogo import Catalogo
from app.db import esquema as e
from app.db.carga import DatosPublicadosModificados, cargar_datos_publicados
from app.db.conexion import transaccion
from app.db.repositorio import cargar_catalogo, leer_datos_base
from ml.datos import cargar_raw
from tests.integracion.apoyo import DATOS, cargar_sinteticos
from tests.integracion.conftest import config_alembic
from tests.ml.sinteticos import escribir_carpeta, generar


def _tablas(url):
    m = create_engine(url, future=True)
    with m.connect() as c:
        r = c.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        nombres = {x[0] for x in r}
    m.dispose()
    return nombres


def test_upgrade_downgrade_upgrade(url_bd_vacia):
    cfg = config_alembic(url_bd_vacia)
    command.upgrade(cfg, "head")
    esperadas = set(e.metadata.tables) | {"alembic_version", "v_predicciones_reales"}
    assert _tablas(url_bd_vacia) == esperadas
    command.downgrade(cfg, "base")
    assert _tablas(url_bd_vacia) == {"alembic_version"}
    command.upgrade(cfg, "head")
    assert _tablas(url_bd_vacia) == esperadas


def test_esquema_migrado_coincide_con_el_modelo(motor):
    with motor.connect() as c:
        diferencias = compare_metadata(MigrationContext.configure(c, opts={"compare_type": True}), e.metadata)
    assert diferencias == []


def test_una_sola_cabeza_y_base_en_head(motor):
    scripts = ScriptDirectory.from_config(config_alembic(str(motor.url)))
    assert len(scripts.get_heads()) == 1
    with motor.connect() as c:
        assert MigrationContext.configure(c).get_current_revision() == scripts.get_current_head()


def test_triggers_de_inmutabilidad_instalados(motor):
    with motor.connect() as c:
        filas = c.execute(text("SELECT tgrelid::regclass::text, tgname FROM pg_trigger WHERE NOT tgisinternal")).all()
    por_tabla = {}
    for tabla, nombre in filas:
        por_tabla.setdefault(tabla, set()).add(nombre)
    coherencia = {"solicitud_familias": {"trg_solicitud_familias_respondida"},
                  "observaciones_historial": {"trg_observaciones_historial_envio_aceptado"}}
    for t in e.TABLAS_SOLO_INSERCION:
        assert por_tabla[t] == {f"trg_{t}_inmutable", f"trg_{t}_sin_truncate"} | coherencia.get(t, set())
    assert por_tabla["observaciones"] == {"trg_observaciones_sin_borrado", "trg_observaciones_sin_truncate",
                                          "trg_observaciones_envio_aceptado"}


# ------------------------------------------------------------------ carga
def test_carga_y_lectura(motor, tmp_path):
    r = cargar_sinteticos(motor, tmp_path)
    assert r["historico_ventas.csv"] == len(DATOS.historico) and r["tiendas.csv"] == 3
    with transaccion(motor) as c:
        leidos = leer_datos_base(c)
        catalogo = cargar_catalogo(c)
    esperado = cargar_raw(tmp_path)
    pd.testing.assert_frame_equal(leidos.historico.reset_index(drop=True),
                                  esperado.historico.sort_values(["fecha", "tienda_id", "familia_id"])
                                  .reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(leidos.costos, esperado.costos.sort_values("familia_id").reset_index(drop=True),
                                  check_dtype=False)
    assert len(leidos.feriados) == len(esperado.feriados)
    csv = Catalogo.desde_csv(tmp_path)
    assert catalogo.tiendas == csv.tiendas and catalogo.familias == csv.familias


def test_segunda_carga_no_repite_trabajo(motor, tmp_path):
    cargar_sinteticos(motor, tmp_path)
    r = cargar_datos_publicados(motor, tmp_path)
    assert set(r.values()) == {"ya_cargado"}
    with motor.connect() as c:
        assert c.execute(select(func.count()).select_from(e.ventas_historicas)).scalar() == len(DATOS.historico)
        assert c.execute(select(func.count()).select_from(e.cargas_datos)).scalar() == 5


def test_archivo_modificado_se_detecta_y_no_se_mezcla(motor, tmp_path):
    cargar_sinteticos(motor, tmp_path)
    h = pd.read_csv(tmp_path / "historico_ventas.csv")
    h.loc[0, "ventas"] += 1
    h.to_csv(tmp_path / "historico_ventas.csv", index=False)
    with pytest.raises(DatosPublicadosModificados, match="historico_ventas.csv cambió"):
        cargar_datos_publicados(motor, tmp_path)


def test_carga_atomica_ante_error_de_integridad(motor, tmp_path):
    """Una venta de una tienda inexistente viola la FK: no queda cargado ningún archivo."""
    d = generar(dias=10)
    escribir_carpeta(d, tmp_path)
    h = pd.read_csv(tmp_path / "historico_ventas.csv")
    h.loc[0, "tienda_id"] = 99
    h.to_csv(tmp_path / "historico_ventas.csv", index=False)
    with pytest.raises(IntegrityError):
        cargar_datos_publicados(motor, tmp_path)
    with motor.connect() as c:
        for t in (e.cargas_datos, e.tiendas, e.familias, e.ventas_historicas):
            assert c.execute(select(func.count()).select_from(t)).scalar() == 0


# ------------------------------------------- verificación: triggers vs migraciones y operación
def test_migraciones_con_datos_y_triggers(url_bd_vacia, tmp_path):
    """Los triggers no impiden revertir ni volver a aplicar el esquema con datos cargados,
    y repetir upgrade sobre una base al día no hace nada."""
    from app.db.conexion import crear_motor
    from tests.integracion.apoyo import observar, payload_observacion, payload_prediccion, predecir_y_registrar
    cfg = config_alembic(url_bd_vacia)
    command.upgrade(cfg, "head")
    m = crear_motor(url_bd_vacia)
    cargar_sinteticos(m, tmp_path)
    predecir_y_registrar(m, payload_prediccion())
    observar(m, payload_observacion())
    observar(m, payload_observacion(ventas={"fam_a": 1, "fam_b": 150, "fam_c": 60}))
    m.dispose()
    command.upgrade(cfg, "head")                         # ya al día: no-op
    assert "solicitudes_prediccion" in _tablas(url_bd_vacia)
    command.downgrade(cfg, "base")                       # DROP no se bloquea por los triggers
    assert _tablas(url_bd_vacia) == {"alembic_version"}
    command.upgrade(cfg, "head")
    m = crear_motor(url_bd_vacia)
    cargar_sinteticos(m, tmp_path / "otra")              # operación normal tras recrear
    r, _, _ = predecir_y_registrar(m, payload_prediccion())
    assert r.solicitud_id == 1
    m.dispose()


def test_funciones_de_triggers_se_eliminan_en_downgrade(url_bd_vacia):
    cfg = config_alembic(url_bd_vacia)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    m = create_engine(url_bd_vacia, future=True)
    with m.connect() as c:
        funciones = c.execute(text("SELECT proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                                   "WHERE n.nspname = 'public'")).scalars().all()
    m.dispose()
    assert funciones == []


# ------------------------------------------------------------- etapa 6 (E6-7)
def test_raw_ausente_con_datos_ya_cargados_se_omite(motor, tmp_path):
    from app.db.carga import DatosPublicadosAusentes, cargar_datos_publicados
    with pytest.raises(DatosPublicadosAusentes, match="colóquelos en la carpeta raw/"):
        cargar_datos_publicados(motor, tmp_path / "vacia")          # primer arranque sin raw/
    cargar_sinteticos(motor, tmp_path / "raw")
    r = cargar_datos_publicados(motor, tmp_path / "no_existe")      # reinicio sin raw/
    assert set(r.values()) == {"ya_cargado_sin_raw"}


def test_triggers_de_las_tablas_de_promocion(motor):
    from sqlalchemy.exc import DBAPIError
    with motor.connect() as c:
        filas = c.execute(text("SELECT tgrelid::regclass::text, tgname FROM pg_trigger WHERE NOT tgisinternal "
                               "AND tgrelid::regclass::text IN ('decisiones_promocion', 'resoluciones_promocion', "
                               "'bloqueo_entrenamiento')")).all()
    por_tabla = {}
    for tabla, nombre in filas:
        por_tabla.setdefault(tabla, set()).add(nombre)
    for t in e.TABLAS_SOLO_INSERCION_PROMOCION:
        assert por_tabla[t] == {f"trg_{t}_inmutable", f"trg_{t}_sin_truncate"}
    assert por_tabla["bloqueo_entrenamiento"] == {"trg_bloqueo_entrenamiento_sin_borrado",
                                                  "trg_bloqueo_entrenamiento_sin_truncate"}
    for sql in ("DELETE FROM bloqueo_entrenamiento", "INSERT INTO bloqueo_entrenamiento (id) VALUES (2)",
                "INSERT INTO bloqueo_entrenamiento (id) VALUES (1)"):
        with pytest.raises(DBAPIError), motor.begin() as c:
            c.execute(text(sql))
    with motor.begin() as c:
        d = c.execute(text("INSERT INTO decisiones_promocion (nombre_modelo, version, run_id, ruta_artefacto, "
                           "sha256_artefacto, epoca, origen) VALUES ('m','1','r','x', repeat('a', 64), 1, 'p') "
                           "RETURNING decision_id")).scalar()
    for sql in (f"UPDATE decisiones_promocion SET version = '2' WHERE decision_id = {d}",
                f"DELETE FROM decisiones_promocion WHERE decision_id = {d}", "TRUNCATE decisiones_promocion CASCADE"):
        with pytest.raises(DBAPIError), motor.begin() as c:
            c.execute(text(sql))
