"""Escribe datos sintéticos en la base y verifica que sigan intactos tras reiniciar.

Usado por ``scripts/verificar_persistencia_docker.sh`` (local y en la CI):

    python -m scripts.verificar_persistencia escribir  --estado /tmp/estado.json
    python -m scripts.verificar_persistencia verificar --estado /tmp/estado.json

Usa ``DATABASE_URL``. No usa datos reales: genera el catálogo y el histórico.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import func, select

from app.calidad import limpiar_solicitud_observacion, limpiar_solicitud_prediccion
from app.config import url_base_datos
from app.db import esquema as e
from app.db.carga import cargar_datos_publicados
from app.db.conexion import crear_motor, transaccion
from app.db.repositorio import PrediccionFamilia, cargar_catalogo, registrar_observacion, registrar_prediccion

RAIZ = Path(__file__).resolve().parents[1]


def _alembic() -> Config:
    cfg = Config(str(RAIZ / "alembic.ini"))
    cfg.set_main_option("script_location", str(RAIZ / "migraciones"))
    cfg.set_main_option("sqlalchemy.url", url_base_datos().replace("%", "%%"))
    return cfg


def _carpeta_sintetica() -> Path:
    from tests.ml.sinteticos import escribir_carpeta, generar
    carpeta = Path(tempfile.mkdtemp(prefix="nm_sinteticos_"))
    escribir_carpeta(generar(dias=40, semilla=7), carpeta)
    return carpeta


def huella(motor) -> dict:
    """Resumen verificable del contenido: conteos y valores clave."""
    with motor.connect() as c:
        conteos = {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in e.metadata.sorted_tables}
        solicitudes = [[r.solicitud_id, r.registrado_en.isoformat(), r.version_modelo, r.resultado]
                       for r in c.execute(select(e.solicitudes_prediccion).order_by("solicitud_id"))]
        obs = [[str(r.fecha), r.tienda_id, r.familia_id, r.ventas]
               for r in c.execute(select(e.observaciones).order_by("fecha", "tienda_id", "familia_id"))]
        supuestas = [[r.solicitud_id, r.familia_id] for r in c.execute(
            select(e.solicitud_familias).where(e.solicitud_familias.c.promocion_supuesta)
            .order_by("solicitud_id", "familia_id"))]
        suma = c.execute(select(func.sum(e.ventas_historicas.c.ventas))).scalar()
    return {"conteos": conteos, "solicitudes": solicitudes, "observaciones": obs,
            "promociones_supuestas": supuestas, "suma_ventas_historicas": int(suma or 0)}


def escribir(motor) -> dict:
    command.upgrade(_alembic(), "head")
    cargar_datos_publicados(motor, _carpeta_sintetica())
    with transaccion(motor) as conn:
        cat = cargar_catalogo(conn)
        payload = {"tienda_id": "T01", "fecha": "2025-06-20",
                   "familias": [{"familia_id": "fam_a", "en_promocion": 3}, {"familia_id": "fam_b", "en_promocion": 0}]}
        sol, reg = limpiar_solicitud_prediccion(payload, cat)            # fam_c falta -> promoción supuesta
        preds = [PrediccionFamilia(f.familia_id, 50.0, 55) for f in sol.familias]
        registrar_prediccion(conn, payload=payload, solicitud=sol, registro=reg, predicciones=preds,
                             nombre_modelo="verificacion", version_modelo="1")
    for ventas in ({"fam_a": 300, "fam_b": 150, "fam_c": 60}, {"fam_a": 300, "fam_b": 175, "fam_c": 60}):
        with transaccion(motor) as conn:
            p = {"tienda_id": 1, "fecha": "2025-06-20",
                 "ventas": [{"familia_id": f, "ventas": v} for f, v in ventas.items()]}
            sol, reg = limpiar_solicitud_observacion(p, cargar_catalogo(conn))
            registrar_observacion(conn, payload=p, solicitud=sol, registro=reg)
    return huella(motor)


# ---------------------------------------------------------------- MLflow (etapa 5)
MODELO_VERIFICACION = "verificacion-persistencia"


def escribir_mlflow(uri: str) -> dict:
    """Registra un modelo pequeño con un artefacto conocido y le asigna el alias."""
    import hashlib

    from app.config import configurar_cliente_mlflow
    configurar_cliente_mlflow()
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("verificacion-persistencia")
    contenido = b"artefacto de verificacion de persistencia\n"
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run() as run:
        ruta = Path(tmp) / "modelo.joblib"
        ruta.write_bytes(contenido)
        mlflow.log_artifact(str(ruta), "modelo/artifacts")
    c = MlflowClient(uri, uri)
    v = mlflow.register_model(f"runs:/{run.info.run_id}/modelo", MODELO_VERIFICACION)
    c.set_registered_model_alias(MODELO_VERIFICACION, "vigente", v.version)
    return {"run_id": run.info.run_id, "version": str(v.version), "sha256": hashlib.sha256(contenido).hexdigest()}


def verificar_mlflow(uri: str, esperado: dict) -> list[str]:
    import hashlib

    from app.config import configurar_cliente_mlflow
    configurar_cliente_mlflow()
    from mlflow import MlflowClient

    from app.modelo.mlflow_util import descargar_de_ejecucion
    c = MlflowClient(uri, uri)
    problemas = []
    mv = c.get_model_version_by_alias(MODELO_VERIFICACION, "vigente")
    if str(mv.version) != esperado["version"] or mv.run_id != esperado["run_id"]:
        problemas.append(f"alias vigente -> versión {mv.version} (run {mv.run_id}); esperado {esperado}")
    with tempfile.TemporaryDirectory() as tmp:
        ruta = descargar_de_ejecucion(uri, esperado["run_id"], "modelo/artifacts/modelo.joblib", tmp)
        if hashlib.sha256(Path(ruta).read_bytes()).hexdigest() != esperado["sha256"]:
            problemas.append("el artefacto registrado cambió")
    if c.get_experiment_by_name("verificacion-persistencia") is None:
        problemas.append("se perdió el experimento")
    return problemas


def verificar(motor, esperado: dict) -> list[str]:
    problemas = []
    actual = huella(motor)
    for clave, valor in esperado.items():
        if actual.get(clave) != valor:
            problemas.append(f"{clave}: esperado {valor!r}, encontrado {actual.get(clave)!r}")
    with motor.connect() as c:
        actual_rev = MigrationContext.configure(c).get_current_revision()
    head = ScriptDirectory.from_config(_alembic()).get_current_head()
    if actual_rev != head:
        problemas.append(f"revisión de migración {actual_rev} != {head}")
    # Un arranque posterior no debe repetir la carga de los datos publicados
    from app.db.carga import ORDEN
    with motor.connect() as c:
        cargados = {r.archivo for r in c.execute(select(e.cargas_datos.c.archivo))}
    if cargados != set(ORDEN):
        problemas.append(f"cargas registradas {sorted(cargados)}")
    return problemas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["escribir", "verificar"])
    ap.add_argument("--estado", required=True, type=Path)
    a = ap.parse_args()
    motor = crear_motor(timeout_consulta=300)
    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        if a.accion == "escribir":
            estado = escribir(motor)
            if uri:
                estado["mlflow"] = escribir_mlflow(uri)
            a.estado.write_text(json.dumps(estado, indent=2), encoding="utf-8")
            print(f"escrito: {estado['conteos']}" + (f"; MLflow versión {estado['mlflow']['version']}" if uri else ""))
            return 0
        esperado = json.loads(a.estado.read_text(encoding="utf-8"))
        esperado_mlflow = esperado.pop("mlflow", None)
        problemas = verificar(motor, esperado)
        if esperado_mlflow is not None:
            problemas += verificar_mlflow(uri, esperado_mlflow)
        if problemas:
            print("FALLO de persistencia:\n  " + "\n  ".join(problemas))
            return 1
        print("persistencia verificada: datos, migración y cargas intactos"
              + ("; MLflow: experimento, versión, alias y artefacto intactos" if esperado_mlflow else ""))
        return 0
    finally:
        motor.dispose()


if __name__ == "__main__":
    sys.exit(main())
