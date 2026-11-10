"""Carga idempotente de los archivos publicados (``raw/``) en la base del producto.

* Cada archivo se registra en ``cargas_datos`` con su SHA-256.
* Si ya se cargó el mismo contenido, se omite (los siguientes arranques no
  repiten el trabajo).
* Si el contenido cambió, se detiene con :class:`DatosPublicadosModificados`:
  mezclar dos versiones del histórico produciría datos inconsistentes.
* Todo ocurre en una transacción: o se cargan todos los archivos o ninguno.

Uso:  python -m app.db.carga --raw raw   (usa DATABASE_URL)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.engine import Connection, Engine

from app.db import esquema as e
from app.db.conexion import crear_motor, transaccion
from ml.datos import cargar_raw

ORDEN = ["tiendas.csv", "familias.csv", "costos.csv", "feriados.csv", "historico_ventas.csv"]


class DatosPublicadosModificados(RuntimeError):
    pass


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _filas(df: pd.DataFrame, columnas: list[str]) -> list[dict]:
    d = df[columnas].copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.date
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in fila.items()}
            for fila in d.to_dict(orient="records")]


def _insertar(conn: Connection, archivo: str, df: pd.DataFrame, sha: str) -> int:
    carga_id = conn.execute(insert(e.cargas_datos).values(archivo=archivo, sha256=sha, filas=len(df))
                            .returning(e.cargas_datos.c.carga_id)).scalar_one()
    if archivo == "tiendas.csv":
        conn.execute(insert(e.tiendas), _filas(df, ["tienda_id", "nombre", "ciudad", "departamento", "tipo",
                                                    "fecha_apertura"]))
    elif archivo == "familias.csv":
        conn.execute(insert(e.familias), _filas(df, ["familia_id", "nombre"]))
    elif archivo == "costos.csv":
        conn.execute(insert(e.costos), _filas(df, ["familia_id", "costo_faltante", "costo_sobrante"]))
    elif archivo == "feriados.csv":
        conn.execute(insert(e.feriados), _filas(df, ["fecha", "nombre", "tipo", "ambito"]))
    elif archivo == "historico_ventas.csv":
        filas = _filas(df, ["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"])
        for f in filas:
            f["carga_id"] = carga_id
        conn.execute(insert(e.ventas_historicas), filas)
    return len(df)


def cargar_datos_publicados(motor: Engine, carpeta: str | Path) -> dict[str, int | str]:
    """Devuelve por archivo el número de filas cargadas o ``"ya_cargado"``."""
    carpeta = Path(carpeta)
    datos = cargar_raw(carpeta)  # valida columnas, tipos, duplicados y negativos
    tablas = {"tiendas.csv": datos.tiendas, "familias.csv": datos.familias, "costos.csv": datos.costos,
              "feriados.csv": datos.feriados, "historico_ventas.csv": datos.historico}
    resumen: dict[str, int | str] = {}
    with transaccion(motor) as conn:
        previas = dict(conn.execute(select(e.cargas_datos.c.archivo, e.cargas_datos.c.sha256)).all())
        for archivo in ORDEN:
            sha = _sha256(carpeta / archivo)
            if archivo in previas:
                if previas[archivo] != sha:
                    raise DatosPublicadosModificados(
                        f"{archivo} cambió desde que se cargó (sha256 {previas[archivo][:12]}… → {sha[:12]}…); "
                        "no se mezclan versiones de los datos publicados")
                resumen[archivo] = "ya_cargado"
                continue
            resumen[archivo] = _insertar(conn, archivo, tablas[archivo], sha)
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="raw", help="carpeta con los archivos publicados")
    a = ap.parse_args()
    motor = crear_motor()
    for archivo, r in cargar_datos_publicados(motor, a.raw).items():
        print(f"{archivo}: {r}")
    motor.dispose()


if __name__ == "__main__":
    main()
