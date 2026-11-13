"""Copia inmutable de los datos de un entrenamiento (DECISIONES.md §14, M12).

Cada ejecución guarda en ``datos/`` todo lo necesario para reconstruir el
entrenamiento y la evaluación **sin PostgreSQL**: ventas y promociones
consolidadas, observaciones, promociones de solicitudes (con las supuestas
marcadas), catálogos y un ``manifiesto.json`` con el SHA-256 del contenido sin
comprimir de cada archivo. ``huella_datos`` es el SHA-256 del manifiesto.

Formato canónico: CSV con columnas fijas, filas ordenadas por clave, fechas ISO,
fin de línea ``\\n``; ``gzip`` con ``mtime=0``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from app.db.conexion import transaccion
from app.db.repositorio import leer_datos_base, leer_observaciones, leer_promociones_solicitudes
from ml.datos import CLAVE, COLUMNAS_SOLICITUDES, consolidar_promociones, consolidar_ventas

FORMATO_COPIA = 1
MANIFIESTO = "manifiesto.json"

# nombre -> (columnas, orden de filas; None = conservar el orden recibido)
ESQUEMA: dict[str, tuple[list[str], list[str] | None]] = {
    "ventas_consolidadas": (CLAVE + ["ventas", "origen", "ventas_observacion", "discrepancia"], CLAVE),
    "promociones_consolidadas": (CLAVE + ["en_promocion", "origen", "en_promocion_solicitud", "discrepancia"], CLAVE),
    "observaciones": (CLAVE + ["ventas"], CLAVE),
    "promociones_solicitudes": (COLUMNAS_SOLICITUDES, ["solicitud_id", "familia_id"]),
    "tiendas": (["tienda_id", "nombre", "ciudad", "departamento", "tipo", "fecha_apertura"], ["tienda_id"]),
    "familias": (["familia_id", "nombre"], None),          # el orden define familia_cod del modelo
    "costos": (["familia_id", "costo_faltante", "costo_sobrante"], ["familia_id"]),
    "feriados": (["fecha", "nombre", "tipo", "ambito"], ["fecha", "nombre", "ambito"]),
}
FECHAS = {"ventas_consolidadas": ["fecha"], "promociones_consolidadas": ["fecha"], "observaciones": ["fecha"],
          "promociones_solicitudes": ["fecha"], "tiendas": ["fecha_apertura"], "feriados": ["fecha"]}


class CopiaDatosInvalida(RuntimeError):
    """La copia de datos no coincide con su manifiesto o su huella."""


@dataclass
class DatosEntrenamiento:
    ventas: pd.DataFrame
    promociones: pd.DataFrame
    observaciones: pd.DataFrame
    solicitudes: pd.DataFrame
    tiendas: pd.DataFrame
    familias: pd.DataFrame
    costos: pd.DataFrame
    feriados: pd.DataFrame
    corte: pd.Timestamp

    @property
    def familia_ids(self) -> list[str]:
        return list(self.familias["familia_id"])

    def tablas(self) -> dict[str, pd.DataFrame]:
        return {"ventas_consolidadas": self.ventas, "promociones_consolidadas": self.promociones,
                "observaciones": self.observaciones, "promociones_solicitudes": self.solicitudes,
                "tiendas": self.tiendas, "familias": self.familias, "costos": self.costos,
                "feriados": self.feriados}


def desde_bd(motor: Engine) -> DatosEntrenamiento:
    """Consolida los datos de la base (mismo criterio que ``entrenar_desde_bd``)."""
    with transaccion(motor) as conn:
        base = leer_datos_base(conn)
        observaciones = leer_observaciones(conn)
        solicitudes = leer_promociones_solicitudes(conn)
    if base.historico.empty:
        raise RuntimeError("la base no tiene datos publicados; ejecute python -m app.db.carga --raw raw")
    ventas = consolidar_ventas(base.historico, observaciones)
    promociones = consolidar_promociones(base.historico, solicitudes)
    corte = ventas["fecha"].max() + pd.Timedelta(days=1)
    return DatosEntrenamiento(ventas, promociones, observaciones, solicitudes, base.tiendas, base.familias,
                              base.costos, base.feriados, corte)


def _canonico(nombre: str, df: pd.DataFrame) -> bytes:
    columnas, orden = ESQUEMA[nombre]
    d = df[columnas].copy()
    if orden:
        d = d.sort_values(orden, kind="mergesort")
    for c in FECHAS.get(nombre, []):
        d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    if "registrado_en" in d:
        d["registrado_en"] = pd.to_datetime(d["registrado_en"], utc=True).map(lambda t: t.isoformat())
    return d.to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def escribir_copia(datos: DatosEntrenamiento, carpeta: str | Path, extra: dict | None = None) -> tuple[dict, str]:
    """Escribe ``carpeta/*.csv.gz`` y ``manifiesto.json``. Devuelve (manifiesto, huella)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    archivos = {}
    for nombre, df in datos.tablas().items():
        crudo = _canonico(nombre, df)
        (carpeta / f"{nombre}.csv.gz").write_bytes(gzip.compress(crudo, mtime=0))
        archivos[nombre] = {"archivo": f"{nombre}.csv.gz", "sha256": sha256(crudo), "filas": int(len(df))}
    manifiesto = {"formato": FORMATO_COPIA, "corte": datos.corte.strftime("%Y-%m-%d"), "archivos": archivos,
                  **(extra or {})}
    texto = json.dumps(manifiesto, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    (carpeta / MANIFIESTO).write_bytes(texto)
    return manifiesto, sha256(texto)


def _leer(nombre: str, crudo: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(crudo), keep_default_na=True)
    for c in FECHAS.get(nombre, []):
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d")
    if nombre == "ventas_consolidadas":
        df["ventas_observacion"] = df["ventas_observacion"].astype("Int64")
    if nombre == "promociones_consolidadas":
        df["en_promocion_solicitud"] = df["en_promocion_solicitud"].astype("Int64")
    if nombre == "promociones_solicitudes" and len(df):
        df["registrado_en"] = pd.to_datetime(df["registrado_en"], utc=True)
    return df


def leer_copia(carpeta: str | Path, huella_esperada: str | None = None) -> tuple[DatosEntrenamiento, dict]:
    """Lee y **verifica** la copia: huella del manifiesto (si se indica) y SHA-256
    de cada archivo, antes de devolver ningún dato."""
    carpeta = Path(carpeta)
    try:
        texto = (carpeta / MANIFIESTO).read_bytes()
        manifiesto = json.loads(texto)
    except (OSError, ValueError) as exc:
        raise CopiaDatosInvalida("el manifiesto de la copia de datos falta o no se puede leer") from exc
    if huella_esperada is not None and sha256(texto) != huella_esperada:
        raise CopiaDatosInvalida("el manifiesto no coincide con la huella registrada (huella_datos)")
    if manifiesto.get("formato") != FORMATO_COPIA or set(manifiesto.get("archivos", {})) != set(ESQUEMA):
        raise CopiaDatosInvalida("el manifiesto no tiene el formato esperado")
    tablas = {}
    for nombre, info in manifiesto["archivos"].items():
        try:
            crudo = gzip.decompress((carpeta / info["archivo"]).read_bytes())
        except (OSError, gzip.BadGzipFile, EOFError) as exc:
            raise CopiaDatosInvalida(f"no se puede leer {info['archivo']}") from exc
        if sha256(crudo) != info["sha256"]:
            raise CopiaDatosInvalida(f"{info['archivo']} no coincide con su SHA-256 del manifiesto")
        tablas[nombre] = _leer(nombre, crudo)
    datos = DatosEntrenamiento(
        tablas["ventas_consolidadas"], tablas["promociones_consolidadas"], tablas["observaciones"],
        tablas["promociones_solicitudes"], tablas["tiendas"], tablas["familias"], tablas["costos"],
        tablas["feriados"], pd.Timestamp(manifiesto["corte"]))
    return datos, manifiesto
