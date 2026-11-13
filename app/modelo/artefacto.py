"""Validación del artefacto del modelo: la **única referencia** es ``modelo.joblib``.

El mismo archivo (mismos bytes) se calcula al entrenar (``sha256_artefacto``), se
registra en MLflow, se descarga, se guarda en la caché y se carga. Orden de
validación (DECISIONES.md §14):

1. SHA-256 de los bytes **antes** de deserializar (deserializar ejecuta código);
2. formato, versiones de bibliotecas y tipo ``Pronosticador``;
3. predicción de humo (valores finitos, venta ≥ 0, cantidades enteras ≥ 0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.modelo.proveedores import FORMATO_ARTEFACTO, VersionNoUtilizable, versiones_librerias
from ml.pronosticador import Pronosticador

NOMBRE_ARCHIVO = "modelo.joblib"


class ArtefactoInvalido(VersionNoUtilizable):
    """El archivo no supera alguna validación; nunca se usa."""


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: str | Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_artefacto(ruta: str | Path, sha_esperado: str | None) -> dict:
    """Valida y deserializa ``modelo.joblib``. Si ``sha_esperado`` se indica, se
    comprueba **antes** de ``joblib.load``."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ArtefactoInvalido(f"no existe el artefacto {ruta.name}")
    if sha_esperado is not None:
        real = sha256_archivo(ruta)
        if real != sha_esperado:
            raise ArtefactoInvalido("el SHA-256 del artefacto no coincide con el registrado")
    try:
        artefacto = joblib.load(ruta)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura invalida el artefacto
        raise ArtefactoInvalido(f"no se pudo leer el artefacto del modelo: {type(exc).__name__}") from exc
    if not isinstance(artefacto, dict) or artefacto.get("formato") != FORMATO_ARTEFACTO:
        raise ArtefactoInvalido("el artefacto del modelo no tiene el formato esperado")
    if artefacto.get("librerias") != versiones_librerias():
        raise ArtefactoInvalido(f"el artefacto se creó con {artefacto.get('librerias')}, "
                                f"pero el servicio usa {versiones_librerias()}")
    if not isinstance(artefacto.get("pronosticador"), Pronosticador):
        raise ArtefactoInvalido("el artefacto no contiene un Pronosticador")
    return artefacto


def prueba_de_humo(p: Pronosticador) -> None:
    """Predice todas las familias de una tienda sin ventas previas (usa el nivel de
    respaldo): debe dar venta esperada finita ≥ 0 y cantidades enteras ≥ 0."""
    try:
        tienda = int(sorted(p.tiendas["tienda_id"])[0])
        objetivos = pd.DataFrame({"fecha": pd.Timestamp(p.info["corte"]), "tienda_id": tienda,
                                  "familia_id": list(p.familias), "en_promocion": 0})
        vacias = pd.DataFrame({"fecha": pd.Series(dtype="datetime64[ns]"), "tienda_id": pd.Series(dtype=int),
                               "familia_id": pd.Series(dtype=str), "ventas": pd.Series(dtype=int)})
        r = p.predecir(objetivos, vacias)
    except Exception as exc:  # noqa: BLE001
        raise ArtefactoInvalido(f"la predicción de humo falló: {type(exc).__name__}") from exc
    venta = r["venta_esperada"].to_numpy(float)
    cantidad = r["cantidad_sugerida"].to_numpy()
    if (len(r) != len(p.familias) or not np.isfinite(venta).all() or (venta < 0).any()
            or cantidad.dtype.kind not in "iu" or (cantidad < 0).any()):
        raise ArtefactoInvalido("la predicción de humo produjo valores fuera del contrato")


def validar_archivo(ruta: str | Path, sha_esperado: str | None) -> dict:
    artefacto = cargar_artefacto(ruta, sha_esperado)
    prueba_de_humo(artefacto["pronosticador"])
    return artefacto
