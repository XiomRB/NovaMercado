"""Origen del modelo vigente. Etapa 4: artefacto local; etapa 5: MLflow Model Registry.

Un proveedor entrega un :class:`ModeloCargado` y una ``huella`` barata de
calcular que cambia cuando hay un modelo nuevo (el gestor la consulta
periódicamente para recargar sin reiniciar la API).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml.pronosticador import Pronosticador

FORMATO_ARTEFACTO = 1
NOMBRE_MODELO = "novamercado-ventas"


class ModeloNoDisponible(RuntimeError):
    """No hay un modelo cargado o el artefacto no se puede usar. La API responde 503."""


@dataclass(frozen=True)
class ModeloCargado:
    pronosticador: Pronosticador
    nombre: str
    alias: str
    version: str


class ProveedorModelo(Protocol):
    def huella(self) -> str | None:
        """Identificador barato del modelo disponible; ``None`` si no hay ninguno."""

    def obtener(self) -> ModeloCargado:
        """Carga el modelo disponible o lanza :class:`ModeloNoDisponible`."""


def versiones_librerias() -> dict[str, str]:
    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}


class ProveedorArchivo:
    """Artefacto ``joblib`` producido por ``python -m app.modelo.entrenar_local``.

    Se rechaza (ModeloNoDisponible) si falta, si no tiene el formato esperado o
    si fue creado con otras versiones de scikit-learn, numpy o pandas: un modelo
    deserializado con otras bibliotecas puede fallar o predecir distinto.
    """

    alias = "local"

    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)

    def huella(self) -> str | None:
        try:
            st = os.stat(self.ruta)
        except OSError:
            return None
        return f"{st.st_mtime_ns}-{st.st_size}"

    def obtener(self) -> ModeloCargado:
        if not self.ruta.is_file():
            raise ModeloNoDisponible(f"no existe el artefacto del modelo ({self.ruta.name}); "
                                     "ejecute python -m app.modelo.entrenar_local")
        try:
            artefacto = joblib.load(self.ruta)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura deja el modelo no disponible
            raise ModeloNoDisponible(f"no se pudo leer el artefacto del modelo: {type(exc).__name__}") from exc
        if not isinstance(artefacto, dict) or artefacto.get("formato") != FORMATO_ARTEFACTO:
            raise ModeloNoDisponible("el artefacto del modelo no tiene el formato esperado")
        if artefacto.get("librerias") != versiones_librerias():
            raise ModeloNoDisponible(f"el artefacto se creó con {artefacto.get('librerias')}, "
                                     f"pero el servicio usa {versiones_librerias()}")
        p = artefacto.get("pronosticador")
        if not isinstance(p, Pronosticador):
            raise ModeloNoDisponible("el artefacto no contiene un Pronosticador")
        return ModeloCargado(p, artefacto["nombre"], self.alias, str(artefacto["version"]))
