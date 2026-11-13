"""Caché local de modelos registrados (para predecir con MLflow caído).

Estructura: ``<dir>/<versión>/modelo.joblib`` + ``metadatos.json`` (nombre, alias,
versión, ``run_id``, ``ruta_artefacto``, ``sha256_artefacto`` copiado del tag del
Registry). La escritura es atómica (directorio temporal + ``rename``) y se
conservan las ``conservar`` versiones más recientes.

Validación al leer, versión por versión (de la más reciente a la más antigua):
metadatos legibles → SHA-256 **recalculado** igual al de los metadatos (antes de
deserializar) → formato y bibliotecas → predicción de humo. Se usa la primera
válida; si ninguna lo es, ModeloNoDisponible.

Limitación documentada (DECISIONES.md §14): sin MLflow no se detecta una
modificación **coherente** del artefacto y de sus metadatos a la vez. Con MLflow
disponible, el SHA siempre se contrasta con el tag del Registry.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.modelo.artefacto import NOMBRE_ARCHIVO, ArtefactoInvalido, sha256_archivo, validar_archivo
from app.modelo.proveedores import ModeloNoDisponible

logger = logging.getLogger(__name__)

METADATOS = "metadatos.json"
CAMPOS = ("nombre", "alias", "version", "run_id", "ruta_artefacto", "sha256_artefacto")


@dataclass(frozen=True)
class EntradaValida:
    artefacto: dict
    metadatos: dict


def _orden(version: str) -> tuple:
    return (0, int(version)) if version.isdigit() else (-1, version)


class CacheModelos:
    def __init__(self, directorio: str | Path, conservar: int = 3):
        self.dir = Path(directorio)
        self.conservar = conservar

    def versiones(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted((p.name for p in self.dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=_orden, reverse=True)

    def ruta(self, version: str) -> Path:
        return self.dir / str(version) / NOMBRE_ARCHIVO

    def archivo_si_coincide(self, version: str, sha: str) -> Path | None:
        """Ruta del artefacto de ``version`` si su SHA **recalculado** coincide con ``sha``."""
        ruta = self.ruta(version)
        try:
            return ruta if ruta.is_file() and sha256_archivo(ruta) == sha else None
        except OSError:
            return None

    def guardar(self, origen: str | Path, metadatos: dict) -> None:
        faltan = [c for c in CAMPOS if c not in metadatos]
        if faltan:
            raise ValueError(f"metadatos incompletos: {faltan}")
        version = str(metadatos["version"])
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=self.dir, prefix=".tmp-"))
        try:
            shutil.copyfile(origen, tmp / NOMBRE_ARCHIVO)
            if sha256_archivo(tmp / NOMBRE_ARCHIVO) != metadatos["sha256_artefacto"]:
                raise ArtefactoInvalido("el artefacto a guardar en caché no coincide con su SHA-256")
            (tmp / METADATOS).write_text(json.dumps(metadatos, sort_keys=True, indent=2), encoding="utf-8")
            destino = self.dir / version
            if destino.exists():
                shutil.rmtree(destino)
            os.replace(tmp, destino)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._limpiar()

    def _limpiar(self) -> None:
        for v in self.versiones()[self.conservar:]:
            shutil.rmtree(self.dir / v, ignore_errors=True)

    def validar(self, version: str) -> EntradaValida:
        carpeta = self.dir / version
        try:
            metadatos = json.loads((carpeta / METADATOS).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtefactoInvalido("metadatos de la caché ausentes o ilegibles") from exc
        if not isinstance(metadatos, dict) or any(c not in metadatos for c in CAMPOS):
            raise ArtefactoInvalido("metadatos de la caché incompletos")
        if str(metadatos["version"]) != version:
            raise ArtefactoInvalido("la versión de los metadatos no corresponde a la carpeta")
        artefacto = validar_archivo(carpeta / NOMBRE_ARCHIVO, metadatos["sha256_artefacto"])
        return EntradaValida(artefacto, metadatos)

    def primera_valida(self) -> EntradaValida:
        motivos = []
        for v in self.versiones():
            try:
                return self.validar(v)
            except ModeloNoDisponible as exc:
                motivos.append(f"versión {v}: {exc}")
                logger.warning("caché: versión %s descartada: %s", v, exc)
        detalle = "; ".join(motivos) if motivos else "la caché está vacía"
        raise ModeloNoDisponible(f"no hay una versión válida en la caché ({detalle})")
