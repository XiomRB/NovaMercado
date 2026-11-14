"""Estado reanudable de una ejecución (``trabajo/``).

* ``estado.json``: se reescribe de forma atómica (temporal + ``os.replace``).
* ``eventos.jsonl`` y ``respuestas.jsonl``: solo se agregan líneas (``fsync`` en cada una).
  Cada solicitud se registra como **intención** antes de enviarse y como **resultado**
  después: una intención sin resultado identifica lo que pudo quedar a medias.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def escribir_json_atomico(ruta: Path, datos: dict) -> None:
    ruta = Path(ruta)
    tmp = ruta.with_name(f".{ruta.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, ruta)


def leer_json(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


class Registro:
    """Archivo JSONL de solo agregado."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)

    def agregar(self, entrada: dict) -> None:
        with open(self.ruta, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entrada, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def leer(self) -> list[dict]:
        if not self.ruta.exists():
            return []
        out = []
        for linea in self.ruta.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(linea))
            except json.JSONDecodeError:
                break                     # línea final truncada por una interrupción
        return out


class Estado:
    def __init__(self, trabajo: Path):
        self.trabajo = Path(trabajo)
        self.ruta = self.trabajo / "estado.json"
        self.datos: dict = leer_json(self.ruta) if self.ruta.exists() else {}
        self.eventos = Registro(self.trabajo / "eventos.jsonl")
        self.respuestas = Registro(self.trabajo / "respuestas.jsonl")

    def __getitem__(self, k):
        return self.datos[k]

    def get(self, k, defecto=None):
        return self.datos.get(k, defecto)

    def __setitem__(self, k, v):
        self.datos[k] = v

    def guardar(self) -> None:
        escribir_json_atomico(self.ruta, self.datos)

    def evento(self, tipo: str, **datos) -> dict:
        e = {"t": time.time(), "tipo": tipo, **datos}
        self.eventos.agregar(e)
        return e
