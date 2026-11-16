"""Cliente HTTP del simulador (biblioteca estándar). Mide la duración de cada llamada."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Respuesta:
    codigo: int | None          # None: sin respuesta (conexión rechazada, tiempo agotado)
    cuerpo: object
    inicio: float               # time.time() del equipo
    fin: float
    error: str | None = None

    @property
    def duracion(self) -> float:
        return self.fin - self.inicio


class Cliente:
    def __init__(self, puerto: int, timeout: float = 30.0):
        self.base = f"http://127.0.0.1:{puerto}"
        self.timeout = timeout

    def llamar(self, metodo: str, ruta: str, datos: bytes | None = None, cabeceras: dict | None = None,
               timeout: float | None = None) -> Respuesta:
        h = {"Content-Type": "application/json", **(cabeceras or {})}
        req = urllib.request.Request(self.base + ruta, data=datos, method=metodo, headers=h)
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                codigo, texto = r.status, r.read()
        except urllib.error.HTTPError as e:
            codigo, texto = e.code, e.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            return Respuesta(None, None, t, time.time(), f"{type(e).__name__}: {e}")
        try:
            cuerpo = json.loads(texto) if texto else None
        except json.JSONDecodeError:
            cuerpo = {"_no_json": texto[:200].decode("utf-8", "replace")}
        return Respuesta(codigo, cuerpo, t, time.time())

    def get(self, ruta: str, timeout: float | None = None) -> Respuesta:
        return self.llamar("GET", ruta, timeout=timeout)

    def post(self, ruta: str, datos: bytes | None = None, cabeceras: dict | None = None) -> Respuesta:
        return self.llamar("POST", ruta, datos if datos is not None else b"", cabeceras)
