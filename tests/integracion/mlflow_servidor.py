"""Servidor MLflow real para las pruebas (proceso local, SQLite y artefactos en un
directorio temporal). Se puede detener y volver a iniciar con el mismo
almacenamiento para simular caídas y reinicios.

``mlflow server`` lanza gunicorn como procesos aparte: se inicia en su propio
grupo de procesos y se termina el grupo completo.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


def puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServidorMLflow:
    def __init__(self, carpeta: Path, puerto: int | None = None):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.puerto = puerto or puerto_libre()
        self.uri = f"http://127.0.0.1:{self.puerto}"
        self._proc: subprocess.Popen | None = None

    def iniciar(self, espera: float = 90) -> None:
        log = open(self.carpeta / "servidor.log", "ab")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(self.puerto),
             "--backend-store-uri", f"sqlite:///{self.carpeta / 'mlflow.db'}",
             "--artifacts-destination", str(self.carpeta / "artefactos"), "--serve-artifacts", "--workers", "1"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=self.carpeta)
        fin = time.monotonic() + espera
        while time.monotonic() < fin:
            if self.disponible():
                return
            if self._proc.poll() is not None:
                break
            time.sleep(0.3)
        raise RuntimeError(f"el servidor MLflow no arrancó; ver {self.carpeta / 'servidor.log'}")

    def disponible(self) -> bool:
        try:
            return requests.get(f"{self.uri}/health", timeout=1).status_code == 200
        except requests.RequestException:
            return False

    def detener(self) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
            self._proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._proc = None
        fin = time.monotonic() + 15
        while time.monotonic() < fin and self.disponible():
            time.sleep(0.2)

    def reiniciar(self) -> None:
        self.detener()
        self.iniciar()
