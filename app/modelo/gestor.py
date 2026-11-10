"""Modelo vigente en memoria, con recarga en segundo plano sin reiniciar la API.

* ``actual()`` devuelve el modelo cargado (referencia inmutable; el cambio de
  modelo es atómico) o lanza ModeloNoDisponible.
* ``refrescar()`` consulta la huella del proveedor y, si cambió, carga el nuevo
  modelo. Si la carga falla, **se conserva el modelo vigente** (requisito: la
  API sigue prediciendo aunque el origen del modelo no esté disponible).
"""

from __future__ import annotations

import logging
import threading

from app.modelo.proveedores import ModeloCargado, ModeloNoDisponible, ProveedorModelo

logger = logging.getLogger(__name__)


class GestorModelo:
    def __init__(self, proveedor: ProveedorModelo, intervalo: float = 30.0):
        self.proveedor = proveedor
        self.intervalo = intervalo
        self._modelo: ModeloCargado | None = None
        self._huella: str | None = None
        self._candado = threading.Lock()
        self.ultimo_error: str | None = None
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None

    def actual(self) -> ModeloCargado:
        modelo = self._modelo
        if modelo is None:
            raise ModeloNoDisponible(self.ultimo_error or "no hay un modelo cargado")
        return modelo

    def cargado(self) -> bool:
        return self._modelo is not None

    def refrescar(self) -> bool:
        """``True`` si se cargó un modelo nuevo."""
        with self._candado:
            try:
                huella = self.proveedor.huella()
            except Exception as exc:  # noqa: BLE001
                self.ultimo_error = f"no se pudo consultar el origen del modelo: {type(exc).__name__}"
                logger.warning(self.ultimo_error)
                return False
            if huella is None:
                if self._modelo is None:
                    self.ultimo_error = "no hay un modelo disponible"
                return False
            if huella == self._huella and self._modelo is not None:
                return False
            try:
                nuevo = self.proveedor.obtener()
            except ModeloNoDisponible as exc:
                self.ultimo_error = str(exc)
                logger.warning("modelo no cargado: %s", exc)
                return False
            anterior = self._modelo
            self._modelo, self._huella, self.ultimo_error = nuevo, huella, None
            logger.info("modelo cargado: %s versión %s (antes: %s)", nuevo.nombre, nuevo.version,
                        anterior.version if anterior else "ninguno")
            return True

    def _bucle(self) -> None:
        while not self._detener.wait(self.intervalo):
            self.refrescar()

    def iniciar(self) -> None:
        self.refrescar()
        self._detener.clear()
        self._hilo = threading.Thread(target=self._bucle, name="gestor-modelo", daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
