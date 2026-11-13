"""Modelo vigente en memoria, con recarga en segundo plano sin reiniciar la API.

* ``actual()`` devuelve el modelo cargado (referencia inmutable; el cambio de
  modelo es atómico) o lanza ModeloNoDisponible.
* ``refrescar()`` consulta la huella del proveedor y, si cambió, carga el nuevo
  modelo. Si la carga falla, **se conserva el modelo vigente** (requisito: la
  API sigue prediciendo aunque el origen del modelo no esté disponible).
* Si el origen no responde y **todavía no hay modelo**, se usa
  ``proveedor.respaldo()`` cuando existe (caché validada del proveedor MLflow).
* ``actualizado()`` es ``False`` si el origen responde pero no ofrece una versión
  utilizable (alias inválido, versión inválida): ``/health`` lo informa (M9).
"""

from __future__ import annotations

import logging
import threading

from app.modelo.proveedores import ModeloCargado, ModeloNoDisponible, OrigenNoDisponible, ProveedorModelo

logger = logging.getLogger(__name__)


class GestorModelo:
    def __init__(self, proveedor: ProveedorModelo, intervalo: float = 30.0):
        self.proveedor = proveedor
        self.intervalo = intervalo
        self._modelo: ModeloCargado | None = None
        self._huella: str | None = None
        self._candado = threading.Lock()
        self.ultimo_error: str | None = None
        self.actualizacion_fallida: str | None = None
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None

    def actual(self) -> ModeloCargado:
        modelo = self._modelo
        if modelo is None:
            raise ModeloNoDisponible(self.ultimo_error or "no hay un modelo cargado")
        return modelo

    def cargado(self) -> bool:
        return self._modelo is not None

    def actualizado(self) -> bool:
        return self.actualizacion_fallida is None

    def _cargar(self, nuevo: ModeloCargado, huella: str) -> bool:
        anterior = self._modelo
        self._modelo, self._huella, self.ultimo_error, self.actualizacion_fallida = nuevo, huella, None, None
        logger.info("modelo cargado: %s versión %s (antes: %s)", nuevo.nombre, nuevo.version,
                    anterior.version if anterior else "ninguno")
        return True

    def _origen_caido(self, motivo: str) -> bool:
        """El origen no responde: se conserva el modelo; sin modelo, se intenta el respaldo."""
        self.ultimo_error = motivo
        logger.warning(motivo)
        respaldo = getattr(self.proveedor, "respaldo", None)
        if self._modelo is not None or respaldo is None:
            return False
        try:
            modelo, huella = respaldo()
        except ModeloNoDisponible as exc:
            self.ultimo_error = f"{motivo}; {exc}"
            logger.warning("sin modelo de respaldo: %s", exc)
            return False
        logger.warning("origen del modelo no disponible: se usa la versión %s de la caché", modelo.version)
        return self._cargar(modelo, huella)

    def _fallo_de_actualizacion(self, motivo: str) -> bool:
        self.ultimo_error = motivo
        if self._modelo is not None:
            self.actualizacion_fallida = motivo
        logger.warning("no se pudo actualizar el modelo: %s", motivo)
        return False

    def refrescar(self) -> bool:
        """``True`` si se cargó un modelo nuevo."""
        with self._candado:
            try:
                huella = self.proveedor.huella()
            except OrigenNoDisponible as exc:
                return self._origen_caido(f"origen del modelo no disponible: {exc}")
            except ModeloNoDisponible as exc:
                return self._fallo_de_actualizacion(str(exc))
            except Exception as exc:  # noqa: BLE001
                return self._origen_caido(f"no se pudo consultar el origen del modelo: {type(exc).__name__}")
            if huella is None:
                if self._modelo is None:
                    self.ultimo_error = "no hay un modelo disponible"
                    return False
                return self._fallo_de_actualizacion("el origen ya no ofrece un modelo")
            if huella == self._huella and self._modelo is not None:
                self.actualizacion_fallida = None
                return False
            try:
                nuevo = self.proveedor.obtener()
            except OrigenNoDisponible as exc:
                return self._origen_caido(f"origen del modelo no disponible: {exc}")
            except ModeloNoDisponible as exc:
                return self._fallo_de_actualizacion(str(exc))
            return self._cargar(nuevo, huella)

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
