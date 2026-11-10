"""Monitor de salud en segundo plano (contrato ``GET /health``).

* Cada ``intervalo`` segundos se ejecutan **en paralelo** las verificaciones de
  los componentes, con un **límite global** ``limite`` por ciclo: lo que no
  responde a tiempo cuenta como falla. Una verificación que sigue colgada de un
  ciclo anterior no se vuelve a lanzar (se informa como "sin respuesta").
* ``GET /health`` solo lee el último resultado: nunca espera a una dependencia.
* Transiciones: un componente que cae (o vuelve) se refleja en a lo sumo
  ``intervalo + limite`` segundos (10 + 3 por defecto; el contrato pide < 60 s
  para degradar y < 120 s para volver a ok).
* El modelo se consulta en vivo en cada lectura (es una comprobación en memoria).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

OK, FALLA, NO_CONFIGURADO, PENDIENTE = "ok", "falla", "no_configurado", "pendiente"

MOTIVOS = {
    "base_datos": "base de datos no disponible",
    "modelo": "no hay un modelo cargado",
    "mlflow": "MLflow no disponible",
    "airflow": "Airflow no disponible",
}


@dataclass(frozen=True)
class Diagnostico:
    estado: str                       # "ok" | "degradado"
    motivo: str | None
    componentes: dict[str, str]
    modelo_cargado: bool

    @property
    def codigo_http(self) -> int:
        return 200 if self.modelo_cargado else 503

    def como_respuesta(self) -> dict:
        r = {"estado": self.estado, "componentes": self.componentes}
        if self.motivo:
            r["motivo"] = self.motivo
        return r


@dataclass
class MonitorSalud:
    verificaciones: dict[str, Callable[[], bool]]
    requeridos: Iterable[str]
    modelo_cargado: Callable[[], bool]
    intervalo: float = 10.0
    limite: float = 3.0
    _resultados: dict[str, str] = field(default_factory=dict)
    _en_curso: dict[str, Future] = field(default_factory=dict)

    def __post_init__(self):
        self.requeridos = tuple(self.requeridos)
        self._ejecutor = ThreadPoolExecutor(max_workers=max(2 * len(self.verificaciones), 2),
                                            thread_name_prefix="salud")
        self._candado = threading.Lock()
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None

    @staticmethod
    def _seguro(fn: Callable[[], bool]) -> bool:
        try:
            return bool(fn())
        except Exception as exc:  # noqa: BLE001 - una verificación nunca debe tumbar el monitor
            logger.warning("verificación de salud falló: %s", type(exc).__name__)
            return False

    def ejecutar_ciclo(self) -> dict[str, str]:
        """Ejecuta las verificaciones en paralelo con límite global. Devuelve los resultados."""
        lanzadas: dict[str, Future] = {}
        resultados: dict[str, str] = {}
        for nombre, fn in self.verificaciones.items():
            previa = self._en_curso.get(nombre)
            if previa is not None and not previa.done():
                resultados[nombre] = FALLA          # sigue colgada desde un ciclo anterior
                continue
            lanzadas[nombre] = self._en_curso[nombre] = self._ejecutor.submit(self._seguro, fn)
        hechas, _ = wait(lanzadas.values(), timeout=self.limite)
        for nombre, fut in lanzadas.items():
            resultados[nombre] = OK if fut in hechas and fut.result() else FALLA
        with self._candado:
            self._resultados = resultados
        return resultados

    def diagnostico(self) -> Diagnostico:
        with self._candado:
            res = dict(self._resultados)
        modelo = bool(self._seguro(self.modelo_cargado))
        componentes = {"api": OK, "modelo": OK if modelo else FALLA}
        for nombre in ("base_datos", "mlflow", "airflow"):
            if nombre in self.verificaciones:
                componentes[nombre] = res.get(nombre, PENDIENTE)
            else:
                componentes[nombre] = FALLA if nombre in self.requeridos else NO_CONFIGURADO
        problemas = []
        for nombre in self.requeridos:
            estado = componentes.get(nombre, FALLA)
            if estado == PENDIENTE:
                problemas.append(f"{nombre}: verificación en curso")
            elif estado != OK:
                problemas.append(MOTIVOS.get(nombre, f"{nombre} no disponible"))
        if problemas:
            return Diagnostico("degradado", "; ".join(problemas), componentes, modelo)
        return Diagnostico("ok", None, componentes, modelo)

    def _bucle(self) -> None:
        while not self._detener.wait(self.intervalo):
            self.ejecutar_ciclo()

    def iniciar(self) -> None:
        self.ejecutar_ciclo()        # primer diagnóstico antes de atender solicitudes (acotado por ``limite``)
        self._detener.clear()
        self._hilo = threading.Thread(target=self._bucle, name="monitor-salud", daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
        self._ejecutor.shutdown(wait=False, cancel_futures=True)
