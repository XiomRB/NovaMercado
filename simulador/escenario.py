"""Escenarios de la simulación (DECISIONES.md §17, plan v3 de la etapa 8).

El día ``d`` (1..dias) corresponde a la fecha ``corte + d``. Cada día: observaciones del
día anterior (el día 1 no envía: el día ``corte`` ya está en el histórico inicial), la
solicitud de reentrenamiento si toca y los pronósticos del día. Un paso de cierre envía
las observaciones del último día.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta


@dataclass(frozen=True)
class Escenario:
    nombre: str
    fuente: str                        # "raw" (reproducción de datos publicados) o "sintetico"
    corte: str                         # último día del histórico inicial (AAAA-MM-DD)
    dias: int
    reentrenos: tuple[int, ...]        # días con reentrenamiento programado (con Idempotency-Key)
    dia_repetir_clave: int             # repite la solicitud de ese día con la misma clave
    dia_mlflow: int                    # MLflow detenido; reentrenamiento extra sin clave (fallido)
    dia_stop_start: int
    dia_down_up: int
    semilla: int = 20251201
    tasa_defecto_prediccion: float = 0.25
    tasa_defecto_observacion: float = 0.20
    tasa_repeticion: float = 0.05
    cambio_demanda: dict | None = None     # {"familia", "factor", "desde_dia"} (solo S2)
    criterio_costo: bool = False           # S9 solo en la reproducción real de 90 días
    plazos: dict = field(default_factory=lambda: dict(PLAZOS))

    def fecha(self, dia: int) -> date:
        return date.fromisoformat(self.corte) + timedelta(days=dia)

    @property
    def inicio(self) -> date:
        return self.fecha(1)

    @property
    def fin(self) -> date:
        return self.fecha(self.dias)

    def como_dict(self) -> dict:
        d = asdict(self)
        d["reentrenos"] = list(self.reentrenos)
        return d

    def huella(self) -> str:
        texto = json.dumps(self.como_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(texto.encode()).hexdigest()


# Plazos del enunciado (segundos): no se relajan.
PLAZOS = {
    "arranque": 600,              # /health ok tras levantar
    "reentrenamiento": 600,       # 202 -> resultado
    "promocion": 120,             # fin del entrenamiento -> la API sirve la versión
    "degradado": 60,              # MLflow detenido -> /health degradado
    "recuperado": 120,            # MLflow de vuelta -> /health ok
}

INTERVALO_SONDEO = 5.0            # consulta de ejecuciones pendientes


def _programados(dias: int) -> tuple[int, ...]:
    # día 7k+1: ya están registradas las observaciones de 7k días (R2 exige 7)
    return tuple(d for d in range(8, dias + 1, 7))


COMPLETO = Escenario(nombre="completo", fuente="raw", corte="2025-11-30", dias=90,
                     reentrenos=_programados(90), dia_repetir_clave=8, dia_mlflow=47,
                     dia_stop_start=30, dia_down_up=60, criterio_costo=True)

# CI: datos sintéticos (tests/ml/sinteticos.generar_contrato), 400 días de histórico y 14
# simulados, como el e2e de la etapa 7.
CORTO = Escenario(nombre="corto", fuente="sintetico", corte="2026-04-13", dias=14,
                  reentrenos=(8,), dia_repetir_clave=8, dia_mlflow=11, dia_stop_start=10,
                  dia_down_up=13)

# S2: independiente y opcional; nunca entra en el informe ni en los criterios de COMPLETO.
CAMBIO_DEMANDA = Escenario(nombre="cambio_demanda", fuente="raw", corte="2025-11-30", dias=90,
                           reentrenos=_programados(90), dia_repetir_clave=8, dia_mlflow=47,
                           dia_stop_start=30, dia_down_up=60,
                           cambio_demanda={"familia": "bebidas", "factor": 1.3, "desde_dia": 46})

ESCENARIOS = {e.nombre: e for e in (COMPLETO, CORTO, CAMBIO_DEMANDA)}

# Histórico sintético del modo corto (mismos valores que el e2e)
SINTETICO = {"fin": "2026-04-27", "dias": 414, "semilla": 0}
