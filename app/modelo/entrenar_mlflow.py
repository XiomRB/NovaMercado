"""Entrena, evalúa y registra el modelo en MLflow (entrenamiento manual de producción).

Uso:  python -m app.modelo.entrenar_mlflow [--promover si_no_hay|siempre|nunca] [--esperar SEG]

* Primer paso: la **procedencia** del código (``app.modelo.procedencia``): en un
  repositorio Git, árbol limpio (etapa 5); en un contenedor, ``procedencia.json``
  (código y dependencias). Si no se cumple, termina sin crear nada. No hay opción
  para omitirla.
* El entrenamiento corre bajo la concesión de entrenamiento y en un proceso hijo
  (``app.modelo.ciclo_entrenamiento``). Solo el supervisor autoriza y promueve.
* ``--promover``: ``si_no_hay`` (por defecto) promueve solo si PostgreSQL no tiene
  un vigente; ``siempre`` lo reemplaza (acción manual explícita); ``nunca`` solo
  registra y valida. Ninguno omite validaciones.
* ``--esperar``: segundos de espera si otro entrenamiento tiene la concesión (0).

Códigos de salida: 0 candidato válido · 3 candidato no válido · 2 procedencia ·
4 resultado del alias indeterminado (promoción o reconciliación) · 5 otro
entrenamiento en curso · 6 concesión perdida · 1 error (incluida una reconciliación
cuyo alias no se aplicó).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Solo módulos sin dependencias del resto del proyecto hasta verificar la procedencia
from app.modelo.codigos_salida import ERROR, INDETERMINADA, OCUPADO, PERDIDA, PROCEDENCIA
from app.modelo.git_estado import RepositorioNoValido


def ejecutar(modo: str, promover: str, esperar: float | None) -> int:
    from app.modelo.procedencia import verificar_procedencia
    try:
        proc = verificar_procedencia()                 # antes de leer la base o tocar MLflow
    except RepositorioNoValido as exc:
        print(str(exc), file=sys.stderr)
        return PROCEDENCIA
    os.chdir(proc.raiz)

    from app.config import mlflow_alias, mlflow_modelo_nombre, mlflow_tracking_uri, url_base_datos
    from app.modelo.bloqueo import BloqueoOcupado, ConcesionPerdida
    from app.modelo.ciclo_entrenamiento import ejecutar_entrenamiento
    from app.modelo.promocion import PromocionIndeterminada, ReconciliacionIncompleta
    try:
        r = ejecutar_entrenamiento(url_bd=url_base_datos(), tracking_uri=mlflow_tracking_uri(),
                                   nombre=mlflow_modelo_nombre(), alias=mlflow_alias(), modo=modo,
                                   promover=promover, git_commit=proc.commit, etiquetas=proc.etiquetas(),
                                   espera=esperar)
    except BloqueoOcupado as exc:
        print(f"otro entrenamiento en curso: {exc}", file=sys.stderr)
        return OCUPADO
    except ConcesionPerdida as exc:
        print(f"se perdió la concesión de entrenamiento: {exc}", file=sys.stderr)
        return PERDIDA
    except PromocionIndeterminada as exc:
        print(str(exc), file=sys.stderr)
        return INDETERMINADA
    except ReconciliacionIncompleta as exc:
        # no se entrena; se distingue el alias que no se aplicó del que no se pudo determinar
        print(str(exc), file=sys.stderr)
        return INDETERMINADA if exc.resultado == "indeterminada" else ERROR
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("el entrenamiento falló")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return ERROR
    print(r.como_json())
    return r.codigo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entrena, evalúa y registra el modelo en MLflow")
    ap.add_argument("--promover", choices=["si_no_hay", "siempre", "nunca"], default="si_no_hay")
    ap.add_argument("--esperar", type=float, default=None,
                    help="segundos de espera si otro entrenamiento tiene la concesión (por defecto 0)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return ejecutar("manual", a.promover, a.esperar)


if __name__ == "__main__":
    sys.exit(main())
