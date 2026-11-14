"""Reconciliación periódica (DAG ``novamercado_reconciliacion``, etapa 7, cada 5 min).

Uso:  python -m app.modelo.reconciliacion

Procedencia primero; luego ``ejecutar_entrenamiento(modo="reconciliacion")`` toma la
concesión **sin esperar**: si otro proceso la tiene, código 5 (la tarea queda
``skipped``, nunca compite con un entrenamiento). Con la concesión, la reconciliación de
la etapa 6 (anula decisiones huérfanas y refleja el vigente de PostgreSQL en el alias).
"""

from __future__ import annotations

import logging
import sys

from app.modelo.entrenar_mlflow import ejecutar


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return ejecutar("reconciliacion", "nunca", 0.0)


if __name__ == "__main__":
    sys.exit(main())
