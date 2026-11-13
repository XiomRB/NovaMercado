"""Primer modelo del producto (servicio ``entrenamiento-inicial`` de Compose, plan §6).

Uso:  python -m app.modelo.arranque

1. Verifica la procedencia (código 2 si falla).
2. Bajo la concesión de entrenamiento (espera hasta 15 min si otro la tiene):
   reconciliación y consulta del **vigente en PostgreSQL**. Si existe, termina con 0
   sin entrenar (cada reinicio del producto).
3. Entrena el candidato (``si_no_hay``). Si falla **solo** C1–C3 y sigue sin haber
   vigente, entrena y promueve el **respaldo baseline** en su propia ejecución.

Códigos de salida: 0 hay vigente · 3 sin modelo utilizable (D1, T1–T4b o el respaldo
también falló) · 2 procedencia · 4 promoción indeterminada · 5 concesión ocupada ·
6 concesión perdida · 1 error (PostgreSQL o MLflow no disponibles, excepción).
"""

from __future__ import annotations

import logging
import sys

from app.modelo.entrenar_mlflow import ejecutar


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return ejecutar("arranque", "si_no_hay", None)


if __name__ == "__main__":
    sys.exit(main())
