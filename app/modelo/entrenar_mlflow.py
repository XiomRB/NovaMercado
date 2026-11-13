"""Entrena, evalúa y registra el modelo en MLflow (entrenamiento de producción).

Uso:  python -m app.modelo.entrenar_mlflow [--promover si_no_hay|siempre|nunca]

* Primer paso: el repositorio del código debe estar **limpio** (sin cambios sin
  confirmar). Si no, termina con un mensaje claro y sin crear nada en MLflow. No
  hay opción para omitir esta comprobación.
* ``--promover``: ``si_no_hay`` (por defecto) asigna el alias solo si no existe;
  ``siempre`` lo reemplaza (acción manual explícita); ``nunca`` solo registra.
  En todos los casos el alias solo se mueve si el candidato es **válido**.

Códigos de salida: 0 candidato válido · 3 candidato registrado pero no válido ·
2 repositorio no limpio o no verificable · 4 promoción indeterminada (falló la
asignación del alias y no se pudo consultar; ver el alias en MLflow) · 1 error.
"""

from __future__ import annotations

import argparse
import os
import sys

from app.modelo.git_estado import RepositorioNoValido, verificar_arbol_limpio


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entrena, evalúa y registra el modelo en MLflow")
    ap.add_argument("--promover", choices=["si_no_hay", "siempre", "nunca"], default="si_no_hay")
    a = ap.parse_args(argv)
    try:
        git = verificar_arbol_limpio()                 # antes de leer la base o tocar MLflow
    except RepositorioNoValido as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.chdir(git.raiz)                                 # trabaja explícitamente desde la raíz

    from app.config import mlflow_alias, mlflow_modelo_nombre, mlflow_tracking_uri
    from app.db.conexion import crear_motor
    from app.modelo.registro import PromocionIndeterminada, entrenar_y_registrar, resumen_json

    motor = crear_motor(timeout_consulta=300)          # proceso por lotes: límite amplio
    try:
        r = entrenar_y_registrar(motor, tracking_uri=mlflow_tracking_uri(), nombre=mlflow_modelo_nombre(),
                                 alias=mlflow_alias(), promover=a.promover, git_commit=git.commit)
    except PromocionIndeterminada as exc:
        print(str(exc), file=sys.stderr)
        return 4
    finally:
        motor.dispose()
    print(resumen_json(r))
    return 0 if r.valido else 3


if __name__ == "__main__":
    sys.exit(main())
