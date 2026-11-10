"""Entrena el modelo con la base del producto y guarda el artefacto local.

Uso:  python -m app.modelo.entrenar_local [--salida modelos/modelo_local.joblib]

Se ejecuta **explícitamente**: la API nunca entrena al iniciar. La API en
ejecución detecta el artefacto nuevo y lo carga sin reiniciarse. La escritura
es atómica (archivo temporal + rename) para que la API nunca lea un archivo a
medias. ``modelos/`` está excluida de git.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from app.config import ruta_modelo
from app.db.conexion import crear_motor
from app.modelo.entrenamiento import entrenar_desde_bd


def guardar_atomico(datos: bytes, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=destino.parent, prefix=".modelo_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(datos)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, destino)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", default=None, help="ruta del artefacto (por defecto MODELO_RUTA)")
    a = ap.parse_args()
    destino = Path(a.salida) if a.salida else ruta_modelo()
    motor = crear_motor()
    try:
        r = entrenar_desde_bd(motor)
    finally:
        motor.dispose()
    guardar_atomico(r.artefacto, destino)
    info = r.pronosticador.info
    print(f"modelo {r.version} guardado en {destino}")
    print(f"  corte={info['corte'].date()} filas_entrenamiento={info['filas_entrenamiento']} "
          f"filas_sin_promocion={info['filas_sin_promocion']} "
          f"discrepancias_promocion={info['discrepancias_promocion']}")


if __name__ == "__main__":
    main()
