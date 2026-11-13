"""Reproduce una ejecución registrada usando solo lo guardado en MLflow.

Uso:  python -m app.modelo.reproducir --run-id <id>

1. Descarga la copia de datos y verifica su huella y cada SHA-256.
2. Repite los cortes del backtest y compara con los registrados (T4a).
3. Reentrena el modelo definitivo y compara con la versión registrada:
   * ``identica``: mismo SHA-256 de ``modelo.joblib`` y backtest igual;
   * ``equivalente``: SHA distinto, pero predicciones (últimos ``horizonte`` días
     de la copia y prueba de humo) dentro de la tolerancia, cantidades iguales y
     backtest igual;
   * ``no_reproduce``: cualquier otra cosa.

No registra nada ni toca el alias. Advierte si las bibliotecas o el commit del
código actual difieren de los registrados.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import configurar_cliente_mlflow, mlflow_tracking_uri
from app.modelo.artefacto import NOMBRE_ARCHIVO, prueba_de_humo, sha256_bytes, validar_archivo
from app.modelo.copia_datos import CopiaDatosInvalida, leer_copia
from app.modelo.evaluacion import TOL_ABS, TOL_REL, comparar_metricas, evaluar
from app.modelo.mlflow_util import descargar_de_ejecucion
from app.modelo.registro import (
    HILOS_POR_DEFECTO,
    config_desde_manifiesto,
    descargar_backtest,
    descargar_copia,
    entorno,
    hilos_fijos,
    modelos_de_manifiesto,
)
from ml.datos import CLAVE, filas_supervisadas


@dataclass
class ResultadoReproduccion:
    estado: str                       # identica | equivalente | no_reproduce
    backtest_igual: bool
    sha_registrado: str | None
    sha_reproducido: str | None
    diferencias: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)


def _objetivos(datos, horizonte: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sup, _ = filas_supervisadas(datos.ventas, datos.promociones)
    desde = sup["fecha"].max() - pd.Timedelta(days=horizonte - 1)
    obj = sup[sup["fecha"] >= desde][CLAVE + ["en_promocion"]].reset_index(drop=True)
    return obj, datos.ventas[CLAVE + ["ventas"]]


def comparar_predicciones(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    difs = []
    if not (a["cantidad_sugerida"].to_numpy() == b["cantidad_sugerida"].to_numpy()).all():
        difs.append("cantidad_sugerida distinta en al menos una fila")
    va, vb = a["venta_esperada"].to_numpy(float), b["venta_esperada"].to_numpy(float)
    malas = [i for i in range(len(va)) if not math.isclose(va[i], vb[i], rel_tol=TOL_REL, abs_tol=TOL_ABS)]
    if malas:
        difs.append(f"venta_esperada fuera de tolerancia en {len(malas)} filas (máx. dif. "
                    f"{np.max(np.abs(va[malas] - vb[malas])):.3g})")
    return difs


def reproducir(run_id: str, tracking_uri: str, cliente=None) -> ResultadoReproduccion:
    configurar_cliente_mlflow()
    from mlflow import MlflowClient

    from app.modelo.registro import entrenar_definitivo
    cliente = cliente or MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    run = cliente.get_run(run_id)
    tags = run.data.tags
    versiones = cliente.search_model_versions(f"run_id='{run_id}'")
    if not versiones:
        return ResultadoReproduccion("no_reproduce", False, None, None, ["la ejecución no registró una versión"])
    mv = versiones[0]
    vt = dict(mv.tags or {})
    sha_reg = vt.get("sha256_artefacto")
    with tempfile.TemporaryDirectory(prefix="nm_reproducir_") as tmp:
        tmp = Path(tmp)
        try:
            datos, manifiesto = leer_copia(descargar_copia(tracking_uri, run_id, tmp / "copia"), tags["huella_datos"])
            registrada = descargar_backtest(tracking_uri, run_id, tmp / "bt", vt["sha256_backtest"])
        except (CopiaDatosInvalida, ValueError, KeyError) as exc:
            return ResultadoReproduccion("no_reproduce", False, sha_reg, None, [f"copia o backtest inválidos: {exc}"])
        advertencias = []
        actual = entorno()["librerias"]
        if manifiesto["entorno"]["librerias"] != actual:
            advertencias.append(f"bibliotecas distintas: registradas {manifiesto['entorno']['librerias']}, "
                                f"actuales {actual}")
        try:
            from app.modelo.git_estado import RepositorioNoValido, verificar_arbol_limpio
            git = verificar_arbol_limpio()
            if git.commit != manifiesto.get("git_commit"):
                advertencias.append(f"el código actual ({git.commit[:12]}) no es el commit registrado "
                                    f"({str(manifiesto.get('git_commit'))[:12]})")
        except RepositorioNoValido as exc:
            advertencias.append(f"no se pudo verificar el código actual: {str(exc).splitlines()[0]}")

        cfg, ev = config_desde_manifiesto(manifiesto)
        with hilos_fijos(manifiesto.get("hilos_entrenamiento", HILOS_POR_DEFECTO)):   # mismos hilos que el original
            difs = comparar_metricas(registrada, evaluar(datos, ev, cfg, modelos_de_manifiesto(manifiesto)).por_corte)
            p, art = entrenar_definitivo(datos, cfg, mv.name)
        backtest_igual = not difs
        sha_rep = sha256_bytes(art.artefacto)
        if backtest_igual and sha_rep == sha_reg:
            return ResultadoReproduccion("identica", True, sha_reg, sha_rep, [], advertencias)

        ruta = descargar_de_ejecucion(tracking_uri, run_id, vt["ruta_artefacto"], str(tmp / "reg"))
        original = validar_archivo(ruta, sha_reg)["pronosticador"]
        obj, ventas = _objetivos(datos, ev.horizonte)
        difs += comparar_predicciones(original.predecir(obj, ventas), p.predecir(obj, ventas))
        prueba_de_humo(p)
        estado = "equivalente" if not difs else "no_reproduce"
        return ResultadoReproduccion(estado, backtest_igual, sha_reg, sha_rep, difs, advertencias)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce una ejecución de entrenamiento registrada en MLflow")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    r = reproducir(a.run_id, mlflow_tracking_uri())
    print(f"reproducción: {r.estado}")
    print(f"  backtest igual: {r.backtest_igual}")
    print(f"  {NOMBRE_ARCHIVO}: registrado {r.sha_registrado} · reproducido {r.sha_reproducido}")
    for d in r.diferencias:
        print(f"  diferencia: {d}")
    for w in r.advertencias:
        print(f"  advertencia: {w}")
    return 0 if r.estado in ("identica", "equivalente") else 1


if __name__ == "__main__":
    sys.exit(main())
