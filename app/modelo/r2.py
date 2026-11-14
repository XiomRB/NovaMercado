"""Criterio R2 de reemplazo: retador frente al campeón vigente (DECISIONES.md §16, plan
v2 §5, v3 §5.8, v4 §5.3).

* ``corte_c``: corte del candidato (snapshot). ``corte_v``: corte del vigente, leído del
  ``manifiesto.json`` de **su** ejecución, verificado por su huella; la ejecución la
  identifica la decisión confirmada en PostgreSQL (no el alias).
* Ventana ``W``: ``max(corte_v, corte_c − 28 d) ≤ d < corte_c``; cortes semanales
  ``c_k = inicio(W) + 7·k``; bloques ``[c_k, min(c_k + 7, corte_c))``.
* Filas comparables ``F_W``: filas con venta real **y** promoción real del snapshot
  (``filas_supervisadas``) con fecha en ``W``. Una sola tabla de objetivos por bloque.
* Campeón: el ``modelo.joblib`` vigente congelado (SHA de PostgreSQL verificado antes de
  deserializar, formato y humo). Retador (efímero, nunca se registra): por bloque, un
  ``Pronosticador`` con la configuración del candidato entrenado con fecha < ``c_k``.
  Ambos predicen la misma tabla con ``Pronosticador.predecir(objetivos, historia)``,
  con la misma historia (ventas reales del snapshot; las variables solo usan fechas
  anteriores a cada fila) y los mismos costos (``costos.csv`` del snapshot).
* ``R2 = True`` si ``C_ret ≤ C_camp`` (empate con ``isclose(rel=1e-9)``); ``False`` si
  no; ``"no_evaluable"`` con menos de 7 días evaluables; ``"no_disponible"`` si el
  campeón o ``corte_v`` no se pueden obtener o verificar.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.modelo.copia_datos import MANIFIESTO, DatosEntrenamiento, sha256
from ml.datos import CLAVE, filas_supervisadas
from ml.metricas import agregar_costo
from ml.pronosticador import ConfigEntrenamiento, Pronosticador

DIAS_VENTANA = 28
DIAS_BLOQUE = 7
DIAS_MINIMOS = 7
TOL_REL = 1e-9
RUTA_R2 = "r2"
ARCHIVO_R2 = "resultados_por_bloque.csv"
COLUMNAS = ["bloque", "inicio", "fin", "filas", "dias", "costo_retador", "costo_vigente"]
NO_EVALUABLE = "no_evaluable"
NO_DISPONIBLE = "no_disponible"


class CampeonNoDisponible(RuntimeError):
    """No se pudo obtener o verificar el campeón o su ``corte_v``."""


@dataclass
class ResultadoR2:
    resultado: bool | str                 # True | False | "no_evaluable" | "no_disponible"
    motivo: str
    costo_retador: float | None = None
    costo_vigente: float | None = None
    dias: int = 0
    filas: int = 0
    corte_v: str | None = None
    por_bloque: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNAS))

    def tabla_csv(self) -> bytes:
        return self.por_bloque[COLUMNAS].to_csv(index=False, lineterminator="\n").encode("utf-8")

    def metricas(self) -> dict[str, float]:
        m = {"r2_dias": float(self.dias), "r2_filas": float(self.filas)}
        if self.costo_retador is not None:
            m.update(r2_costo_retador=float(self.costo_retador), r2_costo_vigente=float(self.costo_vigente))
        return m


def no_disponible(motivo: str) -> ResultadoR2:
    return ResultadoR2(NO_DISPONIBLE, f"R2 no disponible: {motivo}")


def ventana(corte_v: pd.Timestamp, corte_c: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    inicio = max(pd.Timestamp(corte_v), pd.Timestamp(corte_c) - pd.Timedelta(days=DIAS_VENTANA))
    return inicio, pd.Timestamp(corte_c)


def bloques(inicio: pd.Timestamp, fin: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out, c = [], pd.Timestamp(inicio)
    while c < fin:
        out.append((c, min(c + pd.Timedelta(days=DIAS_BLOQUE), fin)))
        c += pd.Timedelta(days=DIAS_BLOQUE)
    return out


def _costo(pred: pd.DataFrame, objetivos: pd.DataFrame, costos: pd.DataFrame) -> float:
    claves_pred = pred[CLAVE].sort_values(CLAVE).reset_index(drop=True)
    claves_obj = objetivos[CLAVE].sort_values(CLAVE).reset_index(drop=True)
    if not claves_pred.equals(claves_obj):
        raise ValueError("filas no comparables")
    d = pred.merge(objetivos[CLAVE + ["real"]], on=CLAVE, how="inner")
    return float(agregar_costo(d, costos)["costo"].sum())


def calcular(datos: DatosEntrenamiento, campeon: Pronosticador, corte_v: pd.Timestamp,
             config: ConfigEntrenamiento) -> ResultadoR2:
    """R2 desde el snapshot del candidato (``datos``) y el campeón congelado."""
    corte_c = pd.Timestamp(datos.corte)
    corte_v = pd.Timestamp(corte_v)
    inicio, fin = ventana(corte_v, corte_c)
    sup, _ = filas_supervisadas(datos.ventas, datos.promociones)
    fw = sup[(sup["fecha"] >= inicio) & (sup["fecha"] < fin)].rename(columns={"ventas": "real"})
    dias = int(fw["fecha"].nunique())
    base = {"corte_v": corte_v.strftime("%Y-%m-%d"), "dias": dias, "filas": int(len(fw))}
    if dias < DIAS_MINIMOS:
        return ResultadoR2(NO_EVALUABLE, f"datos nuevos insuficientes para comparar con el vigente "
                                         f"({dias} < {DIAS_MINIMOS} días)", **base)
    historia = datos.ventas[CLAVE + ["ventas"]].copy()
    historia["fecha"] = pd.to_datetime(historia["fecha"])
    filas = []
    for k, (a, b) in enumerate(bloques(inicio, fin)):
        obj = fw[(fw["fecha"] >= a) & (fw["fecha"] < b)].sort_values(CLAVE).reset_index(drop=True)
        if obj.empty:
            continue                                   # bloque sin filas: se omite para ambos
        objetivos = obj[CLAVE + ["en_promocion"]]
        retador = Pronosticador.entrenar(datos.ventas, datos.promociones, datos.tiendas, datos.feriados,
                                         datos.costos, datos.familia_ids, a, config)
        hist = historia[historia["fecha"] < b]
        c_ret = _costo(retador.predecir(objetivos, hist), obj, datos.costos)
        c_camp = _costo(campeon.predecir(objetivos, hist), obj, datos.costos)
        filas.append({"bloque": k, "inicio": a.strftime("%Y-%m-%d"), "fin": b.strftime("%Y-%m-%d"),
                      "filas": int(len(obj)), "dias": int(obj["fecha"].nunique()),
                      "costo_retador": c_ret, "costo_vigente": c_camp})
    tabla = pd.DataFrame(filas, columns=COLUMNAS)
    c_ret, c_camp = float(tabla["costo_retador"].sum()), float(tabla["costo_vigente"].sum())
    cumple = c_ret <= c_camp or math.isclose(c_ret, c_camp, rel_tol=TOL_REL)
    motivo = (f"Costo en validación {c_ret:,.0f} del candidato vs {c_camp:,.0f} del vigente "
              f"({dias} días, {len(fw)} filas)")
    return ResultadoR2(bool(cumple), motivo, c_ret, c_camp, por_bloque=tabla, **base)


# --------------------------------------------------------------- campeón verificado
def obtener_campeon(tracking_uri: str, campeon: dict, destino: Path) -> tuple[Pronosticador, pd.Timestamp]:
    """Descarga y verifica el campeón de la decisión confirmada que pasa el supervisor:
    ``modelo.joblib`` con el SHA de PostgreSQL (antes de deserializar), formato y humo;
    ``corte_v`` del manifiesto de su ejecución, verificado con su ``huella_datos`` y
    coherente con el corte guardado en el propio artefacto."""
    from mlflow import MlflowClient

    from app.modelo.artefacto import validar_archivo
    from app.modelo.mlflow_util import descargar_de_ejecucion
    from app.modelo.proveedores import VersionNoUtilizable
    try:
        ruta = descargar_de_ejecucion(tracking_uri, campeon["run_id"], campeon["ruta_artefacto"],
                                      str(destino / "campeon"))
        p = validar_archivo(ruta, campeon["sha256_artefacto"])["pronosticador"]
    except VersionNoUtilizable as exc:
        raise CampeonNoDisponible(f"el artefacto vigente no supera la verificación ({exc})") from exc
    except Exception as exc:  # noqa: BLE001
        raise CampeonNoDisponible(f"no se pudo descargar el artefacto vigente ({type(exc).__name__})") from exc
    try:
        huella = MlflowClient(tracking_uri=tracking_uri).get_run(campeon["run_id"]).data.tags.get("huella_datos")
        man = descargar_de_ejecucion(tracking_uri, campeon["run_id"], f"datos/{MANIFIESTO}",
                                     str(destino / "manifiesto_vigente"))
        texto = Path(man).read_bytes()
    except Exception as exc:  # noqa: BLE001
        raise CampeonNoDisponible(f"la ejecución del vigente no tiene una copia de datos legible "
                                  f"({type(exc).__name__})") from exc
    if not huella or sha256(texto) != huella:
        raise CampeonNoDisponible("el manifiesto del vigente no coincide con su huella_datos")
    try:
        corte_v = pd.Timestamp(json.loads(texto)["corte"])
    except (ValueError, KeyError) as exc:
        raise CampeonNoDisponible("el manifiesto del vigente no tiene corte") from exc
    if pd.Timestamp(p.info.get("corte")) != corte_v:
        raise CampeonNoDisponible("el corte del artefacto vigente no coincide con el de su manifiesto")
    return p, corte_v


def evaluar_con_campeon(datos: DatosEntrenamiento, config: ConfigEntrenamiento, tracking_uri: str,
                        campeon: dict | None) -> ResultadoR2:
    if not campeon:
        return no_disponible("no hay un vigente confirmado")
    with tempfile.TemporaryDirectory(prefix="nm_r2_") as tmp:
        try:
            p, corte_v = obtener_campeon(tracking_uri, campeon, Path(tmp))
        except CampeonNoDisponible as exc:
            return no_disponible(str(exc))
    try:
        return calcular(datos, p, corte_v, config)
    except ValueError as exc:
        return ResultadoR2(False, f"R2 falla cerrado: {exc}")


# ----------------------------------------------------------------- reproducción
def leer_tabla_csv(crudo: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(crudo), dtype={"inicio": str, "fin": str})


def comparar(registrada: pd.DataFrame, reproducida: pd.DataFrame) -> list[str]:
    """Enteros y fechas exactos; costos con tolerancia (como T4a)."""
    a = registrada.reset_index(drop=True)
    b = reproducida[COLUMNAS].reset_index(drop=True)
    if len(a) != len(b):
        return [f"R2: {len(a)} bloques registrados frente a {len(b)} reproducidos"]
    difs = []
    for col in ("bloque", "inicio", "fin", "filas", "dias"):
        x, y = a[col].astype(str).to_numpy(), b[col].astype(str).to_numpy()
        for i in np.flatnonzero(x != y):
            difs.append(f"R2 bloque {i} {col}: {x[i]} != {y[i]}")
    for col in ("costo_retador", "costo_vigente"):
        for i in range(len(a)):
            x, y = float(a.at[i, col]), float(b.at[i, col])
            if not math.isclose(x, y, rel_tol=TOL_REL, abs_tol=1e-9):
                difs.append(f"R2 bloque {i} {col}: {x!r} != {y!r}")
    return difs
