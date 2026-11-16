"""Costo de las sugerencias frente a ``baseline_dow4`` (S9, plan v2 §5).

* Conjunto: cada tienda abierta x cada familia x cada fecha simulada (``escenario.json``).
* Servicio: ``cantidad_sugerida`` de la **primera respuesta 200** de cada tienda y fecha
  (la primera solicitud del día puede llevar defectos; si recibió 422, la del reenvío limpio).
* Referencia: media de la venta real de la misma tienda y familia en d-7, d-14, d-21 y
  d-28, redondeo aritmético (0,5 hacia arriba), mínimo 0. Solo ventas con fecha < d.
* costo = cf * max(real - q, 0) + cs * max(q - real, 0), suma en coma flotante de 64 bits
  en orden fijo (fecha, tienda, familia), sin redondear.

``recalcular`` solo lee ``evidencia/`` (verificada con su manifiesto): nunca ``raw/`` ni
``datos/``.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

from simulador import evidencia


class CostoNoCalculable(Exception):
    pass


def referencia(ventas: dict, tienda: int, familia: str, d: date) -> int:
    s = 0
    for k in (7, 14, 21, 28):
        clave = (d - timedelta(days=k), tienda, familia)
        if clave not in ventas:
            raise CostoNoCalculable(f"falta la venta de {clave} para la referencia")
        s += ventas[clave]
    return (s + 2) // 4               # redondeo aritmético de s/4 (s >= 0)


def costo_fila(real: int, q: int, cf: float, cs: float) -> float:
    return cf * max(real - q, 0) + cs * max(q - real, 0)


def sugerencias(respuestas: list[dict]) -> dict[tuple[date, int], dict]:
    """(fecha, tienda) -> {"cantidades": {familia: q}, "version": v, "seq": n} de la primera 200."""
    out: dict = {}
    for r in sorted((r for r in respuestas if r.get("fase") == "resultado" and r.get("tipo") == "prediccion"
                     and r.get("codigo") == 200), key=lambda r: r["seq"]):
        k = (date.fromisoformat(r["fecha"]), r["tienda"])
        if k in out:
            continue
        out[k] = {"cantidades": {p["familia_id"]: p["cantidad_sugerida"] for p in r["respuesta"]["predicciones"]},
                  "version": r["respuesta"]["version_modelo"], "seq": r["seq"]}
    return out


def calcular(ventas: dict, costos: dict[str, tuple[float, float]], conjunto: dict,
             respuestas: list[dict]) -> dict:
    sug = sugerencias(respuestas)
    inicio, fin = date.fromisoformat(conjunto["inicio"]), date.fromisoformat(conjunto["fin"])
    aperturas = {int(t): date.fromisoformat(a) for t, a in conjunto["aperturas"].items()}
    total_s = total_b = 0.0
    filas = 0
    desglose = {"familia": {}, "tienda": {}, "semana": {}, "version": {}}
    faltan = []
    d = inicio
    while d <= fin:
        for t in conjunto["tiendas"]:
            if aperturas[t] > d:
                continue
            s = sug.get((d, t))
            if s is None:
                faltan.append(f"{d} tienda {t}")
                continue
            for f in conjunto["familias"]:
                real = ventas[(d, t, f)]
                cf, cs = costos[f]
                cs_ = costo_fila(real, s["cantidades"][f], cf, cs)
                cb_ = costo_fila(real, referencia(ventas, t, f, d), cf, cs)
                total_s += cs_
                total_b += cb_
                filas += 1
                semana = (d - inicio).days // 7 + 1
                for eje, clave in (("familia", f), ("tienda", str(t)), ("semana", str(semana)),
                                   ("version", s["version"])):
                    acc = desglose[eje].setdefault(clave, [0.0, 0.0, 0])
                    acc[0] += cs_
                    acc[1] += cb_
                    acc[2] += 1
        d += timedelta(days=1)
    if faltan:
        raise CostoNoCalculable(f"{len(faltan)} tienda-días sin respuesta 200 (p. ej. {faltan[0]})")
    return {"costo_servicio": total_s, "costo_referencia": total_b, "filas": filas,
            "reduccion": (1 - total_s / total_b) if total_b else None,
            "desglose": {e: {k: {"servicio": v[0], "referencia": v[1], "filas": v[2]} for k, v in sorted(m.items())}
                         for e, m in desglose.items()}}


def leer_evidencia(carpeta: Path, respuestas: Path | None = None) -> tuple[dict, dict, dict, list[dict]]:
    """Lee ``ventas_referencia.csv``, ``costos.csv`` y ``escenario.json`` de ``carpeta`` y las
    respuestas de ``carpeta/respuestas.jsonl`` (o de ``respuestas`` antes del sellado)."""
    carpeta = Path(carpeta)
    ventas = {}
    with open(carpeta / "ventas_referencia.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ventas[(date.fromisoformat(r["fecha"]), int(r["tienda_id"]), r["familia_id"])] = int(r["ventas"])
    with open(carpeta / "costos.csv", encoding="utf-8", newline="") as fh:
        costos = {r["familia_id"]: (float(r["costo_faltante"]), float(r["costo_sobrante"])) for r in csv.DictReader(fh)}
    conjunto = json.loads((carpeta / "escenario.json").read_text())["conjunto"]
    ruta = Path(respuestas) if respuestas is not None else carpeta / "respuestas.jsonl"
    respuestas = [json.loads(x) for x in ruta.read_text().splitlines() if x.strip()]
    return ventas, costos, conjunto, respuestas


def recalcular(carpeta_evidencia: Path) -> dict:
    """Verifica el manifiesto y recalcula solo con ``evidencia/``."""
    evidencia.verificar(carpeta_evidencia)
    return calcular(*leer_evidencia(carpeta_evidencia))
