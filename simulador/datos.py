"""Preparación de los datos de una ejecución (S1, S2, S10).

* ``datos/``: los cinco archivos que carga el producto (``app/db/carga.ORDEN``). El
  histórico se corta en ``corte`` (fechas ≤ corte); ``tiendas``, ``familias``, ``costos`` y
  ``feriados`` se copian byte a byte (``feriados.csv`` completo: calendario conocido de
  antemano, única excepción declarada, S10).
* ``trabajo/dias_simulados.csv``: ventas y promociones de los días simulados (la "verdad").
* ``evidencia/``: lo mínimo para ``costos --recalcular`` (ver :mod:`simulador.evidencia`).

Solo biblioteca estándar; el modo sintético usa ``tests/ml/sinteticos.py``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from simulador import evidencia
from simulador.escenario import SINTETICO, Escenario

ARCHIVOS = ("tiendas.csv", "familias.csv", "costos.csv", "feriados.csv", "historico_ventas.csv")
DIAS_REFERENCIA = 28              # baseline_dow4 usa d-7, d-14, d-21 y d-28


class DatosInvalidos(Exception):
    pass


def sha256_archivo(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def redondear(x: Decimal) -> int:
    """Redondeo aritmético (0,5 hacia arriba), como DECISIONES.md §3."""
    return int(x.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def generar_fuente_sintetica(destino: Path) -> None:
    from tests.ml.sinteticos import escribir_carpeta, generar_contrato
    d = generar_contrato(fin=SINTETICO["fin"], dias=SINTETICO["dias"], semilla=SINTETICO["semilla"])
    escribir_carpeta(d, destino)


def _leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def leer_catalogo(carpeta: Path) -> tuple[dict[int, date], list[tuple[str, str]]]:
    tiendas = {int(r["tienda_id"]): date.fromisoformat(r["fecha_apertura"])
               for r in _leer_csv(carpeta / "tiendas.csv")}
    familias = [(r["familia_id"], r["nombre"]) for r in _leer_csv(carpeta / "familias.csv")]
    return tiendas, familias


def _escribir_csv(columnas: list[str], filas: list[tuple]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columnas)
    w.writerows(filas)
    return buf.getvalue().encode()


def preparar(esc: Escenario, fuente: Path, carpeta: Path, commit: str) -> dict:
    """Escribe ``datos/``, ``trabajo/dias_simulados.csv`` y la evidencia previa al sellado.
    Devuelve un resumen con las comprobaciones de S1."""
    fuente, carpeta = Path(fuente), Path(carpeta)
    faltan = [a for a in ARCHIVOS if not (fuente / a).is_file()]
    if faltan:
        raise DatosInvalidos(f"faltan archivos en {fuente}: {', '.join(faltan)}")
    datos, trabajo, evid = carpeta / "datos", carpeta / "trabajo", carpeta / "evidencia"
    for d in (datos, trabajo, evid):
        d.mkdir(mode=0o700, parents=True, exist_ok=False)
    # el contenedor (otro usuario) lee datos/ por el bind mount; la carpeta de la ejecución
    # sigue en 0700, así que ningún otro usuario del equipo puede llegar a ella
    datos.chmod(0o755)

    tiendas, familias = leer_catalogo(fuente)
    ids_familia = [f for f, _ in familias]
    corte, inicio, fin = date.fromisoformat(esc.corte), esc.inicio, esc.fin
    desde_ref = inicio - timedelta(days=DIAS_REFERENCIA)

    texto = (fuente / "historico_ventas.csv").read_text(encoding="utf-8")
    lineas = texto.splitlines()
    cabecera = lineas[0]
    if cabecera.split(",") != ["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"]:
        raise DatosInvalidos(f"cabecera inesperada en historico_ventas.csv: {cabecera}")
    inicial, verdad = [cabecera], {}
    for r in csv.DictReader(io.StringIO(texto)):
        f = date.fromisoformat(r["fecha"])
        clave = (f, int(r["tienda_id"]), r["familia_id"])
        if f <= corte:
            inicial.append(",".join(r[c] for c in ("fecha", "tienda_id", "familia_id", "ventas", "en_promocion")))
        if desde_ref <= f <= fin:
            verdad[clave] = (int(r["ventas"]), int(r["en_promocion"]))

    cambio = esc.cambio_demanda
    if cambio:
        desde = esc.fecha(cambio["desde_dia"])
        factor = Decimal(str(cambio["factor"]))
        for (f, t, fam), (v, p) in list(verdad.items()):
            if fam == cambio["familia"] and f >= desde:
                verdad[(f, t, fam)] = (redondear(Decimal(v) * factor), p)

    # cuadrícula completa: cada tienda abierta x cada familia, cada día de la ventana
    huecos = []
    d = desde_ref
    while d <= fin:
        for t, apertura in tiendas.items():
            if apertura <= d:
                huecos += [(d, t, fam) for fam in ids_familia if (d, t, fam) not in verdad]
        d += timedelta(days=1)
    if huecos:
        raise DatosInvalidos(f"faltan {len(huecos)} ventas en {desde_ref}..{fin} (p. ej. {huecos[0]}); "
                             "la referencia y la simulación necesitan la cuadrícula completa")

    (datos / "historico_ventas.csv").write_text("\n".join(inicial) + "\n", encoding="utf-8")
    for a in ARCHIVOS[:-1]:
        (datos / a).write_bytes((fuente / a).read_bytes())

    # S1: ninguna venta posterior al corte en la carpeta que se monta
    fechas = [date.fromisoformat(r["fecha"]) for r in _leer_csv(datos / "historico_ventas.csv")]
    if not fechas or max(fechas) != corte:
        raise DatosInvalidos(f"el histórico preparado termina en {max(fechas) if fechas else None}, no en {corte}")

    simulados = sorted((k, v) for k, v in verdad.items() if inicio <= k[0] <= fin)
    (trabajo / "dias_simulados.csv").write_bytes(_escribir_csv(
        ["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"],
        [(f.isoformat(), t, fam, v, p) for (f, t, fam), (v, p) in simulados]))

    referencia = sorted(verdad.items())
    evidencia.escribir_unica(evid / "ventas_referencia.csv", _escribir_csv(
        ["fecha", "tienda_id", "familia_id", "ventas"],
        [(f.isoformat(), t, fam, v) for (f, t, fam), (v, _) in referencia]))
    evidencia.escribir_unica(evid / "costos.csv", (fuente / "costos.csv").read_bytes())
    conjunto = {"inicio": inicio.isoformat(), "fin": fin.isoformat(),
                "tiendas": sorted(tiendas), "familias": ids_familia,
                "aperturas": {str(t): a.isoformat() for t, a in sorted(tiendas.items())}}
    evidencia.escribir_unica(evid / "escenario.json", json.dumps(
        {"escenario": esc.como_dict(), "huella": esc.huella(), "commit": commit, "conjunto": conjunto},
        ensure_ascii=False, indent=2, sort_keys=True).encode())
    origen = {"fuente": {a: sha256_archivo(fuente / a) for a in ARCHIVOS},
              "datos": {a: sha256_archivo(datos / a) for a in ARCHIVOS},
              "nota": "trazabilidad; costos --recalcular no usa estos archivos"}
    evidencia.escribir_unica(evid / "origen.json", json.dumps(origen, indent=2, sort_keys=True).encode())
    maximos = maximos_promocion(datos)
    return {"corte": corte.isoformat(), "max_fecha_historico": max(fechas).isoformat(),
            "filas_historico": len(fechas), "filas_simuladas": len(simulados),
            "filas_referencia": len(referencia), "sha256_datos": origen["datos"], "maximos_promocion": maximos}


def maximos_promocion(datos: Path) -> dict[str, dict[str, int]]:
    """Máximo de ``en_promocion`` por familia en todo el histórico inicial y en sus últimos
    365 días (cotas del oráculo de promociones atípicas, DECISIONES.md §9)."""
    filas = _leer_csv(Path(datos) / "historico_ventas.csv")
    ultima = max(date.fromisoformat(r["fecha"]) for r in filas)
    desde = ultima - timedelta(days=365)
    total: dict[str, int] = {}
    reciente: dict[str, int] = {}
    for r in filas:
        p, fam = int(r["en_promocion"]), r["familia_id"]
        total[fam] = max(total.get(fam, 0), p)
        if date.fromisoformat(r["fecha"]) > desde:
            reciente[fam] = max(reciente.get(fam, 0), p)
    return {"total": total, "reciente": reciente}


def leer_dias(trabajo: Path) -> dict[date, dict[int, dict[str, tuple[int, int]]]]:
    """fecha -> tienda -> familia -> (ventas, en_promocion), en el orden del archivo."""
    out: dict = {}
    for r in _leer_csv(Path(trabajo) / "dias_simulados.csv"):
        f = date.fromisoformat(r["fecha"])
        out.setdefault(f, {}).setdefault(int(r["tienda_id"]), {})[r["familia_id"]] = (
            int(r["ventas"]), int(r["en_promocion"]))
    return out


def ultimas_ventas_iniciales(datos: Path) -> dict[tuple[int, str], date]:
    out: dict = {}
    for r in _leer_csv(Path(datos) / "historico_ventas.csv"):
        k = (int(r["tienda_id"]), r["familia_id"])
        f = date.fromisoformat(r["fecha"])
        if k not in out or f > out[k]:
            out[k] = f
    return out
