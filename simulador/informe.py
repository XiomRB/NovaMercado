"""Informe de una ejecución: ``trabajo/informe/{resumen.md, metricas.json, ejemplos.json}``.

Solo agregados. ``ejemplos.json`` trae solicitudes y respuestas de **predicciones** (sin
ventas reales) y solo las respuestas de observaciones (sin sus cuerpos), así puede
publicarse (S5) con ``python -m simulador exportar``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

from simulador import defectos

CRITERIOS = {
    1: "Sin intervención manual, desde un solo comando",
    2: "Respuestas conforme al Anexo; sin 5xx inesperados",
    3: "Defectos con el resultado de DECISIONES.md §3 (oráculo)",
    4: "version_modelo coherente con /modelo y registrada igual en PostgreSQL",
    5: "Observaciones idempotentes y guardadas como se esperaba",
    6: "Reentrenamientos: ≤ 10 min, promoción ≤ 120 s, vigente intacto si no promueve o falla",
    7: "MLflow detenido: degradado < 60 s, 200, reentrenamiento fallido, ok < 120 s",
    8: "Reinicios: ok ≤ 10 min, sin reentrenar, datos y ejecuciones conservados",
    9: "Costo del servicio < referencia baseline_dow4 (90 días reales)",
    10: "costos --recalcular reproduce el costo solo con la evidencia, tras la limpieza",
    11: "Imágenes propias del commit; las :local no cambian",
}


def _fmt(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


def construir(estado: dict, esc, respuestas: list[dict]) -> tuple[str, dict, dict]:
    fallos = estado.get("fallos", [])
    por_criterio = Counter(f["criterio"] for f in fallos)
    cerrada = estado.get("fase") == "cerrada"
    aplica = {n: True for n in CRITERIOS}
    aplica[9] = esc.criterio_costo
    resultado = "APROBADA" if cerrada and not fallos else ("NO APROBADA" if cerrada else "INCOMPLETA")
    res = [r for r in respuestas if r.get("fase") == "resultado"]
    conteo = Counter((r["tipo"], r["rol"], r["codigo"]) for r in res)
    defectuosas = [r for r in res if r["defectos"]]
    nombres = Counter(n for r in defectuosas for n in r["defectos"])
    combinadas = sum(1 for r in defectuosas if len(r["defectos"]) > 1)
    c = estado.get("costos") or {}

    m = [f"# Simulación de operación — escenario `{esc.nombre}`", "",
         f"Resultado: **{resultado}**", "",
         "| | |", "|---|---|",
         f"| Ejecución | `{estado['ejecucion']}` (proyecto `{estado['proyecto']}`) |",
         f"| Commit | `{estado['commit']}` |",
         f"| Fuente | {esc.fuente}; histórico inicial hasta {esc.corte}; días {esc.inicio} a {esc.fin} ({esc.dias}) |",
         f"| Duración | {((estado.get('fin') or time.time()) - estado['inicio']) / 60:.0f} min |",
         f"| Reanudaciones | {estado.get('reanudaciones', 0)} |", ""]
    if esc.nombre == "cambio_demanda":
        m += ["> **Escenario sintético: no son ventas reales.** Independiente y opcional (S2); no "
              "interviene en el informe ni en los criterios de la reproducción de 90 días.", ""]
    m += ["## Criterios de aceptación", "", "| # | Criterio | Resultado |", "|---|---|---|"]
    for n, texto in CRITERIOS.items():
        r = "no aplica" if not aplica[n] else ("cumple" if por_criterio[n] == 0 and cerrada else
                                               (f"no cumple ({por_criterio[n]})" if por_criterio[n] else "pendiente"))
        m.append(f"| {n} | {texto} | {r} |")
    m += ["", "Los criterios 10 y 11 se completan con las pruebas I1–I7 de la CI (ver README).", ""]
    s1 = estado.get("s1", {})
    v = s1.get("verificacion", {})
    m += ["## Datos iniciales (S1, S10)", "",
          f"* Histórico preparado: {s1.get('filas_historico')} filas, última fecha {s1.get('max_fecha_historico')}.",
          f"* PostgreSQL antes del día 1: última fecha {v.get('max_fecha')}, {v.get('observaciones')} observaciones; "
          f"corte del primer modelo `{v.get('corte_modelo')}`.",
          "* `feriados.csv` completo: calendario conocido de antemano, única excepción declarada (S10).", ""]
    m += ["## Solicitudes", "", "| Tipo | Rol | Código | Cantidad |", "|---|---|---|---|"]
    m += [f"| {t} | {rol} | {cod} | {n} |" for (t, rol, cod), n in sorted(conteo.items(), key=str)]
    m += ["", f"Con defectos: {len(defectuosas)} ({combinadas} combinadas); tipos distintos: {len(nombres)}.", "",
          "| Defecto | Catálogo | Veces |", "|---|---|---|"]
    m += [f"| {n} | {defectos.DEFECTOS[n][1]} | {k} |" for n, k in sorted(nombres.items())]
    m += ["", "## Reentrenamientos", "",
          "| Día | Tipo | id | Estado | Promovido | Versión | Duración (s) | Promoción (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for x in estado.get("ejecuciones", []):
        t = x.get("terminal") or {}
        m.append(f"| {x['dia']} | {x['tipo']} | `{x.get('id')}` | {t.get('estado')} | {t.get('promovido')} | "
                 f"{t.get('version')} | {x.get('duracion', 0):.0f} | {x.get('promocion_s', '—')} |")
    m += ["", "## Mediciones", ""]
    m += [f"* `{k}`: {v}" for k, v in sorted(estado.get("mediciones", {}).items())]
    m += ["", "## Versiones servidas", ""]
    m += [f"* {x['version']} (vista primero en {x['origen']})" for x in estado.get("versiones_vistas", [])]
    if c:
        m += ["", "## Costo (S9)", "",
              f"* Filas evaluadas: {c['filas']}.",
              f"* Servicio: **{_fmt(c['costo_servicio'])}**; referencia `baseline_dow4`: "
              f"{_fmt(c['costo_referencia'])}; "
              f"reducción {100 * (c['reduccion'] or 0):.1f} %.", ""]
        for eje in ("familia", "tienda", "semana", "version"):
            filas = estado.get("costos_desglose", {}).get(eje, {})
            m += [f"| {eje.capitalize()} | Servicio | Referencia | Reducción |", "|---|---|---|---|"]
            m += [f"| {k} | {_fmt(x['servicio'])} | {_fmt(x['referencia'])} | "
                  f"{100 * (1 - x['servicio'] / x['referencia']) if x['referencia'] else 0:.1f} % |"
                  for k, x in filas.items()]
            m.append("")
    if estado.get("manifiesto"):
        rec = estado.get("recalculo", {})
        m += ["## Evidencia", "",
              f"* SHA-256 de `evidencia.sha256`: `{estado['manifiesto']}`.",
              f"* Limpieza: {estado.get('limpieza', {})}.",
              f"* `costos --recalcular` (solo con la evidencia): servicio {_fmt(rec.get('costo_servicio', 0))}, "
              f"referencia {_fmt(rec.get('costo_referencia', 0))}, {rec.get('filas')} filas.", ""]
    m += ["## Imágenes (S11)", ""]
    m += [f"* {t}: `{x['etiqueta']}` → `{x['id'][:19]}`" for t, x in estado.get("imagenes", {}).items()]
    m += [f"* `:local` antes: {estado.get('protegidas_antes')}; después: {estado.get('protegidas_despues')}", ""]
    m += ["## Fallos", ""]
    m += [f"* criterio {f['criterio']} (día {f['dia']}): {f['detalle']}" for f in fallos] or ["Ninguno."]

    metricas = {"resultado": resultado, "escenario": esc.como_dict(), "commit": estado["commit"],
                "fallos_por_criterio": dict(por_criterio), "costos": c,
                "costos_desglose": estado.get("costos_desglose"),
                "mediciones": estado.get("mediciones"), "ejecuciones": estado.get("ejecuciones"),
                "solicitudes": {f"{t}/{rol}/{cod}": n for (t, rol, cod), n in conteo.items()},
                "defectos": dict(nombres), "manifiesto": estado.get("manifiesto"), "recalculo": estado.get("recalculo")}
    ejemplos: dict = {"prediccion": {}, "observacion": {}}
    for r in res:
        clave = "+".join(r["defectos"]) or "limpia"
        if clave in ejemplos[r["tipo"]]:
            continue
        if r["tipo"] == "prediccion":
            ejemplos["prediccion"][clave] = {"solicitud": r["cuerpo"] if r["texto"] is None else r["texto"],
                                             "codigo": r["codigo"], "respuesta": r["respuesta"]}
        else:
            ejemplos["observacion"][clave] = {"codigo": r["codigo"], "respuesta": r["respuesta"]}
    return "\n".join(m) + "\n", metricas, ejemplos


def escribir(carpeta: Path, estado: dict, esc, respuestas: list[dict]) -> Path:
    texto, metricas, ejemplos = construir(estado, esc, respuestas)
    destino = Path(carpeta) / "trabajo" / "informe"
    destino.mkdir(exist_ok=True)
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
    (destino / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8")
    (destino / "ejemplos.json").write_text(json.dumps(ejemplos, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


def exportar(carpeta: Path, destino: Path) -> list[Path]:
    origen = Path(carpeta) / "trabajo" / "informe"
    Path(destino).mkdir(parents=True, exist_ok=True)
    out = []
    for nombre in ("resumen.md", "metricas.json", "ejemplos.json"):
        shutil.copyfile(origen / nombre, Path(destino) / nombre)
        out.append(Path(destino) / nombre)
    return out
