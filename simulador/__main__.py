"""Simulador de operación (etapa 8). Ver README.md, sección "Simulador de operación".

    python -m simulador ejecutar --escenario completo --datos raw
    python -m simulador ejecutar --escenario corto            (datos sintéticos, CI)
    python -m simulador reanudar <carpeta>
    python -m simulador limpiar <carpeta>
    python -m simulador costos --recalcular <carpeta>
    python -m simulador exportar <carpeta> reportes/simulacion
    python -m simulador borrar-evidencia <carpeta> --confirmar

Códigos de salida: 0 aprobada; 1 algún criterio no se cumple; 2 error de uso o de entorno;
3 imagen no válida (S11); 10 interrumpida a propósito (--interrumpir-tras-dia).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from simulador import costos, evidencia, imagenes, informe, simulacion
from simulador import entorno as ent
from simulador import estado as est
from simulador.escenario import ESCENARIOS

SALIDA_OK, SALIDA_FALLOS, SALIDA_USO, SALIDA_IMAGEN, SALIDA_INTERRUMPIDA = 0, 1, 2, 3, 10


def _correr(carpeta: Path, esc, interrumpir) -> int:
    estado, entorno = simulacion.abrir(carpeta, esc)
    sim = simulacion.Simulacion(carpeta, esc, estado, entorno, interrumpir_tras=interrumpir)
    try:
        if estado["fase"] == "preparada":
            sim.arrancar()
        else:
            estado["reanudaciones"] = estado.get("reanudaciones", 0) + 1
            sim.restablecer()
        sim.operar()
        simulacion.cerrar(sim)
    except simulacion.Interrumpida as exc:
        print(f"{exc}; reanude con: python -m simulador reanudar {carpeta}")
        return SALIDA_INTERRUMPIDA
    except imagenes.ImagenNoValida as exc:
        print(f"IMAGEN NO VÁLIDA: {exc}", file=sys.stderr)
        informe.escribir(carpeta, estado.datos, esc, estado.respuestas.leer())
        return SALIDA_IMAGEN
    except simulacion.Detener as exc:
        print(f"DETENIDA: {exc}", file=sys.stderr)
        destino = informe.escribir(carpeta, estado.datos, esc, estado.respuestas.leer())
        print(f"informe parcial: {destino / 'resumen.md'}; el entorno se conserva (limpiar con: "
              f"python -m simulador limpiar {carpeta})")
        return SALIDA_FALLOS
    destino = informe.escribir(carpeta, estado.datos, esc, estado.respuestas.leer())
    print((destino / "resumen.md").read_text())
    return SALIDA_OK if not estado["fallos"] else SALIDA_FALLOS


def _escenario_de(carpeta: Path):
    datos = est.leer_json(carpeta / "trabajo" / "estado.json")
    return ESCENARIOS[datos["escenario"]], datos


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m simulador", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="orden", required=True)
    e = sub.add_parser("ejecutar")
    e.add_argument("--escenario", choices=sorted(ESCENARIOS), default="completo")
    e.add_argument("--datos", type=Path, help="carpeta con los archivos publicados (escenarios con fuente raw)")
    e.add_argument("--base", type=Path, default=None, help="carpeta de ejecuciones (fuera del repositorio)")
    e.add_argument("--puerto-base", type=int, default=None, help="desplaza los cuatro puertos (57000 por defecto)")
    e.add_argument("--interrumpir-tras-dia", type=int, default=None, help="solo para pruebas de reanudación")
    r = sub.add_parser("reanudar")
    r.add_argument("carpeta", type=Path)
    r.add_argument("--interrumpir-tras-dia", type=int, default=None)
    li = sub.add_parser("limpiar")
    li.add_argument("carpeta", type=Path)
    c = sub.add_parser("costos")
    c.add_argument("--recalcular", type=Path, required=True, metavar="CARPETA")
    ex = sub.add_parser("exportar")
    ex.add_argument("carpeta", type=Path)
    ex.add_argument("destino", type=Path)
    b = sub.add_parser("borrar-evidencia")
    b.add_argument("carpeta", type=Path)
    b.add_argument("--confirmar", action="store_true")
    a = p.parse_args(argv)

    try:
        if a.orden == "ejecutar":
            esc = ESCENARIOS[a.escenario]
            if esc.fuente == "raw" and not a.datos:
                p.error(f"el escenario {esc.nombre} necesita --datos (carpeta con los archivos publicados)")
            base = a.base or simulacion.carpeta_base()
            if ent.RAIZ in base.resolve().parents or base.resolve() == ent.RAIZ:
                p.error("la carpeta de ejecuciones debe estar fuera del repositorio")
            puertos = dict(ent.PUERTOS_POR_DEFECTO)
            if a.puerto_base:
                puertos = {k: a.puerto_base + (v - 57000) for k, v in puertos.items()}
            carpeta, _ = simulacion.nueva(esc, a.datos, base, puertos)
            return _correr(carpeta, esc, a.interrumpir_tras_dia)
        if a.orden == "reanudar":
            esc, datos = _escenario_de(a.carpeta)
            if datos["fase"] in ("sellada", "cerrada"):
                print("la ejecución ya terminó")
                return SALIDA_USO
            return _correr(a.carpeta, esc, a.interrumpir_tras_dia)
        if a.orden == "limpiar":
            esc, _ = _escenario_de(a.carpeta)
            estado, entorno = simulacion.abrir(a.carpeta, esc)
            print(json.dumps(entorno.limpiar(), ensure_ascii=False))
            shutil.rmtree(a.carpeta / "datos", ignore_errors=True)
            print("se conservan trabajo/ y evidencia/")
            return SALIDA_OK
        if a.orden == "costos":
            carpeta = a.recalcular / "evidencia" if (a.recalcular / "evidencia").is_dir() else a.recalcular
            res = costos.recalcular(carpeta)
            print(json.dumps({k: res[k] for k in ("costo_servicio", "costo_referencia", "filas", "reduccion")}))
            return SALIDA_OK
        if a.orden == "exportar":
            for x in informe.exportar(a.carpeta, a.destino):
                print(x)
            return SALIDA_OK
        if a.orden == "borrar-evidencia":
            evidencia.borrar(a.carpeta / "evidencia", a.confirmar)
            return SALIDA_OK
    except imagenes.ImagenNoValida as exc:
        print(f"IMAGEN NO VÁLIDA: {exc}", file=sys.stderr)
        return SALIDA_IMAGEN
    except (ent.EntornoInvalido, evidencia.EvidenciaInvalida, costos.CostoNoCalculable,
            FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return SALIDA_USO
    return SALIDA_USO


if __name__ == "__main__":
    sys.exit(main())
