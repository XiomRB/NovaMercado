"""Simulación corta con datos sintéticos y pruebas de aislamiento I5–I7 (etapa 8).

1. Imágenes centinela: si no existen ``novamercado-app:local`` y ``novamercado-airflow:local``
   (CI), se crean etiquetando una imagen pequeña ya descargada. Si existen (equipo local),
   se usan tal cual: nunca se reetiquetan las del usuario.
2. ``ejecutar --escenario corto --interrumpir-tras-dia 2`` → código 10, entorno conservado.
3. I7: se etiqueta como ``novamercado-app:sim-<commit>`` una imagen de otro origen →
   ``reanudar`` se niega (código 3) sin reiniciar ni crear contenedores; se restaura la etiqueta.
4. I6: ``reanudar`` → termina (código 0) sin construir, con los IDs registrados.
5. Limpieza: ningún recurso del proyecto, ``datos/`` borrada, ``evidencia/`` sellada y
   ``costos --recalcular`` idéntico al informe. I5: las ``:local`` siguen en los mismos IDs.

Uso:  python scripts/verificar_simulador_docker.py      (requiere Docker y el árbol limpio)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LOCALES = ("novamercado-app:local", "novamercado-airflow:local")
PEQUENA = "postgres:16.14-alpine"
RESULTADOS: list[str] = []


def ok(msg):
    RESULTADOS.append(msg)
    print(f"  OK  {msg}", flush=True)


def falla(msg):
    print(f"  FALLA  {msg}", flush=True)
    raise SystemExit(1)


def comprobar(cond, msg):
    ok(msg) if cond else falla(msg)


def run(*cmd, check=False, **kw):
    return subprocess.run(list(cmd), capture_output=True, text=True, **kw)


def id_imagen(etiqueta):
    r = run("docker", "image", "inspect", "-f", "{{.Id}}", etiqueta)
    return r.stdout.strip() if r.returncode == 0 else None


def sim(*args):
    print(f"$ python -m simulador {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", "simulador", *args], cwd=RAIZ, text=True, capture_output=True)
    print(r.stdout[-6000:], r.stderr[-3000:], sep="\n", flush=True)
    return r.returncode


PROPIOS = {"api", "preparar", "entrenamiento-inicial", "airflow-init", "airflow-scheduler", "airflow-webserver"}


def contenedores(proyecto):
    """nombre -> (inicio, imagen, servicio) de todos los contenedores del proyecto."""
    r = run("docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={proyecto}", "--format", "{{.ID}}")
    out = {}
    for cid in r.stdout.split():
        i = run("docker", "inspect", "-f",
                '{{.Name}} {{.State.StartedAt}} {{.Image}} {{index .Config.Labels "com.docker.compose.service"}}',
                cid).stdout.split()
        out[i[0]] = (i[1], i[2], i[3])
    return out


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="nm_sim_ci_"))
    if run("docker", "image", "inspect", PEQUENA).returncode != 0:
        run("docker", "pull", PEQUENA)
    creadas = []
    for e in LOCALES:
        if id_imagen(e) is None:
            run("docker", "tag", PEQUENA, e)
            creadas.append(e)
    centinelas = {e: id_imagen(e) for e in LOCALES}
    ok(f"centinelas :local {'creadas' if creadas else 'existentes (no se tocan)'}: "
       f"{ {k: v[:19] for k, v in centinelas.items()} }")
    try:
        codigo = sim("ejecutar", "--escenario", "corto", "--base", str(base), "--interrumpir-tras-dia", "2")
        comprobar(codigo == 10, f"interrupción tras el día 2: código {codigo} (10)")
        carpeta = next(base.iterdir())
        estado = json.loads((carpeta / "trabajo" / "estado.json").read_text())
        proyecto, registradas = estado["proyecto"], estado["imagenes"]
        comprobar(estado["dia_completo"] == 2 and (carpeta / "datos").is_dir(), "estado en el día 2; datos/ conservada")

        # I7: imagen de otro origen con la etiqueta del simulador
        antes = contenedores(proyecto)
        etiqueta = registradas["app"]["etiqueta"]
        run("docker", "tag", PEQUENA, etiqueta)
        codigo = sim("reanudar", str(carpeta))
        despues = contenedores(proyecto)
        ajena = id_imagen(PEQUENA)
        comprobar(codigo == 3, f"I7: reanudar con {etiqueta} ajena → código {codigo} (3)")
        if antes != despues:
            print({k: (antes.get(k), despues.get(k)) for k in set(antes) | set(despues)
                   if antes.get(k) != despues.get(k)})
        # la imagen ajena es la de postgres: solo cuentan los servicios con imagen propia
        comprobar(antes == despues and not any(img == ajena for _, img, srv in despues.values() if srv in PROPIOS),
                  "I7: ningún contenedor se creó ni se reinició; ninguno de la aplicación usa la imagen ajena")
        run("docker", "tag", registradas["app"]["id"], etiqueta)
        comprobar(id_imagen(etiqueta) == registradas["app"]["id"], "etiqueta restaurada al ID registrado")

        # I6: reanudación sin construir
        n_eventos = len((carpeta / "trabajo" / "eventos.jsonl").read_text().splitlines())
        codigo = sim("reanudar", str(carpeta))
        comprobar(codigo == 0, f"I6: la reanudación terminó la simulación (código {codigo})")
        eventos = [json.loads(x) for x in (carpeta / "trabajo" / "eventos.jsonl").read_text().splitlines()]
        nuevos = [e for e in eventos[n_eventos:] if e["tipo"] == "imagen"]
        estado = json.loads((carpeta / "trabajo" / "estado.json").read_text())
        comprobar(not nuevos and estado["imagenes"] == registradas,
                  "I6: sin construcciones al reanudar; los IDs son los registrados")
        comprobar(not estado["fallos"], f"simulación corta sin fallos ({len(estado['fallos'])})")

        # limpieza, evidencia y recálculo
        quedan = {t: run("docker", t, "ls" if t != "ps" else "-a", "--filter",
                         f"label=com.docker.compose.project={proyecto}", "-q").stdout.split()
                  for t in ("ps", "volume", "network")}
        comprobar(not any(quedan.values()), f"limpieza: sin contenedores, volúmenes ni redes de {proyecto}")
        comprobar(not (carpeta / "datos").exists() and (carpeta / "evidencia" / "evidencia.sha256").exists(),
                  "datos/ borrada; evidencia/ sellada y conservada")
        r = subprocess.run([sys.executable, "-m", "simulador", "costos", "--recalcular", str(carpeta)], cwd=RAIZ,
                           capture_output=True, text=True)
        rec = json.loads(r.stdout) if r.returncode == 0 else {}
        comprobar(rec.get("costo_servicio") == estado["costos"]["costo_servicio"]
                  and rec.get("costo_referencia") == estado["costos"]["costo_referencia"],
                  f"costos --recalcular (solo evidencia) = informe: {rec}")
        comprobar({e: id_imagen(e) for e in LOCALES} == centinelas, "I5: las :local siguen en los IDs centinela")
    finally:
        for c in base.glob("*"):
            subprocess.run([sys.executable, "-m", "simulador", "limpiar", str(c)], cwd=RAIZ, capture_output=True)
            if (c / "evidencia").exists():
                (c / "evidencia").chmod(0o755)
        shutil.rmtree(base, ignore_errors=True)
        for e in creadas:
            run("docker", "rmi", e)
    print(f"\nSimulación corta y aislamiento de imágenes: {len(RESULTADOS)} comprobaciones OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
