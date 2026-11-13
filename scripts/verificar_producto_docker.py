"""Verificación de extremo a extremo del producto con Docker Compose (etapa 6, plan §9).

Usa datos SINTÉTICOS (``tests/ml/sinteticos.py``) en una carpeta temporal montada como
``RAW_DIR``: nunca lee ``raw/``. Proyecto de Compose y volúmenes propios, que se borran
al terminar. Requiere el árbol de Git limpio (la imagen debe poder entrenar).

Comprueba: imagen (sin datos, .env ni bytecode, usuario sin privilegios, procedencia del
commit, dependencias verificadas); arranque desde cero hasta /health ok (tiempo medido);
contrato (válida 200, defectuosa 422, observaciones idempotentes); reinicios con
stop/start y down/up sin -v (sin entrenar de nuevo, datos conservados); MLflow detenido
(degradado < 60 s, respuestas 200) y de vuelta (ok < 120 s); E-ANT tras una promoción con
MLflow caído; procedencia que no coincide (bind mount, imagen derivada) → código 2.

Uso:  python scripts/verificar_producto_docker.py      (o scripts/verificar_producto_docker.sh)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
PROYECTO = os.environ.get("E2E_PROYECTO", f"nm-e2e-{os.getpid()}")
P_API = int(os.environ.get("E2E_PUERTO_API", "58000"))
P_DB = int(os.environ.get("E2E_PUERTO_DB", "55433"))
P_MLFLOW = int(os.environ.get("E2E_PUERTO_MLFLOW", "55000"))
IMAGEN = "novamercado-app:local"
CLAVE = "Cl@ve:/e2e%x?"                          # caracteres especiales: la URL se arma por partes
FIX = RAIZ / "tests" / "fixtures" / "ejemplos"
RESULTADOS: list[str] = []
VIGENTE: dict = {}                               # versión vigente tras el arranque


def ok(msg: str) -> None:
    RESULTADOS.append(msg)
    print(f"  OK  {msg}", flush=True)


def falla(msg: str) -> None:
    print(f"  FALLA  {msg}", flush=True)
    raise SystemExit(1)


def comprobar(cond: bool, msg: str) -> None:
    ok(msg) if cond else falla(msg)


def run(*args, check=True, capturar=True, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(list(args), capture_output=capturar, text=True, **kw)
    if check and r.returncode != 0:
        print(r.stdout[-3000:] if r.stdout else "", r.stderr[-3000:] if r.stderr else "", sep="\n")
        falla(f"falló: {' '.join(args)[:200]}")
    return r


def dc(*args, **kw) -> subprocess.CompletedProcess:
    return run("docker", "compose", "-p", PROYECTO, "--env-file", str(ENV), *args, cwd=RAIZ, **kw)


def http(metodo: str, ruta: str, cuerpo=None, puerto: int = P_API, timeout: float = 10):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{puerto}{ruta}", data=datos, method=metodo,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None, None


def esperar(cond, limite: float, paso: float = 1.0) -> float | None:
    t = time.monotonic()
    while time.monotonic() - t < limite:
        if cond():
            return time.monotonic() - t
        time.sleep(paso)
    return None


def salud():
    return http("GET", "/health")


def psql(sql: str) -> str:
    r = dc("exec", "-T", "db", "psql", "-U", "nova", "-d", "novamercado", "-tAc", sql)
    return r.stdout.strip()


def mlflow_json(ruta: str, cuerpo=None):
    _, j = http("POST" if cuerpo is not None else "GET", ruta, cuerpo, puerto=P_MLFLOW)
    return j or {}


def versiones_mlflow() -> list[dict]:
    return mlflow_json("/api/2.0/mlflow/model-versions/search?filter=name%3D%27novamercado-ventas%27").get(
        "model_versions", [])


def ejecuciones_mlflow() -> int:
    exp = mlflow_json("/api/2.0/mlflow/experiments/get-by-name?experiment_name=novamercado-entrenamiento")
    if "experiment" not in exp:
        return 0
    r = mlflow_json("/api/2.0/mlflow/runs/search", {"experiment_ids": [exp["experiment"]["experiment_id"]],
                                                   "max_results": 1000})
    return len(r.get("runs", []))


def log(servicio: str) -> str:
    return dc("logs", "--no-color", servicio, check=False).stdout


# ------------------------------------------------------------------- etapas
def preparar_entorno() -> None:
    from tests.ml.sinteticos import escribir_carpeta, generar_contrato
    # 400 días: la evaluación de producción (7 cortes de 28 días + historia mínima y
    # calibración) necesita historia suficiente; con menos, D1 falla y no hay modelo.
    escribir_carpeta(generar_contrato(dias=400), TMP / "raw")
    texto = (RAIZ / ".env.example").read_text()
    reemplazos = {"POSTGRES_USER": "nova", "POSTGRES_DB": "novamercado", "POSTGRES_PASSWORD": f"'{CLAVE}'",
                  "AIRFLOW_DB_PASSWORD": "'otra-clave-e2e'", "PRODUCTO_DB_PUERTO_HOST": str(P_DB),
                  "MLFLOW_PUERTO_HOST": str(P_MLFLOW), "API_PUERTO_HOST": str(P_API), "RAW_DIR": str(TMP / "raw"),
                  "API_DIRECCION_PUBLICACION": "127.0.0.1"}
    lineas = []
    for ln in texto.splitlines():
        clave = ln.split("=", 1)[0]
        lineas.append(f"{clave}={reemplazos.pop(clave)}" if clave in reemplazos and "=" in ln else ln)
    lineas += [f"{k}={v}" for k, v in reemplazos.items()]
    ENV.write_text("\n".join(lineas) + "\n")


def verificar_imagen() -> None:
    print("== Imagen", flush=True)
    commit = run("git", "rev-parse", "HEAD", cwd=RAIZ).stdout.strip()
    sucio = run("git", "status", "--porcelain", "--untracked-files=normal", cwd=RAIZ).stdout.strip()
    if sucio:
        falla("el árbol de Git tiene cambios sin confirmar: la imagen no podría entrenar")
    t = time.monotonic()
    dc("build")
    ok(f"imagen construida en {time.monotonic() - t:.0f} s (no cuenta para los 10 minutos)")
    r = run("docker", "run", "--rm", IMAGEN, "python", "-c",
            "import json,os;d=json.load(open('/app/procedencia.json'));print(d['commit'],d['limpio'],os.getuid())")
    c, limpio, uid = r.stdout.split()
    comprobar(c == commit and limpio == "True", f"procedencia.json: commit {c[:12]} del checkout, limpio")
    comprobar(uid == "10001", "el proceso corre con un usuario sin privilegios (uid 10001)")
    r = run("docker", "run", "--rm", IMAGEN, "sh", "-c",
            "test ! -e /app/raw && test ! -e /app/.env && test ! -e /app/tests && "
            "test -z \"$(find / -xdev \\( -name 'historico_ventas.csv' -o -name '.env' -o -name "
            "'instrucciones_lab2.md' -o -name 'Anexo_ContratoDeServicio.md' \\) -print -quit 2>/dev/null)\" && "
            "test -z \"$(find /usr/local/lib/python3.11 /app \\( -name '*.pyc' -o -name '*.pyo' \\) -print -quit)\" "
            "&& echo limpio", check=False)
    comprobar(r.stdout.strip() == "limpio", "la imagen no contiene datos, .env, el enunciado ni bytecode")
    r = run("docker", "run", "--rm", IMAGEN, "python", "-m", "app.modelo.procedencia", "--verificar", check=False)
    comprobar(r.returncode == 0, f"procedencia y dependencias verificadas en la imagen: {r.stdout.strip()}")


def arranque_desde_cero() -> float:
    print("== Arranque desde cero", flush=True)
    t = time.monotonic()
    dc("up", "-d")
    seg = esperar(lambda: salud()[0] == 200 and salud()[1]["estado"] == "ok", 600, 2)
    if seg is None:
        print(log("preparar")[-3000:], log("entrenamiento-inicial")[-5000:], log("api")[-3000:], sep="\n")
        falla("el producto no llegó a /health ok en 10 minutos")
    total = time.monotonic() - t
    ok(f"/health ok {total:.0f} s después de 'docker compose up' (límite: 600 s)")
    _, m = http("GET", "/modelo")
    comprobar(m["es_vigente"] and m["estado_vigencia"] == "vigente", f"/modelo: {m}")
    VIGENTE["version"] = m["version"]
    arr = log("entrenamiento-inicial")
    respaldo = '"criterio": "respaldo"' in arr
    comprobar('"promovido": true' in arr, f"entrenamiento-inicial entrenó y promovió la versión {m['version']}"
              + (" (respaldo baseline: el candidato falló solo C1–C3)" if respaldo else " (candidato)"))
    puerto = dc("port", "api", "8000").stdout.strip()
    comprobar(puerto.startswith("127.0.0.1:"), f"la API se publica solo en 127.0.0.1 ({puerto})")
    tags = {t["key"]: t["value"] for v in versiones_mlflow() if v["version"] == m["version"]
            for t in v.get("tags", [])}
    commit = run("git", "rev-parse", "HEAD", cwd=RAIZ).stdout.strip()
    comprobar(tags.get("git_commit") == commit and tags.get("procedencia") == "imagen"
              and tags.get("codigo_verificado") == "true", "la versión registra procedencia=imagen y el commit")
    return total


def contrato() -> None:
    print("== Contrato", flush=True)
    pred = json.loads((FIX / "prediccion_valida.json").read_text())
    s, r = http("POST", "/v1/prediccion", pred)
    comprobar(s == 200 and r["version_modelo"] == VIGENTE["version"] and r["modelo_es_vigente"] is True,
              "predicción válida: 200 con version_modelo y modelo_es_vigente")
    s, r = http("POST", "/v1/prediccion", json.loads((FIX / "errores" / "E01_tienda_texto.json").read_text()))
    comprobar(s == 200 and r["calidad"]["estado"] == "corregido", "solicitud reparable: 200 corregido")
    s, r = http("POST", "/v1/prediccion", json.loads((FIX / "errores" / "E06_familia_repetida.json").read_text()))
    comprobar(s == 422 and set(r) == {"detail"}, "solicitud defectuosa: 422 con detail")
    obs = json.loads((FIX / "observacion_valida.json").read_text())
    s1, r1 = http("POST", "/v1/observaciones", obs)
    s2, r2 = http("POST", "/v1/observaciones", obs)
    comprobar(s1 == s2 == 200 and r1["registros_guardados"] > 0 and r2["registros_guardados"] == 0,
              f"observaciones idempotentes ({r1['registros_guardados']} y luego 0)")


def estado_persistente() -> dict:
    return {"ejecuciones": ejecuciones_mlflow(), "versiones": len(versiones_mlflow()),
            "solicitudes": psql("SELECT count(*) FROM solicitudes_prediccion"),
            "observaciones": psql("SELECT count(*) FROM observaciones"),
            "resoluciones": psql("SELECT count(*) FROM resoluciones_promocion")}


def reinicio(como: str, antes: dict) -> None:
    print(f"== Reinicio ({como})", flush=True)
    if como == "stop/start":
        dc("stop")
    else:
        dc("down")                                                        # sin -v: se conservan los volúmenes
    dc("up", "-d")
    seg = esperar(lambda: salud()[0] == 200 and salud()[1]["estado"] == "ok", 600, 2)
    comprobar(seg is not None, f"vuelve a /health ok en {seg or 0:.0f} s")
    comprobar("ya_cargado" in log("preparar"), "preparar: la carga responde ya_cargado")
    t = time.monotonic()
    r = dc("wait", "entrenamiento-inicial", check=False, timeout=900)      # termina después de que la API
    comprobar(r.returncode == 0, "entrenamiento-inicial terminó con código 0 "
              f"en {time.monotonic() - t:.0f} s")                          # ya responde con el modelo
    ultima = [ln for ln in log("entrenamiento-inicial").splitlines() if '"codigo"' in ln][-1]
    comprobar("ya existe un vigente confirmado" in ultima, "entrenamiento-inicial: no entrena de nuevo")
    despues = estado_persistente()
    comprobar(despues == antes, f"se conservan ejecuciones, versiones, solicitudes y observaciones: {despues}")
    comprobar(http("GET", "/modelo")[1]["version"] == VIGENTE["version"],
              f"se sigue sirviendo la versión {VIGENTE['version']}")


def mlflow_detenido() -> None:
    print("== MLflow detenido", flush=True)
    pred = json.loads((FIX / "prediccion_valida.json").read_text())
    dc("stop", "mlflow")
    seg = esperar(lambda: salud()[1] and salud()[1]["estado"] == "degradado", 60, 1)
    comprobar(seg is not None, f"/health degradado {seg or 0:.0f} s después de detener MLflow (< 60 s)")
    comprobar(all(http("POST", "/v1/prediccion", pred)[0] == 200 for _ in range(5)), "5 de 5 predicciones 200")
    dc("start", "mlflow")
    seg = esperar(lambda: salud()[1] and salud()[1]["estado"] == "ok", 120, 1)
    comprobar(seg is not None, f"/health ok {seg or 0:.0f} s después de volver MLflow (< 120 s)")


def e_ant() -> None:
    print("== Promoción con MLflow caído (E-ANT)", flush=True)
    pred = json.loads((FIX / "prediccion_valida.json").read_text())
    dc("pause", "api")                                     # la API no llega a descargar la versión nueva
    r = dc("run", "--rm", "--no-deps", "entrenamiento-inicial", "python", "-m", "app.modelo.entrenar_mlflow",
           "--promover", "siempre", check=False)
    comprobar(r.returncode == 0 and '"promovido": true' in r.stdout,
              "entrenamiento manual en el contenedor (--promover siempre): versión nueva promovida")
    nueva = json.loads(r.stdout.strip().splitlines()[-1])["version_vigente"]
    v = VIGENTE["version"]
    dc("stop", "mlflow")
    dc("unpause", "api")
    seg = esperar(lambda: http("GET", "/modelo")[1].get("estado_vigencia") == "anterior", 60, 1)
    comprobar(seg is not None, f"la API observa la versión {nueva}, no puede obtenerla y queda en 'anterior'")
    _, m = http("GET", "/modelo")
    comprobar((m["version"], m["es_vigente"], m["version_vigente"]) == (v, False, nueva), f"/modelo: {m}")
    s, r = http("POST", "/v1/prediccion", pred)
    comprobar(s == 200 and r["version_modelo"] == v and r["modelo_es_vigente"] is False,
              f"las predicciones siguen con la versión {v}, marcadas como no vigente")
    comprobar(salud()[1]["estado"] == "degradado", "/health degradado")
    dc("start", "mlflow")
    seg = esperar(lambda: http("GET", "/modelo")[1].get("version") == nueva, 120, 1)
    comprobar(seg is not None, f"al volver MLflow, la versión {nueva} se activa sin reinicio en {seg or 0:.0f} s")


def procedencia_no_coincide() -> None:
    print("== Procedencia que no coincide con lo ejecutado", flush=True)
    antes = ejecuciones_mlflow()
    modificado = TMP / "costos.py"
    modificado.write_text((RAIZ / "ml" / "costos.py").read_text() + "\n# cambio no confirmado\n")
    r = dc("run", "--rm", "--no-deps", "-v", f"{modificado}:/app/ml/costos.py:ro", "entrenamiento-inicial",
           check=False)
    comprobar(r.returncode == 2 and "ml/costos.py no coincide con el commit" in r.stderr,
              "un archivo de código sustituido (bind mount): código 2")
    for nombre, paso in (("extra", "echo 'x = 1' > /usr/local/lib/python3.11/site-packages/ajeno.py"),
                         ("modificado", "echo '#' >> /usr/local/lib/python3.11/site-packages/numpy/version.py")):
        d = TMP / f"derivada_{nombre}"
        d.mkdir()
        (d / "Dockerfile").write_text(f"FROM {IMAGEN}\nUSER root\nRUN {paso}\nUSER app\n")
        run("docker", "build", "-q", "-t", f"novamercado-derivada-{nombre}:e2e", str(d))
        r = run("docker", "run", "--rm", "--network", "none", f"novamercado-derivada-{nombre}:e2e",
                "python", "-m", "app.modelo.arranque", check=False)
        comprobar(r.returncode == 2 and "dependencias:" in r.stderr,
                  f"imagen derivada con un archivo {nombre} en site-packages: código 2")
        run("docker", "rmi", "-f", f"novamercado-derivada-{nombre}:e2e", check=False)
    comprobar(ejecuciones_mlflow() == antes, "ninguno creó ejecuciones en MLflow")


def main() -> int:
    global TMP, ENV
    TMP = Path(tempfile.mkdtemp(prefix="nm_e2e_"))
    ENV = TMP / "env"
    preparar_entorno()
    try:
        verificar_imagen()
        arranque_desde_cero()
        contrato()
        antes = estado_persistente()
        reinicio("stop/start", antes)
        reinicio("down/up sin -v", antes)
        mlflow_detenido()
        e_ant()
        procedencia_no_coincide()
    except SystemExit:
        for s in ("preparar", "entrenamiento-inicial", "api"):
            print(f"--- logs de {s} ---\n{log(s)[-4000:]}")
        raise
    finally:
        dc("down", "-v", "--remove-orphans", check=False)
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nVerificación del producto con Docker Compose: {len(RESULTADOS)} comprobaciones OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
