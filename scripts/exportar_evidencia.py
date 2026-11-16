"""Exporta la evidencia de funcionamiento de un producto levantado (etapa 9, F1).

Solo lectura: ``GET`` a la API y a la API REST de Airflow, consultas de búsqueda a la API
REST de MLflow (``runs/search`` es un POST de lectura) y ``SELECT`` en PostgreSQL con
``default_transaction_read_only=on``. No envía solicitudes de pronóstico, observaciones ni
reentrenamientos, y no escribe en ningún servicio.

Escribe en ``--salida``:

* ``mlflow.json`` / ``mlflow.md``: experimentos, ejecuciones (métricas, parámetros y tags de
  linaje), versiones registradas con sus tags, alias, decisiones de promoción en PostgreSQL y
  ``GET /modelo``.
* ``airflow.json`` / ``airflow.md``: ejecuciones del DAG ``novamercado_reentrenamiento`` con sus
  tareas, cruzadas con ``GET /admin/reentrenar/{id}`` y con el resultado en PostgreSQL.

Las credenciales se leen del archivo de entorno (``--env-file``) y nunca se escriben. Se
omiten los parámetros y tags cuyo nombre sugiera un secreto. Solo biblioteca estándar.

Uso::

    python scripts/exportar_evidencia.py --env-file /ruta/.env --repo /ruta/clon \\
        --salida reportes/entrega [--proyecto <nombre de Compose>]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DAG_ID = "novamercado_reentrenamiento"
EXPERIMENTO = "novamercado-entrenamiento"
SECRETO = re.compile(r"(?i)(password|clave|secret|token|fernet|credencial)")
TIMEOUT = 15


def leer_env(ruta: Path) -> dict[str, str]:
    out = {}
    for ln in Path(ruta).read_text(encoding="utf-8").splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def pedir(url: str, metodo: str = "GET", cuerpo: dict | None = None, auth: tuple[str, str] | None = None):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode()
    cab = {"Accept": "application/json"}
    if datos is not None:
        cab["Content-Type"] = "application/json"
    if auth:
        cab["Authorization"] = "Basic " + base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    req = urllib.request.Request(url, data=datos, method=metodo, headers=cab)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"null")
        except ValueError:
            return exc.code, None


def sin_secretos(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items()) if not SECRETO.search(k)}


class Psql:
    """``SELECT`` dentro del contenedor ``db`` del proyecto, en una transacción de solo lectura."""

    def __init__(self, repo: Path, env_file: Path, proyecto: str | None, usuario: str, base: str):
        self.base_cmd = ["docker", "compose", "-f", str(repo / "docker-compose.yml"), "--project-directory", str(repo),
                         "--env-file", str(env_file)] + (["-p", proyecto] if proyecto else [])
        self.usuario, self.base = usuario, base

    def json(self, sql: str):
        cmd = self.base_cmd + ["exec", "-T", "-e", "PGOPTIONS=-c default_transaction_read_only=on", "db", "psql",
                               "-X", "-q", "-v", "ON_ERROR_STOP=1", "-U", self.usuario, "-d", self.base, "-tAc",
                               f"SELECT coalesce(json_agg(x), '[]') FROM ({sql}) x"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"psql: {r.stderr.strip()[-300:]}")
        return json.loads(r.stdout.strip() or "[]")


# ------------------------------------------------------------------ MLflow
def exportar_mlflow(mlflow: str, modelo: str, psql: Psql, api: str) -> dict:
    base = mlflow.rstrip("/") + "/api/2.0/mlflow"
    _, exps = pedir(base + "/experiments/search", "POST", {"max_results": 100})
    experimentos = [{"id": e["experiment_id"], "nombre": e["name"], "estado": e.get("lifecycle_stage")}
                    for e in (exps or {}).get("experiments", [])]
    ejecuciones = []
    ids = [e["id"] for e in experimentos if e["nombre"] == EXPERIMENTO]
    token = None
    while ids:
        cuerpo = {"experiment_ids": ids, "max_results": 500, "order_by": ["attributes.start_time ASC"]}
        if token:
            cuerpo["page_token"] = token
        _, r = pedir(base + "/runs/search", "POST", cuerpo)
        for run in (r or {}).get("runs", []):
            info, data = run["info"], run.get("data", {})
            ejecuciones.append({
                "run_id": info["run_id"], "nombre": info.get("run_name"), "estado": info.get("status"),
                "inicio": _ms(info.get("start_time")), "fin": _ms(info.get("end_time")),
                "metricas": {m["key"]: m["value"] for m in sorted(data.get("metrics", []), key=lambda m: m["key"])},
                "parametros": sin_secretos({p["key"]: p["value"] for p in data.get("params", [])}),
                "tags": sin_secretos({t["key"]: t["value"] for t in data.get("tags", [])
                                      if not t["key"].startswith("mlflow.log-model")}),
            })
        token = (r or {}).get("next_page_token")
        if not token:
            break
    _, rm = pedir(base + "/registered-models/get?" + urllib.parse.urlencode({"name": modelo}))
    reg = (rm or {}).get("registered_model", {})
    alias = {a["alias"]: a["version"] for a in reg.get("aliases", [])}
    _, mv = pedir(base + "/model-versions/search?" + urllib.parse.urlencode(
        {"filter": f"name='{modelo}'", "max_results": 1000}))
    versiones = sorted(({"version": int(v["version"]), "run_id": v.get("run_id"), "estado": v.get("status"),
                         "creada": _ms(v.get("creation_timestamp")),
                         "tags": sin_secretos({t["key"]: t["value"] for t in v.get("tags", [])})}
                        for v in (mv or {}).get("model_versions", [])), key=lambda v: v["version"])
    decisiones = psql.json(
        "SELECT d.decision_id, d.version, d.run_id, d.sha256_artefacto, d.origen, d.creada_en, "
        "r.resultado, r.motivo, r.secuencia, r.resuelta_en FROM decisiones_promocion d "
        "LEFT JOIN resoluciones_promocion r USING (decision_id) ORDER BY d.creada_en")
    _, modelo_api = pedir(api.rstrip("/") + "/modelo")
    return {"experimentos": experimentos, "ejecuciones": ejecuciones, "modelo_registrado": modelo,
            "alias": alias, "versiones": versiones, "decisiones_postgresql": decisiones, "api_modelo": modelo_api}


def _ms(v) -> str | None:
    if v in (None, "", 0, "0"):
        return None
    return datetime.fromtimestamp(int(v) / 1000, tz=timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ Airflow
def exportar_airflow(airflow: str, auth: tuple[str, str], api: str, psql: Psql) -> dict:
    base = airflow.rstrip("/") + "/api/v1"
    _, dag = pedir(f"{base}/dags/{DAG_ID}", auth=auth)
    _, runs = pedir(f"{base}/dags/{DAG_ID}/dagRuns?" + urllib.parse.urlencode({"limit": 100, "order_by": "start_date"}),
                    auth=auth)
    pg = {r["id"]: r for r in psql.json(
        "SELECT s.id, s.clave_idempotencia IS NOT NULL AS con_clave, s.solicitado_en, s.origen, t.iniciada_en, "
        "r.estado, r.promovido, r.version, r.motivo, r.escrito_por, r.terminado_en "
        "FROM solicitudes_reentrenamiento s LEFT JOIN tareas_reentrenamiento t USING (id) "
        "LEFT JOIN resultados_reentrenamiento r USING (id) ORDER BY s.solicitado_en")}
    ejecuciones = []
    for run in (runs or {}).get("dag_runs", []):
        rid = run["dag_run_id"]
        _, ti = pedir(f"{base}/dags/{DAG_ID}/dagRuns/{urllib.parse.quote(rid, safe='')}/taskInstances", auth=auth)
        _, admin = pedir(api.rstrip("/") + "/admin/reentrenar/" + urllib.parse.quote(rid, safe=""))
        ejecuciones.append({
            "dag_run_id": rid, "estado": run.get("state"), "inicio": run.get("start_date"), "fin": run.get("end_date"),
            "segundos": _segundos(run.get("start_date"), run.get("end_date")),
            "tareas": [{"tarea": t["task_id"], "estado": t.get("state"), "segundos": t.get("duration")}
                       for t in sorted((ti or {}).get("task_instances", []), key=lambda t: t.get("start_date") or "")],
            "admin_reentrenar": admin, "postgresql": pg.get(rid),
        })
    return {"dag": {"dag_id": DAG_ID, "pausado": (dag or {}).get("is_paused"), "programacion":
                    (dag or {}).get("schedule_interval")},
            "ejecuciones": ejecuciones,
            "solicitudes_sin_ejecucion": [v for k, v in pg.items() if k not in {e["dag_run_id"] for e in ejecuciones}]}


def _segundos(a, b):
    try:
        return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ Markdown
METRICAS_MD = ("costo_modelo", "costo_baseline", "mejora_costo", "segundos_total", "segundos_r2")


def md_mlflow(d: dict) -> str:
    m = ["# Evidencia de MLflow", "",
         f"Modelo registrado `{d['modelo_registrado']}`; alias: "
         + (", ".join(f"`{a}` → versión {v}" for a, v in sorted(d["alias"].items())) or "ninguno") + ".", "",
         f"`GET /modelo`: `{json.dumps(d['api_modelo'], ensure_ascii=False)}`", "",
         "## Experimentos", "", "| id | nombre | estado |", "|---|---|---|"]
    m += [f"| {e['id']} | {e['nombre']} | {e['estado']} |" for e in d["experimentos"]]
    m += ["", f"## Ejecuciones de `{EXPERIMENTO}` ({len(d['ejecuciones'])})", "",
          "| run_id | nombre | estado | inicio | verificación | versión | commit |", "|---|---|---|---|---|---|---|"]
    for r in d["ejecuciones"]:
        t = r["tags"]
        commit = str(t.get("commit", t.get("mlflow.source.git.commit", "—")))[:12]
        m.append(f"| `{r['run_id'][:12]}` | {r['nombre']} | {r['estado']} | {r['inicio']} "
                 f"| {t.get('verificacion', '—')} | {t.get('version_registrada', '—')} | `{commit}` |")
    m += ["", "Métricas y parámetros completos en `mlflow.json`.", "", "## Versiones registradas", "",
          "| versión | run_id | estado (tag) | sha256 del artefacto | creada |", "|---|---|---|---|---|"]
    for v in d["versiones"]:
        m.append(f"| {v['version']} | `{str(v['run_id'])[:12]}` | {v['tags'].get('estado', '—')} "
                 f"| `{str(v['tags'].get('sha256_artefacto', '—'))[:16]}` | {v['creada']} |")
    m += ["", "## Decisiones de promoción (PostgreSQL, fuente de verdad)", "",
          "| versión | origen | resultado | secuencia | motivo |", "|---|---|---|---|---|"]
    for x in d["decisiones_postgresql"]:
        m.append(f"| {x['version']} | {x['origen']} | {x['resultado']} | {x['secuencia']} | {x['motivo'] or ''} |")
    return "\n".join(m) + "\n"


def md_airflow(d: dict) -> str:
    m = ["# Evidencia de Airflow", "", f"DAG `{d['dag']['dag_id']}` (pausado: {d['dag']['pausado']}, "
         f"programación: {d['dag']['programacion']}).", "",
         "| dag_run_id | Airflow | s | tareas | /admin/reentrenar | promovido | versión |",
         "|---|---|---|---|---|---|---|"]
    for e in d["ejecuciones"]:
        a = e["admin_reentrenar"] if isinstance(e["admin_reentrenar"], dict) else {}
        tareas = ", ".join(f"{t['tarea']}={t['estado']}" for t in e["tareas"])
        m.append(f"| `{e['dag_run_id']}` | {e['estado']} | {e['segundos']} | {tareas} | {a.get('estado')} "
                 f"| {a.get('promovido')} | {a.get('version_modelo', a.get('version'))} |")
    if d["solicitudes_sin_ejecucion"]:
        m += ["", f"Solicitudes sin ejecución en Airflow: {len(d['solicitudes_sin_ejecucion'])} (ver `airflow.json`)."]
    return "\n".join(m) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="carpeta con docker-compose.yml del producto")
    ap.add_argument("--proyecto", default=None, help="nombre del proyecto de Compose (por defecto, el de la carpeta)")
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    env = leer_env(a.env_file)
    api = f"http://127.0.0.1:{env.get('API_PUERTO_HOST', '8000')}"
    mlflow = f"http://127.0.0.1:{env.get('MLFLOW_PUERTO_HOST', '5000')}"
    airflow = f"http://127.0.0.1:{env.get('AIRFLOW_PUERTO_HOST', '8080')}"
    psql = Psql(a.repo.resolve(), a.env_file.resolve(), a.proyecto, env["POSTGRES_USER"], env["POSTGRES_DB"])
    a.salida.mkdir(parents=True, exist_ok=True)
    ml = exportar_mlflow(mlflow, env.get("MLFLOW_MODELO_NOMBRE", "novamercado-ventas"), psql, api)
    af = exportar_airflow(airflow, (env["AIRFLOW_API_USUARIO"], env["AIRFLOW_API_CLAVE"]), api, psql)
    for nombre, datos, md in (("mlflow", ml, md_mlflow), ("airflow", af, md_airflow)):
        (a.salida / f"{nombre}.json").write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str) + "\n",
                                                 encoding="utf-8")
        (a.salida / f"{nombre}.md").write_text(md(datos), encoding="utf-8")
    print(f"{len(ml['ejecuciones'])} ejecuciones de MLflow, {len(ml['versiones'])} versiones, "
          f"{len(af['ejecuciones'])} ejecuciones del DAG -> {a.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
