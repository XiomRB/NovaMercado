"""Etapa 7 sin servidores: arranque verificado y lanzador (plan v3 §1, E7-14), copia
idéntica del SQL del DAG y presupuestos de tiempo estáticos (G1, G2, vigilante)."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from app.db import cierre
from app.modelo import reentrenar

RAIZ = Path(__file__).resolve().parents[2]
ARRANQUE = RAIZ / "docker" / "arranque_verificado.py"
LANZADOR = RAIZ / "docker" / "ejecutar"


# ------------------------------------------------------------------ copia del SQL
def test_sql_del_dag_es_copia_identica():
    assert (RAIZ / "dags" / "nm_cierre.py").read_bytes() == (RAIZ / "app" / "db" / "cierre.py").read_bytes()


def test_cierre_solo_usa_biblioteca_estandar_y_psycopg2():
    arbol = ast.parse((RAIZ / "app" / "db" / "cierre.py").read_text())
    modulos = {a.name.split(".")[0] for n in ast.walk(arbol) if isinstance(n, ast.Import) for a in n.names}
    modulos |= {n.module.split(".")[0] for n in ast.walk(arbol) if isinstance(n, ast.ImportFrom) and n.module}
    assert modulos <= {"__future__", "hashlib", "os", "socket", "threading", "time", "dataclasses", "urllib",
                       "psycopg2"}, modulos


def test_los_dag_no_importan_el_proyecto():
    for f in (RAIZ / "dags").glob("*.py"):
        arbol = ast.parse(f.read_text())
        for n in ast.walk(arbol):
            nombres = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            assert not any(x.split(".")[0] in ("app", "ml") for x in nombres), f"{f.name} importa {nombres}"


# --------------------------------------------------------------------- presupuestos
def _constantes_dag():
    arbol = ast.parse((RAIZ / "dags" / "novamercado_reentrenamiento.py").read_text())
    out = {}
    for n in arbol.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call):
            if getattr(n.value.func, "id", None) == "timedelta":
                out[n.targets[0].id] = timedelta(**{k.arg: k.value.value for k in n.value.keywords})
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
            out[n.targets[0].id] = n.value.value
    return out


def _entorno_airflow():
    c = yaml.safe_load((RAIZ / "docker-compose.yml").read_text())
    return c["x-entorno-airflow"]


def test_presupuesto_del_vigilante_34_38_40():
    peor = (cierre.VIG_CONEXION_S + cierre.VIG_LECTURA_S + cierre.VIG_FASE_BLOQUEO_S + cierre.VIG_FUNCION_S
            + cierre.VIG_RECONEXION_S + cierre.VIG_VERIFICACION_S
            + cierre.VIG_ESPERA_BLOQUEO_S + 3 * cierre.VIG_SECCION_CRITICA_OP_S)
    assert peor == 34
    assert cierre.VIG_FASE_BLOQUEO_S == 3 * cierre.VIG_ESPERA_BLOQUEO_S
    d = _constantes_dag()
    assert peor < cierre.VIG_RESPALDO_S == 38 < d["TIMEOUT_VIGILANTE"].total_seconds() == 40
    # 4 intentos con 30 s entre ellos: 250 s efectivos
    intentos = d["REINTENTOS_VIGILANTE"] + 1
    assert intentos * 40 + d["REINTENTOS_VIGILANTE"] * d["ESPERA_REINTENTO_VIGILANTE"].total_seconds() == 250


def test_presupuesto_g1_g2():
    d, env = _constantes_dag(), _entorno_airflow()
    limpieza = int(env["AIRFLOW__CORE__KILLED_TASK_CLEANUP_TIME"])
    muerte = d["TIMEOUT_REENTRENAR"].total_seconds() + limpieza
    assert muerte == 8 * 60 + 45                                       # nada de reentrenar después de 8:45
    assert reentrenar.PROMOCION_S < reentrenar.HIJO_S < reentrenar.RECONCILIACION_S < reentrenar.RESULTADO_S
    assert reentrenar.HIJO_S + 10 <= 8 * 60 + 10                         # G1: hijo detenido + gracia
    assert reentrenar.RESULTADO_TRAS_SIGTERM_S < muerte
    d_max, t_reg = 40, cierre.CIERRE_TOTAL_S + 1                        # demora del scheduler y peor registro
    assert muerte + d_max + t_reg == 9 * 60 + 38 < 10 * 60               # G2-normal con 22 s de margen
    assert d["TIMEOUT_REGISTRAR"].total_seconds() >= t_reg
    assert int(env["AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC"]) <= 5


def test_configuracion_airflow_aprobada():
    env = _entorno_airflow()
    assert env["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor" and int(env["AIRFLOW__CORE__PARALLELISM"]) >= 4
    assert env["AIRFLOW__CORE__LOAD_EXAMPLES"] == "false" and env["AIRFLOW__WEBSERVER__WORKERS"] == "2"
    assert "basic_auth" in env["AIRFLOW__API__AUTH_BACKENDS"]
    c = yaml.safe_load((RAIZ / "docker-compose.yml").read_text())
    puertos = c["services"]["airflow-webserver"]["ports"]
    assert puertos == ["${AIRFLOW_DIRECCION_PUBLICACION:-127.0.0.1}:${AIRFLOW_PUERTO_HOST:-8080}:8080"]
    assert c["services"]["entrenamiento-inicial"]["command"] == ["/app/docker/ejecutar", "arranque"]   # E7-14


# ------------------------------------------------------------------- lanzador
@pytest.fixture
def arbol_lanzador(tmp_path):
    """Copia del lanzador con un 'python' falso que imprime su entorno y argumentos."""
    (tmp_path / "docker").mkdir()
    shutil.copy(LANZADOR, tmp_path / "docker" / "ejecutar")
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps({{'argv': sys.argv[1:], "
                  "'env': dict(os.environ)}))\n")
    py.chmod(0o755)
    return tmp_path


def test_lanzador_entorno_cerrado(arbol_lanzador):
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": "/malo", "PYTHONHOME": "/malo", "AIRFLOW_HOME": "/x",
           "AIRFLOW__CORE__FERNET_KEY": "secreto", "LD_PRELOAD": "/malo.so", "POSTGRES_PASSWORD": "c l@v'e\"$",
           "DB_HOST": "db", "NM_CIERRE_PASSWORD": "k", "NM_VIGILANTE_PASSWORD": "no-debe-pasar"}
    r = subprocess.run([str(arbol_lanzador / "docker" / "ejecutar"), "reentrenar", "--ejecucion", "a b"],
                       env=env, capture_output=True, text=True, check=True)
    salida = json.loads(r.stdout)
    assert salida["argv"] == ["-I", "-S", "-B", str(arbol_lanzador / "docker" / "arranque_verificado.py"),
                              "app.modelo.reentrenar", "--ejecucion", "a b"]
    e = salida["env"]
    assert e["POSTGRES_PASSWORD"] == "c l@v'e\"$" and e["DB_HOST"] == "db" and e["NM_CIERRE_PASSWORD"] == "k"
    assert e["PYTHONNOUSERSITE"] == "1" and e["PYTHONDONTWRITEBYTECODE"] == "1"
    for prohibida in ("PYTHONPATH", "PYTHONHOME", "AIRFLOW_HOME", "AIRFLOW__CORE__FERNET_KEY", "LD_PRELOAD",
                      "NM_VIGILANTE_PASSWORD"):
        assert prohibida not in e


@pytest.mark.parametrize("comando", ["", "otro", "app.modelo.reentrenar", "../reentrenar"])
def test_lanzador_rechaza_comandos(arbol_lanzador, comando):
    r = subprocess.run([str(arbol_lanzador / "docker" / "ejecutar"), *([comando] if comando else [])],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "no permitido" in r.stderr


# ----------------------------------------------------------- arranque verificado
def blob(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


@pytest.fixture
def arbol(tmp_path):
    """Árbol mínimo con su procedencia.json; el código 'malicioso' crea un marcador al importarse."""
    archivos = {"app/__init__.py": b"", "app/modelo/__init__.py": b"",
                "app/modelo/reentrenar.py": b"open(__file__ + '.ejecutado', 'w').close()\n",
                "requirements/api.lock.txt": b"", "docker/arranque_verificado.py": ARRANQUE.read_bytes()}
    for rel, b in archivos.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b)
    proc = {"formato": 1, "commit": "a" * 40, "limpio": True, "sin_git": False, "cambios": [],
            "clase_a": {k: blob(v) for k, v in archivos.items()}, "clase_b": {},
            "directorios_a": ["app", "requirements", "docker"]}
    (tmp_path / "procedencia.json").write_text(json.dumps(proc))
    return tmp_path


def correr(arbol, *flags, modulo="app.modelo.reentrenar"):
    return subprocess.run([sys.executable, *flags, str(arbol / "docker" / "arranque_verificado.py"), modulo],
                          capture_output=True, text=True, cwd=arbol, timeout=60)


@pytest.mark.parametrize("alteracion", ["init", "agregado", "pth_en_app", "procedencia_sucia"])
def test_codigo_alterado_no_se_ejecuta(arbol, alteracion):
    marcador = arbol / "marcador"
    malicioso = f"open({str(marcador)!r}, 'w').close()\n"
    if alteracion == "init":
        (arbol / "app" / "__init__.py").write_text(malicioso)
    elif alteracion == "agregado":
        (arbol / "app" / "sitecustomize.py").write_text(malicioso)
    elif alteracion == "pth_en_app":
        (arbol / "app" / "x.pth").write_text("import os; " + malicioso)
    else:
        d = json.loads((arbol / "procedencia.json").read_text())
        d["limpio"] = False
        (arbol / "procedencia.json").write_text(json.dumps(d))
    r = correr(arbol, "-I", "-S", "-B")
    assert r.returncode == 2 and "No se puede ejecutar" in r.stderr
    assert not marcador.exists() and not (arbol / "app" / "modelo" / "reentrenar.py.ejecutado").exists()


def test_codigo_intacto_pasa_el_paso_1(arbol):
    """El paso 1 (solo biblioteca estándar) acepta el árbol intacto; el paso 2 falla aquí
    porque el entorno de pruebas no es el del lock: aun así nada del proyecto se ejecuta
    antes de verificar las dependencias."""
    r = correr(arbol, "-I", "-S", "-B")
    assert r.returncode == 2 and "packaging" in r.stderr and "commit" not in r.stderr
    assert not (arbol / "app" / "modelo" / "reentrenar.py.ejecutado").exists()


def test_exige_python_aislado_y_modulo_permitido(arbol):
    assert correr(arbol).returncode == 2 and "-I -S" in correr(arbol).stderr
    r = correr(arbol, "-I", "-S", "-B", modulo="app.modelo.procedencia")
    assert r.returncode == 2 and "no permitido" in r.stderr
