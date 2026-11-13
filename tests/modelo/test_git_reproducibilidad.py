"""Árbol de Git limpio (resuelto desde el código) y reproducibilidad entre procesos."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import app.modelo.git_estado as ge
from app.modelo.git_estado import RepositorioNoValido, verificar_arbol_limpio

RAIZ = Path(__file__).resolve().parents[2]


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """Repositorio temporal con un 'módulo de código' confirmado y .gitignore."""
    r = tmp_path / "repo"
    (r / "app" / "modelo").mkdir(parents=True)
    (r / "app" / "modelo" / "codigo.py").write_text("x = 1\n")
    (r / ".gitignore").write_text("raw/\n.env\nmodelos/\n")
    git("init", "-q", "-b", "main", cwd=r)
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".", cwd=r)
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "inicial", cwd=r)
    return r


def codigo(repo):
    return repo / "app" / "modelo" / "codigo.py"


def head(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()


def test_arbol_limpio(repo):
    e = verificar_arbol_limpio(codigo(repo))
    assert e.commit == head(repo) and e.raiz == repo.resolve()


@pytest.mark.parametrize("cambio", ["modificado", "preparado", "nuevo"])
def test_cambios_sin_confirmar_se_rechazan(repo, cambio):
    if cambio == "modificado":
        codigo(repo).write_text("x = 2\n")
    elif cambio == "preparado":
        codigo(repo).write_text("x = 3\n")
        git("add", ".", cwd=repo)
    else:
        (repo / "app" / "nuevo.py").write_text("y = 1\n")
    with pytest.raises(RepositorioNoValido, match="cambios sin confirmar") as exc:
        verificar_arbol_limpio(codigo(repo))
    assert "Confirme o descarte" in str(exc.value)


def test_solo_ignorados_no_cuentan(repo):
    for f in ("raw/datos.csv", ".env", "modelos/m.joblib"):
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("x")
    assert verificar_arbol_limpio(codigo(repo)).commit == head(repo)


def test_lista_acotada_de_cambios(repo):
    for i in range(15):
        (repo / f"n{i}.py").write_text("x")
    with pytest.raises(RepositorioNoValido, match=r"\(y 5 más\)"):
        verificar_arbol_limpio(codigo(repo))


def test_codigo_fuera_de_un_repositorio(tmp_path):
    f = tmp_path / "suelto" / "codigo.py"
    f.parent.mkdir()
    f.write_text("x")
    with pytest.raises(RepositorioNoValido, match="no está dentro de un repositorio"):
        verificar_arbol_limpio(f)


def test_git_no_disponible(repo, monkeypatch):
    monkeypatch.setattr(ge.shutil, "which", lambda _: None)
    with pytest.raises(RepositorioNoValido, match="git no está disponible"):
        verificar_arbol_limpio(codigo(repo))


def test_head_cambia_durante_la_verificacion(repo, monkeypatch):
    real, llamadas = ge._git, []

    def git_que_cambia(g, raiz, *args):
        salida = real(g, raiz, *args)
        if args[:1] == ("rev-parse",) and args[1:] == ("HEAD",):
            llamadas.append(1)
            if len(llamadas) == 2:
                return "0" * 40 + "\n"
        return salida
    monkeypatch.setattr(ge, "_git", git_que_cambia)
    with pytest.raises(RepositorioNoValido, match="cambió durante la verificación"):
        verificar_arbol_limpio(codigo(repo))


def test_se_usa_el_repositorio_del_codigo_no_el_del_directorio_actual(repo, tmp_path, monkeypatch):
    otro = tmp_path / "otro"
    otro.mkdir()
    git("init", "-q", cwd=otro)                          # repositorio limpio y vacío
    monkeypatch.chdir(otro)
    codigo(repo).write_text("x = 9\n")                   # el del código está sucio
    with pytest.raises(RepositorioNoValido, match="cambios sin confirmar"):
        verificar_arbol_limpio(codigo(repo))


def test_lanzado_desde_otro_directorio(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert verificar_arbol_limpio(codigo(repo)).commit == head(repo)


def test_worktree(repo, tmp_path):
    wt = tmp_path / "wt"
    git("worktree", "add", "-q", str(wt), cwd=repo)
    e = verificar_arbol_limpio(wt / "app" / "modelo" / "codigo.py")
    assert e.raiz == wt.resolve() and e.commit == head(repo)


def test_cli_rechaza_arbol_sucio_sin_crear_nada(tmp_path):
    """El comando real (copiado a un repositorio temporal sucio) sale con código 2 antes
    de leer la base o contactar MLflow (las URL apuntan a servidores inexistentes)."""
    import shutil
    r = tmp_path / "copia"
    for d in ("app", "ml"):
        shutil.copytree(RAIZ / d, r / d, ignore=shutil.ignore_patterns("__pycache__"))
    git("init", "-q", "-b", "main", cwd=r)
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".", cwd=r)
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "c", cwd=r)
    (r / "app" / "modelo" / "entrenamiento.py").write_text("# cambio sin confirmar\n", encoding="utf-8")
    entorno = {**os.environ, "DATABASE_URL": "postgresql+psycopg2://u@127.0.0.1:1/x",
               "MLFLOW_TRACKING_URI": "http://127.0.0.1:1"}
    p = subprocess.run([sys.executable, "-m", "app.modelo.entrenar_mlflow"], cwd=tmp_path / "copia" / "..",
                       env={**entorno, "PYTHONPATH": str(r)}, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, p.stderr
    assert "cambios sin confirmar" in p.stderr and "app/modelo/entrenamiento.py" in p.stderr
    assert "Traceback" not in p.stderr


# --------------------------------------------------------------- reproducibilidad
SCRIPT = """
import sys
from app.modelo.copia_datos import leer_copia
from app.modelo.registro import config_desde_manifiesto, entrenar_definitivo, hilos_fijos
from app.modelo.artefacto import sha256_bytes
datos, manifiesto = leer_copia(sys.argv[1])
config, _ = config_desde_manifiesto(manifiesto)      # el mismo camino que registro y reproducir
with hilos_fijos(manifiesto["hilos_entrenamiento"]):
    p, art = entrenar_definitivo(datos, config, "prueba")
print(sha256_bytes(art.artefacto))
"""


def test_dos_entrenamientos_independientes_dan_el_mismo_artefacto(tmp_path):
    """Dos procesos, con PYTHONHASHSEED y OMP_NUM_THREADS distintos, misma copia y
    configuración: mismo SHA-256 de modelo.joblib (los hilos se fijan en el proceso)."""
    from dataclasses import asdict

    from app.modelo.copia_datos import escribir_copia
    from app.modelo.evaluacion import ConfigEvaluacion
    from tests.modelo.conftest import CONFIG_RAPIDA, datos_entrenamiento
    escribir_copia(datos_entrenamiento(), tmp_path / "copia", {
        "config": asdict(CONFIG_RAPIDA), "evaluacion": ConfigEvaluacion().como_dict(), "hilos_entrenamiento": 4})
    shas = []
    for semilla, hilos in (("0", "1"), ("12345", "8")):
        env = {**os.environ, "PYTHONHASHSEED": semilla, "OMP_NUM_THREADS": hilos, "PYTHONPATH": str(RAIZ)}
        p = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "copia")], env=env, cwd=RAIZ,
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        shas.append(p.stdout.strip().splitlines()[-1])
    assert shas[0] == shas[1]
