"""Procedencia y dependencias sin servidores (plan §2): lector del lock en tres fases,
RECORD, archivos sin dueño, bytecode, sys.path, blobs de Git y verificación de la imagen."""

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import app.modelo.dependencias as dep
from app.modelo import procedencia as pr

H = "a" * 64
H2 = "b" * 64


# ----------------------------------------------------------------------- lock
def test_lector_del_formato_completo_de_pip_compile():
    texto = (FIXT / "lock_pip_compile.txt").read_text()
    e = dep.leer_lock(texto)
    assert [(x.nombre, str(x.version)) for x in e] == [("alembic", "1.14.1"), ("mako", "1.3.8"),
                                                      ("markupsafe", "3.0.2"), ("pywin32", "308"),
                                                      ("uvicorn", "0.34.0")]
    assert e[0].hashes == (H, H2) and e[3].marcador == 'sys_platform == "win32"'
    assert sorted(dep.aplicables(e)) == ["alembic", "mako", "markupsafe", "uvicorn"]   # el marcador no aplica


@pytest.mark.parametrize("texto, error", [
    (f"x==1 \\\n    --hash=sha256:{H} \\", "sin línea siguiente"),
    (f"x==1 --hash=sha256:{H}\r\n", "CRLF"),
    ("x==1\n", "no tiene --hash"),
    (f"x==1 --hash=md5:{H[:32]}\n", "hash inválido"),
    (f"x>=1 --hash=sha256:{H}\n", "exactamente un '==X'"),
    (f"x==1.* --hash=sha256:{H}\n", "exactamente un '==X'"),
    (f"x===1 --hash=sha256:{H}\n", "exactamente un '==X'"),
    (f"x @ https://e.org/x.whl --hash=sha256:{H}\n", "URL"),
    ("-e .\n", "opción no permitida"),
    ("--index-url https://otro\n", "opción no permitida"),
    ("-r otro.txt\n", "opción no permitida"),
    (f"x==1 --hash=sha256:{H} --no-binary=x\n", "opción no permitida"),
])
def test_lock_invalido(texto, error):
    with pytest.raises(dep.LockInvalido, match=error):
        dep.aplicables(dep.leer_lock(texto))


def test_nombre_duplicado_aplicable_es_ambiguo():
    with pytest.raises(dep.LockInvalido, match="ambiguo"):
        dep.aplicables(dep.leer_lock(f"Foo_Bar==1 --hash=sha256:{H}\nfoo.bar==1 --hash=sha256:{H}\n"))


FIXT = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------- site-packages simulado
def _hash(datos: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(datos).digest()).rstrip(b"=").decode()


def instalar(sp: Path, nombre: str, version: str, archivos: dict[str, bytes] | None = None,
             record_extra: list[str] | None = None) -> Path:
    sp.mkdir(parents=True, exist_ok=True)
    archivos = archivos or {f"{nombre}/__init__.py": b"x = 1\n"}
    info = f"{nombre}-{version}.dist-info"
    archivos = {**archivos, f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {nombre}\nVersion: {version}\n".encode(),
                f"{info}/INSTALLER": b"pip\n"}
    filas = []
    for rel, datos in archivos.items():
        (sp / rel).parent.mkdir(parents=True, exist_ok=True)
        (sp / rel).write_bytes(datos)
        filas.append(f"{rel},{_hash(datos)},{len(datos)}")
    filas += record_extra or []
    filas.append(f"{info}/RECORD,,")
    (sp / info / "RECORD").write_text("\n".join(filas) + "\n")
    return sp / info


def lock(tmp_path, *entradas):
    f = tmp_path / "api.lock.txt"
    f.write_text("".join(f"{n}=={v} \\\n    --hash=sha256:{H}\n" for n, v in entradas))
    return f


def verificar(tmp_path, sp, *entradas):
    return dep.verificar_dependencias(lock(tmp_path, *entradas), rutas=[str(sp)], comprobar_sys_path=False)


def test_instalacion_igual_al_lock(tmp_path):
    sp = tmp_path / "site-packages"
    instalar(sp, "demo", "1.0")
    instalar(sp, "Otro_Paquete", "2.0.0")
    assert verificar(tmp_path, sp, ("demo", "1.0"), ("otro.paquete", "2.0")) == []   # nombres y versiones normalizados


@pytest.mark.parametrize("caso, esperado", [
    ("adicional", "instalado pero no está en el lock"),
    ("version", "versión 1.1 instalada"),
    ("faltante", "falta (el lock exige 2.0)"),
    ("local", "versión 1.0+cpu instalada"),
])
def test_diferencias_con_el_lock(tmp_path, caso, esperado):
    sp = tmp_path / "site-packages"
    entradas = [("demo", "1.0")]
    if caso == "adicional":
        instalar(sp, "demo", "1.0")
        instalar(sp, "extra", "1.0")
    elif caso == "version":
        instalar(sp, "demo", "1.1")
    elif caso == "faltante":
        instalar(sp, "demo", "1.0")
        entradas.append(("otro", "2.0"))
    else:
        instalar(sp, "demo", "1.0+cpu")
    problemas = verificar(tmp_path, sp, *entradas)
    assert any(esperado in p for p in problemas), problemas


def test_duplicado_en_dos_rutas(tmp_path):
    a, b = tmp_path / "a" / "site-packages", tmp_path / "b" / "site-packages"
    instalar(a, "demo", "1.0")
    instalar(b, "demo", "1.0")
    p = dep.verificar_dependencias(lock(tmp_path, ("demo", "1.0")), rutas=[str(a), str(b)], comprobar_sys_path=False)
    assert any("instalado más de una vez" in x for x in p)


@pytest.mark.parametrize("caso, esperado", [
    ("modificado", "archivo modificado: demo/__init__.py"),
    ("sin_hash", "archivo no verificable"),
    ("pth_ajeno", "archivo sin dueño en site-packages: ajeno.pth"),
    ("modulo_agregado", "archivo sin dueño en site-packages: demo/agregado.py"),
    ("bytecode", "bytecode no permitido"),
    ("direct_url", "direct_url.json"),
    ("installer", "INSTALLER no es pip"),
])
def test_integridad_de_lo_instalado(tmp_path, caso, esperado):
    sp = tmp_path / "site-packages"
    extra = {"sin_hash": ["demo/sin_hash.py,,"]}.get(caso)
    info = instalar(sp, "demo", "1.0", record_extra=extra)
    if caso == "modificado":
        (sp / "demo" / "__init__.py").write_bytes(b"x = 2\n")
    elif caso == "sin_hash":
        (sp / "demo" / "sin_hash.py").write_text("")
    elif caso == "pth_ajeno":
        (sp / "ajeno.pth").write_text("/tmp\n")
    elif caso == "modulo_agregado":
        (sp / "demo" / "agregado.py").write_text("")
    elif caso == "bytecode":
        (sp / "demo" / "__pycache__").mkdir()
        (sp / "demo" / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"\0")
    elif caso == "direct_url":
        (info / "direct_url.json").write_text("{}")
        rec = info / "RECORD"
        rec.write_text(rec.read_text() + f"{info.name}/direct_url.json,{_hash(b'{}')},2\n")
    else:
        (info / "INSTALLER").write_bytes(b"otro\n")
    problemas = verificar(tmp_path, sp, ("demo", "1.0"))
    assert any(esperado in p for p in problemas), problemas


def test_entradas_sin_hash_permitidas_solo_las_previstas(tmp_path):
    sp = tmp_path / "site-packages"
    info = instalar(sp, "demo", "1.0")
    (info / "REQUESTED").write_bytes(b"")
    rec = info / "RECORD"
    rec.write_text(rec.read_text() + f"{info.name}/REQUESTED,,\n")
    assert verificar(tmp_path, sp, ("demo", "1.0")) == []           # RECORD y REQUESTED vacío sin hash


def test_sys_path_y_sitio_de_usuario(tmp_path, monkeypatch):
    sp = tmp_path / "site-packages"
    instalar(sp, "demo", "1.0")
    monkeypatch.setattr(dep.site, "ENABLE_USER_SITE", True)
    p = dep.verificar_dependencias(lock(tmp_path, ("demo", "1.0")), rutas=[str(sp), "/home/x/.local/lib"],
                                   comprobar_sys_path=True)
    assert any("sitio de usuario" in x for x in p)
    assert any("ruta no permitida: '/home/x/.local/lib'" in x for x in p)
    assert any(str(sp) in x and "no permitida" in x for x in p)      # un site-packages ajeno tampoco


def test_rutas_permitidas_de_la_biblioteca_estandar():
    import sysconfig
    permitidas = dep.rutas_permitidas()
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    assert dep._permitida(str(stdlib), permitidas)
    assert dep._permitida(str(stdlib / "lib-dynload"), permitidas)
    assert dep._permitida(str(stdlib.parent / "python311.zip"), permitidas)
    assert not dep._permitida(str(Path(sysconfig.get_paths()["purelib"]) / "otro"), permitidas)
    assert not dep._permitida("", permitidas)


def test_sin_bytecode(tmp_path):
    (tmp_path / "a").mkdir()
    assert dep.sin_bytecode([str(tmp_path)]) == []
    (tmp_path / "a" / "m.pyc").write_bytes(b"\0")
    assert dep.sin_bytecode([str(tmp_path)]) == [str(tmp_path / "a" / "m.pyc")]


# ---------------------------------------------------------------- procedencia
def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("requiere git")
    r = tmp_path / "repo"
    for rel, contenido in {"app/m.py": "x = 1\n", "ml/f.py": "y = 2\n", "migraciones/env.py": "",
                           "alembic.ini": "[alembic]\n", "requirements/comun.txt": "numpy==1\n",
                           "requirements/api.txt": "-r comun.txt\n", "requirements/api.lock.txt": "",
                           "Dockerfile": "FROM python:3.11.18-slim-bookworm@sha256:" + "c" * 64 + "\n",
                           "docker-compose.yml": "", ".dockerignore": "", ".gitattributes": "* text=auto eol=lf\n",
                           "scripts/generar_procedencia.sh": "", "README.md": ""}.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(contenido)
    git(r.parent, "init", "-q", str(r))
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "inicial")
    return r


def generar(repo, salida):
    script = Path(pr.RAIZ) / "scripts" / "generar_procedencia.sh"
    subprocess.run(["sh", str(script), str(repo), str(salida)], check=True, capture_output=True,
                   env={"HOME": str(salida.parent), "PATH": "/usr/bin:/bin"})
    datos = pr.generar(salida)
    (repo / "procedencia.json").write_text(json.dumps(datos))
    return datos


def test_blob_igual_a_git_hash_object(repo):
    f = repo / "app" / "m.py"
    assert pr.blob_git(f.read_bytes()) == git(repo, "hash-object", str(f)).strip()


def test_procedencia_limpia_se_verifica(repo, tmp_path):
    d = generar(repo, tmp_path / "git")
    assert d["limpio"] and d["commit"] == git(repo, "rev-parse", "HEAD").strip()
    assert set(d["clase_a"]) == {"app/m.py", "ml/f.py", "migraciones/env.py", "alembic.ini",
                                 "requirements/comun.txt", "requirements/api.txt", "requirements/api.lock.txt"}
    assert "Dockerfile" in d["clase_b"] and d["imagen_base"].startswith("python:3.11.18-slim-bookworm@sha256:")
    p = pr.verificar_imagen(repo, dependencias=False)
    assert p.tipo == "imagen" and p.commit == d["commit"]
    assert p.etiquetas()["codigo_verificado"] == "true"


@pytest.mark.parametrize("caso, esperado", [
    ("modificado", "app/m.py no coincide con el commit"),
    ("agregado", "archivo que no está en el commit: ml/nuevo.py"),
    ("agregado_no_py", "archivo que no está en el commit: app/datos.bin"),
    ("borrado", "falta ml/f.py"),
])
def test_codigo_que_no_coincide_se_rechaza(repo, tmp_path, caso, esperado):
    generar(repo, tmp_path / "git")
    if caso == "modificado":
        (repo / "app" / "m.py").write_text("x = 999\n")
    elif caso == "agregado":
        (repo / "ml" / "nuevo.py").write_text("")
    elif caso == "agregado_no_py":
        (repo / "app" / "datos.bin").write_bytes(b"\0")
    else:
        (repo / "ml" / "f.py").unlink()
    with pytest.raises(pr.ProcedenciaInvalida, match=esperado):
        pr.verificar_imagen(repo, dependencias=False)


def test_arbol_sucio_al_construir_no_puede_entrenar(repo, tmp_path):
    (repo / "app" / "m.py").write_text("x = 2\n")
    d = generar(repo, tmp_path / "git")
    assert not d["limpio"] and any("app/m.py" in c for c in d["cambios"])
    with pytest.raises(pr.ProcedenciaInvalida, match="cambios sin confirmar"):
        pr.verificar_imagen(repo, dependencias=False)


def test_sin_git_en_el_contexto_se_rechaza(tmp_path):
    salida = tmp_path / "git"
    (tmp_path / "sin_git").mkdir()
    subprocess.run(["sh", str(Path(pr.RAIZ) / "scripts" / "generar_procedencia.sh"), str(tmp_path / "sin_git"),
                    str(salida)], check=True, capture_output=True,
                   env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "GIT_CEILING_DIRECTORIES": str(tmp_path)})
    d = pr.generar(salida)
    assert d["sin_git"] and d["commit"] is None
    (tmp_path / "sin_git" / "procedencia.json").write_text(json.dumps(d))
    with pytest.raises(pr.ProcedenciaInvalida, match="sin .git"):
        pr.verificar_imagen(tmp_path / "sin_git", dependencias=False)


def test_sin_repositorio_ni_procedencia_se_rechaza(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "_hay_repositorio", lambda codigo: False)
    with pytest.raises(pr.ProcedenciaInvalida, match="No hay un commit verificable"):
        pr.verificar_procedencia(archivo=tmp_path / "no_existe.json")


def test_crlf_cambia_el_blob_y_gitattributes_lo_evita():
    lf, crlf = b"x = 1\n", b"x = 1\r\n"
    assert pr.blob_git(lf) != pr.blob_git(crlf)
    reglas = (Path(pr.RAIZ) / ".gitattributes").read_text()
    assert "* text=auto eol=lf" in reglas


def test_dockerfile_fija_la_imagen_base_por_digest():
    import re
    texto = (Path(pr.RAIZ) / "Dockerfile").read_text()
    for linea in (ln for ln in texto.splitlines() if ln.startswith("FROM ")):
        assert re.search(r"@sha256:[0-9a-f]{64}", linea), linea


def test_lock_coherente_con_las_dependencias_directas():
    """Cada dependencia directa de comun.txt y api.txt está en el lock con la MISMA versión
    (no se exige regenerar el lock byte a byte: el versionado es la referencia)."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    raiz = Path(pr.RAIZ) / "requirements"
    lock = dep.aplicables(dep.leer_lock((raiz / "api.lock.txt").read_text()))
    directas = {}
    for archivo in ("comun.txt", "api.txt"):
        for ln in (raiz / archivo).read_text().splitlines():
            ln = ln.split("#")[0].strip()
            if ln and not ln.startswith("-"):
                r = Requirement(ln)
                directas[canonicalize_name(r.name)] = str(next(iter(r.specifier)).version)
    assert {"numpy", "pandas", "scikit-learn", "mlflow", "pip", "setuptools", "wheel"} <= set(directas)
    for nombre, version in directas.items():
        assert nombre in lock and str(lock[nombre].version) == version, nombre


def test_casos_reales_de_los_wheels(tmp_path):
    """Hallazgos al construir la imagen: un wheel con bytecode propio en su RECORD (numpy),
    un INSTALLER vendorizado distinto de pip (setuptools/_vendor) y el README.txt de CPython."""
    sp = tmp_path / "site-packages"
    vendor = b"uv\n"
    instalar(sp, "demo", "1.0", archivos={"demo/__init__.py": b"x = 1\n",
                                          "demo/_vendor/otro-1.0.dist-info/INSTALLER": vendor},
             record_extra=["demo/__pycache__/m.cpython-311.pyc,sha256=abc,3"])
    (sp / "README.txt").write_text(dep.README_CPYTHON)
    assert verificar(tmp_path, sp, ("demo", "1.0")) == []
    (sp / "demo" / "__pycache__").mkdir()
    (sp / "demo" / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"\0\0\0")      # si existe, se rechaza
    (sp / "README.txt").write_text("otro contenido\n")
    problemas = verificar(tmp_path, sp, ("demo", "1.0"))
    assert any("bytecode no permitido (demo/__pycache__/m.cpython-311.pyc)" in p for p in problemas)
    assert any("archivo sin dueño en site-packages: README.txt" in p for p in problemas)
    (sp / "demo" / "_vendor" / "otro-1.0.dist-info" / "INSTALLER").write_bytes(b"pip\n")   # vendorizado: por hash
    assert any("archivo modificado: demo/_vendor/otro-1.0.dist-info/INSTALLER" in p
               for p in verificar(tmp_path, sp, ("demo", "1.0")))
