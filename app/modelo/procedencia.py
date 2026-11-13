"""Procedencia del código que entrena (etapa 6, E6-2/E6-3, plan §2).

Capas (DECISIONES.md §15):

* **Código propio (clase A)**: ``app/``, ``ml/``, ``migraciones/``, ``alembic.ini``,
  ``requirements/comun.txt``, ``api.txt`` y ``api.lock.txt``. Se compara el blob de
  Git de cada archivo (``sha1("blob <n>\\0" + contenido)``) con ``procedencia.json`` y
  se rechaza cualquier archivo de más.
* **Construcción (clase B)**: ``Dockerfile`` (con el digest de la base),
  ``.dockerignore``, ``.gitattributes``, ``scripts/generar_procedencia.sh`` y
  ``docker-compose.yml``. Se registran al construir; su garantía es el árbol limpio.
* **Dependencias**: ``app.modelo.dependencias`` (lock + RECORD).
* **Intérprete y biblioteca estándar**: solo el digest de la imagen base.

:func:`verificar_procedencia` elige según dónde corre el código, sin opción para
omitirla: con un repositorio Git → la verificación de la etapa 5 (árbol limpio);
con ``procedencia.json`` → la de la imagen; sin ninguno → rechazo (código 2).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.modelo.git_estado import RepositorioNoValido

RAIZ = Path(__file__).resolve().parents[2]
ARCHIVO = RAIZ / "procedencia.json"
FORMATO = 1
DIRECTORIOS_A = ("app", "ml", "migraciones", "requirements")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_CAMBIOS = 10


class ProcedenciaInvalida(RepositorioNoValido):
    """El código o las dependencias no corresponden a un commit limpio verificable."""


@dataclass(frozen=True)
class Procedencia:
    commit: str
    tipo: str                         # "git" | "imagen"
    raiz: Path
    datos: dict = field(default_factory=dict)

    def etiquetas(self) -> dict[str, str]:
        e = {"procedencia": self.tipo, "codigo_verificado": "true"}
        if self.tipo == "imagen":
            e["imagen_base"] = str(self.datos.get("imagen_base", ""))
        return e


def blob_git(datos: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(datos) + datos).hexdigest()


# ----------------------------------------------------------------- generación
def _leer_ls_files(texto: str) -> dict[str, str]:
    """``git ls-files -s``: ``<modo> <blob> <etapa>\\t<ruta>``."""
    out = {}
    for ln in texto.splitlines():
        if not ln.strip():
            continue
        meta, ruta = ln.split("\t", 1)
        _, blob, _ = meta.split()
        out[ruta] = blob
    return out


def generar(carpeta_git: Path) -> dict:
    """Convierte la salida de ``scripts/generar_procedencia.sh`` en el contenido de
    ``procedencia.json`` (etapa final del Dockerfile)."""
    estado = (carpeta_git / "estado").read_text().strip()
    if estado != "con_git":
        return {"formato": FORMATO, "commit": None, "limpio": False, "sin_git": True,
                "cambios": ["el contexto de construcción no incluía .git"], "clase_a": {}, "clase_b": {},
                "imagen_base": None}
    cambios = [c for c in (carpeta_git / "cambios").read_text().splitlines() if c.strip()]
    return {"formato": FORMATO, "commit": (carpeta_git / "commit").read_text().strip(), "limpio": not cambios,
            "sin_git": False, "cambios": cambios[:MAX_CAMBIOS] + ([f"(y {len(cambios) - MAX_CAMBIOS} más)"]
                                                                  if len(cambios) > MAX_CAMBIOS else []),
            "clase_a": _leer_ls_files((carpeta_git / "clase_a").read_text()),
            "clase_b": _leer_ls_files((carpeta_git / "clase_b").read_text()),
            "imagen_base": (carpeta_git / "imagen_base").read_text().strip() or None}


# --------------------------------------------------------------- verificación
def verificar_construccion(raiz: Path = RAIZ, archivo: Path | None = None) -> list[str]:
    """Al construir: las dependencias siempre; el código solo si el árbol estaba limpio.
    Con el árbol sucio (o sin .git) la imagen se construye igual, marcada como no apta
    para entrenar (``limpio=false``): sus archivos son los del árbol de trabajo y no
    pueden coincidir con los blobs del índice."""
    from app.modelo.dependencias import sin_bytecode, verificar_dependencias
    problemas = [f"dependencias: {p}" for p in verificar_dependencias()]
    problemas += [f"bytecode no permitido: {p}" for p in sin_bytecode()]
    datos = json.loads(Path(archivo or raiz / "procedencia.json").read_text(encoding="utf-8"))
    if datos.get("limpio") and not datos.get("sin_git"):
        try:
            verificar_imagen(raiz, archivo, dependencias=False)
        except ProcedenciaInvalida as exc:
            problemas.append(str(exc))
    return problemas


def verificar_imagen(raiz: Path = RAIZ, archivo: Path | None = None, *, dependencias: bool = True) -> Procedencia:
    archivo = archivo or raiz / "procedencia.json"
    try:
        datos = json.loads(Path(archivo).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProcedenciaInvalida(f"no se pudo leer {archivo.name}: {exc}") from exc
    problemas = []
    if datos.get("sin_git"):
        raise ProcedenciaInvalida("la imagen se construyó sin .git: no hay un commit verificable. Construya "
                                  "la imagen desde un clon del repositorio (git clone), no desde un ZIP.")
    commit = str(datos.get("commit") or "")
    if not COMMIT_RE.match(commit):
        problemas.append("commit inválido en procedencia.json")
    if not datos.get("limpio"):
        cambios = datos.get("cambios") or []
        raise ProcedenciaInvalida(
            "No se puede entrenar: la imagen se construyó con cambios sin confirmar en el repositorio:\n  "
            + "\n  ".join(cambios) + "\nConfirme o descarte los cambios y reconstruya la imagen.")
    esperados: dict[str, str] = datos.get("clase_a") or {}
    if not esperados:
        problemas.append("procedencia.json no lista archivos de código")
    for ruta, blob in sorted(esperados.items()):
        f = raiz / ruta
        if not f.is_file():
            problemas.append(f"falta {ruta}")
        elif blob_git(f.read_bytes()) != blob:
            problemas.append(f"{ruta} no coincide con el commit {commit[:12]}")
    for d in DIRECTORIOS_A:
        base = raiz / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_dir() or "__pycache__" in f.parts:
                continue
            rel = f.relative_to(raiz).as_posix()
            if rel not in esperados:
                problemas.append(f"archivo que no está en el commit: {rel}")
    if dependencias:
        from app.modelo.dependencias import sin_bytecode, verificar_dependencias
        problemas += [f"dependencias: {p}" for p in verificar_dependencias()]
        problemas += [f"bytecode no permitido: {p}" for p in sin_bytecode()]
    if problemas:
        lista = problemas[:20] + ([f"(y {len(problemas) - 20} más)"] if len(problemas) > 20 else [])
        raise ProcedenciaInvalida("No se puede entrenar: el código o las dependencias de la imagen no "
                                  "corresponden al commit registrado:\n  " + "\n  ".join(lista))
    return Procedencia(commit, "imagen", raiz, datos)


def _hay_repositorio(codigo: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    r = subprocess.run([git, "-C", str(codigo.parent), "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def verificar_procedencia(codigo: str | Path | None = None, archivo: Path | None = None) -> Procedencia:
    """Repositorio Git → árbol limpio (etapa 5). ``procedencia.json`` → imagen. Ninguno →
    :class:`ProcedenciaInvalida`. No hay opción para omitirla."""
    codigo = Path(codigo or __file__).resolve()
    if _hay_repositorio(codigo):
        from app.modelo.git_estado import verificar_arbol_limpio
        g = verificar_arbol_limpio(codigo)
        return Procedencia(g.commit, "git", Path(g.raiz))
    archivo = archivo or ARCHIVO
    if archivo.is_file():
        return verificar_imagen(RAIZ, archivo)
    raise ProcedenciaInvalida("No se puede entrenar: el código no está en un repositorio Git ni tiene "
                              "procedencia.json (imagen construida desde un clon). No hay un commit verificable.")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Procedencia del código de la imagen")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--generar", metavar="CARPETA", help="salida de scripts/generar_procedencia.sh")
    g.add_argument("--verificar", action="store_true", help="verifica la imagen (código y dependencias)")
    g.add_argument("--verificar-construccion", action="store_true",
                   help="paso final del Dockerfile: dependencias siempre; código si el árbol estaba limpio")
    ap.add_argument("--salida", default=str(ARCHIVO))
    a = ap.parse_args(argv)
    if a.generar:
        datos = generar(Path(a.generar))
        Path(a.salida).write_text(json.dumps(datos, indent=2, sort_keys=True), encoding="utf-8")
        print(f"procedencia: commit {datos['commit']} limpio={datos['limpio']}")
        return 0
    if a.verificar_construccion:
        problemas = verificar_construccion(RAIZ)
        for x in problemas:
            print(x, file=sys.stderr)
        datos = json.loads(ARCHIVO.read_text(encoding="utf-8"))
        if not datos.get("limpio"):
            print("AVISO: imagen construida con cambios sin confirmar o sin .git: no podrá entrenar",
                  file=sys.stderr)
        return 1 if problemas else 0
    try:
        p = verificar_imagen(RAIZ)
    except ProcedenciaInvalida as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"procedencia verificada: commit {p.commit} ({len(p.datos['clase_a'])} archivos)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
