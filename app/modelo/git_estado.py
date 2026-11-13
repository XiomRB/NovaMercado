"""Verificación de Git antes de un entrenamiento de producción (DECISIONES.md §14).

* La raíz se resuelve **desde la ubicación del código que se ejecuta**
  (``Path(__file__)``), nunca desde el directorio actual, y se exige que el
  código esté dentro de esa raíz.
* ``HEAD`` se lee antes y después de ``git status``: si cambió, se rechaza.
* El árbol debe estar limpio: sin archivos versionados modificados, sin cambios
  preparados y sin archivos nuevos no ignorados. Los ignorados (``raw/``,
  ``.env``, ``modelos/``) no cuentan.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_LISTADOS = 10


class RepositorioNoValido(RuntimeError):
    """No se puede garantizar que el commit registrado corresponda al código ejecutado."""


@dataclass(frozen=True)
class EstadoGit:
    raiz: Path
    commit: str


def _git(git: str, raiz: Path, *args: str) -> str:
    r = subprocess.run([git, "-C", str(raiz), *args], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RepositorioNoValido(f"git {' '.join(args)} falló")
    return r.stdout


def verificar_arbol_limpio(codigo: str | Path | None = None) -> EstadoGit:
    git = shutil.which("git")
    if git is None:
        raise RepositorioNoValido("No se puede entrenar para producción: git no está disponible para "
                                  "verificar que el código corresponde a un commit.")
    archivo = Path(codigo if codigo is not None else __file__).resolve()
    try:
        raiz = Path(_git(git, archivo.parent, "rev-parse", "--show-toplevel").strip()).resolve()
    except RepositorioNoValido as exc:
        raise RepositorioNoValido("No se puede entrenar para producción: el código no está dentro de un "
                                  "repositorio Git, así que no hay un commit que lo identifique.") from exc
    if not archivo.is_relative_to(raiz):
        raise RepositorioNoValido("No se puede entrenar para producción: el código ejecutado no pertenece "
                                  "al repositorio resuelto.")
    antes = _git(git, raiz, "rev-parse", "HEAD").strip()
    estado = _git(git, raiz, "status", "--porcelain", "--untracked-files=normal")
    despues = _git(git, raiz, "rev-parse", "HEAD").strip()
    if antes != despues:
        raise RepositorioNoValido("No se puede entrenar para producción: el commit del repositorio cambió "
                                  "durante la verificación. Vuelva a intentarlo.")
    lineas = [ln for ln in estado.splitlines() if ln.strip()]
    if lineas:
        listado = "\n".join(f"  {ln}" for ln in lineas[:MAX_LISTADOS])
        if len(lineas) > MAX_LISTADOS:
            listado += f"\n  (y {len(lineas) - MAX_LISTADOS} más)"
        raise RepositorioNoValido(
            "No se puede entrenar para producción: el repositorio tiene cambios sin confirmar.\n"
            "El commit registrado en MLflow debe corresponder exactamente al código usado.\n"
            f"Confirme o descarte estos cambios y vuelva a intentarlo:\n{listado}")
    return EstadoGit(raiz, antes)
