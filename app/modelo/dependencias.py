"""Verificación de las dependencias instaladas contra ``requirements/api.lock.txt``
(etapa 6, E6-8, plan §2.3). Se ejecuta al construir la imagen (la construcción falla
ante cualquier diferencia) y antes de cada entrenamiento en un contenedor.

1. **Lock** en tres fases (no se interpreta línea por línea): líneas lógicas (``\\``
   final une con la siguiente; un ``\\`` huérfano o CRLF son errores) → comentarios
   (``#`` al inicio o precedido por espacio) → clasificación (vacía; opción global,
   ``-r``/``-c``/``-e``, ruta o URL son errores; entrada = requisito con exactamente
   un ``==X`` + ``--hash=sha256:<64 hex>`` como única opción, al menos una).
   Marcadores evaluados en el intérprete en ejecución; dos entradas aplicables con el
   mismo nombre son un error.
2. **Instalado**: ``importlib.metadata.distributions()`` sobre ``sys.path``.
3. **Comparación exacta**: nombres PEP 503 y ``packaging.version.Version`` (incluida
   la etiqueta local); faltantes, adicionales, otra versión y duplicados.
4. **RECORD**: cada entrada con hash se verifica por SHA-256 y tamaño; sin hash solo
   ``RECORD`` (y ``.jws``/``.p7s``) e ``INSTALLER``/``REQUESTED`` con contenido
   validado; ``direct_url.json`` debe estar ausente. Todo archivo de
   ``site-packages`` debe pertenecer a algún RECORD. Sin bytecode.
5. **sys.path**: solo biblioteca estándar, ``site-packages`` y la raíz de la
   aplicación; ``site.ENABLE_USER_SITE`` falso; todo ``.pth`` pertenece a un RECORD.

Límite documentado: una imagen derivada que reescriba a la vez los archivos y su
RECORD no se detecta. La biblioteca estándar no figura en ningún RECORD: su
integridad depende solo del digest de la imagen base.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import shlex
import site
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

RAIZ = Path(__file__).resolve().parents[2]
LOCK = RAIZ / "requirements" / "api.lock.txt"
SIN_HASH_PERMITIDOS = {"RECORD", "RECORD.jws", "RECORD.p7s", "INSTALLER", "REQUESTED"}
HASH_RE = re.compile(r"^--hash=sha256:([0-9a-f]{64})$")
MAX_DIFERENCIAS = 20


class LockInvalido(ValueError):
    pass


@dataclass(frozen=True)
class EntradaLock:
    nombre: str                 # normalizado (PEP 503)
    version: Version
    marcador: str | None
    hashes: tuple[str, ...]
    linea: int


# --------------------------------------------------------------------------- lock
def _lineas_logicas(texto: str) -> list[tuple[int, str]]:
    if "\r" in texto:
        raise LockInvalido("el lock contiene CRLF o CR: debe usar finales de línea LF")
    fisicas = texto.split("\n")
    logicas, actual, inicio = [], None, 0
    for i, ln in enumerate(fisicas, start=1):
        if actual is None:
            actual, inicio = "", i
        if ln.endswith("\\"):
            actual += ln[:-1] + " "
            if i == len(fisicas) or (i == len(fisicas) - 1 and fisicas[-1] == ""):
                raise LockInvalido(f"línea {inicio}: '\\' final sin línea siguiente")
            continue
        actual += ln
        logicas.append((inicio, actual))
        actual = None
    return logicas


def _sin_comentario(linea: str) -> str:
    if linea.lstrip().startswith("#"):
        return ""
    m = re.search(r"\s#", linea)
    return linea[:m.start()] if m else linea


def leer_lock(texto: str) -> list[EntradaLock]:
    entradas = []
    for n, logica in _lineas_logicas(texto):
        linea = _sin_comentario(logica).strip()
        if not linea:
            continue
        if linea.startswith("-"):
            raise LockInvalido(f"línea {n}: opción no permitida en el lock: {linea.split()[0]}")
        # El requisito es todo lo anterior a la primera opción (" -"); los marcadores
        # conservan sus comillas. Las opciones se separan por espacios.
        m = re.search(r"\s-", linea)
        requisito = (linea[:m.start()] if m else linea).strip()
        try:
            opciones = shlex.split(linea[m.start():], posix=True) if m else []
        except ValueError as exc:
            raise LockInvalido(f"línea {n}: {exc}") from exc
        if "/" in requisito.split(";")[0] or "@" in requisito.split(";")[0]:
            raise LockInvalido(f"línea {n}: rutas y URL no están permitidas: {requisito}")
        try:
            req = Requirement(requisito)
        except InvalidRequirement as exc:
            raise LockInvalido(f"línea {n}: requisito inválido: {exc}") from exc
        especs = list(req.specifier)
        if len(especs) != 1 or especs[0].operator != "==" or "*" in especs[0].version:
            raise LockInvalido(f"línea {n}: {req.name} debe fijarse con exactamente un '==X'")
        hashes = []
        for op in opciones:
            m = HASH_RE.match(op)
            if not m:
                raise LockInvalido(f"línea {n}: opción no permitida o hash inválido: {op}")
            hashes.append(m.group(1))
        if not hashes:
            raise LockInvalido(f"línea {n}: {req.name} no tiene --hash=sha256")
        try:
            version = Version(especs[0].version)
        except InvalidVersion as exc:
            raise LockInvalido(f"línea {n}: versión inválida {especs[0].version}") from exc
        entradas.append(EntradaLock(canonicalize_name(req.name), version,
                                    str(req.marker) if req.marker else None, tuple(hashes), n))
    return entradas


def aplicables(entradas: list[EntradaLock]) -> dict[str, EntradaLock]:
    from packaging.markers import Marker
    out: dict[str, EntradaLock] = {}
    for e in entradas:
        if e.marcador and not Marker(e.marcador).evaluate():
            continue
        if e.nombre in out:
            raise LockInvalido(f"lock ambiguo: {e.nombre} aparece dos veces (líneas {out[e.nombre].linea} y "
                               f"{e.linea})")
        out[e.nombre] = e
    return out


# ------------------------------------------------------------------- instalado
def _hash_record(ruta: Path) -> tuple[str, int]:
    datos = ruta.read_bytes()
    return base64.urlsafe_b64encode(hashlib.sha256(datos).digest()).rstrip(b"=").decode(), len(datos)


def _verificar_record(dist, problemas: list[str], propios: set[Path]) -> None:
    nombre = dist.metadata["Name"]
    texto = dist.read_text("RECORD")
    if texto is None:
        problemas.append(f"{nombre}: sin RECORD")
        return
    base = Path(dist.locate_file("")).resolve()
    for fila in csv.reader(io.StringIO(texto)):
        if not fila:
            continue
        ruta_rel, hash_, tam = (fila + ["", "", ""])[:3]
        ruta = (base / ruta_rel).resolve()
        propios.add(ruta)
        if ruta.suffix in (".pyc", ".pyo"):
            problemas.append(f"{nombre}: el RECORD lista bytecode ({ruta_rel})")
            continue
        if ruta.name == "direct_url.json":
            problemas.append(f"{nombre}: instalado desde una URL o ruta (direct_url.json)")
            continue
        if not hash_:
            if ruta.name not in SIN_HASH_PERMITIDOS:
                problemas.append(f"{nombre}: archivo no verificable (sin hash en RECORD): {ruta_rel}")
            elif ruta.name == "INSTALLER" and ruta.is_file() and ruta.read_text().strip() != "pip":
                problemas.append(f"{nombre}: INSTALLER no es pip")
            elif ruta.name == "REQUESTED" and ruta.is_file() and ruta.read_bytes().strip():
                problemas.append(f"{nombre}: REQUESTED no está vacío")
            continue
        if not ruta.is_file():
            problemas.append(f"{nombre}: falta {ruta_rel}")
            continue
        algoritmo, _, valor = hash_.partition("=")
        if algoritmo != "sha256":
            problemas.append(f"{nombre}: hash no sha256 en RECORD: {ruta_rel}")
            continue
        real, tam_real = _hash_record(ruta)
        if real != valor or (tam and int(tam) != tam_real):
            problemas.append(f"{nombre}: archivo modificado: {ruta_rel}")
        if ruta.name == "INSTALLER" and ruta.read_text().strip() != "pip":
            problemas.append(f"{nombre}: INSTALLER no es pip")


def rutas_permitidas() -> set[Path]:
    p = sysconfig.get_paths()
    return {Path(x).resolve() for x in (p["stdlib"], p["platstdlib"], p["purelib"], p["platlib"], RAIZ)}


def _permitida(entrada: str, permitidas: set[Path]) -> bool:
    """Biblioteca estándar (y su ``lib-dynload`` y ``pythonXY.zip``), ``site-packages``
    exactos y la raíz de la aplicación. Nada dentro de ``site-packages`` ni fuera."""
    if not entrada:
        return False
    ruta = Path(entrada).resolve()
    if ruta in permitidas:
        return True
    p = sysconfig.get_paths()
    stdlib = Path(p["stdlib"]).resolve()
    paquetes = {Path(p["purelib"]).resolve(), Path(p["platlib"]).resolve()}
    if ruta.is_relative_to(stdlib) and not any(ruta.is_relative_to(q) for q in paquetes):
        return True                                       # lib-dynload
    return ruta.suffix == ".zip" and ruta.parent == stdlib.parent


def verificar_dependencias(lock: Path = LOCK, rutas: list[str] | None = None,
                           comprobar_sys_path: bool = True) -> list[str]:
    """Devuelve la lista de problemas (vacía = todo coincide)."""
    problemas: list[str] = []
    try:
        esperado = aplicables(leer_lock(Path(lock).read_text(encoding="utf-8")))
    except (OSError, LockInvalido) as exc:
        return [f"lock: {exc}"]
    rutas = list(sys.path if rutas is None else rutas)

    if comprobar_sys_path:
        if site.ENABLE_USER_SITE:
            problemas.append("el sitio de usuario está habilitado (PYTHONNOUSERSITE debe ser 1)")
        permitidas = rutas_permitidas()
        for entrada in rutas:
            if not _permitida(entrada, permitidas):
                problemas.append(f"sys.path contiene una ruta no permitida: {entrada!r}")

    instaladas: dict[str, list] = {}
    for d in metadata.distributions(path=rutas):
        instaladas.setdefault(canonicalize_name(d.metadata["Name"]), []).append(d)
    for nombre, dists in sorted(instaladas.items()):
        if len(dists) > 1:
            lugares = sorted({str(Path(d.locate_file("")).resolve()) for d in dists})
            problemas.append(f"{nombre}: instalado más de una vez ({', '.join(lugares)})")
    for nombre, e in sorted(esperado.items()):
        if nombre not in instaladas:
            problemas.append(f"{nombre}: falta (el lock exige {e.version})")
            continue
        v = Version(instaladas[nombre][0].version)
        if v != e.version:
            problemas.append(f"{nombre}: versión {v} instalada, el lock exige {e.version}")
    for nombre in sorted(set(instaladas) - set(esperado)):
        problemas.append(f"{nombre}: instalado pero no está en el lock")

    propios: set[Path] = set()
    for dists in instaladas.values():
        for d in dists:
            _verificar_record(d, problemas, propios)
    for sp in sorted({Path(r).resolve() for r in rutas if r and Path(r).name in ("site-packages", "dist-packages")}):
        if not sp.is_dir():
            continue
        for f in sp.rglob("*"):
            if f.is_dir():
                continue
            if f.suffix in (".pyc", ".pyo"):
                problemas.append(f"bytecode no permitido: {f}")
            elif f.resolve() not in propios:
                problemas.append(f"archivo sin dueño en site-packages: {f.relative_to(sp)}"
                                 + (" (.pth)" if f.suffix == ".pth" else ""))
    return problemas


def sin_bytecode(rutas: list[str] | None = None) -> list[str]:
    """Rechaza cualquier ``.pyc``/``.pyo`` en las rutas de ``sys.path`` (incluida la
    biblioteca estándar)."""
    hallados = []
    for r in (sys.path if rutas is None else rutas):
        p = Path(r)
        if not r or not p.is_dir():
            continue
        for f in p.rglob("*"):
            if f.suffix in (".pyc", ".pyo"):
                hallados.append(str(f))
                if len(hallados) >= MAX_DIFERENCIAS:
                    return hallados
    return hallados


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Verifica las dependencias instaladas contra el lock")
    ap.add_argument("--lock", default=str(LOCK))
    a = ap.parse_args(argv)
    problemas = verificar_dependencias(Path(a.lock)) + [f"bytecode: {x}" for x in sin_bytecode()]
    for p in problemas[:200]:
        print(p, file=sys.stderr)
    if problemas:
        print(f"{len(problemas)} problemas en las dependencias instaladas", file=sys.stderr)
        return 1
    print(f"dependencias verificadas: {len(aplicables(leer_lock(Path(a.lock).read_text())))} paquetes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
