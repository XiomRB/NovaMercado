"""Arranque verificado de los comandos que entrenan (etapa 7, plan v3 §1 y E7-14).

Lo ejecuta el lanzador ``docker/ejecutar`` con ``python -I -S``:

* ``-I``: ignora las variables ``PYTHON*``, el sitio de usuario y el directorio actual;
* ``-S``: no importa ``site``: ningún ``.pth`` se procesa y ``site-packages`` no está en
  ``sys.path``.

Pasos (si alguno falla: código 2, sin haber importado nada no verificado):

1. **Solo biblioteca estándar**: lee ``procedencia.json`` (exige ``limpio``), recalcula
   el blob de Git de **todos** los archivos de clase A (incluidos los ``__init__.py``,
   ``dags/``, este archivo y el lanzador) y rechaza archivos de más en sus directorios.
2. Verifica con la biblioteca estándar el ``RECORD`` de ``packaging`` (lo importa la
   verificación de dependencias) y su versión en el lock. Agrega a ``sys.path`` la raíz
   y el ``site-packages`` del intérprete **sin** procesar ``.pth``; importa
   ``app.modelo.dependencias`` (ya verificado) y comprueba lock, ``RECORD``, archivos
   sin dueño, bytecode y aislamiento (``find_spec("airflow") is None``; en un entorno
   virtual, ``include-system-site-packages = false``).
3. ``site.addsitedir`` (procesa los ``.pth`` ya verificados) y ejecuta el módulo con
   ``runpy`` como ``__main__``. Solo módulos de una lista cerrada.

Ancla de confianza: este archivo y ``procedencia.json`` vienen en la imagen, son de root y
de solo lectura para el usuario que ejecuta (límite documentado en DECISIONES.md §16).
"""

import base64
import hashlib
import json
import os
import sys
import sysconfig

PROCEDENCIA = 2
MODULOS = ("app.modelo.reentrenar", "app.modelo.reconciliacion", "app.modelo.entrenar_mlflow",
           "app.modelo.arranque")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTORIOS_A = ("app", "ml", "migraciones", "requirements")


class Rechazo(Exception):
    pass


def blob_git(datos: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(datos) + datos).hexdigest()


def verificar_codigo(raiz: str) -> dict:
    """Paso 1 (solo biblioteca estándar). Devuelve ``procedencia.json``."""
    try:
        with open(os.path.join(raiz, "procedencia.json"), encoding="utf-8") as fh:
            datos = json.load(fh)
    except (OSError, ValueError) as exc:
        raise Rechazo(f"no se pudo leer procedencia.json: {exc}") from exc
    if datos.get("sin_git"):
        raise Rechazo("la imagen se construyó sin .git: no hay un commit verificable")
    if not datos.get("limpio"):
        raise Rechazo("la imagen se construyó con cambios sin confirmar: "
                      + "; ".join(datos.get("cambios") or []))
    esperados = datos.get("clase_a") or {}
    if not esperados:
        raise Rechazo("procedencia.json no lista archivos de código")
    problemas = []
    for ruta, blob in sorted(esperados.items()):
        f = os.path.join(raiz, ruta)
        if not os.path.isfile(f):
            problemas.append(f"falta {ruta}")
            continue
        with open(f, "rb") as fh:
            if blob_git(fh.read()) != blob:
                problemas.append(f"{ruta} no coincide con el commit")
    for d in datos.get("directorios_a") or DIRECTORIOS_A:
        base = os.path.join(raiz, d)
        for actual, dirs, archivos in os.walk(base):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for a in archivos:
                rel = os.path.relpath(os.path.join(actual, a), raiz).replace(os.sep, "/")
                if rel not in esperados:
                    problemas.append(f"archivo que no está en el commit: {rel}")
    if problemas:
        raise Rechazo("el código no corresponde al commit registrado: " + "; ".join(problemas[:20]))
    return datos


def _version_en_lock(raiz: str, nombre: str) -> str | None:
    with open(os.path.join(raiz, "requirements", "api.lock.txt"), encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith(nombre + "=="):
                return ln.split("==", 1)[1].split()[0].rstrip("\\").strip()
    return None


def verificar_packaging(raiz: str, site_packages: str) -> None:
    """``packaging`` se importa antes de verificar las dependencias: su RECORD y su
    versión se comprueban aquí con la biblioteca estándar."""
    esperada = _version_en_lock(raiz, "packaging")
    info = os.path.join(site_packages, f"packaging-{esperada}.dist-info")
    if esperada is None or not os.path.isdir(info):
        raise Rechazo(f"packaging no coincide con el lock (se esperaba {esperada})")
    import csv
    with open(os.path.join(info, "RECORD"), encoding="utf-8", newline="") as fh:
        filas = list(csv.reader(fh))
    for fila in filas:
        if len(fila) < 2 or not fila[1]:
            continue
        algoritmo, valor = fila[1].split("=", 1)
        if algoritmo != "sha256":
            raise Rechazo(f"packaging: hash no admitido en RECORD ({algoritmo})")
        with open(os.path.join(site_packages, fila[0]), "rb") as fh:
            real = base64.urlsafe_b64encode(hashlib.sha256(fh.read()).digest()).rstrip(b"=").decode()
        if real != valor:
            raise Rechazo(f"packaging: {fila[0]} no coincide con su RECORD")


def verificar_entorno(raiz: str, site_packages: str) -> None:
    """Paso 2, con el código ya verificado."""
    verificar_packaging(raiz, site_packages)
    for r in (raiz, site_packages):
        if r not in sys.path:
            sys.path.append(r)                       # sin procesar .pth
    from importlib.util import find_spec

    from app.modelo.dependencias import sin_bytecode, verificar_dependencias
    problemas = verificar_dependencias() + [f"bytecode: {x}" for x in sin_bytecode()]
    if find_spec("airflow") is not None:
        problemas.append("airflow es importable desde el entorno de entrenamiento")
    if sys.prefix != sys.base_prefix:                # entorno virtual (imagen de Airflow)
        if os.path.realpath(site_packages) != os.path.realpath(
                os.path.join(sys.prefix, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}",
                             "site-packages")):
            problemas.append(f"site-packages fuera del entorno virtual: {site_packages}")
        cfg = os.path.join(sys.prefix, "pyvenv.cfg")
        with open(cfg, encoding="utf-8") as fh:
            lineas = [x.replace(" ", "").lower() for x in fh.read().splitlines()]
        if "include-system-site-packages=false" not in lineas:
            problemas.append("el entorno virtual hereda los paquetes del sistema")
    if problemas:
        raise Rechazo("dependencias o aislamiento: " + "; ".join(problemas[:20]))


def entorno_virtual() -> str | None:
    """Con ``-S`` Python 3.11 no aplica el entorno virtual (lo hace ``site.venv``):
    ``sys.prefix`` queda en la base. Si el intérprete es el de un entorno virtual
    (``pyvenv.cfg`` junto a ``bin/``), se fija su prefijo como haría ``site``, para que
    ``sysconfig`` dé su ``site-packages`` y nunca el de la base."""
    raiz = os.path.dirname(os.path.dirname(sys.executable))
    cfg = os.path.join(raiz, "pyvenv.cfg")
    if not os.path.isfile(cfg):
        return None
    sys.prefix = sys.exec_prefix = raiz
    return cfg


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in MODULOS:
        print(f"arranque verificado: módulo no permitido (permitidos: {', '.join(MODULOS)})", file=sys.stderr)
        return PROCEDENCIA
    if not (sys.flags.isolated and sys.flags.no_site):
        print("arranque verificado: debe ejecutarse con python -I -S", file=sys.stderr)
        return PROCEDENCIA
    entorno_virtual()                                # antes de la primera llamada a sysconfig
    site_packages = sysconfig.get_paths()["purelib"]
    try:
        verificar_codigo(RAIZ)
        verificar_entorno(RAIZ, site_packages)
    except Rechazo as exc:
        print(f"No se puede ejecutar: {exc}", file=sys.stderr)
        return PROCEDENCIA
    import runpy
    import site
    site.addsitedir(site_packages)                   # .pth ya verificados (pertenecen a un RECORD)
    os.chdir(RAIZ)
    sys.argv = [argv[0], *argv[1:]]
    try:
        runpy.run_module(argv[0], run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        codigo = exc.code
        return codigo if isinstance(codigo, int) else (0 if codigo is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
