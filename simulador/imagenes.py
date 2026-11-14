"""Imágenes del simulador (S11).

``docker-compose.yml`` usa ``${NM_IMAGEN_APP:-novamercado-app:local}`` y
``${NM_IMAGEN_AIRFLOW:-novamercado-airflow:local}``: sin esas variables nada cambia. El
simulador las define (solo en su archivo de entorno) como ``…:sim-<commit de 12>``.

* Construye solo con ``docker compose build`` explícito, una imagen a la vez, con el árbol
  limpio y ``HEAD`` igual al commit de la ejecución; registra el ID de cada imagen y el
  commit que la imagen declara en su procedencia (etapa 6).
* Antes de cada arranque o reanudación: la etiqueta existe con el ID registrado y su
  procedencia es el commit de la ejecución (= ``HEAD``). Si no, se niega (``ImagenNoValida``).
  Única excepción: la etiqueta desapareció y el commit coincide → se reconstruye.
* Las etiquetas ``:local`` del producto nunca se crean, mueven ni borran: este módulo no
  ejecuta ``docker tag`` ni ``docker rmi``.
"""

from __future__ import annotations

PROTEGIDAS = {"app": "novamercado-app:local", "airflow": "novamercado-airflow:local"}
SERVICIOS = {"app": ("api", "preparar", "entrenamiento-inicial"),
             "airflow": ("airflow-init", "airflow-scheduler", "airflow-webserver")}
VARIABLES = {"app": "NM_IMAGEN_APP", "airflow": "NM_IMAGEN_AIRFLOW"}
CONSTRUIR = {"app": "api", "airflow": "airflow-webserver"}          # un servicio por imagen
_PROCEDENCIA = {
    "app": ["python", "-c", "import json;d=json.load(open('/app/procedencia.json'));print(d['commit'],d['limpio'])"],
    "airflow": ["--entrypoint", "/opt/novamercado/venv/bin/python", None, "-c",
                "import json;d=json.load(open('/opt/novamercado/procedencia.json'));print(d['commit'],d['limpio'])"],
}


class ImagenNoValida(Exception):
    pass


def etiquetas(commit: str) -> dict[str, str]:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ImagenNoValida(f"commit inválido: {commit!r}")
    return {"app": f"novamercado-app:sim-{commit[:12]}", "airflow": f"novamercado-airflow:sim-{commit[:12]}"}


def id_imagen(ejecutar, etiqueta: str) -> str | None:
    r = ejecutar(["docker", "image", "inspect", "-f", "{{.Id}}", etiqueta])
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def procedencia(ejecutar, tipo: str, etiqueta: str) -> tuple[str | None, bool]:
    plantilla = _PROCEDENCIA[tipo]
    if tipo == "app":
        cmd = ["docker", "run", "--rm", "--network", "none", etiqueta, *plantilla]
    else:
        cmd = ["docker", "run", "--rm", "--network", "none", *[etiqueta if x is None else x for x in plantilla]]
    r = ejecutar(cmd)
    partes = r.stdout.split()
    if r.returncode != 0 or len(partes) != 2:
        return None, False
    return partes[0], partes[1] == "True"


def ids_protegidas(ejecutar) -> dict[str, str | None]:
    return {t: id_imagen(ejecutar, e) for t, e in PROTEGIDAS.items()}


def preparar(entorno, estado, commit: str, head: str, arbol_limpio: bool) -> list[dict]:
    """Deja verificadas las dos imágenes de la ejecución. Devuelve los eventos ocurridos
    (``construida``, ``reutilizada``, ``reconstruida``). Lanza ``ImagenNoValida``."""
    if head != commit:
        raise ImagenNoValida(f"HEAD {head[:12]} no es el commit de la ejecución {commit[:12]}")
    registradas = estado.get("imagenes") or {}
    eventos = []
    for tipo, etiqueta in etiquetas(commit).items():
        previo = registradas.get(tipo)
        actual = id_imagen(entorno.ejecutar, etiqueta)
        if previo and actual and actual != previo["id"]:
            raise ImagenNoValida(f"{etiqueta} apunta a {actual[:19]}, no a la imagen registrada {previo['id'][:19]}")
        accion = "reutilizada" if actual else ("reconstruida" if previo else "construida")
        if not actual:
            if not arbol_limpio:
                raise ImagenNoValida("el árbol de Git tiene cambios: no se construye una imagen")
            r = entorno.compose("build", CONSTRUIR[tipo])
            if r.returncode != 0:
                raise ImagenNoValida(f"falló la construcción de {etiqueta}: {r.stderr[-800:]}")
            actual = id_imagen(entorno.ejecutar, etiqueta)
            if not actual:
                raise ImagenNoValida(f"la construcción no dejó la etiqueta {etiqueta}")
        c, limpio = procedencia(entorno.ejecutar, tipo, etiqueta)
        if c != commit or not limpio:
            raise ImagenNoValida(f"{etiqueta}: procedencia del commit {str(c)[:12]} (limpio={limpio}), "
                                 f"no de {commit[:12]}")
        registradas[tipo] = {"etiqueta": etiqueta, "id": actual}
        eventos.append({"imagen": tipo, "etiqueta": etiqueta, "id": actual, "accion": accion})
    estado["imagenes"] = registradas
    estado.guardar()
    return eventos


def verificar_contenedores(entorno, estado) -> list[str]:
    """Cada contenedor de un servicio con imagen propia usa exactamente el ID registrado."""
    ids = {t: v["id"] for t, v in estado["imagenes"].items()}
    errores = []
    r = entorno.ejecutar(["docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={entorno.proyecto}",
                          "--format", '{{.ID}} {{.Label "com.docker.compose.service"}}'])
    for linea in r.stdout.splitlines():
        cid, _, servicio = linea.partition(" ")
        tipo = next((t for t, ss in SERVICIOS.items() if servicio in ss), None)
        if tipo is None:
            continue
        img = entorno.ejecutar(["docker", "inspect", "-f", "{{.Image}}", cid]).stdout.strip()
        if img != ids[tipo]:
            errores.append(f"{servicio} usa {img[:19]}, no {ids[tipo][:19]}")
    return errores
