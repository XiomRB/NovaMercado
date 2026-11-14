"""Proyecto de Compose aislado del simulador (plan v2 §3).

* Siempre ``docker compose -p nm-sim-<ejecución> -f <repo>/docker-compose.yml
  --project-directory <repo> --env-file <trabajo>/entorno.env``: volúmenes, red y
  contenedores llevan el prefijo del proyecto (el archivo no fija ``name:`` en volúmenes
  ni ``container_name``).
* El archivo de entorno (0600) se genera con claves aleatorias, los puertos propios, el
  ``RAW_DIR`` de la ejecución y las etiquetas de imagen del simulador. Compose corre con un
  entorno mínimo: nunca se leen el ``.env`` del usuario ni las variables de su terminal.
* ``up`` siempre con ``--no-build``. La limpieza solo toca recursos con la etiqueta del
  proyecto y verifica su prefijo antes de ``down -v``; nunca ``prune``.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PUERTOS_POR_DEFECTO = {"API_PUERTO_HOST": 57000, "PRODUCTO_DB_PUERTO_HOST": 57432,
                       "MLFLOW_PUERTO_HOST": 57500, "AIRFLOW_PUERTO_HOST": 57080}
_ENTORNO_PERMITIDO = ("PATH", "HOME", "LANG", "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT",
                      "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY")


class EntornoInvalido(Exception):
    pass


def entorno_minimo() -> dict[str, str]:
    return {k: os.environ[k] for k in _ENTORNO_PERMITIDO if k in os.environ}


def ejecutar_proceso(cmd: list[str], timeout: float = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=entorno_minimo(), cwd=RAIZ, timeout=timeout)


def puertos_libres(puertos: dict[str, int]) -> list[int]:
    ocupados = []
    for p in puertos.values():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                ocupados.append(p)
    return ocupados


def escribir_entorno(ruta: Path, raw_dir: Path, puertos: dict[str, int], imagenes: dict[str, str]) -> dict:
    """Genera el archivo de entorno a partir de ``.env.example``. Devuelve lo que el
    simulador necesita leer (usuario y base de PostgreSQL, credenciales de Airflow)."""
    clave = lambda: secrets.token_urlsafe(18)                                  # noqa: E731
    valores = {
        "POSTGRES_DB": "novamercado", "POSTGRES_USER": "novamercado", "POSTGRES_PASSWORD": clave(),
        "AIRFLOW_DB_PASSWORD": clave(), "NM_CIERRE_PASSWORD": clave(), "NM_VIGILANTE_PASSWORD": clave(),
        "AIRFLOW_API_USUARIO": "simulador-api", "AIRFLOW_API_CLAVE": clave(), "AIRFLOW_ADMIN_CLAVE": clave(),
        "AIRFLOW_FERNET_KEY": base64.urlsafe_b64encode(os.urandom(32)).decode(),
        "AIRFLOW_SECRET_KEY": secrets.token_hex(24), "RAW_DIR": str(raw_dir),
        "API_DIRECCION_PUBLICACION": "127.0.0.1", "AIRFLOW_DIRECCION_PUBLICACION": "127.0.0.1",
        **{k: str(v) for k, v in puertos.items()},
        "NM_IMAGEN_APP": imagenes["app"], "NM_IMAGEN_AIRFLOW": imagenes["airflow"],
    }
    valores["DATABASE_URL"] = (f"postgresql+psycopg2://{valores['POSTGRES_USER']}:{valores['POSTGRES_PASSWORD']}"
                               f"@localhost:{puertos['PRODUCTO_DB_PUERTO_HOST']}/{valores['POSTGRES_DB']}")
    valores["MLFLOW_TRACKING_URI"] = f"http://localhost:{puertos['MLFLOW_PUERTO_HOST']}"
    pendientes = dict(valores)
    lineas = []
    for ln in (RAIZ / ".env.example").read_text().splitlines():
        k = ln.split("=", 1)[0]
        lineas.append(f"{k}={pendientes.pop(k)}" if "=" in ln and not ln.startswith("#") and k in pendientes else ln)
    lineas += [f"{k}={v}" for k, v in pendientes.items()]
    fd = os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lineas) + "\n")
    return {"db_usuario": valores["POSTGRES_USER"], "db_nombre": valores["POSTGRES_DB"],
            "airflow_usuario": valores["AIRFLOW_API_USUARIO"]}


class Entorno:
    def __init__(self, proyecto: str, env_file: Path, puertos: dict[str, int], acceso: dict, ejecutar=None):
        if not proyecto.startswith("nm-sim-"):
            raise EntornoInvalido(f"proyecto {proyecto!r}: el simulador solo usa proyectos nm-sim-*")
        self.proyecto, self.env_file, self.puertos, self.acceso = proyecto, Path(env_file), puertos, acceso
        self.ejecutar = ejecutar or ejecutar_proceso

    def comando(self, *args: str) -> list[str]:
        return ["docker", "compose", "-p", self.proyecto, "-f", str(RAIZ / "docker-compose.yml"),
                "--project-directory", str(RAIZ), "--env-file", str(self.env_file), *args]

    def compose(self, *args: str) -> subprocess.CompletedProcess:
        if args and args[0] == "up" and "--no-build" not in args:
            raise EntornoInvalido("up sin --no-build: el simulador nunca construye de forma implícita")
        return self.ejecutar(self.comando(*args))

    # -- ciclo de vida
    def levantar(self):
        return self.compose("up", "-d", "--no-build")

    def detener(self):
        return self.compose("stop")

    def iniciar(self):
        return self.compose("start")

    def bajar(self):
        return self.compose("down")                       # sin -v: se conservan los volúmenes

    def detener_servicio(self, servicio: str):
        return self.compose("stop", servicio)

    def iniciar_servicio(self, servicio: str):
        return self.compose("start", servicio)

    # -- lectura interna (S8, solo lectura)
    def psql_json(self, sql: str):
        r = self.compose("exec", "-T", "db", "psql", "-X", "-q", "-U", self.acceso["db_usuario"],
                         "-d", self.acceso["db_nombre"], "-tAc", f"SELECT coalesce(json_agg(x), '[]') FROM ({sql}) x")
        if r.returncode != 0:
            raise EntornoInvalido(f"psql falló: {r.stderr[-400:]}")
        return json.loads(r.stdout.strip() or "[]")

    def logs(self, servicio: str) -> str:
        return self.compose("logs", "--no-color", servicio).stdout

    def guardar_logs(self, destino: Path) -> dict:
        """Diagnóstico sin tocar el entorno: logs de cada servicio (con marcas de tiempo), el
        estado de los contenedores y la lista de recursos del proyecto, en ``destino`` (0700)."""
        destino = Path(destino)
        destino.mkdir(mode=0o700, parents=True, exist_ok=True)
        servicios = self.compose("config", "--services").stdout.split()
        guardados = {}
        for s in servicios:
            r = self.compose("logs", "--no-color", "--timestamps", s)
            (destino / f"{s}.log").write_text(r.stdout + r.stderr)
            guardados[s] = len(r.stdout)
        (destino / "ps.txt").write_text(self.compose("ps", "-a").stdout)
        (destino / "recursos.json").write_text(json.dumps(self.recursos(), indent=1))
        return guardados

    # -- recursos y limpieza
    def recursos(self) -> dict[str, list[str]]:
        filtro = f"label=com.docker.compose.project={self.proyecto}"
        out = {}
        for tipo, cmd in (("contenedores", ["docker", "ps", "-a", "--filter", filtro, "--format", "{{.Names}}"]),
                          ("volumenes", ["docker", "volume", "ls", "--filter", filtro, "--format", "{{.Name}}"]),
                          ("redes", ["docker", "network", "ls", "--filter", filtro, "--format", "{{.Name}}"])):
            r = self.ejecutar(cmd)
            out[tipo] = sorted(x for x in r.stdout.split() if x)
        return out

    def limpiar(self) -> dict[str, list[str]]:
        """Borra contenedores, volúmenes y redes del proyecto (y nada más). Devuelve lo borrado."""
        recursos = self.recursos()
        ajenos = [n for ns in recursos.values() for n in ns
                  if not (n.startswith(self.proyecto + "_") or n.startswith(self.proyecto + "-"))]
        if ajenos:
            raise EntornoInvalido(f"recursos con la etiqueta del proyecto pero otro prefijo: {ajenos}")
        r = self.compose("down", "-v", "--remove-orphans")
        if r.returncode != 0:
            raise EntornoInvalido(f"down -v falló: {r.stderr[-400:]}")
        quedan = self.recursos()
        if any(quedan.values()):
            raise EntornoInvalido(f"quedaron recursos del proyecto: {quedan}")
        return recursos
