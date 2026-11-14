"""Cierre de reentrenamientos y decisión del vigilante (etapa 7, plan v7–v12).

Este archivo usa SOLO la biblioteca estándar y psycopg2: no importa nada del proyecto.
El DAG usa una copia idéntica byte a byte (``dags/nm_cierre.py``; una prueba exige que
sean iguales y ambas están cubiertas por la procedencia), así el entorno de Airflow
nunca importa código del proyecto.

* :func:`token` — identificador corto y determinista de los escritores
  (``application_name = nm-cierre:<32 hex>``, 42 bytes < 63).
* :func:`cerrar` — escribe el resultado de un reentrenamiento con UNA sentencia
  condicional después de ``SELECT … FOR UPDATE`` de la solicitud: nunca hay dos
  resultados (``UNIQUE``) y nunca un ``exitoso`` después de un veredicto de ausencia.
  Presupuesto global (por defecto 12 s): fase A de escritura (8 s) con un único
  vigilante de cliente que cubre conexión, bloqueo, sentencias, COMMIT y red; si la
  fase A no termina confirmada (incluido un COMMIT sin respuesta), fase B de
  verificación por id hasta los 12 s, sin reintentar la escritura.
* :func:`vigilar` — un intento del vigilante del DAG (rol ``nm_vigilante``): decide por
  el resultado persistido o, bajo el bloqueo de la solicitud, inserta el veredicto de
  ausencia; si no puede, ``indeterminado`` (nunca concluye ausencia sin el bloqueo).
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
import time
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import psycopg2
import psycopg2.errors

ROL_CIERRE = "nm_cierre"

# Cotas del vigilante (plan v11–v12, §2–§3); las lee la prueba estática del presupuesto
VIG_CONEXION_S = 2.0
VIG_LECTURA_S = 1.0
VIG_ESPERA_BLOQUEO_S = 5.0
VIG_FASE_BLOQUEO_S = 15.0
VIG_SECCION_CRITICA_OP_S = 1.0
VIG_FUNCION_S = 2.0
VIG_RECONEXION_S = 2.0
VIG_VERIFICACION_S = 4.0
VIG_RESPALDO_S = 38.0
VIG_POLL_S = 0.5

# Presupuesto de cierre (plan v6–v7)
CIERRE_TOTAL_S = 12.0
CIERRE_ESCRITURA_S = 8.0


# --------------------------------------------------------------------- identidad
def token(id_: str) -> str:
    return hashlib.sha256(id_.encode("utf-8")).hexdigest()[:32]


def app_cierre(id_: str) -> str:
    return "nm-cierre:" + token(id_)


def app_verif(id_: str) -> str:
    return "nm-verif:" + token(id_)


def parametros_desde_url(url: str) -> dict:
    """``postgresql[+driver]://usuario:clave@host:puerto/base`` → parámetros de psycopg2."""
    u = urlsplit(url)
    return {"host": u.hostname, "port": u.port or 5432, "user": unquote(u.username or ""),
            "password": unquote(u.password or ""), "dbname": unquote(u.path.lstrip("/"))}


# ------------------------------------------------------------- conexión y corte
class Corte:
    """Vigilante de cliente GLOBAL: a ``limite`` segundos cierra el socket de la
    conexión vigilada, para que cualquier operación en curso (conexión, espera de
    bloqueo, sentencia, COMMIT o red) termine con error."""

    def __init__(self, limite: float):
        self.fin = time.monotonic() + limite
        self.conn = None
        self.disparado = False
        self._candado = threading.Lock()
        self._t = threading.Timer(max(limite, 0.0), self._cortar)
        self._t.daemon = True
        self._t.start()

    def restante(self) -> float:
        return max(self.fin - time.monotonic(), 0.0)

    def vigilar(self, conn) -> None:
        with self._candado:
            self.conn = conn
            if self.disparado:
                _cortar_socket(conn)

    def _cortar(self) -> None:
        with self._candado:
            self.disparado = True
            if self.conn is not None:
                _cortar_socket(self.conn)

    def cancelar(self) -> None:
        self._t.cancel()


def _cortar_socket(conn) -> None:
    try:
        with socket.socket(fileno=os.dup(conn.fileno())) as s:
            s.shutdown(socket.SHUT_RDWR)
    except (OSError, psycopg2.Error):
        pass


def conectar(params: dict, application_name: str, connect_timeout: float):
    conn = psycopg2.connect(**params, application_name=application_name,
                            connect_timeout=max(int(connect_timeout + 0.999), 1),
                            keepalives=1, keepalives_idle=10, keepalives_interval=3, keepalives_count=3)
    conn.autocommit = True
    return conn


def _ms(segundos: float) -> int:
    return max(int(segundos * 1000), 1)


def _cerrar_conexion(conn) -> None:
    try:
        if conn is not None:
            conn.close()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- SQL
SQL_BLOQUEAR = "SELECT id FROM solicitudes_reentrenamiento WHERE id = %(id)s FOR UPDATE"

SQL_VIGENTE_VERSION = """
SELECT d.version FROM resoluciones_promocion r
JOIN decisiones_promocion d ON d.decision_id = r.decision_id
WHERE r.resultado = 'confirmada' AND d.nombre_modelo = %(nombre)s
ORDER BY r.secuencia DESC LIMIT 1
"""

SQL_PROMOVIDO_EN_SU_EPOCA = """
EXISTS (SELECT 1 FROM reclamaciones_reentrenamiento rc
        JOIN decisiones_promocion d ON d.epoca = rc.epoca AND d.nombre_modelo = %(nombre)s
        JOIN resoluciones_promocion r ON r.decision_id = d.decision_id AND r.resultado = 'confirmada'
        WHERE rc.id = %(id)s)
"""

# Una sola sentencia condicional: no hay resultado previo y, si es "exitoso", no hay
# veredicto de ausencia (plan v10 §3). La versión es el vigente AL ESCRIBIR.
SQL_INSERTAR = f"""
INSERT INTO resultados_reentrenamiento (id, estado, promovido, version, motivo, escrito_por)
SELECT %(id)s, %(estado)s,
       COALESCE(%(promovido)s, {SQL_PROMOVIDO_EN_SU_EPOCA}),
       ({SQL_VIGENTE_VERSION}), %(motivo)s, %(escrito_por)s
WHERE NOT EXISTS (SELECT 1 FROM resultados_reentrenamiento WHERE id = %(id)s)
  AND (%(estado)s = 'fallido' OR NOT EXISTS (SELECT 1 FROM veredictos_reentrenamiento WHERE id = %(id)s))
"""

SQL_ESTADO = """
SELECT (SELECT estado FROM resultados_reentrenamiento WHERE id = %(id)s),
       EXISTS (SELECT 1 FROM veredictos_reentrenamiento WHERE id = %(id)s),
       EXISTS (SELECT 1 FROM pg_catalog.pg_stat_activity
               WHERE datname = pg_catalog.current_database() AND usename = %(rol)s
                 AND application_name = %(app)s AND pid <> pg_catalog.pg_backend_pid())
"""


@dataclass
class ResultadoCierre:
    codigo: str            # escrito | ya_registrado | rechazado_por_veredicto | no_escrito | indeterminado
    estado_persistido: str | None = None
    detalle: str = ""

    @property
    def persistido(self) -> bool:
        return self.estado_persistido is not None


def _estado(conn, id_: str) -> tuple[str | None, bool, bool]:
    with conn.cursor() as cur:
        cur.execute(SQL_ESTADO, {"id": id_, "rol": ROL_CIERRE, "app": app_cierre(id_)})
        r, v, vivos = cur.fetchone()
    return r, bool(v), bool(vivos)


def cerrar(params: dict, id_: str, *, estado: str, motivo: str, escrito_por: str, nombre_modelo: str,
           promovido: bool | None = None, total: float = CIERRE_TOTAL_S,
           escritura: float = CIERRE_ESCRITURA_S, poll: float = VIG_POLL_S) -> ResultadoCierre:
    """Escribe el resultado (rol nm_cierre). Nunca reintenta la escritura: tras un error
    solo verifica por id. ``promovido=None`` lo calcula de las resoluciones confirmadas
    de la época de su reclamación."""
    if estado not in ("exitoso", "fallido"):
        raise ValueError(estado)
    inicio = time.monotonic()
    fin_total = inicio + total
    corte = Corte(escritura)
    conn = None
    commit_enviado = False
    error = ""
    try:
        conn = conectar(params, app_cierre(id_), min(3.0, corte.restante()))
        corte.vigilar(conn)
        with conn.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute("SELECT set_config('lock_timeout', %s, true), set_config('statement_timeout', %s, true)",
                        (str(_ms(corte.restante())), str(_ms(corte.restante() + 1))))
            cur.execute(SQL_BLOQUEAR, {"id": id_})
            if cur.fetchone() is None:
                cur.execute("ROLLBACK")
                return ResultadoCierre("no_escrito", None, "la solicitud no existe")
            cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(_ms(corte.restante())),))
            cur.execute(SQL_INSERTAR, {"id": id_, "estado": estado, "promovido": promovido,
                                       "motivo": motivo[:2000], "escrito_por": escrito_por,
                                       "nombre": nombre_modelo})
            n = cur.rowcount
            commit_enviado = True
            cur.execute("COMMIT")
        corte.cancelar()
        if n == 1:
            return ResultadoCierre("escrito", estado)
        r, v, _ = _estado(conn, id_)
        if r is not None:
            return ResultadoCierre("ya_registrado", r)
        return ResultadoCierre("rechazado_por_veredicto" if v else "no_escrito", None,
                               "veredicto de ausencia" if v else "")
    except psycopg2.Error as exc:
        error = (("COMMIT sin respuesta; " if commit_enviado else "")
                 + f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}")
    finally:
        corte.cancelar()
        _cerrar_conexion(conn)

    # Fase B: verificación por id, hasta el total (nunca reintenta la escritura)
    verif = None
    try:
        while time.monotonic() < fin_total:
            try:
                if verif is None:
                    verif = conectar(params, app_verif(id_), min(1.5, max(fin_total - time.monotonic(), 0.1)))
                with verif.cursor() as cur:
                    cur.execute("SELECT set_config('statement_timeout', %s, false)",
                                (str(_ms(max(fin_total - time.monotonic(), 0.05))),))
                r, v, vivos = _estado(verif, id_)
                if r is not None:
                    return ResultadoCierre("ya_registrado", r, error)
                if not vivos:
                    if v and estado == "exitoso":
                        return ResultadoCierre("rechazado_por_veredicto", None, error)
                    return ResultadoCierre("no_escrito", None, error)
            except psycopg2.Error:
                _cerrar_conexion(verif)
                verif = None
            time.sleep(min(poll, max(fin_total - time.monotonic(), 0)))
        return ResultadoCierre("indeterminado", None, error)
    finally:
        _cerrar_conexion(verif)


# ---------------------------------------------------------------------- vigilante
@dataclass
class DecisionVigilante:
    tipo: str                      # persistido | ausencia | indeterminado
    estado: str | None = None      # estado persistido (si tipo == persistido)
    detalle: str = ""
    fases: dict | None = None      # duración de cada fase (para las pruebas del presupuesto)


def _intento_bloqueo(conn, id_: str) -> DecisionVigilante | None:
    """BEGIN; lock_timeout 5 s (statement_timeout 6 s solo para el FOR UPDATE); con el
    bloqueo, statement_timeout 1 s para cada operación de la sección crítica (SELECT,
    INSERT del veredicto, COMMIT). ``None`` si no se obtuvo el bloqueo."""
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        cur.execute("SELECT set_config('lock_timeout', %s, true), set_config('statement_timeout', %s, true)",
                    (str(_ms(VIG_ESPERA_BLOQUEO_S)), str(_ms(VIG_ESPERA_BLOQUEO_S + 1))))
        try:
            cur.execute(SQL_BLOQUEAR, {"id": id_})
        except psycopg2.errors.LockNotAvailable:
            cur.execute("ROLLBACK")
            return None
        if cur.fetchone() is None:
            cur.execute("ROLLBACK")
            return DecisionVigilante("indeterminado", None, "la solicitud no existe")
        cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(_ms(VIG_SECCION_CRITICA_OP_S)),))
        cur.execute("SELECT (SELECT estado FROM resultados_reentrenamiento WHERE id = %(id)s), "
                    "EXISTS (SELECT 1 FROM veredictos_reentrenamiento WHERE id = %(id)s)", {"id": id_})
        r, v = cur.fetchone()
        if r is not None:
            cur.execute("COMMIT")
            return DecisionVigilante("persistido", r)
        if not v:
            cur.execute("INSERT INTO veredictos_reentrenamiento (id, veredicto) VALUES (%(id)s, 'sin_resultado')",
                        {"id": id_})
        cur.execute("COMMIT")
        return DecisionVigilante("ausencia", None, "veredicto registrado bajo el bloqueo")


def vigilar(params: dict, id_: str) -> DecisionVigilante:
    """Un intento del vigilante (rol nm_vigilante). Peor caso 34 s con PostgreSQL
    respondiendo; respaldo de cliente a los 38 s; nunca concluye ausencia sin el
    bloqueo de la solicitud."""
    t0 = time.monotonic()
    fases: dict[str, float] = {}
    marca = [t0]

    def fase(nombre):
        ahora = time.monotonic()
        fases[nombre] = round(ahora - marca[0], 3)
        marca[0] = ahora

    corte = Corte(VIG_RESPALDO_S)
    conn = None
    try:
        try:
            conn = conectar(params, app_verif(id_), VIG_CONEXION_S)
            corte.vigilar(conn)
            fase("conexion")
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', %s, false)", (str(_ms(VIG_LECTURA_S)),))
                cur.execute("SELECT (SELECT estado FROM resultados_reentrenamiento WHERE id = %(id)s), "
                            "EXISTS (SELECT 1 FROM veredictos_reentrenamiento WHERE id = %(id)s)", {"id": id_})
                r, v = cur.fetchone()
            fase("lectura")
            if r is not None:
                return DecisionVigilante("persistido", r, "", fases)
            if v:
                return DecisionVigilante("ausencia", None, "veredicto ya registrado", fases)
            fin_bloqueo = time.monotonic() + VIG_FASE_BLOQUEO_S
            while time.monotonic() + VIG_ESPERA_BLOQUEO_S <= fin_bloqueo + 0.01:
                d = _intento_bloqueo(conn, id_)
                if d is not None:
                    fase("bloqueo")
                    d.fases = fases
                    return d
            fase("bloqueo")
        except psycopg2.Error as exc:
            return DecisionVigilante("indeterminado", None, f"{type(exc).__name__}", fases)

        # Terminación controlada del escritor que retiene el bloqueo (cota de cliente 2 s)
        llamada = ""
        sub = Corte(VIG_FUNCION_S)
        try:
            sub.vigilar(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', %s, false)", (str(_ms(VIG_FUNCION_S)),))
                cur.execute("SELECT nm_control.nm_terminar_escritor(%s)", (id_,))
                llamada = f"terminados={cur.fetchone()[0]}"
        except psycopg2.Error as exc:
            llamada = f"resultado indeterminado de la terminación ({type(exc).__name__})"
        finally:
            sub.cancelar()
        fase("funcion")

        # Verificación obligatoria con el token, cualquiera que haya sido el resultado
        try:
            if conn.closed or llamada.startswith("resultado indeterminado"):
                _cerrar_conexion(conn)
                conn = conectar(params, app_verif(id_), VIG_RECONEXION_S)
                corte.vigilar(conn)
                fase("reconexion")
            fin_verif = time.monotonic() + VIG_VERIFICACION_S
            vivo = True
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', %s, false)", (str(_ms(VIG_LECTURA_S)),))
            while True:
                _, _, vivo = _estado(conn, id_)
                if not vivo or time.monotonic() + VIG_POLL_S > fin_verif:
                    break
                time.sleep(VIG_POLL_S)
            fase("verificacion")
            if vivo:
                return DecisionVigilante("indeterminado", None, f"el escritor sigue vivo ({llamada})", fases)
            d = _intento_bloqueo(conn, id_)
            fase("ultimo_intento")
            if d is None:
                return DecisionVigilante("indeterminado", None, "no se obtuvo el bloqueo tras la terminación",
                                         fases)
            d.fases = fases
            d.detalle = (d.detalle + f"; {llamada}").strip("; ")
            return d
        except psycopg2.Error as exc:
            return DecisionVigilante("indeterminado", None, f"verificación imposible ({type(exc).__name__})",
                                     fases)
    finally:
        corte.cancelar()
        _cerrar_conexion(conn)
