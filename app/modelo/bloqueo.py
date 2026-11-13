"""Exclusión de entrenamientos: advisory lock + concesión con latido (DECISIONES.md §15).

Es el **único** código del proyecto que toca el lock y la concesión. Lo usa solo el
supervisor (``app.modelo.ciclo_entrenamiento``); el proceso hijo únicamente lee la
fila de la concesión desde su vigilante (:func:`vigilar_concesion`).

* **Advisory lock de sesión** (una clave ``bigint``; las observaciones usan la forma
  de dos claves ``int4``, así que nunca chocan): evita que dos procesos empiecen a
  la vez. Se obtiene con ``pg_try_advisory_lock`` cada 2 s hasta el plazo del modo.
* **Concesión** (fila única de ``bloqueo_entrenamiento``): para tomarla debe estar
  liberada o vencida (``latido`` anterior a ``now() - 60 s``, reloj del servidor);
  al tomarla ``epoca = epoca + 1``. Un reinicio de PostgreSQL libera el advisory
  lock, pero la fila con el último latido persiste: un nuevo titular debe esperar a
  que venza.
* **Conexión dedicada** psycopg2 sin pool ni reconexión
  (:func:`app.db.conexion.abrir_conexion_dedicada`). Al adquirir se registran
  ``pg_backend_pid()`` y ``pg_postmaster_start_time()``; cada sondeo los compara,
  así que una sesión nueva o un servidor reiniciado se detectan como pérdida.
* **Transacciones condicionadas**: bloquean la fila de la concesión con
  ``FOR UPDATE`` y exigen época propia, latido vigente, identidad de la sesión y
  advisory lock. Las inserciones de resoluciones asignan ``secuencia`` con el
  contador de esa misma fila (§5.3): una transacción no puede obtener su secuencia
  mientras otra que ya incrementó el contador no termine en COMMIT o ROLLBACK.

Garantías y límites (modelo de fallos): DECISIONES.md §15. Las cifras de 22/42/60 s
son cotas bajo el modelo cubierto, no garantías absolutas.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import psycopg2

from app.db.conexion import BaseDatosNoDisponible, abrir_conexion_dedicada
from app.db.promociones import SQL_DECISIONES_SIN_RESOLVER, Vigente, leer_vigente_cursor

logger = logging.getLogger(__name__)

CLAVE_BLOQUEO = 1313669126          # "NM" + 6; < 2**31: classid = 0, objid = clave en pg_locks
VENCIMIENTO_S = 60                  # una concesión sin latido durante 60 s se puede tomar
SONDEO_S = 5                        # intervalo de sondeo del supervisor y del vigilante del hijo
LATIDO_MAX_HIJO_S = 30              # el hijo se termina si el latido tiene más de 30 s
REINTENTO_S = 2                     # espera entre intentos de adquisición

_EN_POSESION = threading.Lock()     # sin reentrada dentro del mismo proceso


class BloqueoOcupado(RuntimeError):
    """Otro entrenamiento tiene el lock o una concesión vigente (código de salida 5)."""


class BloqueoYaTomado(RuntimeError):
    """Este proceso ya tiene la concesión: un segundo intento sería una autoexclusión."""


class ConcesionPerdida(RuntimeError):
    """Se perdió la conexión, la sesión, el advisory lock o la concesión (código 6)."""


SQL_SONDEO = """
UPDATE bloqueo_entrenamiento SET latido = now()
WHERE id = 1 AND epoca = %(epoca)s AND latido IS NOT NULL
  AND latido > now() - make_interval(secs => %(vence)s)
  AND pg_backend_pid() = %(pid)s AND pg_postmaster_start_time() = %(inicio)s
  AND EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid() AND granted
              AND objsubid = 1 AND classid = 0 AND objid = %(clave)s::oid)
RETURNING 1
"""


class ConcesionEntrenamiento:
    """Lock y concesión de entrenamiento. Uso: ``with ConcesionEntrenamiento(...) as c:``."""

    def __init__(self, url: str | None = None, *, modo: str = "manual", espera: float = 0.0,
                 vencimiento: float = VENCIMIENTO_S, reintento: float = REINTENTO_S):
        self.url = url
        self.modo = modo
        self.espera = espera
        self.vencimiento = vencimiento
        self.reintento = reintento
        self.titular = f"{socket.gethostname()}:{os.getpid()}:{modo}"
        self.conn = None
        self.epoca: int | None = None
        self._pid: int | None = None
        self._inicio = None
        self.perdida = False
        self._tomado = False

    # ------------------------------------------------------------ ciclo de vida
    def __enter__(self) -> ConcesionEntrenamiento:
        self.adquirir()
        return self

    def __exit__(self, *exc) -> bool:
        self.liberar()
        return False

    def adquirir(self) -> None:
        if not _EN_POSESION.acquire(blocking=False):
            raise BloqueoYaTomado("este proceso ya tiene la concesión de entrenamiento")
        self._tomado = True
        try:
            self._adquirir()
        except BaseException:
            self._cerrar()
            raise

    def _adquirir(self) -> None:
        limite = time.monotonic() + self.espera
        self.conn = abrir_conexion_dedicada(self.url)
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid(), pg_postmaster_start_time()")
            self._pid, self._inicio = cur.fetchone()
            while True:
                cur.execute("SELECT pg_try_advisory_lock(%s)", (CLAVE_BLOQUEO,))
                if cur.fetchone()[0]:
                    break
                if time.monotonic() >= limite:
                    raise BloqueoOcupado("otro entrenamiento tiene el lock de entrenamiento")
                time.sleep(self.reintento)
            while True:
                cur.execute("BEGIN")
                try:
                    cur.execute("SELECT epoca, latido IS NULL OR latido < now() - make_interval(secs => %s) "
                                "FROM bloqueo_entrenamiento WHERE id = 1 FOR UPDATE", (self.vencimiento,))
                    fila = cur.fetchone()
                    if fila is None:
                        raise RuntimeError("falta la fila de bloqueo_entrenamiento (¿migraciones aplicadas?)")
                    if fila[1]:
                        cur.execute("UPDATE bloqueo_entrenamiento SET epoca = epoca + 1, titular = %s, "
                                    "adquirido_en = now(), latido = now() WHERE id = 1 RETURNING epoca",
                                    (self.titular,))
                        self.epoca = int(cur.fetchone()[0])
                        cur.execute("COMMIT")
                        break
                    cur.execute("ROLLBACK")
                except BaseException:
                    self._rollback(cur)
                    raise
                if time.monotonic() >= limite:
                    raise BloqueoOcupado("la concesión anterior todavía no venció (otro titular pudo "
                                         "perder la conexión hace menos de "
                                         f"{self.vencimiento:g} s)")
                time.sleep(self.reintento)
        logger.info("concesión de entrenamiento tomada: época %s (%s)", self.epoca, self.titular)

    def liberar(self) -> None:
        """Marca la concesión como liberada, libera el lock y cierra la conexión (mejor esfuerzo)."""
        try:
            if self.conn is not None and not self.perdida and self.epoca is not None:
                with self.conn.cursor() as cur:
                    cur.execute("UPDATE bloqueo_entrenamiento SET latido = NULL, titular = NULL "
                                "WHERE id = 1 AND epoca = %s", (self.epoca,))
                    cur.execute("SELECT pg_advisory_unlock(%s)", (CLAVE_BLOQUEO,))
        except Exception as exc:  # noqa: BLE001 - al cerrar la sesión el servidor libera el lock
            logger.warning("no se pudo liberar la concesión limpiamente: %s", type(exc).__name__)
        finally:
            self._cerrar()

    def _cerrar(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        self.conn = None
        if self._tomado:
            self._tomado = False
            _EN_POSESION.release()

    @staticmethod
    def _rollback(cur) -> None:
        try:
            cur.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ sondeo
    def _params(self) -> dict:
        return {"epoca": self.epoca, "vence": self.vencimiento, "pid": self._pid, "inicio": self._inicio,
                "clave": CLAVE_BLOQUEO}

    def sondear(self) -> bool:
        """Renueva el latido si todo sigue en orden. ``False`` = concesión perdida."""
        if self.perdida or self.conn is None:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute(SQL_SONDEO, self._params())
                ok = cur.rowcount == 1
        except (psycopg2.Error, BaseDatosNoDisponible, OSError) as exc:
            logger.error("sondeo de la concesión falló: %s", type(exc).__name__)
            ok = False
        if not ok:
            self.perdida = True
            logger.error("concesión de entrenamiento perdida (época %s)", self.epoca)
        return ok

    def verificar(self) -> None:
        """Sondea y lanza :class:`ConcesionPerdida` si se perdió. Se llama antes de cada
        escritura en MLflow."""
        if not self.sondear():
            raise ConcesionPerdida(f"se perdió la concesión de entrenamiento (época {self.epoca})")

    # ----------------------------------------------------- transacción condicionada
    def condicionada(self, fn: Callable[[Any, int], Any]) -> Any:
        """Ejecuta ``fn(cursor, ultima_secuencia)`` en una transacción que bloquea la fila
        de la concesión y exige época, latido, sesión y lock propios. La transacción se
        confirma solo si ``fn`` termina sin error."""
        if self.perdida or self.conn is None:
            raise ConcesionPerdida(f"la concesión de la época {self.epoca} ya se perdió")
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN")
            cur.execute("SELECT epoca, latido IS NOT NULL AND latido > now() - make_interval(secs => %s), "
                        "ultima_secuencia, pg_backend_pid(), pg_postmaster_start_time() "
                        "FROM bloqueo_entrenamiento WHERE id = 1 FOR UPDATE", (self.vencimiento,))
            epoca, vigente, ultima, pid, inicio = cur.fetchone()
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                        "AND pid = pg_backend_pid() AND granted AND objsubid = 1 AND classid = 0 "
                        "AND objid = %s::oid)", (CLAVE_BLOQUEO,))
            lock = cur.fetchone()[0]
            if not (epoca == self.epoca and vigente and pid == self._pid and inicio == self._inicio and lock):
                raise ConcesionPerdida(f"la concesión de la época {self.epoca} ya no es vigente "
                                       f"(época actual {epoca})")
            cur.execute("UPDATE bloqueo_entrenamiento SET latido = now() WHERE id = 1")
            resultado = fn(cur, int(ultima))
            cur.execute("COMMIT")
            return resultado
        except ConcesionPerdida:
            self._rollback(cur)
            self.perdida = True
            raise
        except (psycopg2.OperationalError, psycopg2.InterfaceError, BaseDatosNoDisponible) as exc:
            self._rollback(cur)
            self.perdida = True
            raise ConcesionPerdida(f"se perdió la conexión de la concesión: {type(exc).__name__}") from exc
        except BaseException:
            self._rollback(cur)
            raise
        finally:
            try:
                cur.close()
            except Exception:  # noqa: BLE001
                pass

    def insertar_decision(self, *, nombre: str, version: str, run_id: str, ruta: str, sha256: str,
                          origen: str) -> int:
        def fn(cur, _ultima):
            cur.execute("INSERT INTO decisiones_promocion (nombre_modelo, version, run_id, ruta_artefacto, "
                        "sha256_artefacto, epoca, origen) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "RETURNING decision_id", (nombre, str(version), run_id, ruta, sha256, self.epoca, origen))
            return int(cur.fetchone()[0])
        return self.condicionada(fn)

    def insertar_resolucion(self, decision_id: int, resultado: str, motivo: str) -> int:
        """Resolución con ``secuencia = ultima_secuencia + 1`` asignada dentro de la
        transacción que tiene la fila de la concesión bloqueada (§5.3)."""
        def fn(cur, ultima):
            s = ultima + 1
            cur.execute("UPDATE bloqueo_entrenamiento SET ultima_secuencia = %s WHERE id = 1", (s,))
            cur.execute("INSERT INTO resoluciones_promocion (decision_id, resultado, epoca, motivo, secuencia) "
                        "VALUES (%s, %s, %s, %s, %s)", (decision_id, resultado, self.epoca, motivo[:2000], s))
            return s
        return self.condicionada(fn)

    # ------------------------------------------------------------------ lectura
    def leer_vigente(self, nombre: str) -> Vigente | None:
        if self.perdida or self.conn is None:
            raise ConcesionPerdida(f"la concesión de la época {self.epoca} ya se perdió")
        try:
            with self.conn.cursor() as cur:
                return leer_vigente_cursor(cur, nombre)
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            self.perdida = True
            raise ConcesionPerdida("se perdió la conexión de la concesión") from exc

    def decisiones_sin_resolver(self, nombre: str) -> list[tuple[int, str, int]]:
        """Decisiones de épocas anteriores sin resolución (para la reconciliación)."""
        with self.conn.cursor() as cur:
            cur.execute(SQL_DECISIONES_SIN_RESOLVER, {"nombre": nombre, "epoca": self.epoca})
            return [(int(a), str(b), int(c)) for a, b, c in cur.fetchall()]

    def resolucion_de(self, decision_id: int) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT resultado FROM resoluciones_promocion WHERE decision_id = %s", (decision_id,))
            fila = cur.fetchone()
        return fila[0] if fila else None


# ------------------------------------------------------- vigilante del proceso hijo
def estado_concesion(conn) -> tuple[int, float | None]:
    """(época, segundos desde el último latido o ``None`` si está liberada)."""
    with conn.cursor() as cur:
        cur.execute("SELECT epoca, EXTRACT(EPOCH FROM now() - latido) FROM bloqueo_entrenamiento WHERE id = 1")
        epoca, edad = cur.fetchone()
    return int(epoca), (None if edad is None else float(edad))


def vigilar_concesion(url: str | None, epoca: int, al_perder: Callable[[str], None], *,
                      intervalo: float = SONDEO_S, latido_max: float = LATIDO_MAX_HIJO_S,
                      detener: threading.Event | None = None) -> None:
    """Bucle del vigilante del hijo (§4.2): con su propia conexión directa lee la fila
    cada ``intervalo`` s y llama a ``al_perder`` si la época cambió, si el latido tiene
    más de ``latido_max`` s o si la base no responde. ``al_perder`` termina el proceso
    (``os._exit(6)``) aunque el hilo principal esté dentro de código C."""
    detener = detener or threading.Event()
    conn = None
    while not detener.is_set():
        try:
            if conn is None:
                conn = abrir_conexion_dedicada(url)
            actual, edad = estado_concesion(conn)
            if actual != epoca:
                al_perder(f"la época cambió ({epoca} → {actual})")
                return
            if edad is None or edad > latido_max:
                al_perder("la concesión fue liberada" if edad is None else f"el último latido tiene {edad:.0f} s")
                return
        except Exception as exc:  # noqa: BLE001 - sin respuesta = pérdida
            al_perder(f"la base no responde ({type(exc).__name__})")
            return
        detener.wait(intervalo)
