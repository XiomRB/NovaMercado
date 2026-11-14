"""Orquestador Airflow de ``/admin/reentrenar`` (etapa 7, DECISIONES.md §16, plan v3–v12 §4).

**Iniciar** (``POST /admin/reentrenar``):

1. Con ``Idempotency-Key`` (≤ 128 de ``[A-Za-z0-9_-]``): la misma clave con la misma
   huella devuelve el mismo id (y lo reenvía a Airflow si sigue pendiente; 202 aunque
   ya haya terminado); con otra huella → 422. Sin clave, cada POST es una solicitud nueva.
2. El id (``nm-<uuid4>``) se registra en ``solicitudes_reentrenamiento`` **antes** de
   llamar a Airflow; si no se puede registrar → 503 y no se llama a Airflow.
3. ``POST dagRuns`` con ``dag_run_id = id``. 200 solo cuenta si el cuerpo trae ese
   ``dag_run_id`` y ``dag_id``; 409 o una respuesta perdida se confirman con
   ``GET dagRuns/{id}`` (mismos dos campos). 404 o sin confirmación → 503 y la solicitud
   queda **pendiente** (la reenvía un reintento con la misma clave o la cierra el plazo
   de cola).

**Estado** (``GET /admin/reentrenar/{id}``): solo PostgreSQL. Antes de responder ejecuta la
recuperación de ese id (G2-lectura). Resultado → terminal; solicitud sin resultado →
``en_curso``; id desconocido → 404.

**Recuperación** (hilo cada 15 s y en cada GET), con el rol ``nm_cierre`` y la condición
evaluada bajo el bloqueo de la solicitud (``app.db.cierre.CONDICIONES``):

* sin inicio de tarea y 30 min desde ``solicitado_en`` → ``fallido`` (plazo de cola);
* con inicio y t0 + 10:00 → ``fallido`` "la ejecución se interrumpió" (G2);
* con veredicto de ausencia → ``fallido`` (cierre del veredicto del DAG).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

from sqlalchemy import text

from app.db.conexion import BaseDatosNoDisponible, transaccion
from app.orquestador import OrquestadorNoDisponible
from app.schemas.contrato import EstadoReentrenamiento

logger = logging.getLogger(__name__)

DAG_ID = "novamercado_reentrenamiento"
CLAVE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
INTERVALO_RECUPERACION_S = 15.0
TIMEOUT_HTTP_S = 5.0
CONFIRMACIONES = 3
ESPERA_CONFIRMACION_S = 1.5
VERSION_SIN_VIGENTE = ""
MOTIVOS = {
    "cola_vencida": "la ejecución no empezó dentro del plazo de cola (30 min)",
    "plazo_vencido": "la ejecución se interrumpió (sin resultado a los 10 min de iniciada)",
    "con_veredicto": "la ejecución terminó sin resultado (veredicto del vigilante)",
}


class ClaveInvalida(ValueError):
    """``Idempotency-Key`` inválida o reutilizada con otra solicitud (422)."""


@dataclass(frozen=True)
class ConfigAirflow:
    url: str
    usuario: str
    clave: str
    dag_id: str = DAG_ID
    timeout: float = TIMEOUT_HTTP_S


def huella(cuerpo: bytes) -> str:
    return hashlib.sha256(b"POST /admin/reentrenar\n" + cuerpo).hexdigest()


def validar_clave(clave: str | None) -> str | None:
    if clave is None:
        return None
    if not CLAVE_RE.fullmatch(clave):
        raise ClaveInvalida("Idempotency-Key debe tener entre 1 y 128 caracteres de [A-Za-z0-9_-]")
    return clave


class ClienteAirflow:
    """API REST de Airflow 2 con autenticación básica (urllib, sin dependencias)."""

    def __init__(self, cfg: ConfigAirflow):
        self.cfg = cfg
        token = base64.b64encode(f"{cfg.usuario}:{cfg.clave}".encode()).decode()
        self._cabeceras = {"Authorization": f"Basic {token}", "Content-Type": "application/json",
                           "Accept": "application/json"}

    def _pedir(self, metodo: str, ruta: str, cuerpo: dict | None = None) -> tuple[int, dict]:
        datos = None if cuerpo is None else json.dumps(cuerpo).encode()
        req = urllib.request.Request(self.cfg.url.rstrip("/") + ruta, data=datos, method=metodo,
                                     headers=self._cabeceras)
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                return r.status, _json(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _json(exc.read())

    def crear_ejecucion(self, id_: str) -> tuple[int, dict]:
        return self._pedir("POST", f"/api/v1/dags/{self.cfg.dag_id}/dagRuns", {"dag_run_id": id_, "conf": {}})

    def leer_ejecucion(self, id_: str) -> tuple[int, dict]:
        from urllib.parse import quote
        return self._pedir("GET", f"/api/v1/dags/{self.cfg.dag_id}/dagRuns/{quote(id_, safe='')}")

    def salud(self) -> bool:
        try:
            req = urllib.request.Request(self.cfg.url.rstrip("/") + "/health", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=min(self.cfg.timeout, 3.0)) as r:
                d = _json(r.read())
            return all(d.get(k, {}).get("status") == "healthy" for k in ("metadatabase", "scheduler"))
        except Exception:  # noqa: BLE001
            return False


def _json(crudo: bytes) -> dict:
    try:
        d = json.loads(crudo or b"{}")
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


class OrquestadorAirflow:
    def __init__(self, motor, cfg: ConfigAirflow, *, nombre_modelo: str, parametros_cierre,
                 cliente: ClienteAirflow | None = None, intervalo: float = INTERVALO_RECUPERACION_S,
                 espera_confirmacion: float = ESPERA_CONFIRMACION_S):
        self.motor = motor
        self.cfg = cfg
        self.cliente = cliente or ClienteAirflow(cfg)
        self.nombre_modelo = nombre_modelo
        self.parametros_cierre = parametros_cierre        # callable → dict (claves fuera del código)
        self.intervalo = intervalo
        self.espera_confirmacion = espera_confirmacion
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None

    # ------------------------------------------------------------------ iniciar
    def iniciar(self, clave: str | None = None, cuerpo: bytes = b"") -> str:
        clave = validar_clave(clave)
        id_, pendiente = self._registrar(clave, huella(cuerpo) if clave else None)
        if not pendiente:
            return id_                                   # ya terminó: 202 con el mismo id
        if self._enviar(id_):
            return id_
        raise OrquestadorNoDisponible("no se pudo confirmar el inicio en Airflow; la solicitud quedó pendiente "
                                      "(reintente con la misma Idempotency-Key)")

    def _registrar(self, clave: str | None, h: str | None) -> tuple[str, bool]:
        try:
            with transaccion(self.motor) as c:
                if clave is not None:
                    fila = c.execute(text("SELECT id, huella_solicitud FROM solicitudes_reentrenamiento "
                                          "WHERE clave_idempotencia = :k FOR UPDATE"), {"k": clave}).first()
                    if fila is not None:
                        if fila.huella_solicitud != h:
                            raise ClaveInvalida("la Idempotency-Key ya se usó con otra solicitud")
                        cerrada = c.execute(text(
                            "SELECT EXISTS (SELECT 1 FROM resultados_reentrenamiento WHERE id = :i) "
                            "OR EXISTS (SELECT 1 FROM veredictos_reentrenamiento WHERE id = :i) "
                            "OR EXISTS (SELECT 1 FROM tareas_reentrenamiento WHERE id = :i)"), {"i": fila.id}).scalar()
                        return fila.id, not cerrada
                id_ = f"nm-{uuid.uuid4()}"
                c.execute(text("INSERT INTO solicitudes_reentrenamiento (id, clave_idempotencia, huella_solicitud, "
                               "origen) VALUES (:i, :k, :h, 'api') ON CONFLICT (clave_idempotencia) DO NOTHING"),
                          {"i": id_, "k": clave, "h": h})
                existe = c.execute(text("SELECT 1 FROM solicitudes_reentrenamiento WHERE id = :i"),
                                   {"i": id_}).scalar()
            if not existe:                               # otra solicitud con la misma clave ganó la carrera
                return self._registrar(clave, h)
            return id_, True
        except ClaveInvalida:
            raise
        except BaseDatosNoDisponible as exc:
            raise OrquestadorNoDisponible("no se pudo registrar la solicitud: la base de datos no está "
                                          "disponible") from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("no se pudo registrar la solicitud de reentrenamiento: %s", type(exc).__name__)
            raise OrquestadorNoDisponible("no se pudo registrar la solicitud de reentrenamiento") from exc

    def _coincide(self, cuerpo: dict, id_: str) -> bool:
        return cuerpo.get("dag_run_id") == id_ and cuerpo.get("dag_id") == self.cfg.dag_id

    def _enviar(self, id_: str) -> bool:
        try:
            codigo, cuerpo = self.cliente.crear_ejecucion(id_)
        except Exception as exc:  # noqa: BLE001 - respuesta perdida: se confirma
            logger.warning("POST dagRuns sin respuesta (%s): se confirma con GET", type(exc).__name__)
            return self._confirmar(id_)
        if codigo == 200:
            if self._coincide(cuerpo, id_):
                return True
            logger.error("Airflow respondió 200 con otra ejecución: se confirma con GET")
            return self._confirmar(id_)
        if codigo == 409:
            return self._confirmar(id_)
        logger.error("Airflow rechazó la ejecución %s con %s", id_, codigo)
        return False

    def _confirmar(self, id_: str) -> bool:
        import time
        for i in range(CONFIRMACIONES):
            try:
                codigo, cuerpo = self.cliente.leer_ejecucion(id_)
                if codigo == 200:
                    return self._coincide(cuerpo, id_)
                if codigo == 404:
                    return False
            except Exception as exc:  # noqa: BLE001
                logger.warning("no se pudo confirmar la ejecución %s: %s", id_, type(exc).__name__)
            if i + 1 < CONFIRMACIONES:
                time.sleep(self.espera_confirmacion)
        return False

    # ------------------------------------------------------------------- estado
    def estado(self, id_: str) -> EstadoReentrenamiento | None:
        if len(id_) > 250:
            return None
        self.recuperar(id_)
        with transaccion(self.motor) as c:
            existe = c.execute(text("SELECT 1 FROM solicitudes_reentrenamiento WHERE id = :i"), {"i": id_}).scalar()
            if not existe:
                return None
            r = c.execute(text("SELECT estado, promovido, version, motivo FROM resultados_reentrenamiento "
                               "WHERE id = :i"), {"i": id_}).first()
        if r is None:
            return EstadoReentrenamiento(estado="en_curso")
        return EstadoReentrenamiento(estado=r.estado, promovido=bool(r.promovido),
                                     version=r.version if r.version is not None else VERSION_SIN_VIGENTE,
                                     motivo=r.motivo)

    def disponible(self) -> bool:
        return self.cliente.salud()

    # -------------------------------------------------------------- recuperación
    SQL_PENDIENTES = """
    SELECT s.id,
           EXISTS (SELECT 1 FROM veredictos_reentrenamiento v WHERE v.id = s.id) AS con_veredicto,
           t.id IS NULL AND clock_timestamp() >= s.solicitado_en + interval '30 minutes' AS cola_vencida,
           t.id IS NOT NULL AND clock_timestamp() >= t.iniciada_en + interval '10 minutes' AS plazo_vencido
    FROM solicitudes_reentrenamiento s
    LEFT JOIN tareas_reentrenamiento t ON t.id = s.id
    WHERE NOT EXISTS (SELECT 1 FROM resultados_reentrenamiento r WHERE r.id = s.id)
      AND (:id IS NULL OR s.id = :id)
    ORDER BY s.solicitado_en
    LIMIT 50
    """

    def recuperar(self, id_: str | None = None) -> list[tuple[str, str]]:
        """Cierra lo vencido. Devuelve ``[(id, codigo)]``. Lanza ``BaseDatosNoDisponible``
        si PostgreSQL no responde (el GET devuelve 503)."""
        with transaccion(self.motor) as c:
            filas = c.execute(text(self.SQL_PENDIENTES), {"id": id_}).mappings().all()
        hechos = []
        for f in filas:
            condicion = next((k for k in ("con_veredicto", "plazo_vencido", "cola_vencida") if f[k]), None)
            if condicion is None:
                continue
            try:
                from app.db.cierre import cerrar
                r = cerrar(self.parametros_cierre(), f["id"], estado="fallido", motivo=MOTIVOS[condicion],
                           escrito_por="api", nombre_modelo=self.nombre_modelo, condicion=condicion)
                hechos.append((f["id"], r.codigo))
                logger.warning("recuperación de %s (%s): %s", f["id"], condicion, r.codigo)
            except Exception as exc:  # noqa: BLE001
                logger.error("no se pudo cerrar %s: %s", f["id"], type(exc).__name__)
        return hechos

    def iniciar_hilos(self) -> None:
        if self._hilo is not None:
            return
        self._parar.clear()
        self._hilo = threading.Thread(target=self._ciclo, name="recuperacion-reentrenamientos", daemon=True)
        self._hilo.start()

    def detener_hilos(self) -> None:
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=20)
            self._hilo = None

    def _ciclo(self) -> None:
        while not self._parar.is_set():                 # primer ciclo al arrancar, antes de esperar
            try:
                self.recuperar()
            except Exception as exc:  # noqa: BLE001
                logger.warning("recuperación de reentrenamientos omitida: %s", type(exc).__name__)
            self._parar.wait(self.intervalo)
