"""Modelo vigente en memoria, con recarga en segundo plano sin reiniciar la API.

* ``actual()`` devuelve el modelo cargado (referencia inmutable; el cambio de
  modelo es atómico) o lanza ModeloNoDisponible.
* ``refrescar()`` consulta la huella del proveedor y, si cambió, carga el nuevo
  modelo. Si la carga falla, **se conserva el modelo vigente** (requisito: la
  API sigue prediciendo aunque el origen del modelo no esté disponible).
* Si el origen no responde y **todavía no hay modelo**, se usa
  ``proveedor.respaldo()`` cuando existe (caché validada del proveedor MLflow).
* ``actualizado()`` es ``False`` si el origen responde pero no ofrece una versión
  utilizable (alias inválido, versión inválida): ``/health`` lo informa (M9).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace

from app.modelo.proveedores import ModeloCargado, ModeloNoDisponible, OrigenNoDisponible, ProveedorModelo

logger = logging.getLogger(__name__)


class GestorModelo:
    def __init__(self, proveedor: ProveedorModelo, intervalo: float = 30.0):
        self.proveedor = proveedor
        self.intervalo = intervalo
        self._modelo: ModeloCargado | None = None
        self._huella: str | None = None
        self._candado = threading.Lock()
        self.ultimo_error: str | None = None
        self.actualizacion_fallida: str | None = None
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None

    def actual(self) -> ModeloCargado:
        modelo = self._modelo
        if modelo is None:
            raise ModeloNoDisponible(self.ultimo_error or "no hay un modelo cargado")
        return modelo

    def cargado(self) -> bool:
        return self._modelo is not None

    def servido(self) -> ModeloServido:
        return ModeloServido(self.actual(), VIGENTE, self.actual().version, None)

    def motivos_degradacion(self) -> list[str]:
        return []

    def actualizado(self) -> bool:
        return self.actualizacion_fallida is None

    def _cargar(self, nuevo: ModeloCargado, huella: str) -> bool:
        anterior = self._modelo
        self._modelo, self._huella, self.ultimo_error, self.actualizacion_fallida = nuevo, huella, None, None
        logger.info("modelo cargado: %s versión %s (antes: %s)", nuevo.nombre, nuevo.version,
                    anterior.version if anterior else "ninguno")
        return True

    def _origen_caido(self, motivo: str) -> bool:
        """El origen no responde: se conserva el modelo; sin modelo, se intenta el respaldo."""
        self.ultimo_error = motivo
        logger.warning(motivo)
        respaldo = getattr(self.proveedor, "respaldo", None)
        if self._modelo is not None or respaldo is None:
            return False
        try:
            modelo, huella = respaldo()
        except ModeloNoDisponible as exc:
            self.ultimo_error = f"{motivo}; {exc}"
            logger.warning("sin modelo de respaldo: %s", exc)
            return False
        logger.warning("origen del modelo no disponible: se usa la versión %s de la caché", modelo.version)
        return self._cargar(modelo, huella)

    def _fallo_de_actualizacion(self, motivo: str) -> bool:
        self.ultimo_error = motivo
        if self._modelo is not None:
            self.actualizacion_fallida = motivo
        logger.warning("no se pudo actualizar el modelo: %s", motivo)
        return False

    def refrescar(self) -> bool:
        """``True`` si se cargó un modelo nuevo."""
        with self._candado:
            try:
                huella = self.proveedor.huella()
            except OrigenNoDisponible as exc:
                return self._origen_caido(f"origen del modelo no disponible: {exc}")
            except ModeloNoDisponible as exc:
                return self._fallo_de_actualizacion(str(exc))
            except Exception as exc:  # noqa: BLE001
                return self._origen_caido(f"no se pudo consultar el origen del modelo: {type(exc).__name__}")
            if huella is None:
                if self._modelo is None:
                    self.ultimo_error = "no hay un modelo disponible"
                    return False
                return self._fallo_de_actualizacion("el origen ya no ofrece un modelo")
            if huella == self._huella and self._modelo is not None:
                self.actualizacion_fallida = None
                return False
            try:
                nuevo = self.proveedor.obtener()
            except OrigenNoDisponible as exc:
                return self._origen_caido(f"origen del modelo no disponible: {exc}")
            except ModeloNoDisponible as exc:
                return self._fallo_de_actualizacion(str(exc))
            return self._cargar(nuevo, huella)

    def _bucle(self) -> None:
        while not self._detener.wait(self.intervalo):
            self.refrescar()

    def iniciar(self) -> None:
        self.refrescar()
        self._detener.clear()
        self._hilo = threading.Thread(target=self._bucle, name="gestor-modelo", daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)


# ============================================================ etapa 6: GestorVigente
VIGENTE, ACTUALIZANDO, ANTERIOR, SIN_VERIFICAR = "vigente", "actualizando", "anterior", "sin_verificar"
MOTIVO_SIN_VERIFICAR = "no se pudo verificar la vigencia del modelo (PostgreSQL no disponible desde el arranque)"
INTENTOS_POR_CICLO = 3
REFRESCOS_ALIAS_DESINCRONIZADO = 2


@dataclass(frozen=True)
class ModeloServido:
    """Lo que la API informa en cada respuesta (una sola referencia inmutable: el modelo
    y su estado de vigencia siempre corresponden entre sí)."""
    modelo: ModeloCargado
    estado_vigencia: str = VIGENTE
    version_vigente: str | None = None
    motivo: str | None = None

    @property
    def es_vigente(self) -> bool:
        return self.estado_vigencia == VIGENTE


@dataclass(frozen=True)
class _Estado:
    modelo: ModeloCargado | None = None
    decision: object | None = None        # Vigente de la versión servida (None si vino de la caché sin base)
    sha: str | None = None                 # SHA-256 verificado al activar
    validado: bool = False                 # superó SHA (antes de deserializar) y T3 al activar
    vigencia: str = VIGENTE
    observado: object | None = None       # vigente de la última consulta exitosa
    motivo: str | None = None


class GestorVigente:
    """Modelo servido según PostgreSQL (E6-9, plan §7). Estados de vigencia (§7.4):

    * ``vigente``: la versión servida coincide con el vigente de la **última consulta
      exitosa** (incluye la actualización pendiente: una promoción todavía no observada).
    * ``actualizando``: se observó otro vigente B y el intento de activarlo está en curso
      (o pendiente porque la base no responde). No degrada ``/health`` por sí mismo.
    * ``anterior`` (E-ANT): se observó B, se intentó activarlo y no se pudo; se conserva el
      modelo ya cargado (vigente confirmado en su momento). Degrada ``/health``.
    * ``sin_verificar`` (revisión del PR #6): PostgreSQL no respondió **ninguna** consulta
      desde el arranque y se sirve una entrada de la caché confirmada en su momento
      (``decision_id``, ``secuencia`` y SHA verificados). ``version_vigente`` se desconoce.
      Degrada ``/health``. Solo se entra desde el arranque sin modelo: si la base deja
      de responder después de una consulta exitosa se conserva el último estado conocido.

    ``version``/``version_modelo`` informan siempre el modelo que realmente predice;
    ``version_vigente``, el vigente observado.
    """

    def __init__(self, proveedor, intervalo: float = 15.0, intentos: int = INTENTOS_POR_CICLO):
        self.proveedor = proveedor
        self.intervalo = intervalo
        self.intentos = intentos
        self._e = _Estado()
        self._candado = threading.Lock()
        self.ultimo_error: str | None = None
        self.alias_desincronizado = 0
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None
        # gancho para pruebas: se ejecuta entre la segunda lectura y el intercambio
        self._antes_del_intercambio = None

    # ------------------------------------------------------------------ lectura
    def servido(self) -> ModeloServido:
        e = self._e
        if e.modelo is None:
            raise ModeloNoDisponible(self.ultimo_error or "no hay un modelo cargado")
        vv = e.observado.version if e.observado is not None else (e.modelo.version if e.vigencia == VIGENTE
                                                                    else None)
        return ModeloServido(e.modelo, e.vigencia, vv, e.motivo)

    def actual(self) -> ModeloCargado:
        return self.servido().modelo

    def cargado(self) -> bool:
        return self._e.modelo is not None

    def actualizado(self) -> bool:
        return self._e.vigencia not in (ANTERIOR, SIN_VERIFICAR)

    def vigencia(self) -> str | None:
        return self._e.vigencia if self._e.modelo is not None else None

    def motivos_degradacion(self) -> list[str]:
        e = self._e
        motivos = []
        if e.modelo is not None and e.vigencia == ANTERIOR:
            motivos.append(e.motivo or "se sirve un modelo anterior")
        if e.modelo is not None and e.vigencia == SIN_VERIFICAR:
            motivos.append(MOTIVO_SIN_VERIFICAR)
        if self.alias_desincronizado >= REFRESCOS_ALIAS_DESINCRONIZADO:
            motivos.append("el alias de MLflow no coincide con la promoción confirmada")
        return motivos

    # --------------------------------------------------------------- transiciones
    def _poner(self, **cambios) -> None:
        self._e = replace(self._e, **cambios)

    def _retirar(self, motivo: str) -> None:
        if self._e.modelo is not None:
            logger.error("se retira el modelo %s: %s", self._e.modelo.version, motivo)
        self._e = _Estado()
        self.ultimo_error = motivo

    def _anterior(self, observado, motivo: str) -> bool:
        e = self._e
        if e.modelo is None:
            self.ultimo_error = motivo
            return False
        texto = f"se sirve la versión {e.modelo.version}; la vigente {observado.version} no se pudo cargar: {motivo}"
        self._poner(vigencia=ANTERIOR, observado=observado, motivo=texto)
        logger.warning(texto)
        return False

    def _activar(self, o) -> None:
        self._e = _Estado(o.modelo, o.vigente, o.vigente.sha256_artefacto, True, VIGENTE, o.vigente, None)
        self.ultimo_error = None
        logger.info("modelo activado: versión %s (decisión %s, secuencia %s)", o.modelo.version,
                    o.vigente.decision_id, o.vigente.secuencia)

    def _revisar_alias(self, v) -> None:
        try:
            alias = self.proveedor.alias_mlflow()
        except Exception:  # noqa: BLE001 - MLflow caído: no se cambia el contador
            return
        objetivo = v.version if v is not None else None
        self.alias_desincronizado = 0 if alias == objetivo else self.alias_desincronizado + 1

    def refrescar(self) -> bool:
        """``True`` si se activó (o adoptó) un modelo nuevo."""
        with self._candado:
            return self._refrescar()

    def _refrescar(self) -> bool:
        from app.modelo.proveedor_mlflow import ConsultaIndeterminada
        try:
            v = self.proveedor.consultar_vigente()
        except ConsultaIndeterminada as exc:
            return self._sin_base(str(exc))
        if v is None:
            self._retirar("PostgreSQL no tiene un modelo vigente confirmado")
            self._revisar_alias(None)
            return False
        self._revisar_alias(v)
        e = self._e
        # misma decisión (decision_id, secuencia) y mismo SHA verificado: es el vigente. Desde
        # sin_verificar, es la primera consulta exitosa que confirma la entrada de la caché.
        if (e.modelo is not None and e.decision is not None and e.decision.clave == v.clave
                and e.sha == v.sha256_artefacto):
            if e.vigencia == SIN_VERIFICAR:
                logger.info("PostgreSQL confirma que la versión %s de la caché es el vigente", e.modelo.version)
            self._poner(vigencia=VIGENTE, observado=v, motivo=None)
            return False

        for _ in range(self.intentos):
            e = self._e
            # Adopción sin descarga (§7.3.4): misma versión y mismo SHA, otra decisión
            if (e.modelo is not None and e.validado and e.sha == v.sha256_artefacto
                    and e.modelo.version == v.version):
                try:
                    v2 = self.proveedor.consultar_vigente()
                except ConsultaIndeterminada as exc:
                    self._poner(vigencia=ACTUALIZANDO, observado=v)
                    self.ultimo_error = str(exc)
                    return False
                if v2 is None:
                    self._retirar("PostgreSQL no tiene un modelo vigente confirmado")
                    return False
                if v2.clave == v.clave:
                    self._poner(decision=v, vigencia=VIGENTE, observado=v, motivo=None)
                    logger.info("decisión %s adoptada sin descarga (versión %s, mismo SHA)", v.decision_id,
                                v.version)
                    return True
                v = v2
                continue
            if e.modelo is not None:
                self._poner(vigencia=ACTUALIZANDO, observado=v)
            try:
                o = self.proveedor.obtener(v)
            except OrigenNoDisponible as exc:
                return self._anterior(v, f"no se pudo obtener ({exc})")
            except ModeloNoDisponible as exc:
                return self._anterior(v, f"no supera la validación ({exc})")
            try:
                v2 = self.proveedor.consultar_vigente()           # segunda lectura (§7.3)
            except ConsultaIndeterminada as exc:
                o.limpiar()
                self.ultimo_error = str(exc)
                return False                                       # se conserva el estado (v9)
            if v2 is None:
                o.limpiar()
                self._retirar("PostgreSQL no tiene un modelo vigente confirmado")
                return False
            if v2.clave == v.clave:
                if self._antes_del_intercambio is not None:
                    self._antes_del_intercambio()
                self._activar(o)                                   # intercambio atómico, sin E/S
                self.proveedor.guardar_en_cache(o)
                o.limpiar()
                return True
            o.limpiar()
            logger.info("el vigente cambió durante la carga (%s → %s): se descarta lo descargado", v.clave, v2.clave)
            v = v2
        return self._anterior(v, f"la revalidación no confirmó el vigente tras {self.intentos} intentos")

    def _sin_base(self, motivo: str) -> bool:
        """PostgreSQL inaccesible: con un modelo cargado se conservan el modelo, el estado y
        el vigente observado (nunca se pasa a ``sin_verificar``). Sin modelo, la entrada de
        la caché de mayor secuencia (§7.2), en estado ``sin_verificar``: solo entradas
        confirmadas en PostgreSQL en su momento, con ``decision_id``, ``secuencia`` y SHA
        verificados (``CacheModelos.validar``)."""
        self.ultimo_error = motivo
        if self._e.modelo is not None:
            return False
        try:
            modelo, meta = self.proveedor.respaldo_sin_base()
        except ModeloNoDisponible as exc:
            self.ultimo_error = f"{motivo}; {exc}"
            return False
        from app.db.promociones import Vigente
        decision = Vigente(int(meta["decision_id"]), int(meta["secuencia"]), str(meta["version"]),
                           str(meta["run_id"]), str(meta["ruta_artefacto"]), str(meta["sha256_artefacto"]), 0)
        texto = (f"PostgreSQL no disponible desde el arranque: se sirve la versión {modelo.version} de la caché "
                 f"(decisión {decision.decision_id}, secuencia {decision.secuencia}); no se pudo verificar el "
                 "vigente actual")
        self._e = _Estado(modelo, decision, decision.sha256_artefacto, True, SIN_VERIFICAR, None, texto)
        logger.warning(texto)
        return True

    # ------------------------------------------------------------------ hilo
    def _bucle(self) -> None:
        while not self._detener.wait(self.intervalo):
            try:
                self.refrescar()
            except Exception:  # noqa: BLE001 - el hilo nunca muere
                logger.exception("error inesperado al refrescar el modelo")

    def iniciar(self) -> None:
        try:
            self.refrescar()
        except Exception:  # noqa: BLE001
            logger.exception("error inesperado al cargar el modelo")
        self._detener.clear()
        self._hilo = threading.Thread(target=self._bucle, name="gestor-vigente", daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
