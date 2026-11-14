"""Autorización, promoción y reconciliación (etapa 6, DECISIONES.md §15, plan §5.4–§5.5).

PostgreSQL es la fuente de verdad del vigente (E6-9). MLflow es registro y reflejo:

* :func:`autorizar_y_promover` es la **única** función que escribe ``estado=valido``
  y decide promociones. Orden: comprobar el resultado del hijo contra MLflow →
  **decisión** en PostgreSQL (transacción condicionada) → reflejo en MLflow (tags y
  alias, comprobando la concesión antes de cada escritura) → **confirmación** en
  PostgreSQL solo si el alias quedó aplicado. Recién entonces la versión es vigente.
* :func:`_escribir_alias` es la **única** llamada a ``set_registered_model_alias`` /
  ``delete_registered_model_alias`` del proyecto. La usan solo
  :func:`autorizar_y_promover` (promoción) y :func:`reconciliar` (reflejo del vigente
  ya confirmado en PostgreSQL). Una prueba estática (AST) lo verifica.
* :func:`reconciliar` (con la concesión tomada) anula decisiones de épocas anteriores
  sin resolución, invalida versiones huérfanas o anuladas, marca ``FAILED`` las
  ejecuciones huérfanas y vuelve a apuntar el alias al vigente de PostgreSQL.

MLflow no admite *fencing tokens*: una escritura tardía de un supervisor que perdió
su concesión puede dejar MLflow desfasado hasta la siguiente reconciliación, pero
nunca vuelve servible una versión (la API sirve solo lo confirmado en PostgreSQL).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.modelo.bloqueo import ConcesionEntrenamiento, ConcesionPerdida
from app.modelo.registro import EXPERIMENTO, MODOS_PROMOVER, REQUERIDOS, ResultadoRegistro

logger = logging.getLogger(__name__)


class ErrorPromocion(RuntimeError):
    """El alias no se aplicó: la decisión se anuló y la versión quedó inválida."""


class PromocionIndeterminada(RuntimeError):
    """Falló la asignación del alias y tampoco se pudo consultar su valor. La decisión
    se anula en PostgreSQL (la versión nunca es servible); el alias de MLflow puede
    haber quedado en la versión nueva hasta la siguiente reconciliación."""

    def __init__(self, nombre: str, alias: str, version: str, causa: BaseException | None = None):
        self.nombre, self.alias, self.version = nombre, alias, version
        super().__init__(
            f"Promoción en estado indeterminado: la asignación del alias '{alias}' de '{nombre}' a la "
            f"versión {version} falló{f' ({type(causa).__name__})' if causa else ''} y no se pudo consultar "
            "el alias después. La decisión se anuló en PostgreSQL: la API no servirá esta versión; el alias "
            "de MLflow no está garantizado hasta la siguiente reconciliación.")


class ReconciliacionIncompleta(RuntimeError):
    """La reconciliación no pudo confirmar el alias de MLflow. ``resultado`` distingue
    ``no_aplicada`` (el alias quedó en otra versión) de ``indeterminada`` (no se pudo
    consultar después del error: puede haberse aplicado o no). En ambos casos el
    entrenamiento no empieza."""

    def __init__(self, nombre: str, alias: str, objetivo: str | None, resultado: str, observado: str | None):
        self.nombre, self.alias, self.objetivo, self.resultado, self.observado = (nombre, alias, objetivo,
                                                                                  resultado, observado)
        destino = f"la versión {objetivo}" if objetivo else "ninguna versión (borrar el alias)"
        if resultado == "indeterminada":
            detalle = ("falló y no se pudo consultar el alias después: puede haberse aplicado o no; "
                       "no está garantizado")
        else:
            detalle = f"no se aplicó (el alias apunta a {observado or 'ninguna versión'})"
        super().__init__(f"Reconciliación incompleta: el reflejo del vigente en el alias '{alias}' de "
                         f"'{nombre}' hacia {destino} {detalle}. No se entrena.")


@dataclass
class ResultadoPromocion:
    autorizada: bool                  # la versión quedó estado=valido
    promovido: bool                   # resolución confirmada en PostgreSQL
    version: str | None
    motivo: str
    decision_id: int | None = None
    secuencia: int | None = None


# ------------------------------------------------------------------ alias de MLflow
def _alias_actual(cliente, nombre: str, alias: str) -> str | None:
    from mlflow.exceptions import MlflowException
    try:
        return str(cliente.get_model_version_by_alias(nombre, alias).version)
    except MlflowException as exc:
        if getattr(exc, "error_code", None) in ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"):
            return None
        raise


def resolver_asignacion_alias(cliente, nombre: str, alias: str, version: str | None) -> tuple[str, str | None]:
    """Tras una excepción al escribir el alias, lo consulta una vez (etapa 5).
    ``("aplicada", v)``, ``("no_aplicada", observado)`` o ``("indeterminada", None)``."""
    try:
        observado = _alias_actual(cliente, nombre, alias)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de la consulta deja el resultado abierto
        logger.error("no se pudo consultar el alias %s de %s tras el fallo: %s", alias, nombre, type(exc).__name__)
        return "indeterminada", None
    esperado = None if version is None else str(version)
    return ("aplicada", observado) if observado == esperado else ("no_aplicada", observado)


def _escribir_alias(cliente, nombre: str, alias: str, version: str | None) -> tuple[str, str | None]:
    """ÚNICO escritor del alias. ``version=None`` lo elimina. Devuelve ``(resultado,
    observado)``: ``aplicada``, ``no_aplicada`` o ``indeterminada`` (el servidor pudo
    aplicar el cambio aunque el cliente reciba un error, así que se consulta antes de
    concluir) y la versión observada en esa consulta."""
    try:
        if version is None:
            cliente.delete_registered_model_alias(nombre, alias)
        else:
            cliente.set_registered_model_alias(nombre, alias, str(version))
        return "aplicada", (None if version is None else str(version))
    except Exception as exc:  # noqa: BLE001
        resultado, observado = resolver_asignacion_alias(cliente, nombre, alias, version)
        logger.warning("la escritura del alias %s → %s lanzó %s; resultado: %s (observado: %s)",
                       alias, version, type(exc).__name__, resultado, observado)
        return resultado, observado


# --------------------------------------------------------------------- autorización
def _motivo_no_autorizable(cliente, c: ConcesionEntrenamiento, nombre: str, r: ResultadoRegistro) -> str | None:
    if r.version is None:
        return f"sin versión registrada: {r.motivo}"
    if not r.verificacion:
        return r.motivo
    faltan = [k for k in REQUERIDOS.get(r.criterio, ("_",)) if not r.condiciones.get(k, False)]
    if faltan:
        return "no cumple " + ", ".join(faltan)
    if r.epoca != c.epoca:
        return f"el resultado es de la época {r.epoca}, no de la {c.epoca}"
    mv = cliente.get_model_version(nombre, r.version)
    t = dict(mv.tags or {})
    try:
        cond_mlflow = json.loads(t.get("condiciones", "null"))
    except ValueError:
        cond_mlflow = None
    comprobaciones = {
        "run_id": mv.run_id == r.run_id,
        "estado=pendiente": t.get("estado") == "pendiente",
        "epoca_bloqueo": t.get("epoca_bloqueo") == str(c.epoca),
        "sha256_artefacto": bool(r.sha256) and t.get("sha256_artefacto") == r.sha256,
        "ruta_artefacto": bool(r.ruta_artefacto) and t.get("ruta_artefacto") == r.ruta_artefacto,
        "verificacion=superada": t.get("verificacion") == "superada",
        "condiciones": cond_mlflow == r.condiciones,
    }
    malas = [k for k, ok in comprobaciones.items() if not ok]
    return None if not malas else "el resultado del hijo no coincide con MLflow: " + ", ".join(malas)


def _marcar_invalida(c: ConcesionEntrenamiento, cliente, nombre: str, version: str | None, motivo: str) -> None:
    """Mejor esfuerzo: bajar a ``invalido`` (siempre seguro)."""
    if version is None:
        return
    try:
        c.verificar()
        cliente.set_model_version_tag(nombre, version, "estado", "invalido")
        cliente.set_model_version_tag(nombre, version, "motivo", motivo[:500])
    except Exception as exc:  # noqa: BLE001
        logger.error("no se pudo marcar la versión %s como inválida: %s", version, type(exc).__name__)


def _anular(c: ConcesionEntrenamiento, cliente, decision_id: int, nombre: str, version: str, motivo: str) -> None:
    try:
        c.insertar_resolucion(decision_id, "anulada", motivo)
    except ConcesionPerdida:
        logger.error("no se pudo anular la decisión %s: la anulará la siguiente reconciliación", decision_id)
    _marcar_invalida(c, cliente, nombre, version, motivo)


def decidir_si_mejora(c: ConcesionEntrenamiento, r: ResultadoRegistro, vigente) -> tuple[bool, str]:
    """Modo ``si_mejora`` (etapa 7, R2): sin vigente se promueve un candidato válido; con
    un vigente normal, solo si R2 es ``true`` y se calculó contra **ese** vigente (misma
    decisión de PostgreSQL); con un vigente ``respaldo_baseline`` (origen
    ``arranque_respaldo`` en PostgreSQL) basta T1–T4b y C1–C3 (E7-6b), y R2 solo se informa."""
    if vigente is None:
        return True, f"{r.motivo}; no había vigente"
    r2 = r.r2 or {}
    if r2.get("decision_id") is None or int(r2["decision_id"]) != vigente.decision_id:
        return False, (f"R2 no se calculó contra el vigente actual (decisión {vigente.decision_id}); "
                       f"{r2.get('motivo', 'sin R2')}")
    texto = r2.get("motivo") or "R2 sin motivo"
    if c.origen_de_decision(vigente.decision_id) == "arranque_respaldo":
        return True, f"el vigente es un respaldo baseline y el candidato cumple C1–C3; {texto}"
    if r2.get("resultado") is True and r.condiciones.get("R2") is True:
        return True, texto
    return False, texto


def autorizar_y_promover(c: ConcesionEntrenamiento, cliente, r: ResultadoRegistro, *, nombre: str, alias: str,
                         promover: str, origen: str) -> ResultadoPromocion:
    """Única función que escribe ``estado=valido`` y decide promociones (§5.4)."""
    if promover not in MODOS_PROMOVER:
        raise ValueError(f"promover debe ser uno de {MODOS_PROMOVER}")
    motivo_rechazo = _motivo_no_autorizable(cliente, c, nombre, r)
    if motivo_rechazo:
        if r.version is not None:
            _marcar_invalida(c, cliente, nombre, r.version, motivo_rechazo)
        return ResultadoPromocion(False, False, r.version, motivo_rechazo)

    vigente = c.leer_vigente(nombre)
    motivo_mejora = None
    if promover == "si_mejora":
        promover_ahora, motivo_mejora = decidir_si_mejora(c, r, vigente)
    else:
        promover_ahora = promover == "siempre" or (promover == "si_no_hay" and vigente is None)
    if not promover_ahora:
        c.verificar()
        cliente.set_model_version_tag(nombre, r.version, "estado", "valido")
        c.verificar()
        cliente.set_model_version_tag(nombre, r.version, "motivo", f"{r.motivo} (validada; no promovida)")
        if motivo_mejora:
            return ResultadoPromocion(True, False, r.version, f"{motivo_mejora}; no se reemplaza el vigente "
                                      f"(versión {vigente.version})")
        return ResultadoPromocion(True, False, r.version, f"{r.motivo}; no se promueve (--promover {promover}"
                                  + (f", vigente {vigente.version})" if vigente else ")"))

    decision_id = c.insertar_decision(nombre=nombre, version=r.version, run_id=r.run_id, ruta=r.ruta_artefacto,
                                      sha256=r.sha256, origen=origen)
    tags = {"estado": "valido", "decision_id": str(decision_id), "epoca_promocion": str(c.epoca),
            "promovido_por": f"{origen}:{promover}"}
    if vigente is not None:
        tags["alias_anterior"] = vigente.version
    try:
        for k, v in tags.items():
            c.verificar()
            cliente.set_model_version_tag(nombre, r.version, k, v)
        c.verificar()
        resultado, _ = _escribir_alias(cliente, nombre, alias, r.version)   # última operación en MLflow
    except ConcesionPerdida:
        raise                  # la decisión queda sin resolver: la anula la reconciliación
    except Exception as exc:
        _anular(c, cliente, decision_id, nombre, r.version, f"promoción anulada: {type(exc).__name__}")
        raise

    if resultado == "aplicada":
        try:
            secuencia = c.insertar_resolucion(decision_id, "confirmada", r.motivo)
        except ConcesionPerdida:
            try:
                cliente.set_model_version_tag(nombre, r.version, "promocion_tardia", "true")
            except Exception:  # noqa: BLE001
                pass
            logger.error("promoción tardía: el alias se escribió pero la concesión se perdió antes de confirmar")
            raise
        logger.info("versión %s promovida y confirmada (decisión %s, secuencia %s)", r.version, decision_id, secuencia)
        return ResultadoPromocion(True, True, r.version, motivo_mejora or r.motivo, decision_id, secuencia)
    if resultado == "no_aplicada":
        _anular(c, cliente, decision_id, nombre, r.version, "promoción anulada: el alias no se aplicó")
        raise ErrorPromocion(f"el alias '{alias}' no se aplicó a la versión {r.version}; la decisión se anuló")
    _anular(c, cliente, decision_id, nombre, r.version, "promoción anulada: resultado del alias indeterminado")
    raise PromocionIndeterminada(nombre, alias, r.version)


# ------------------------------------------------------------------ reconciliación
@dataclass
class ResumenReconciliacion:
    decisiones_anuladas: list[int]
    versiones_invalidadas: list[str]
    ejecuciones_fallidas: list[str]
    alias: str


def reconciliar(c: ConcesionEntrenamiento, cliente, nombre: str, alias: str) -> ResumenReconciliacion:
    """Con la concesión tomada (§5.5). Nunca toca versiones ``valido`` confirmadas ni las
    de la época actual. Solo es exitosa si el alias queda **confirmado**: ``sin cambios``
    (ya coincidía) o ``aplicada``. Si MLflow no responde, o si el reflejo del alias
    resulta ``no_aplicada`` o ``indeterminada`` (:class:`ReconciliacionIncompleta`), la
    excepción se propaga y el entrenamiento no empieza."""
    from mlflow.exceptions import MlflowException

    anuladas, invalidas, fallidas = [], [], []
    for decision_id, version, epoca in c.decisiones_sin_resolver(nombre):
        motivo = f"anulada: la época {epoca} perdió la concesión"
        c.insertar_resolucion(decision_id, "anulada", motivo)
        anuladas.append(decision_id)
        _marcar_invalida(c, cliente, nombre, version, motivo)
        invalidas.append(version)

    try:
        versiones = cliente.search_model_versions(f"name='{nombre}'")
    except MlflowException as exc:
        if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise
        versiones = []
    for v in versiones:
        t = dict(v.tags or {})
        epoca = int(t.get("epoca_bloqueo") or 0)
        motivo = None
        if t.get("estado") == "pendiente" and epoca < c.epoca:
            motivo = f"huérfana: época {epoca} sin autorización"
        elif t.get("estado") == "valido" and t.get("decision_id"):
            if c.resolucion_de(int(t["decision_id"])) != "confirmada":
                motivo = f"anulada: la decisión {t['decision_id']} no fue confirmada"
        if motivo and str(v.version) not in invalidas:
            c.verificar()
            cliente.set_model_version_tag(nombre, v.version, "estado", "invalido")
            cliente.set_model_version_tag(nombre, v.version, "motivo", motivo)
            invalidas.append(str(v.version))

    exp = cliente.get_experiment_by_name(EXPERIMENTO)
    if exp is not None:
        for run in cliente.search_runs([exp.experiment_id], filter_string="attributes.status = 'RUNNING'"):
            epoca = int(run.data.tags.get("epoca_bloqueo") or 0)
            if epoca and epoca < c.epoca:
                c.verificar()
                cliente.set_tag(run.info.run_id, "motivo_fallo", f"huérfana: época {epoca}")
                cliente.set_terminated(run.info.run_id, status="FAILED")
                fallidas.append(run.info.run_id)

    vigente = c.leer_vigente(nombre)
    objetivo = vigente.version if vigente else None
    estado_alias = "sin cambios"
    if _alias_actual(cliente, nombre, alias) != objetivo:
        c.verificar()
        estado_alias, observado = _escribir_alias(cliente, nombre, alias, objetivo)
        logger.warning("reconciliación: alias %s → %s (%s)", alias, objetivo, estado_alias)
        if estado_alias != "aplicada":
            raise ReconciliacionIncompleta(nombre, alias, objetivo, estado_alias, observado)
    if anuladas or invalidas or fallidas:
        logger.warning("reconciliación: decisiones anuladas %s, versiones inválidas %s, ejecuciones fallidas %s",
                       anuladas, invalidas, fallidas)
    return ResumenReconciliacion(anuladas, invalidas, fallidas, estado_alias)
