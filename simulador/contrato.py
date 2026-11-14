"""Validación del contrato del Anexo escrita aparte de ``app/schemas`` (no comparte
errores con el producto). Cada función devuelve la lista de incumplimientos (vacía = ok).
Los campos adicionales están permitidos; los del contrato no pueden faltar ni cambiar de tipo.
"""

from __future__ import annotations

import math
import re
from datetime import date

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ESTADOS_CALIDAD = ("ok", "corregido", "advertencia")
ACCIONES = ("reparar", "advertir")
ESTADOS_EJECUCION = ("en_curso", "exitoso", "fallido")


def _entero(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _numero(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) and math.isfinite(x)


def _fecha(x) -> bool:
    if not isinstance(x, str) or not _ISO.match(x):
        return False
    try:
        date.fromisoformat(x)
    except ValueError:
        return False
    return True


def calidad(c) -> list[str]:
    if not isinstance(c, dict):
        return ["calidad no es un objeto"]
    errores = []
    estado, obs = c.get("estado"), c.get("observaciones")
    if estado not in ESTADOS_CALIDAD:
        errores.append(f"calidad.estado inválido: {estado!r}")
    if not isinstance(obs, list):
        return errores + ["calidad.observaciones no es una lista"]
    for i, o in enumerate(obs):
        if not (isinstance(o, dict) and isinstance(o.get("campo"), str) and o.get("accion") in ACCIONES
                and isinstance(o.get("detalle"), str) and o["detalle"]):
            errores.append(f"calidad.observaciones[{i}] inválida: {o!r}")
    if errores:
        return errores
    acciones = {o["accion"] for o in obs}
    esperado = "advertencia" if "advertir" in acciones else ("corregido" if obs else "ok")
    if estado != esperado:
        errores.append(f"calidad.estado {estado!r} no corresponde a las acciones (debería ser {esperado!r})")
    return errores


def prediccion(r, tienda: int, fecha: str, familias: list[str]) -> list[str]:
    if not isinstance(r, dict):
        return ["la respuesta no es un objeto"]
    errores = []
    if not _entero(r.get("tienda_id")) or r.get("tienda_id") != tienda:
        errores.append(f"tienda_id {r.get('tienda_id')!r} (se esperaba {tienda})")
    if not _fecha(r.get("fecha")) or r.get("fecha") != fecha:
        errores.append(f"fecha {r.get('fecha')!r} (se esperaba {fecha})")
    if not isinstance(r.get("version_modelo"), str) or not r["version_modelo"]:
        errores.append(f"version_modelo {r.get('version_modelo')!r} no es texto")
    preds = r.get("predicciones")
    if not isinstance(preds, list):
        errores.append("predicciones no es una lista")
    else:
        ids = []
        for i, p in enumerate(preds):
            if not isinstance(p, dict):
                errores.append(f"predicciones[{i}] no es un objeto")
                continue
            ids.append(p.get("familia_id"))
            if not _numero(p.get("venta_esperada")) or p["venta_esperada"] < 0:
                errores.append(f"predicciones[{i}].venta_esperada {p.get('venta_esperada')!r}")
            if not _entero(p.get("cantidad_sugerida")) or p["cantidad_sugerida"] < 0:
                errores.append(f"predicciones[{i}].cantidad_sugerida {p.get('cantidad_sugerida')!r} "
                               "no es un entero no negativo")
        if sorted(ids, key=str) != sorted(familias) or len(set(ids)) != len(ids):
            errores.append(f"predicciones no cubre exactamente las {len(familias)} familias: {ids}")
    return errores + calidad(r.get("calidad"))


def observacion(r) -> list[str]:
    if not isinstance(r, dict):
        return ["la respuesta no es un objeto"]
    errores = []
    if not _entero(r.get("registros_guardados")) or r["registros_guardados"] < 0:
        errores.append(f"registros_guardados {r.get('registros_guardados')!r}")
    return errores + calidad(r.get("calidad"))


def rechazo(r) -> list[str]:
    if not isinstance(r, dict) or not isinstance(r.get("detail"), str) or not r["detail"]:
        return [f"un rechazo debe traer detail (texto): {r!r}"]
    return []


def aceptado(r) -> list[str]:
    if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"]:
        return [f"202 sin id de texto: {r!r}"]
    return []


def estado_ejecucion(r) -> list[str]:
    if not isinstance(r, dict) or r.get("estado") not in ESTADOS_EJECUCION:
        return [f"estado de ejecución inválido: {r!r}"]
    if r["estado"] == "en_curso":
        return []
    errores = []
    if not isinstance(r.get("promovido"), bool):
        errores.append("promovido no es booleano")
    if not isinstance(r.get("version"), str):
        errores.append("version no es texto")
    if not isinstance(r.get("motivo"), str):
        errores.append("motivo no es texto")
    return errores


def modelo(r) -> list[str]:
    if not isinstance(r, dict) or not all(isinstance(r.get(k), str) for k in ("nombre", "alias", "version")):
        return [f"/modelo sin nombre, alias y version de texto: {r!r}"]
    return []


def salud(codigo, r) -> list[str]:
    if codigo != 200:
        return [f"/health respondió {codigo} con un modelo cargado"]
    if not isinstance(r, dict) or r.get("estado") not in ("ok", "degradado"):
        return [f"/health estado inválido: {r!r}"]
    if r["estado"] == "degradado" and not isinstance(r.get("motivo"), str):
        return ["/health degradado sin motivo"]
    return []
