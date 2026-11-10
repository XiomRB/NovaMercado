"""Limpieza de solicitudes de ``POST /v1/observaciones``.

A diferencia de las promociones, una venta real nunca se supone: si un valor
no es confiable (negativo, nulo, no numérico) el registro **no se guarda** y se
advierte. La idempotencia del reenvío (E09) se resuelve en la persistencia.
"""

from __future__ import annotations

from pydantic import ValidationError

from app.calidad.campos import ESQUEMA_OBSERVACION, ESQUEMA_OBSERVACION_ITEM, normalizar_campos
from app.calidad.catalogo import Catalogo
from app.calidad.comunes import (exigir_campo, exigir_objeto, limpiar_fecha, limpiar_tienda,
                                 normalizar_lista)
from app.calidad.numeros import NumeroInvalido, describir, interpretar_numero, redondear
from app.calidad.resultado import RegistroCalidad, SolicitudRechazada
from app.schemas.contrato import SolicitudObservacion

_DESCARTADO = None


def limpiar_venta(valor: object, campo: str, registro: RegistroCalidad) -> int | None:
    """E08/E10. Entero >= 0 -> ok; '1,292', '812,0', 812.0 -> reparar.

    Decimal con fracción -> advertir y redondear. Negativo, nulo o no
    interpretable -> advertir y descartar (``None``): no se inventan ventas.
    """
    if valor is None:
        registro.advertir(campo, "sin valor; el registro no se guardó")
        return _DESCARTADO
    try:
        n = interpretar_numero(valor)
    except NumeroInvalido as exc:
        registro.advertir(campo, f"{exc}; el registro no se guardó")
        return _DESCARTADO
    if n.valor < 0:
        registro.advertir(campo, f"venta negativa {describir(valor)} (posible devolución); "
                                 "el registro no se guardó")
        return _DESCARTADO
    entero = redondear(n.valor)
    if not n.es_entero:
        registro.advertir(campo, f"{describir(valor)} no es un número entero de unidades; "
                                 f"se redondeó a {entero}")
    elif not isinstance(valor, int):
        motivo = n.formato or "número decimal sin parte fraccionaria"
        registro.reparar(campo, f"{describir(valor)} se interpretó como {entero} ({motivo})")
    return entero


def limpiar_solicitud_observacion(
    payload: object, catalogo: Catalogo
) -> tuple[SolicitudObservacion, RegistroCalidad]:
    registro = RegistroCalidad()
    datos = normalizar_campos(exigir_objeto(payload), ESQUEMA_OBSERVACION, "", registro)

    tienda = limpiar_tienda(exigir_campo(datos, "tienda_id"), catalogo, registro)
    fecha = limpiar_fecha(exigir_campo(datos, "fecha"), tienda, registro)
    items = normalizar_lista(exigir_campo(datos, "ventas"), "ventas", "ventas", registro)

    ventas: dict[str, int | None] = {}
    origen: dict[str, int] = {}
    for i, item in enumerate(items):
        ruta = f"ventas[{i}]"
        if not isinstance(item, dict):
            registro.advertir(ruta, f"elemento {describir(item)} no es un objeto y se ignoró")
            continue
        item = normalizar_campos(item, ESQUEMA_OBSERVACION_ITEM, ruta, registro)

        resuelta = catalogo.resolver_familia(item.get("familia_id"))
        if resuelta is None:
            registro.advertir(f"{ruta}.familia_id",
                              f"familia {describir(item.get('familia_id'))} no reconocida; "
                              "el registro no se guardó")
            continue
        familia_id, accion, detalle = resuelta
        registro.registrar(f"{ruta}.familia_id", accion, detalle)

        campo_venta = f"{ruta}.ventas"
        if "ventas" not in item:
            registro.advertir(campo_venta, "falta el campo; el registro no se guardó")
            valor = _DESCARTADO
        else:
            valor = limpiar_venta(item["ventas"], campo_venta, registro)

        if familia_id in ventas:
            j = origen[familia_id]
            if ventas[familia_id] != valor:
                raise SolicitudRechazada(
                    f"{ruta}.familia_id",
                    f"la familia '{familia_id}' aparece repetida con valores distintos "
                    f"(ventas[{j}].ventas={ventas[familia_id]}, {campo_venta}={valor}); "
                    "no es posible decidir cuál es el correcto",
                )
            registro.reparar(f"{ruta}.familia_id",
                             f"la familia '{familia_id}' repetía ventas[{j}] con el mismo valor; se eliminó")
            continue
        ventas[familia_id] = valor
        origen[familia_id] = i

    validas = {f: v for f, v in ventas.items() if v is not None}
    if not validas:
        raise SolicitudRechazada("ventas", "ningún registro de ventas es válido; no se guardó nada")

    faltantes = [f for f in catalogo.familia_ids if f not in ventas]
    if faltantes:
        registro.advertir("ventas", "no se recibieron ventas de: " + ", ".join(faltantes)
                          + "; no se registra nada para esas familias")

    try:
        solicitud = SolicitudObservacion.model_validate({
            "tienda_id": tienda.tienda_id,
            "fecha": fecha,
            "ventas": [{"familia_id": f, "ventas": v} for f, v in validas.items()],
        })
    except ValidationError as exc:
        raise SolicitudRechazada(None, f"la solicitud no cumple el contrato: {exc.errors()[0]['msg']}") from exc
    return solicitud, registro
