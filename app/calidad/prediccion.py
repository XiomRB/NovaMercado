"""Limpieza de solicitudes de ``POST /v1/prediccion``.

Orden: campos (E07) -> tienda (E01) -> fecha (E04) -> familias (E02, E03, E06,
promociones atípicas, E05) -> validación estricta con Pydantic.

Además de la solicitud limpia, el registro informa en
``RegistroCalidad.promociones_supuestas`` las familias cuya promoción no se
recibió de forma utilizable y se supuso: al reentrenar no se tratan como dato real.
"""

from __future__ import annotations

from pydantic import ValidationError

from app.calidad.campos import ESQUEMA_PREDICCION, ESQUEMA_PREDICCION_ITEM, normalizar_campos
from app.calidad.catalogo import Catalogo
from app.calidad.comunes import exigir_campo, exigir_objeto, limpiar_fecha, limpiar_tienda, normalizar_lista
from app.calidad.numeros import NumeroInvalido, describir, interpretar_numero, redondear
from app.calidad.resultado import RegistroCalidad, SolicitudRechazada
from app.schemas.contrato import SolicitudPrediccion

PROMOCION_SUPUESTA = 0


def limpiar_promocion(valor: object, campo: str, registro: RegistroCalidad) -> tuple[int, bool]:
    """E03. Devuelve ``(valor, supuesto)``.

    Entero >= 0 -> ok; '3', 2.0, '1,000' -> reparar. Decimal no entero ->
    advertir y redondear (deriva del dato recibido, no es supuesto). Nulo, no
    interpretable o negativo -> advertir y suponer 0 (``supuesto=True``).
    """
    if valor is None:
        registro.advertir(campo, f"sin valor; se asumió {PROMOCION_SUPUESTA}")
        return PROMOCION_SUPUESTA, True
    try:
        n = interpretar_numero(valor)
    except NumeroInvalido as exc:
        registro.advertir(campo, f"{exc}; se asumió {PROMOCION_SUPUESTA}")
        return PROMOCION_SUPUESTA, True
    if n.valor < 0:
        registro.advertir(campo, f"{describir(valor)} es negativo; se asumió {PROMOCION_SUPUESTA}")
        return PROMOCION_SUPUESTA, True
    entero = redondear(n.valor)
    if not n.es_entero:
        registro.advertir(campo, f"{describir(valor)} no es entero; se redondeó a {entero}")
    elif not isinstance(valor, int):
        motivo = n.formato or "número decimal sin parte fraccionaria"
        registro.reparar(campo, f"{describir(valor)} se interpretó como {entero} ({motivo})")
    return entero, False


def advertir_si_atipica(promo: int, familia_id: str, campo: str,
                        maximos: dict[str, int] | None, registro: RegistroCalidad) -> None:
    """Promoción mayor que el máximo visto al entrenar el modelo vigente -> advertir.

    El valor **se conserva**: es un dato válido, pero el modelo no tiene
    ejemplos con más promociones y responde como si fuera el máximo conocido.
    """
    if not maximos or familia_id not in maximos:
        return
    maximo = int(maximos[familia_id])
    if promo > maximo:
        registro.advertir(
            campo,
            f"en_promocion={promo} supera el máximo observado al entrenar el modelo para "
            f"'{familia_id}' ({maximo}); se conservó el valor, pero el modelo no tiene datos "
            f"con más promociones y lo trata como {maximo}",
        )


def limpiar_solicitud_prediccion(
    payload: object, catalogo: Catalogo, maximos_promocion: dict[str, int] | None = None
) -> tuple[SolicitudPrediccion, RegistroCalidad]:
    """``maximos_promocion``: máximo de ``en_promocion`` por familia en los datos de
    entrenamiento del modelo vigente (``Pronosticador.maximos_promocion``)."""
    registro = RegistroCalidad()
    datos = normalizar_campos(exigir_objeto(payload), ESQUEMA_PREDICCION, "", registro)

    tienda = limpiar_tienda(exigir_campo(datos, "tienda_id"), catalogo, registro)
    fecha = limpiar_fecha(exigir_campo(datos, "fecha"), tienda, registro)
    items = normalizar_lista(exigir_campo(datos, "familias"), "familias", "en_promocion", registro)

    promociones: dict[str, int] = {}
    supuestas: dict[str, bool] = {}
    origen: dict[str, int] = {}
    for i, item in enumerate(items):
        ruta = f"familias[{i}]"
        if not isinstance(item, dict):
            registro.advertir(ruta, f"elemento {describir(item)} no es un objeto y se ignoró")
            continue
        item = normalizar_campos(item, ESQUEMA_PREDICCION_ITEM, ruta, registro)

        resuelta = catalogo.resolver_familia(item.get("familia_id"))
        if resuelta is None:
            registro.advertir(f"{ruta}.familia_id",
                              f"familia {describir(item.get('familia_id'))} no reconocida; se ignoró")
            continue
        familia_id, accion, detalle = resuelta
        registro.registrar(f"{ruta}.familia_id", accion, detalle)

        campo_promo = f"{ruta}.en_promocion"
        if "en_promocion" not in item:
            registro.advertir(campo_promo, f"falta el campo; se asumió {PROMOCION_SUPUESTA}")
            promo, supuesta = PROMOCION_SUPUESTA, True
        else:
            promo, supuesta = limpiar_promocion(item["en_promocion"], campo_promo, registro)

        # E06: repetidas idénticas -> reparar; con valores distintos -> rechazo (D2)
        if familia_id in promociones:
            j = origen[familia_id]
            if promociones[familia_id] != promo:
                raise SolicitudRechazada(
                    f"{ruta}.familia_id",
                    f"la familia '{familia_id}' aparece repetida con valores distintos "
                    f"(familias[{j}].en_promocion={promociones[familia_id]}, "
                    f"{campo_promo}={promo}); no es posible decidir cuál es el correcto",
                )
            registro.reparar(f"{ruta}.familia_id",
                             f"la familia '{familia_id}' repetía familias[{j}] con el mismo valor; se eliminó")
            supuestas[familia_id] = supuestas[familia_id] and supuesta
            continue
        advertir_si_atipica(promo, familia_id, campo_promo, maximos_promocion, registro)
        promociones[familia_id] = promo
        supuestas[familia_id] = supuesta
        origen[familia_id] = i

    if not promociones:
        raise SolicitudRechazada("familias", "ninguna familia de la solicitud es reconocible")

    # E05 (D1): se completan las familias faltantes con el valor supuesto
    for familia_id in catalogo.familia_ids:
        if familia_id not in promociones:
            promociones[familia_id] = PROMOCION_SUPUESTA
            supuestas[familia_id] = True
            registro.advertir("familias",
                              f"no se recibió la familia '{familia_id}'; se pronosticó con "
                              f"en_promocion={PROMOCION_SUPUESTA}")

    try:
        solicitud = SolicitudPrediccion.model_validate({
            "tienda_id": tienda.tienda_id,
            "fecha": fecha,
            "familias": [{"familia_id": f, "en_promocion": p} for f, p in promociones.items()],
        })
    except ValidationError as exc:  # no debería ocurrir: la limpieza ya garantizó los tipos
        raise SolicitudRechazada(None, f"la solicitud no cumple el contrato: {exc.errors()[0]['msg']}") from exc
    registro.promociones_supuestas = {f for f, sup in supuestas.items() if sup}
    return solicitud, registro
