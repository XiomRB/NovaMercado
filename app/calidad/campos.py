"""Normalización de nombres de campo (E07 y generalizaciones).

Política, para cada clave recibida:

* igual al nombre del contrato                         -> sin acción
* difiere solo en mayúsculas/tildes/espacios/camelCase   -> reparar
* es un sinónimo conocido (``tienda`` -> ``tienda_id``)  -> reparar
* parecido claro a un campo del contrato (errata)        -> advertir
* no se reconoce                                         -> advertir: se ignora
* dos claves que apuntan al mismo campo:
    - con el mismo valor  -> reparar (se descarta la repetida)
    - con valores distintos -> rechazo (no se sabe cuál es el correcto)
"""

from __future__ import annotations

from app.calidad.resultado import SolicitudRechazada, RegistroCalidad
from app.calidad.texto import mejor_coincidencia, normalizar_clave

# Solo sinónimos de significado inequívoco (ver DECISIONES.md §3). Se excluyen
# a propósito nombres genéricos como "id", "dia", "cantidad", "unidades",
# "categoria" o contenedores como "items"/"detalle": podrían referirse a otro dato.
SINONIMOS_TIENDA = {"tienda", "id_tienda", "tienda_no", "num_tienda", "numero_tienda",
                    "codigo_tienda", "cod_tienda", "store", "store_id"}
SINONIMOS_FECHA = {"date", "fecha_venta", "fecha_prediccion", "fecha_pronostico",
                   "fecha_observacion"}
SINONIMOS_FAMILIA = {"familia", "id_familia", "cod_familia", "codigo_familia", "family",
                     "family_id"}

ESQUEMA_PREDICCION = {
    "tienda_id": SINONIMOS_TIENDA,
    "fecha": SINONIMOS_FECHA,
    "familias": {"familia", "lista_familias", "families"},
}
ESQUEMA_PREDICCION_ITEM = {
    "familia_id": SINONIMOS_FAMILIA,
    "en_promocion": {"promocion", "promociones", "en_promo", "promo", "productos_en_promocion",
                     "num_promocion", "onpromotion", "on_promotion", "promotion"},
}
ESQUEMA_OBSERVACION = {
    "tienda_id": SINONIMOS_TIENDA,
    "fecha": SINONIMOS_FECHA,
    "ventas": {"observaciones", "ventas_reales", "sales"},
}
ESQUEMA_OBSERVACION_ITEM = {
    "familia_id": SINONIMOS_FAMILIA,
    "ventas": {"venta", "ventas_reales", "venta_real", "unidades_vendidas", "sales"},
}


def _indice(esquema: dict[str, set[str]]) -> tuple[dict[str, str], dict[str, str]]:
    exactos = {normalizar_clave(c): c for c in esquema}
    sinonimos = {normalizar_clave(s): c for c, ss in esquema.items() for s in ss}
    return exactos, sinonimos


def normalizar_campos(
    objeto: dict, esquema: dict[str, set[str]], ruta: str, registro: RegistroCalidad
) -> dict:
    """Devuelve un dict con las claves del contrato. Las ausentes no se incluyen."""
    exactos, sinonimos = _indice(esquema)
    resultado: dict = {}
    origen: dict[str, str] = {}
    prefijo = f"{ruta}." if ruta else ""

    for clave, valor in objeto.items():
        destino, accion, detalle = None, None, None
        if not isinstance(clave, str):
            clave = str(clave)
        if clave in esquema:
            destino = clave
        else:
            norm = normalizar_clave(clave)
            if norm in exactos:
                destino, accion = exactos[norm], "reparar"
                detalle = f"el campo '{clave}' se interpretó como '{exactos[norm]}'"
            elif norm in sinonimos:
                destino, accion = sinonimos[norm], "reparar"
                detalle = f"el campo '{clave}' se interpretó como '{sinonimos[norm]}'"
            else:
                aprox = mejor_coincidencia(norm, {**{k: v for k, v in exactos.items()},
                                                  **sinonimos}, umbral=0.85)
                if aprox is not None:
                    destino, accion = aprox, "advertir"
                    detalle = (f"el campo '{clave}' no pertenece al contrato; se asumió "
                               f"'{aprox}' por similitud")
        if destino is None:
            registro.advertir(f"{prefijo}{clave}",
                              f"el campo '{clave}' no pertenece al contrato y se ignoró")
            continue
        if destino in resultado:
            if resultado[destino] != valor:
                raise SolicitudRechazada(
                    f"{prefijo}{destino}",
                    f"se recibió dos veces con valores distintos ('{origen[destino]}' y "
                    f"'{clave}'); no es posible decidir cuál es el correcto",
                )
            registro.reparar(f"{prefijo}{destino}",
                             f"el campo '{clave}' repetía el valor de '{origen[destino]}' y se descartó")
            continue
        resultado[destino] = valor
        origen[destino] = clave
        registro.registrar(f"{prefijo}{destino}", accion, detalle)
    return resultado
