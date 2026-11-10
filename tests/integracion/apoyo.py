"""Datos y atajos compartidos por las pruebas de integración (sin datos reales)."""

from __future__ import annotations

from app.calidad import limpiar_solicitud_observacion, limpiar_solicitud_prediccion
from app.db.carga import cargar_datos_publicados
from app.db.conexion import transaccion
from app.db.repositorio import PrediccionFamilia, cargar_catalogo, registrar_observacion, registrar_prediccion
from tests.ml.sinteticos import FAMILIAS, escribir_carpeta, generar

FECHA = "2025-06-20"          # posterior al histórico sintético de 40 días
DATOS = generar(dias=40)


def cargar_sinteticos(motor, carpeta):
    escribir_carpeta(DATOS, carpeta)
    return cargar_datos_publicados(motor, carpeta)


def payload_prediccion(tienda=1, fecha=FECHA, promos=None, quitar=()):
    promos = promos or {"fam_a": 2, "fam_b": 0, "fam_c": 1}
    return {"tienda_id": tienda, "fecha": fecha,
            "familias": [{"familia_id": f, "en_promocion": promos[f]} for f in FAMILIAS if f not in quitar]}


def payload_observacion(tienda=1, fecha=FECHA, ventas=None):
    ventas = ventas or {"fam_a": 300, "fam_b": 150, "fam_c": 60}
    return {"tienda_id": tienda, "fecha": fecha,
            "ventas": [{"familia_id": f, "ventas": v} for f, v in ventas.items()]}


def predecir_y_registrar(motor, payload, version="1", nombre="novamercado-ventas"):
    """Camino completo: limpieza -> predicción ficticia de prueba -> registro."""
    with transaccion(motor) as conn:
        catalogo = cargar_catalogo(conn)
        solicitud, registro = limpiar_solicitud_prediccion(payload, catalogo)
        predicciones = [PrediccionFamilia(f.familia_id, 10.0 + i, 12 + i) for i, f in enumerate(solicitud.familias)]
        r = registrar_prediccion(conn, payload=payload, solicitud=solicitud, registro=registro,
                                 predicciones=predicciones, nombre_modelo=nombre, version_modelo=version)
    return r, solicitud, registro


def observar(motor, payload):
    with transaccion(motor) as conn:
        catalogo = cargar_catalogo(conn)
        solicitud, registro = limpiar_solicitud_observacion(payload, catalogo)
        return registrar_observacion(conn, payload=payload, solicitud=solicitud, registro=registro)
