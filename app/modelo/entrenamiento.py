"""Entrenamiento con los datos de la base del producto.

Es la función que usa ``entrenar_local`` (etapa 4) y que usará el DAG de
reentrenamiento (etapa 7): ventas = histórico + observaciones (manda el
histórico); promociones = histórico + última solicitud real (las supuestas no
se usan); corte = día siguiente a la última venta conocida.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import joblib
import pandas as pd
from sqlalchemy.engine import Engine

from app.db.conexion import transaccion
from app.db.repositorio import leer_datos_base, leer_observaciones, leer_promociones_solicitudes
from app.modelo.proveedores import FORMATO_ARTEFACTO, NOMBRE_MODELO, versiones_librerias
from ml.datos import consolidar_promociones, consolidar_ventas
from ml.pronosticador import ConfigEntrenamiento, Pronosticador

# Configuración aprobada en la etapa 2 (DECISIONES.md §6–§7)
CONFIG_APROBADA = ConfigEntrenamiento(modelo="hgb_relativo", estrategia="mult_familia")


@dataclass(frozen=True)
class ResultadoEntrenamiento:
    pronosticador: Pronosticador
    version: str
    artefacto: bytes


def entrenar_desde_bd(motor: Engine, config: ConfigEntrenamiento = CONFIG_APROBADA) -> ResultadoEntrenamiento:
    with transaccion(motor) as conn:
        base = leer_datos_base(conn)
        observaciones = leer_observaciones(conn)
        solicitudes = leer_promociones_solicitudes(conn)
    if base.historico.empty:
        raise RuntimeError("la base no tiene datos publicados; ejecute python -m app.db.carga --raw raw")
    ventas = consolidar_ventas(base.historico, observaciones)
    promociones = consolidar_promociones(base.historico, solicitudes)
    corte = ventas["fecha"].max() + pd.Timedelta(days=1)
    p = Pronosticador.entrenar(ventas, promociones, base.tiendas, base.feriados, base.costos,
                               list(base.familias["familia_id"]), corte, config)
    return empaquetar(p)


def empaquetar(p: Pronosticador, nombre: str = NOMBRE_MODELO) -> ResultadoEntrenamiento:
    """Serializa el pronosticador. La versión depende del corte de los datos y del
    contenido del modelo (no del reloj): ``local-AAAAMMDD-<sha256[:12]>``."""
    buf = io.BytesIO()
    joblib.dump(p, buf)
    sha = hashlib.sha256(buf.getvalue()).hexdigest()[:12]
    version = f"local-{pd.Timestamp(p.info['corte']).strftime('%Y%m%d')}-{sha}"
    salida = io.BytesIO()
    joblib.dump({"formato": FORMATO_ARTEFACTO, "nombre": nombre, "version": version,
                 "librerias": versiones_librerias(), "pronosticador": p}, salida)
    return ResultadoEntrenamiento(p, version, salida.getvalue())
