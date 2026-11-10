"""Esquema de la base del producto (SQLAlchemy Core). Diseño: DECISIONES.md §11.

Es la referencia de ``migraciones/``: una prueba verifica que el esquema
migrado coincide con estas tablas. Los triggers de inmutabilidad y la vista se
definen en la migración (no son objetos de ``MetaData``).
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

CONVENCION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=CONVENCION)

TIPOS_EVENTO = ("feriado", "comercial", "pago", "medio_dia")
ESTADOS_CALIDAD = ("ok", "corregido", "advertencia")


def _en(columna: str, valores: tuple[str, ...]) -> str:
    return f"{columna} IN (" + ", ".join(f"'{v}'" for v in valores) + ")"


def _registrado_en() -> Column:
    return Column("registrado_en", DateTime(timezone=True), nullable=False, server_default=func.now())


# ------------------------------------------------------------- datos publicados
cargas_datos = Table(
    "cargas_datos", metadata,
    Column("carga_id", Integer, Identity(always=True), primary_key=True),
    Column("archivo", Text, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("filas", Integer, nullable=False),
    _registrado_en(),
    UniqueConstraint("archivo"),
    CheckConstraint("filas >= 0", name="filas_no_negativas"),
)

tiendas = Table(
    "tiendas", metadata,
    Column("tienda_id", Integer, primary_key=True, autoincrement=False),
    Column("nombre", Text, nullable=False),
    Column("ciudad", Text, nullable=False),
    Column("departamento", Text, nullable=False),
    Column("tipo", Text, nullable=False),
    Column("fecha_apertura", Date, nullable=False),
    CheckConstraint("tienda_id > 0", name="id_positivo"),
)

familias = Table(
    "familias", metadata,
    Column("familia_id", Text, primary_key=True),
    Column("nombre", Text, nullable=False),
)

costos = Table(
    "costos", metadata,
    Column("familia_id", Text, ForeignKey("familias.familia_id"), primary_key=True),
    Column("costo_faltante", Numeric(12, 6), nullable=False),
    Column("costo_sobrante", Numeric(12, 6), nullable=False),
    CheckConstraint("costo_faltante > 0", name="faltante_positivo"),
    CheckConstraint("costo_sobrante > 0", name="sobrante_positivo"),
)

feriados = Table(
    "feriados", metadata,
    Column("feriado_id", Integer, Identity(always=True), primary_key=True),
    Column("fecha", Date, nullable=False),
    Column("nombre", Text, nullable=False),
    Column("tipo", Text, nullable=False),
    Column("ambito", Text, nullable=False),
    UniqueConstraint("fecha", "nombre", "ambito"),
    CheckConstraint(_en("tipo", TIPOS_EVENTO), name="tipo_valido"),
)

ventas_historicas = Table(
    "ventas_historicas", metadata,
    Column("fecha", Date, primary_key=True),
    Column("tienda_id", Integer, ForeignKey("tiendas.tienda_id"), primary_key=True),
    Column("familia_id", Text, ForeignKey("familias.familia_id"), primary_key=True),
    Column("ventas", Integer, nullable=False),
    Column("en_promocion", Integer, nullable=False),
    Column("carga_id", Integer, ForeignKey("cargas_datos.carga_id"), nullable=False),
    CheckConstraint("ventas >= 0", name="ventas_no_negativas"),
    CheckConstraint("en_promocion >= 0", name="promocion_no_negativa"),
)

# ----------------------------------------------------------------- predicciones
solicitudes_prediccion = Table(
    "solicitudes_prediccion", metadata,
    Column("solicitud_id", BigInteger, Identity(always=True), primary_key=True),
    _registrado_en(),
    Column("resultado", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("tienda_id", Integer, ForeignKey("tiendas.tienda_id"), nullable=True),
    Column("fecha", Date, nullable=True),
    Column("nombre_modelo", Text, nullable=True),
    Column("version_modelo", Text, nullable=True),
    Column("estado_calidad", Text, nullable=True),
    Column("detalle_rechazo", Text, nullable=True),
    CheckConstraint(_en("resultado", ("respondida", "rechazada")), name="resultado_valido"),
    CheckConstraint(f"estado_calidad IS NULL OR {_en('estado_calidad', ESTADOS_CALIDAD)}",
                    name="estado_calidad_valido"),
    CheckConstraint(
        "(resultado = 'respondida' AND tienda_id IS NOT NULL AND fecha IS NOT NULL"
        " AND nombre_modelo IS NOT NULL AND version_modelo IS NOT NULL AND estado_calidad IS NOT NULL"
        " AND detalle_rechazo IS NULL)"
        " OR (resultado = 'rechazada' AND detalle_rechazo IS NOT NULL"
        " AND version_modelo IS NULL AND estado_calidad IS NULL)",
        name="coherencia_resultado",
    ),
    Index("ix_solicitudes_prediccion_orden", "tienda_id", "fecha", "registrado_en", "solicitud_id"),
)

solicitud_familias = Table(
    "solicitud_familias", metadata,
    Column("solicitud_id", BigInteger, ForeignKey("solicitudes_prediccion.solicitud_id"), primary_key=True),
    Column("familia_id", Text, ForeignKey("familias.familia_id"), primary_key=True),
    Column("en_promocion", Integer, nullable=False),
    Column("promocion_supuesta", Boolean, nullable=False),
    CheckConstraint("en_promocion >= 0", name="promocion_no_negativa"),
)

predicciones = Table(
    "predicciones", metadata,
    Column("solicitud_id", BigInteger, primary_key=True),
    Column("familia_id", Text, primary_key=True),
    Column("venta_esperada", Float, nullable=False),
    Column("cantidad_sugerida", Integer, nullable=False),
    ForeignKeyConstraint(["solicitud_id", "familia_id"],
                         ["solicitud_familias.solicitud_id", "solicitud_familias.familia_id"]),
    CheckConstraint("venta_esperada >= 0", name="venta_no_negativa"),
    CheckConstraint("cantidad_sugerida >= 0", name="cantidad_no_negativa"),
)

# ---------------------------------------------------------------- observaciones
envios_observacion = Table(
    "envios_observacion", metadata,
    Column("envio_id", BigInteger, Identity(always=True), primary_key=True),
    _registrado_en(),
    Column("resultado", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("tienda_id", Integer, ForeignKey("tiendas.tienda_id"), nullable=True),
    Column("fecha", Date, nullable=True),
    Column("estado_calidad", Text, nullable=True),
    Column("registros_nuevos", Integer, nullable=True),
    Column("registros_modificados", Integer, nullable=True),
    Column("detalle_rechazo", Text, nullable=True),
    CheckConstraint(_en("resultado", ("aceptado", "rechazado")), name="resultado_valido"),
    CheckConstraint(f"estado_calidad IS NULL OR {_en('estado_calidad', ESTADOS_CALIDAD)}",
                    name="estado_calidad_valido"),
    CheckConstraint(
        "(resultado = 'aceptado' AND tienda_id IS NOT NULL AND fecha IS NOT NULL AND estado_calidad IS NOT NULL"
        " AND registros_nuevos >= 0 AND registros_modificados >= 0 AND detalle_rechazo IS NULL)"
        " OR (resultado = 'rechazado' AND detalle_rechazo IS NOT NULL AND estado_calidad IS NULL"
        " AND registros_nuevos IS NULL AND registros_modificados IS NULL)",
        name="coherencia_resultado",
    ),
)

observaciones = Table(
    "observaciones", metadata,
    Column("fecha", Date, primary_key=True),
    Column("tienda_id", Integer, ForeignKey("tiendas.tienda_id"), primary_key=True),
    Column("familia_id", Text, ForeignKey("familias.familia_id"), primary_key=True),
    Column("ventas", Integer, nullable=False),
    Column("envio_id_creacion", BigInteger, ForeignKey("envios_observacion.envio_id"), nullable=False),
    Column("envio_id_modificacion", BigInteger, ForeignKey("envios_observacion.envio_id"), nullable=True),
    Column("creado_en", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("modificado_en", DateTime(timezone=True), nullable=True),
    CheckConstraint("ventas >= 0", name="ventas_no_negativas"),
    CheckConstraint("(envio_id_modificacion IS NULL) = (modificado_en IS NULL)", name="modificacion_coherente"),
)

observaciones_historial = Table(
    "observaciones_historial", metadata,
    Column("historial_id", BigInteger, Identity(always=True), primary_key=True),
    Column("fecha", Date, nullable=False),
    Column("tienda_id", Integer, nullable=False),
    Column("familia_id", Text, nullable=False),
    Column("ventas_anterior", Integer, nullable=False),
    Column("ventas_nueva", Integer, nullable=False),
    Column("envio_id", BigInteger, ForeignKey("envios_observacion.envio_id"), nullable=False),
    _registrado_en(),
    ForeignKeyConstraint(["fecha", "tienda_id", "familia_id"],
                         ["observaciones.fecha", "observaciones.tienda_id", "observaciones.familia_id"]),
    CheckConstraint("ventas_anterior <> ventas_nueva", name="cambio_real"),
    Index("ix_observaciones_historial_clave", "fecha", "tienda_id", "familia_id"),
)

# ---------------------------------------------------------------------- calidad
calidad_observaciones = Table(
    "calidad_observaciones", metadata,
    Column("calidad_id", BigInteger, Identity(always=True), primary_key=True),
    Column("solicitud_id", BigInteger, ForeignKey("solicitudes_prediccion.solicitud_id"), nullable=True),
    Column("envio_id", BigInteger, ForeignKey("envios_observacion.envio_id"), nullable=True),
    Column("orden", Integer, nullable=False),
    Column("campo", Text, nullable=False),
    Column("accion", Text, nullable=False),
    Column("detalle", Text, nullable=False),
    CheckConstraint("(solicitud_id IS NULL) <> (envio_id IS NULL)", name="un_solo_origen"),
    CheckConstraint(_en("accion", ("reparar", "advertir")), name="accion_valida"),
    CheckConstraint("orden >= 0", name="orden_no_negativo"),
    UniqueConstraint("solicitud_id", "orden"),
    UniqueConstraint("envio_id", "orden"),
)

TABLAS_SOLO_INSERCION = ("ventas_historicas", "solicitudes_prediccion", "solicitud_familias", "predicciones",
                         "envios_observacion", "observaciones_historial", "calidad_observaciones")
TABLAS_SIN_BORRADO = ("observaciones",)
