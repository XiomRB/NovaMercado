"""Esquema inicial de la base del producto (DECISIONES.md §11).

Tablas generadas con ``alembic revision --autogenerate`` desde ``app/db/esquema.py``
y revisadas; se agregan a mano los triggers de inmutabilidad, los que impiden
colgar familias, predicciones u observaciones de registros rechazados, y la
vista ``v_predicciones_reales``, que Alembic no detecta.

Revision ID: 0001
Revises:
Create Date: 2026-11-10 04:28:17.724499+00:00
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db.esquema import TABLAS_SIN_BORRADO, TABLAS_SOLO_INSERCION

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('cargas_datos',
    sa.Column('carga_id', sa.Integer(), sa.Identity(always=True), nullable=False),
    sa.Column('archivo', sa.Text(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('filas', sa.Integer(), nullable=False),
    sa.Column('registrado_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('filas >= 0', name=op.f('ck_cargas_datos_filas_no_negativas')),
    sa.PrimaryKeyConstraint('carga_id', name=op.f('pk_cargas_datos')),
    sa.UniqueConstraint('archivo', name=op.f('uq_cargas_datos_archivo'))
    )
    op.create_table('familias',
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('nombre', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('familia_id', name=op.f('pk_familias'))
    )
    op.create_table('feriados',
    sa.Column('feriado_id', sa.Integer(), sa.Identity(always=True), nullable=False),
    sa.Column('fecha', sa.Date(), nullable=False),
    sa.Column('nombre', sa.Text(), nullable=False),
    sa.Column('tipo', sa.Text(), nullable=False),
    sa.Column('ambito', sa.Text(), nullable=False),
    sa.CheckConstraint("tipo IN ('feriado', 'comercial', 'pago', 'medio_dia')", name=op.f('ck_feriados_tipo_valido')),
    sa.PrimaryKeyConstraint('feriado_id', name=op.f('pk_feriados')),
    sa.UniqueConstraint('fecha', 'nombre', 'ambito', name=op.f('uq_feriados_fecha_nombre_ambito'))
    )
    op.create_table('tiendas',
    sa.Column('tienda_id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('nombre', sa.Text(), nullable=False),
    sa.Column('ciudad', sa.Text(), nullable=False),
    sa.Column('departamento', sa.Text(), nullable=False),
    sa.Column('tipo', sa.Text(), nullable=False),
    sa.Column('fecha_apertura', sa.Date(), nullable=False),
    sa.CheckConstraint('tienda_id > 0', name=op.f('ck_tiendas_id_positivo')),
    sa.PrimaryKeyConstraint('tienda_id', name=op.f('pk_tiendas'))
    )
    op.create_table('costos',
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('costo_faltante', sa.Numeric(precision=12, scale=6), nullable=False),
    sa.Column('costo_sobrante', sa.Numeric(precision=12, scale=6), nullable=False),
    sa.CheckConstraint('costo_faltante > 0', name=op.f('ck_costos_faltante_positivo')),
    sa.CheckConstraint('costo_sobrante > 0', name=op.f('ck_costos_sobrante_positivo')),
    sa.ForeignKeyConstraint(['familia_id'], ['familias.familia_id'], name=op.f('fk_costos_familia_id')),
    sa.PrimaryKeyConstraint('familia_id', name=op.f('pk_costos'))
    )
    op.create_table('envios_observacion',
    sa.Column('envio_id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('registrado_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('resultado', sa.Text(), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('tienda_id', sa.Integer(), nullable=True),
    sa.Column('fecha', sa.Date(), nullable=True),
    sa.Column('estado_calidad', sa.Text(), nullable=True),
    sa.Column('registros_nuevos', sa.Integer(), nullable=True),
    sa.Column('registros_modificados', sa.Integer(), nullable=True),
    sa.Column('detalle_rechazo', sa.Text(), nullable=True),
    sa.CheckConstraint("(resultado = 'aceptado' AND tienda_id IS NOT NULL AND fecha IS NOT NULL AND estado_calidad IS NOT NULL AND registros_nuevos >= 0 AND registros_modificados >= 0 AND detalle_rechazo IS NULL) OR (resultado = 'rechazado' AND detalle_rechazo IS NOT NULL AND estado_calidad IS NULL AND registros_nuevos IS NULL AND registros_modificados IS NULL)", name=op.f('ck_envios_observacion_coherencia_resultado')),
    sa.CheckConstraint("estado_calidad IS NULL OR estado_calidad IN ('ok', 'corregido', 'advertencia')", name=op.f('ck_envios_observacion_estado_calidad_valido')),
    sa.CheckConstraint("resultado IN ('aceptado', 'rechazado')", name=op.f('ck_envios_observacion_resultado_valido')),
    sa.ForeignKeyConstraint(['tienda_id'], ['tiendas.tienda_id'], name=op.f('fk_envios_observacion_tienda_id')),
    sa.PrimaryKeyConstraint('envio_id', name=op.f('pk_envios_observacion'))
    )
    op.create_table('solicitudes_prediccion',
    sa.Column('solicitud_id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('registrado_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('resultado', sa.Text(), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('tienda_id', sa.Integer(), nullable=True),
    sa.Column('fecha', sa.Date(), nullable=True),
    sa.Column('nombre_modelo', sa.Text(), nullable=True),
    sa.Column('version_modelo', sa.Text(), nullable=True),
    sa.Column('estado_calidad', sa.Text(), nullable=True),
    sa.Column('detalle_rechazo', sa.Text(), nullable=True),
    sa.CheckConstraint("(resultado = 'respondida' AND tienda_id IS NOT NULL AND fecha IS NOT NULL AND nombre_modelo IS NOT NULL AND version_modelo IS NOT NULL AND estado_calidad IS NOT NULL AND detalle_rechazo IS NULL) OR (resultado = 'rechazada' AND detalle_rechazo IS NOT NULL AND version_modelo IS NULL AND estado_calidad IS NULL)", name=op.f('ck_solicitudes_prediccion_coherencia_resultado')),
    sa.CheckConstraint("estado_calidad IS NULL OR estado_calidad IN ('ok', 'corregido', 'advertencia')", name=op.f('ck_solicitudes_prediccion_estado_calidad_valido')),
    sa.CheckConstraint("resultado IN ('respondida', 'rechazada')", name=op.f('ck_solicitudes_prediccion_resultado_valido')),
    sa.ForeignKeyConstraint(['tienda_id'], ['tiendas.tienda_id'], name=op.f('fk_solicitudes_prediccion_tienda_id')),
    sa.PrimaryKeyConstraint('solicitud_id', name=op.f('pk_solicitudes_prediccion'))
    )
    op.create_index('ix_solicitudes_prediccion_orden', 'solicitudes_prediccion', ['tienda_id', 'fecha', 'registrado_en', 'solicitud_id'], unique=False)
    op.create_table('ventas_historicas',
    sa.Column('fecha', sa.Date(), nullable=False),
    sa.Column('tienda_id', sa.Integer(), nullable=False),
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('ventas', sa.Integer(), nullable=False),
    sa.Column('en_promocion', sa.Integer(), nullable=False),
    sa.Column('carga_id', sa.Integer(), nullable=False),
    sa.CheckConstraint('en_promocion >= 0', name=op.f('ck_ventas_historicas_promocion_no_negativa')),
    sa.CheckConstraint('ventas >= 0', name=op.f('ck_ventas_historicas_ventas_no_negativas')),
    sa.ForeignKeyConstraint(['carga_id'], ['cargas_datos.carga_id'], name=op.f('fk_ventas_historicas_carga_id')),
    sa.ForeignKeyConstraint(['familia_id'], ['familias.familia_id'], name=op.f('fk_ventas_historicas_familia_id')),
    sa.ForeignKeyConstraint(['tienda_id'], ['tiendas.tienda_id'], name=op.f('fk_ventas_historicas_tienda_id')),
    sa.PrimaryKeyConstraint('fecha', 'tienda_id', 'familia_id', name=op.f('pk_ventas_historicas'))
    )
    op.create_table('calidad_observaciones',
    sa.Column('calidad_id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('solicitud_id', sa.BigInteger(), nullable=True),
    sa.Column('envio_id', sa.BigInteger(), nullable=True),
    sa.Column('orden', sa.Integer(), nullable=False),
    sa.Column('campo', sa.Text(), nullable=False),
    sa.Column('accion', sa.Text(), nullable=False),
    sa.Column('detalle', sa.Text(), nullable=False),
    sa.CheckConstraint("accion IN ('reparar', 'advertir')", name=op.f('ck_calidad_observaciones_accion_valida')),
    sa.CheckConstraint('(solicitud_id IS NULL) <> (envio_id IS NULL)', name=op.f('ck_calidad_observaciones_un_solo_origen')),
    sa.CheckConstraint('orden >= 0', name=op.f('ck_calidad_observaciones_orden_no_negativo')),
    sa.ForeignKeyConstraint(['envio_id'], ['envios_observacion.envio_id'], name=op.f('fk_calidad_observaciones_envio_id')),
    sa.ForeignKeyConstraint(['solicitud_id'], ['solicitudes_prediccion.solicitud_id'], name=op.f('fk_calidad_observaciones_solicitud_id')),
    sa.PrimaryKeyConstraint('calidad_id', name=op.f('pk_calidad_observaciones')),
    sa.UniqueConstraint('envio_id', 'orden', name=op.f('uq_calidad_observaciones_envio_id_orden')),
    sa.UniqueConstraint('solicitud_id', 'orden', name=op.f('uq_calidad_observaciones_solicitud_id_orden'))
    )
    op.create_table('observaciones',
    sa.Column('fecha', sa.Date(), nullable=False),
    sa.Column('tienda_id', sa.Integer(), nullable=False),
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('ventas', sa.Integer(), nullable=False),
    sa.Column('envio_id_creacion', sa.BigInteger(), nullable=False),
    sa.Column('envio_id_modificacion', sa.BigInteger(), nullable=True),
    sa.Column('creado_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('modificado_en', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('(envio_id_modificacion IS NULL) = (modificado_en IS NULL)', name=op.f('ck_observaciones_modificacion_coherente')),
    sa.CheckConstraint('ventas >= 0', name=op.f('ck_observaciones_ventas_no_negativas')),
    sa.ForeignKeyConstraint(['envio_id_creacion'], ['envios_observacion.envio_id'], name=op.f('fk_observaciones_envio_id_creacion')),
    sa.ForeignKeyConstraint(['envio_id_modificacion'], ['envios_observacion.envio_id'], name=op.f('fk_observaciones_envio_id_modificacion')),
    sa.ForeignKeyConstraint(['familia_id'], ['familias.familia_id'], name=op.f('fk_observaciones_familia_id')),
    sa.ForeignKeyConstraint(['tienda_id'], ['tiendas.tienda_id'], name=op.f('fk_observaciones_tienda_id')),
    sa.PrimaryKeyConstraint('fecha', 'tienda_id', 'familia_id', name=op.f('pk_observaciones'))
    )
    op.create_table('solicitud_familias',
    sa.Column('solicitud_id', sa.BigInteger(), nullable=False),
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('en_promocion', sa.Integer(), nullable=False),
    sa.Column('promocion_supuesta', sa.Boolean(), nullable=False),
    sa.CheckConstraint('en_promocion >= 0', name=op.f('ck_solicitud_familias_promocion_no_negativa')),
    sa.ForeignKeyConstraint(['familia_id'], ['familias.familia_id'], name=op.f('fk_solicitud_familias_familia_id')),
    sa.ForeignKeyConstraint(['solicitud_id'], ['solicitudes_prediccion.solicitud_id'], name=op.f('fk_solicitud_familias_solicitud_id')),
    sa.PrimaryKeyConstraint('solicitud_id', 'familia_id', name=op.f('pk_solicitud_familias'))
    )
    op.create_table('observaciones_historial',
    sa.Column('historial_id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('fecha', sa.Date(), nullable=False),
    sa.Column('tienda_id', sa.Integer(), nullable=False),
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('ventas_anterior', sa.Integer(), nullable=False),
    sa.Column('ventas_nueva', sa.Integer(), nullable=False),
    sa.Column('envio_id', sa.BigInteger(), nullable=False),
    sa.Column('registrado_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ventas_anterior <> ventas_nueva', name=op.f('ck_observaciones_historial_cambio_real')),
    sa.ForeignKeyConstraint(['envio_id'], ['envios_observacion.envio_id'], name=op.f('fk_observaciones_historial_envio_id')),
    sa.ForeignKeyConstraint(['fecha', 'tienda_id', 'familia_id'], ['observaciones.fecha', 'observaciones.tienda_id', 'observaciones.familia_id'], name=op.f('fk_observaciones_historial_fecha')),
    sa.PrimaryKeyConstraint('historial_id', name=op.f('pk_observaciones_historial'))
    )
    op.create_index('ix_observaciones_historial_clave', 'observaciones_historial', ['fecha', 'tienda_id', 'familia_id'], unique=False)
    op.create_table('predicciones',
    sa.Column('solicitud_id', sa.BigInteger(), nullable=False),
    sa.Column('familia_id', sa.Text(), nullable=False),
    sa.Column('venta_esperada', sa.Float(), nullable=False),
    sa.Column('cantidad_sugerida', sa.Integer(), nullable=False),
    sa.CheckConstraint('cantidad_sugerida >= 0', name=op.f('ck_predicciones_cantidad_no_negativa')),
    sa.CheckConstraint('venta_esperada >= 0', name=op.f('ck_predicciones_venta_no_negativa')),
    sa.ForeignKeyConstraint(['solicitud_id', 'familia_id'], ['solicitud_familias.solicitud_id', 'solicitud_familias.familia_id'], name=op.f('fk_predicciones_solicitud_id')),
    sa.PrimaryKeyConstraint('solicitud_id', 'familia_id', name=op.f('pk_predicciones'))
    )

    _crear_inmutabilidad()
    _crear_coherencia_con_rechazos()
    op.execute(VISTA)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_predicciones_reales")
    op.drop_table('predicciones')
    op.drop_index('ix_observaciones_historial_clave', table_name='observaciones_historial')
    op.drop_table('observaciones_historial')
    op.drop_table('solicitud_familias')
    op.drop_table('observaciones')
    op.drop_table('calidad_observaciones')
    op.drop_table('ventas_historicas')
    op.drop_index('ix_solicitudes_prediccion_orden', table_name='solicitudes_prediccion')
    op.drop_table('solicitudes_prediccion')
    op.drop_table('envios_observacion')
    op.drop_table('costos')
    op.drop_table('tiendas')
    op.drop_table('feriados')
    op.drop_table('familias')
    op.drop_table('cargas_datos')
    op.execute("DROP FUNCTION IF EXISTS impedir_modificacion()")
    op.execute("DROP FUNCTION IF EXISTS exigir_solicitud_respondida()")
    op.execute("DROP FUNCTION IF EXISTS exigir_envio_aceptado()")


FUNCION = """
CREATE FUNCTION impedir_modificacion() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'la tabla % no admite %: los registros históricos no se sobrescriben',
        TG_TABLE_NAME, TG_OP USING ERRCODE = 'restrict_violation';
END
$$;
"""

# Una solicitud rechazada no tiene familias ni predicciones (predicciones depende de
# solicitud_familias por FK), y un envío rechazado no crea ni modifica
# observaciones. Basta verificarlo al insertar: el resultado del padre es inmutable.
COHERENCIA = """
CREATE FUNCTION exigir_solicitud_respondida() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (SELECT resultado FROM solicitudes_prediccion WHERE solicitud_id = NEW.solicitud_id)
       IS DISTINCT FROM 'respondida' THEN
        RAISE EXCEPTION 'la solicitud % no fue respondida: no admite familias ni predicciones',
            NEW.solicitud_id USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE FUNCTION exigir_envio_aceptado() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    envios bigint[];
    envio bigint;
BEGIN
    -- Asignaciones separadas: PL/pgSQL solo resuelve NEW.<campo> de la rama que ejecuta
    IF TG_TABLE_NAME = 'observaciones' THEN
        envios := ARRAY[NEW.envio_id_creacion, NEW.envio_id_modificacion];
    ELSE
        envios := ARRAY[NEW.envio_id];
    END IF;
    FOREACH envio IN ARRAY envios LOOP
        IF envio IS NOT NULL AND (SELECT resultado FROM envios_observacion WHERE envio_id = envio)
           IS DISTINCT FROM 'aceptado' THEN
            RAISE EXCEPTION 'el envío % no fue aceptado: no puede crear ni modificar observaciones',
                envio USING ERRCODE = 'check_violation';
        END IF;
    END LOOP;
    RETURN NEW;
END
$$;

CREATE TRIGGER trg_solicitud_familias_respondida BEFORE INSERT ON solicitud_familias
    FOR EACH ROW EXECUTE FUNCTION exigir_solicitud_respondida();
CREATE TRIGGER trg_observaciones_envio_aceptado BEFORE INSERT OR UPDATE ON observaciones
    FOR EACH ROW EXECUTE FUNCTION exigir_envio_aceptado();
CREATE TRIGGER trg_observaciones_historial_envio_aceptado BEFORE INSERT ON observaciones_historial
    FOR EACH ROW EXECUTE FUNCTION exigir_envio_aceptado();
"""


def _crear_coherencia_con_rechazos() -> None:
    op.execute(COHERENCIA)


VISTA = """
CREATE VIEW v_predicciones_reales AS
SELECT s.solicitud_id, s.registrado_en, s.fecha, s.tienda_id, p.familia_id,
       s.nombre_modelo, s.version_modelo, sf.en_promocion, sf.promocion_supuesta,
       p.venta_esperada, p.cantidad_sugerida,
       o.ventas AS ventas_observadas, vh.ventas AS ventas_historicas,
       COALESCE(vh.ventas, o.ventas) AS ventas_reales
FROM predicciones p
JOIN solicitudes_prediccion s ON s.solicitud_id = p.solicitud_id
JOIN solicitud_familias sf ON sf.solicitud_id = p.solicitud_id AND sf.familia_id = p.familia_id
LEFT JOIN observaciones o
       ON o.fecha = s.fecha AND o.tienda_id = s.tienda_id AND o.familia_id = p.familia_id
LEFT JOIN ventas_historicas vh
       ON vh.fecha = s.fecha AND vh.tienda_id = s.tienda_id AND vh.familia_id = p.familia_id;
"""


def _crear_inmutabilidad() -> None:
    op.execute(FUNCION)
    for tabla in TABLAS_SOLO_INSERCION:
        op.execute(f"CREATE TRIGGER trg_{tabla}_inmutable BEFORE UPDATE OR DELETE ON {tabla} "
                   "FOR EACH ROW EXECUTE FUNCTION impedir_modificacion()")
        op.execute(f"CREATE TRIGGER trg_{tabla}_sin_truncate BEFORE TRUNCATE ON {tabla} "
                   "FOR EACH STATEMENT EXECUTE FUNCTION impedir_modificacion()")
    for tabla in TABLAS_SIN_BORRADO:
        op.execute(f"CREATE TRIGGER trg_{tabla}_sin_borrado BEFORE DELETE ON {tabla} "
                   "FOR EACH ROW EXECUTE FUNCTION impedir_modificacion()")
        op.execute(f"CREATE TRIGGER trg_{tabla}_sin_truncate BEFORE TRUNCATE ON {tabla} "
                   "FOR EACH STATEMENT EXECUTE FUNCTION impedir_modificacion()")
