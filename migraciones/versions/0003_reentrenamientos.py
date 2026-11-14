"""Etapa 7: solicitudes, inicios, reclamaciones, resultados y veredictos de los
reentrenamientos (DECISIONES.md §16). Todas de solo inserción (triggers).

Los roles nm_cierre, nm_vigilante y nm_senal, sus concesiones y la función
nm_control.nm_terminar_escritor NO están aquí: llevan claves (que no se versionan)
y son objetos del clúster. Los crea ``python -m app.db.roles`` (servicio preparar).

Las listas de tablas se escriben de forma literal: esta migración no debe cambiar
si cambian las constantes de ``app/db/esquema.py``.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

SOLO_INSERCION = ("solicitudes_reentrenamiento", "tareas_reentrenamiento", "reclamaciones_reentrenamiento",
                  "resultados_reentrenamiento", "veredictos_reentrenamiento")


def upgrade() -> None:
    op.create_table('solicitudes_reentrenamiento',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('clave_idempotencia', sa.Text(), nullable=True),
    sa.Column('huella_solicitud', sa.String(length=64), nullable=True),
    sa.Column('solicitado_en', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
    sa.Column('origen', sa.Text(), nullable=False),
    sa.CheckConstraint("id ~ '^[A-Za-z0-9_.:+-]{1,250}$'", name=op.f('ck_solicitudes_reentrenamiento_id_valido')),
    sa.CheckConstraint("clave_idempotencia IS NULL OR clave_idempotencia ~ '^[A-Za-z0-9_-]{1,128}$'", name=op.f('ck_solicitudes_reentrenamiento_clave_valida')),
    sa.CheckConstraint('(clave_idempotencia IS NULL) = (huella_solicitud IS NULL)', name=op.f('ck_solicitudes_reentrenamiento_clave_con_huella')),
    sa.CheckConstraint("origen IN ('api', 'airflow')", name=op.f('ck_solicitudes_reentrenamiento_origen_valido')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_solicitudes_reentrenamiento')),
    sa.UniqueConstraint('clave_idempotencia', name=op.f('uq_solicitudes_reentrenamiento_clave_idempotencia'))
    )
    op.create_table('tareas_reentrenamiento',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('iniciada_en', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['id'], ['solicitudes_reentrenamiento.id'], name=op.f('fk_tareas_reentrenamiento_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_tareas_reentrenamiento'))
    )
    op.create_table('reclamaciones_reentrenamiento',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('reclamada_en', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
    sa.Column('epoca', sa.BigInteger(), nullable=False),
    sa.CheckConstraint('epoca > 0', name=op.f('ck_reclamaciones_reentrenamiento_epoca_positiva')),
    sa.ForeignKeyConstraint(['id'], ['solicitudes_reentrenamiento.id'], name=op.f('fk_reclamaciones_reentrenamiento_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_reclamaciones_reentrenamiento'))
    )
    op.create_table('resultados_reentrenamiento',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('estado', sa.Text(), nullable=False),
    sa.Column('promovido', sa.Boolean(), nullable=False),
    sa.Column('version', sa.Text(), nullable=True),
    sa.Column('motivo', sa.Text(), nullable=False),
    sa.Column('escrito_por', sa.Text(), nullable=False),
    sa.Column('terminado_en', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
    sa.CheckConstraint("estado IN ('exitoso', 'fallido')", name=op.f('ck_resultados_reentrenamiento_estado_valido')),
    sa.ForeignKeyConstraint(['id'], ['solicitudes_reentrenamiento.id'], name=op.f('fk_resultados_reentrenamiento_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_resultados_reentrenamiento'))
    )
    op.create_table('veredictos_reentrenamiento',
    sa.Column('id', sa.Text(), nullable=False),
    sa.Column('veredicto', sa.Text(), nullable=False),
    sa.Column('decidido_en', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
    sa.CheckConstraint("veredicto = 'sin_resultado'", name=op.f('ck_veredictos_reentrenamiento_veredicto_valido')),
    sa.ForeignKeyConstraint(['id'], ['solicitudes_reentrenamiento.id'], name=op.f('fk_veredictos_reentrenamiento_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_veredictos_reentrenamiento'))
    )
    # impedir_modificacion() la crea la migración 0001
    for tabla in SOLO_INSERCION:
        op.execute(f"CREATE TRIGGER trg_{tabla}_inmutable BEFORE UPDATE OR DELETE ON {tabla} "
                   "FOR EACH ROW EXECUTE FUNCTION impedir_modificacion()")
        op.execute(f"CREATE TRIGGER trg_{tabla}_sin_truncate BEFORE TRUNCATE ON {tabla} "
                   "FOR EACH STATEMENT EXECUTE FUNCTION impedir_modificacion()")


def downgrade() -> None:
    for tabla in reversed(SOLO_INSERCION):
        op.drop_table(tabla)
