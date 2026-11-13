"""Etapa 6: concesión de entrenamiento y promociones (DECISIONES.md §15).

* ``bloqueo_entrenamiento``: una sola fila (id = 1), mutable por diseño: época,
  titular, latido de la concesión y contador ``ultima_secuencia``. Un trigger
  impide borrarla o truncarla.
* ``decisiones_promocion`` y ``resoluciones_promocion``: solo inserción (triggers).
  PostgreSQL es la fuente de verdad del modelo vigente.

Las listas de tablas se escriben aquí de forma literal: esta migración no debe
cambiar si mañana cambian las constantes de ``app/db/esquema.py``.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

SOLO_INSERCION = ("decisiones_promocion", "resoluciones_promocion")


def upgrade() -> None:
    op.create_table('bloqueo_entrenamiento',
    sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('epoca', sa.BigInteger(), server_default='0', nullable=False),
    sa.Column('titular', sa.Text(), nullable=True),
    sa.Column('adquirido_en', sa.DateTime(timezone=True), nullable=True),
    sa.Column('latido', sa.DateTime(timezone=True), nullable=True),
    sa.Column('ultima_secuencia', sa.BigInteger(), server_default='0', nullable=False),
    sa.CheckConstraint('epoca >= 0', name=op.f('ck_bloqueo_entrenamiento_epoca_no_negativa')),
    sa.CheckConstraint('id = 1', name=op.f('ck_bloqueo_entrenamiento_fila_unica')),
    sa.CheckConstraint('ultima_secuencia >= 0', name=op.f('ck_bloqueo_entrenamiento_secuencia_no_negativa')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_bloqueo_entrenamiento'))
    )
    op.create_table('decisiones_promocion',
    sa.Column('decision_id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.Column('nombre_modelo', sa.Text(), nullable=False),
    sa.Column('version', sa.Text(), nullable=False),
    sa.Column('run_id', sa.Text(), nullable=False),
    sa.Column('ruta_artefacto', sa.Text(), nullable=False),
    sa.Column('sha256_artefacto', sa.String(length=64), nullable=False),
    sa.Column('epoca', sa.BigInteger(), nullable=False),
    sa.Column('origen', sa.Text(), nullable=False),
    sa.Column('creada_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('epoca > 0', name=op.f('ck_decisiones_promocion_epoca_positiva')),
    sa.CheckConstraint("sha256_artefacto ~ '^[0-9a-f]{64}$'", name=op.f('ck_decisiones_promocion_sha256_valido')),
    sa.PrimaryKeyConstraint('decision_id', name=op.f('pk_decisiones_promocion'))
    )
    op.create_table('resoluciones_promocion',
    sa.Column('decision_id', sa.BigInteger(), nullable=False),
    sa.Column('resultado', sa.Text(), nullable=False),
    sa.Column('epoca', sa.BigInteger(), nullable=False),
    sa.Column('motivo', sa.Text(), nullable=False),
    sa.Column('secuencia', sa.BigInteger(), nullable=False),
    sa.Column('resuelta_en', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("resultado IN ('confirmada', 'anulada')", name=op.f('ck_resoluciones_promocion_resultado_valido')),
    sa.CheckConstraint('secuencia > 0', name=op.f('ck_resoluciones_promocion_secuencia_positiva')),
    sa.ForeignKeyConstraint(['decision_id'], ['decisiones_promocion.decision_id'], name=op.f('fk_resoluciones_promocion_decision_id')),
    sa.PrimaryKeyConstraint('decision_id', name=op.f('pk_resoluciones_promocion')),
    sa.UniqueConstraint('secuencia', name=op.f('uq_resoluciones_promocion_secuencia'))
    )
    op.execute("INSERT INTO bloqueo_entrenamiento (id) VALUES (1)")
    # impedir_modificacion() la crea la migración 0001
    for tabla in SOLO_INSERCION:
        op.execute(f"CREATE TRIGGER trg_{tabla}_inmutable BEFORE UPDATE OR DELETE ON {tabla} "
                   "FOR EACH ROW EXECUTE FUNCTION impedir_modificacion()")
        op.execute(f"CREATE TRIGGER trg_{tabla}_sin_truncate BEFORE TRUNCATE ON {tabla} "
                   "FOR EACH STATEMENT EXECUTE FUNCTION impedir_modificacion()")
    op.execute("CREATE TRIGGER trg_bloqueo_entrenamiento_sin_borrado BEFORE DELETE ON bloqueo_entrenamiento "
               "FOR EACH ROW EXECUTE FUNCTION impedir_modificacion()")
    op.execute("CREATE TRIGGER trg_bloqueo_entrenamiento_sin_truncate BEFORE TRUNCATE ON bloqueo_entrenamiento "
               "FOR EACH STATEMENT EXECUTE FUNCTION impedir_modificacion()")


def downgrade() -> None:
    op.drop_table('resoluciones_promocion')
    op.drop_table('decisiones_promocion')
    op.drop_table('bloqueo_entrenamiento')
