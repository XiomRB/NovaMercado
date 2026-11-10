"""Entorno de Alembic. URL: ``sqlalchemy.url`` del config (pruebas) o ``DATABASE_URL``."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.config import url_base_datos
from app.db.esquema import metadata

config = context.config
if config.config_file_name is not None and config.attributes.get("configurar_logging", True):
    fileConfig(config.config_file_name)


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or url_base_datos()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    motor = create_engine(_url(), poolclass=pool.NullPool, future=True)
    with motor.connect() as conn:
        context.configure(connection=conn, target_metadata=metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    motor.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
