"""Roles de PostgreSQL de la etapa 7 y la función controlada de terminación
(DECISIONES.md §16, plan v12 §1). Idempotente; lo ejecuta el servicio ``preparar``
después de las migraciones, como ``POSTGRES_USER`` (superusuario solo en esta tarea
administrativa). Todo ocurre en UNA transacción: la función nunca es visible con
permisos más amplios que los finales.

Uso:  python -m app.db.roles      (claves en NM_CIERRE_PASSWORD y NM_VIGILANTE_PASSWORD)

* ``nm_cierre`` (LOGIN): toda escritura de resultados de reentrenamiento (CLI,
  registrar_resultado, recuperación de la API). SELECT en las tablas que lee;
  INSERT solo en ``resultados_reentrenamiento``; UPDATE en
  ``solicitudes_reentrenamiento`` solo porque PostgreSQL lo exige para
  ``SELECT … FOR UPDATE`` (el trigger de solo inserción rechaza cualquier UPDATE real).
* ``nm_vigilante`` (LOGIN): solo el vigilante del DAG. SELECT, UPDATE (para FOR
  UPDATE) en las solicitudes, INSERT solo en ``veredictos_reentrenamiento`` y
  EXECUTE sobre ``nm_control.nm_terminar_escritor``. Sin pg_signal_backend.
* ``nm_senal`` (NOLOGIN): propietario de la función; su única membresía es
  ``pg_signal_backend`` WITH INHERIT TRUE, SET FALSE, ADMIN FALSE (PostgreSQL 16: el
  privilegio efectivo de una función SECURITY DEFINER exige herencia). Sin miembros.
* La función filtra por base, rol ``nm_cierre`` y ``application_name`` del token del
  id. No filtra por ``backend_type``: para un rol sin pg_read_all_stats esa columna
  de otras sesiones aparece NULL (comprobado en PostgreSQL 16); solo las sesiones
  cliente llevan ese usename y ese application_name.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.conexion import crear_motor

ROLES = ("nm_cierre", "nm_vigilante")
TABLAS_LECTURA = ("solicitudes_reentrenamiento", "tareas_reentrenamiento", "reclamaciones_reentrenamiento",
                  "resultados_reentrenamiento", "veredictos_reentrenamiento", "decisiones_promocion",
                  "resoluciones_promocion")
FUNCION = "nm_control.nm_terminar_escritor(text)"

SQL_FUNCION = r"""
CREATE OR REPLACE FUNCTION nm_control.nm_terminar_escritor(p_id text) RETURNS integer
LANGUAGE plpgsql STRICT SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_app text;
    v_pid integer;
    v_n integer := 0;
BEGIN
    IF pg_catalog.length(p_id) OPERATOR(pg_catalog.<) 1
       OR pg_catalog.length(p_id) OPERATOR(pg_catalog.>) 250
       OR NOT (p_id OPERATOR(pg_catalog.~) '^[A-Za-z0-9_.:+-]+$') THEN
        RAISE EXCEPTION 'identificador de reentrenamiento inválido' USING ERRCODE = '22023';
    END IF;
    v_app := 'nm-cierre:' OPERATOR(pg_catalog.||) pg_catalog.left(
        pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(p_id, 'UTF8')), 'hex'), 32);
    FOR v_pid IN
        SELECT a.pid FROM pg_catalog.pg_stat_activity AS a
        WHERE a.datname OPERATOR(pg_catalog.=) pg_catalog.current_database()
          AND a.usename OPERATOR(pg_catalog.=) 'nm_cierre'::pg_catalog.name
          AND a.application_name OPERATOR(pg_catalog.=) v_app
          AND a.pid OPERATOR(pg_catalog.<>) pg_catalog.pg_backend_pid()
    LOOP
        IF pg_catalog.pg_terminate_backend(v_pid) THEN
            v_n := v_n OPERATOR(pg_catalog.+) 1;
        END IF;
    END LOOP;
    RETURN v_n;
END
$fn$
"""


class RolesInvalidos(RuntimeError):
    pass


def _clave(variable: str) -> str:
    v = os.environ.get(variable, "")
    if not v:
        raise RolesInvalidos(f"falta la variable de entorno {variable}")
    return v


def configurar_roles(motor: Engine, clave_cierre: str, clave_vigilante: str) -> None:
    with motor.begin() as c:
        version = int(c.execute(text("SHOW server_version_num")).scalar())
        if version < 160000:
            raise RolesInvalidos(f"se requiere PostgreSQL 16 o posterior (servidor: {version})")
        base = c.execute(text("SELECT current_database()")).scalar()
        bq = '"' + base.replace('"', '""') + '"'
        for rol, clave in (("nm_cierre", clave_cierre), ("nm_vigilante", clave_vigilante)):
            existe = c.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": rol}).scalar()
            if not existe:
                c.execute(text(f"CREATE ROLE {rol} LOGIN"))
            c.execute(text(f"ALTER ROLE {rol} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
                           "NOBYPASSRLS INHERIT"))
            c.execute(text(f"ALTER ROLE {rol} PASSWORD :p").bindparams(p=clave))
        if not c.execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'nm_senal'")).scalar():
            c.execute(text("CREATE ROLE nm_senal NOLOGIN"))
        c.execute(text("ALTER ROLE nm_senal NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
                       "NOBYPASSRLS INHERIT"))
        c.execute(text("GRANT pg_signal_backend TO nm_senal WITH INHERIT TRUE, SET FALSE, ADMIN FALSE"))
        # nadie es miembro de nm_senal; nm_vigilante no lo es de pg_signal_backend
        for miembro in c.execute(text("SELECT r.rolname FROM pg_auth_members m JOIN pg_roles r "
                                      "ON r.oid = m.member WHERE m.roleid = 'nm_senal'::regrole")).scalars():
            c.execute(text(f'REVOKE nm_senal FROM "{miembro}"'))
        c.execute(text("REVOKE pg_signal_backend FROM nm_vigilante"))

        # base y esquemas: sin CONNECT/TEMPORARY para PUBLIC ni CREATE en public
        c.execute(text(f"REVOKE CONNECT, TEMPORARY ON DATABASE {bq} FROM PUBLIC"))
        c.execute(text(f"GRANT CONNECT ON DATABASE {bq} TO nm_cierre, nm_vigilante"))
        c.execute(text(f"REVOKE TEMPORARY ON DATABASE {bq} FROM nm_cierre, nm_vigilante"))
        c.execute(text("REVOKE CREATE ON SCHEMA public FROM PUBLIC"))
        c.execute(text("GRANT USAGE ON SCHEMA public TO nm_cierre, nm_vigilante"))

        # privilegios de tabla mínimos
        for t in TABLAS_LECTURA:
            c.execute(text(f"REVOKE ALL ON {t} FROM nm_cierre, nm_vigilante"))
            c.execute(text(f"GRANT SELECT ON {t} TO nm_cierre, nm_vigilante"))
        c.execute(text("GRANT INSERT ON resultados_reentrenamiento TO nm_cierre"))
        c.execute(text("GRANT UPDATE ON solicitudes_reentrenamiento TO nm_cierre, nm_vigilante"))
        c.execute(text("GRANT INSERT ON veredictos_reentrenamiento TO nm_vigilante"))

        # esquema de control propiedad del administrador; solo USAGE para el vigilante
        c.execute(text("CREATE SCHEMA IF NOT EXISTS nm_control"))
        c.execute(text("ALTER SCHEMA nm_control OWNER TO CURRENT_USER"))
        c.execute(text("REVOKE ALL ON SCHEMA nm_control FROM PUBLIC, nm_cierre, nm_vigilante, nm_senal"))
        c.execute(text("GRANT USAGE ON SCHEMA nm_control TO nm_vigilante"))
        c.execute(text(SQL_FUNCION))
        c.execute(text(f"ALTER FUNCTION {FUNCION} OWNER TO nm_senal"))
        c.execute(text(f"REVOKE ALL ON FUNCTION {FUNCION} FROM PUBLIC, nm_cierre"))
        c.execute(text(f"GRANT EXECUTE ON FUNCTION {FUNCION} TO nm_vigilante"))


def main() -> int:
    try:
        configurar_roles(crear_motor(timeout_consulta=60), _clave("NM_CIERRE_PASSWORD"),
                         _clave("NM_VIGILANTE_PASSWORD"))
    except RolesInvalidos as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("roles nm_cierre, nm_vigilante y nm_senal configurados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
