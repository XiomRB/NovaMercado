"""Roles de la etapa 7, función de terminación controlada y escritura/veredicto de los
resultados contra PostgreSQL real (DECISIONES.md §16, plan v12 §1–§3).

Garantías demostradas aquí:

* ``nm_senal`` hereda ``pg_signal_backend`` (INHERIT TRUE, SET FALSE, ADMIN FALSE) y esa
  herencia es la que hace funcionar la función SECURITY DEFINER: con INHERIT FALSE falla;
* ``nm_vigilante`` no puede terminar backends directamente; ``nm_cierre`` no puede llamar
  a la función; la función solo termina sesiones de ``nm_cierre`` con el token del id;
* nunca hay un ``exitoso`` después de un veredicto de ausencia; un COMMIT sin respuesta
  se resuelve verificando por id sin reescribir;
* el vigilante nunca concluye ausencia sin el bloqueo, y termina a un escritor colgado.
"""

from __future__ import annotations

import threading
import time

import psycopg2
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.db import cierre
from app.db.roles import configurar_roles
from tests.integracion.proxy import ProxyCongelable

CLAVE_C, CLAVE_V = "clave-cierre-prueba", "clave-vigilante-prueba"
MODELO = "novamercado-ventas"


@pytest.fixture
def bd(motor, url_bd):
    configurar_roles(motor, CLAVE_C, CLAVE_V)
    base = cierre.parametros_desde_url(url_bd)
    return {
        "motor": motor,
        "admin": base,
        "cierre": {**base, "user": "nm_cierre", "password": CLAVE_C},
        "vigilante": {**base, "user": "nm_vigilante", "password": CLAVE_V},
    }


def solicitud(motor, id_="nm-prueba-1"):
    with motor.begin() as c:
        c.execute(text("INSERT INTO solicitudes_reentrenamiento (id, origen) VALUES (:i, 'api')"), {"i": id_})
    return id_


def fila(motor, tabla, id_):
    with motor.connect() as c:
        return c.execute(text(f"SELECT * FROM {tabla} WHERE id = :i"), {"i": id_}).mappings().first()


def sesion_colgada(params, id_, app=None, bloquear=True):
    """Escritor que toma el bloqueo de la solicitud y queda colgado (idle in transaction)."""
    conn = psycopg2.connect(**params, application_name=app or cierre.app_cierre(id_))
    cur = conn.cursor()
    if bloquear:
        cur.execute(cierre.SQL_BLOQUEAR, {"id": id_})
    else:
        cur.execute("SELECT 1")
    return conn


def vivo(motor, pid):
    with motor.connect() as c:
        return bool(c.execute(text("SELECT count(*) FROM pg_stat_activity WHERE pid = :p"), {"p": pid}).scalar())


# --------------------------------------------------------------------------- roles
def test_catalogo_de_roles(bd):
    with bd["motor"].connect() as c:
        m = c.execute(text("""SELECT m.inherit_option, m.set_option, m.admin_option FROM pg_auth_members m
                              WHERE m.roleid = 'pg_signal_backend'::regrole AND m.member = 'nm_senal'::regrole""")
                      ).one()
        assert tuple(m) == (True, False, False)
        assert c.execute(text("SELECT count(*) FROM pg_auth_members WHERE roleid = 'nm_senal'::regrole")).scalar() == 0
        r = {x.rolname: x for x in c.execute(text(
            "SELECT rolname, rolsuper, rolcanlogin, rolinherit, rolcreaterole, rolbypassrls FROM pg_roles "
            "WHERE rolname IN ('nm_cierre', 'nm_vigilante', 'nm_senal')"))}
        assert not any(x.rolsuper or x.rolcreaterole or x.rolbypassrls for x in r.values())
        assert (r["nm_cierre"].rolcanlogin, r["nm_vigilante"].rolcanlogin, r["nm_senal"].rolcanlogin) == \
            (True, True, False)
        priv = lambda rol, p: c.execute(text("SELECT pg_has_role(:r, 'pg_signal_backend', :p)"),  # noqa: E731
                                        {"r": rol, "p": p}).scalar()
        assert priv("nm_senal", "USAGE") is True and priv("nm_senal", "SET") is False
        assert priv("nm_vigilante", "USAGE") is False and priv("nm_cierre", "USAGE") is False
        f = c.execute(text("""SELECT p.prosecdef, pg_get_userbyid(p.proowner) AS dueno, p.proconfig,
                                     has_function_privilege('nm_vigilante', p.oid, 'EXECUTE') AS v,
                                     has_function_privilege('nm_cierre', p.oid, 'EXECUTE') AS c,
                                     has_function_privilege('public', p.oid, 'EXECUTE') AS pub
                              FROM pg_proc p WHERE p.oid = 'nm_control.nm_terminar_escritor(text)'::regprocedure""")
                      ).one()
        assert f.prosecdef and f.dueno == "nm_senal" and f.proconfig == ["search_path=pg_catalog, pg_temp"]
        assert (f.v, f.c, f.pub) == (True, False, False)
        esq = c.execute(text("""SELECT pg_get_userbyid(nspowner) = current_user,
                                       has_schema_privilege('nm_vigilante', oid, 'CREATE'),
                                       has_schema_privilege('nm_senal', oid, 'CREATE')
                                FROM pg_namespace WHERE nspname = 'nm_control'""")).one()
        assert tuple(esq) == (True, False, False)


def test_configurar_roles_es_idempotente(bd):
    configurar_roles(bd["motor"], CLAVE_C, CLAVE_V)
    test_catalogo_de_roles(bd)


def test_vigilante_no_termina_backends_directamente(bd):
    id_ = solicitud(bd["motor"])
    victima = sesion_colgada(bd["cierre"], id_, bloquear=False)
    pid = victima.get_backend_pid()
    v = psycopg2.connect(**bd["vigilante"])
    try:
        with pytest.raises(psycopg2.errors.InsufficientPrivilege), v.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
        v.rollback()
        with pytest.raises(psycopg2.errors.InsufficientPrivilege), v.cursor() as cur:
            cur.execute("SET ROLE nm_senal")
        assert vivo(bd["motor"], pid)
    finally:
        v.close()
        victima.close()


def test_cierre_no_puede_llamar_a_la_funcion_ni_escribir_veredictos(bd):
    id_ = solicitud(bd["motor"])
    c = psycopg2.connect(**bd["cierre"])
    try:
        for sql in ("SELECT nm_control.nm_terminar_escritor('nm-prueba-1')",
                    "INSERT INTO veredictos_reentrenamiento (id, veredicto) VALUES ('nm-prueba-1', 'sin_resultado')",
                    "CREATE TABLE public.x (a int)", "CREATE TEMP TABLE x (a int)",
                    "UPDATE solicitudes_reentrenamiento SET origen = 'api'",
                    "INSERT INTO decisiones_promocion (nombre_modelo, version, run_id, ruta_artefacto, "
                    "sha256_artefacto, epoca, origen) VALUES ('m','1','r','a','x',1,'o')",
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = 'postgres'"):
            with pytest.raises(psycopg2.Error), c.cursor() as cur:
                cur.execute(sql)
            c.rollback()
    finally:
        c.close()
    v = psycopg2.connect(**bd["vigilante"])
    try:
        for sql in (f"INSERT INTO resultados_reentrenamiento (id, estado, promovido, motivo, escrito_por) "
                    f"VALUES ('{id_}', 'exitoso', false, 'x', 'x')", "CREATE TABLE public.x (a int)",
                    "CREATE TEMP TABLE x (a int)", "CREATE FUNCTION nm_control.f() RETURNS int AS 'SELECT 1' "
                    "LANGUAGE sql"):
            with pytest.raises(psycopg2.Error), v.cursor() as cur:
                cur.execute(sql)
            v.rollback()
    finally:
        v.close()


def test_funcion_solo_termina_al_escritor_del_token(bd):
    id_ = solicitud(bd["motor"])
    otro = solicitud(bd["motor"], "nm-prueba-2")
    objetivo = sesion_colgada(bd["cierre"], id_)
    ajenos = [sesion_colgada(bd["cierre"], otro),                                          # otro token
              sesion_colgada(bd["admin"], id_, app=cierre.app_cierre(id_), bloquear=False),  # otro rol
              sesion_colgada(bd["cierre"], id_, app=cierre.app_verif(id_), bloquear=False)]  # verificador
    pids = [s.get_backend_pid() for s in ajenos]
    pid = objetivo.get_backend_pid()
    v = psycopg2.connect(**bd["vigilante"])
    v.autocommit = True
    try:
        with v.cursor() as cur:
            cur.execute("SELECT nm_control.nm_terminar_escritor(%s)", (id_,))
            assert cur.fetchone()[0] == 1
        for _ in range(50):
            if not vivo(bd["motor"], pid):
                break
            time.sleep(0.1)
        assert not vivo(bd["motor"], pid)
        assert all(vivo(bd["motor"], p) for p in pids)
        with v.cursor() as cur:              # sin escritor: 0, sin error
            cur.execute("SELECT nm_control.nm_terminar_escritor(%s)", ("nm-sin-escritor",))
            assert cur.fetchone()[0] == 0
    finally:
        v.close()
        for s in ajenos + [objetivo]:
            s.close()


@pytest.mark.parametrize("id_", ["", "a" * 251, "con espacio", "x'; DROP TABLE y; --", "ñ"])
def test_funcion_rechaza_ids_invalidos(bd, id_):
    v = psycopg2.connect(**bd["vigilante"])
    try:
        with pytest.raises(psycopg2.errors.InvalidParameterValue), v.cursor() as cur:
            cur.execute("SELECT nm_control.nm_terminar_escritor(%s)", (id_,))
    finally:
        v.close()


def test_herencia_efectiva_es_necesaria(bd, servidor):
    """Con INHERIT FALSE la misma función no puede terminar al escritor: lo que la hace
    funcionar es la herencia de pg_signal_backend por nm_senal (PostgreSQL 16)."""
    id_ = solicitud(bd["motor"])
    victima = sesion_colgada(bd["cierre"], id_, bloquear=False)
    v = psycopg2.connect(**bd["vigilante"])
    try:
        with bd["motor"].begin() as c:
            c.execute(text("REVOKE pg_signal_backend FROM nm_senal"))
            c.execute(text("GRANT pg_signal_backend TO nm_senal WITH INHERIT FALSE, SET FALSE"))
        with pytest.raises(psycopg2.errors.InsufficientPrivilege), v.cursor() as cur:
            cur.execute("SELECT nm_control.nm_terminar_escritor(%s)", (id_,))
        assert vivo(bd["motor"], victima.get_backend_pid())
    finally:
        v.close()
        victima.close()
        configurar_roles(bd["motor"], CLAVE_C, CLAVE_V)     # los roles son del clúster


def test_id_de_250_caracteres(bd):
    id_ = solicitud(bd["motor"], "a" * 250)
    assert len(cierre.app_cierre(id_).encode()) <= 63
    r = cierre.cerrar(bd["cierre"], id_, estado="fallido", motivo="m", escrito_por="prueba", nombre_modelo=MODELO)
    assert r.codigo == "escrito"


# --------------------------------------------------------------------- cierre
def test_cierre_escribe_una_sola_vez(bd):
    id_ = solicitud(bd["motor"])
    r1 = cierre.cerrar(bd["cierre"], id_, estado="exitoso", motivo="ok", escrito_por="cli", nombre_modelo=MODELO)
    r2 = cierre.cerrar(bd["cierre"], id_, estado="fallido", motivo="x", escrito_por="dag", nombre_modelo=MODELO)
    assert (r1.codigo, r2.codigo, r2.estado_persistido) == ("escrito", "ya_registrado", "exitoso")
    f = fila(bd["motor"], "resultados_reentrenamiento", id_)
    assert (f["estado"], f["promovido"], f["version"], f["escrito_por"]) == ("exitoso", False, None, "cli")


def test_cierre_sin_solicitud(bd):
    r = cierre.cerrar(bd["cierre"], "nm-no-existe", estado="fallido", motivo="m", escrito_por="x",
                      nombre_modelo=MODELO)
    assert r.codigo == "no_escrito"


def test_exitoso_nunca_despues_del_veredicto(bd):
    id_ = solicitud(bd["motor"])
    d = cierre.vigilar(bd["vigilante"], id_)
    assert d.tipo == "ausencia"
    r = cierre.cerrar(bd["cierre"], id_, estado="exitoso", motivo="ok", escrito_por="cli", nombre_modelo=MODELO)
    assert r.codigo == "rechazado_por_veredicto" and fila(bd["motor"], "resultados_reentrenamiento", id_) is None
    r = cierre.cerrar(bd["cierre"], id_, estado="fallido", motivo="la ejecución se interrumpió",
                      escrito_por="api", nombre_modelo=MODELO)
    assert r.codigo == "escrito"
    assert cierre.vigilar(bd["vigilante"], id_).tipo == "persistido"


def test_vigilante_decide_por_lo_persistido(bd):
    id_ = solicitud(bd["motor"])
    cierre.cerrar(bd["cierre"], id_, estado="exitoso", motivo="ok", escrito_por="cli", nombre_modelo=MODELO)
    d = cierre.vigilar(bd["vigilante"], id_)
    assert (d.tipo, d.estado) == ("persistido", "exitoso")
    assert fila(bd["motor"], "veredictos_reentrenamiento", id_) is None


def test_commit_sin_respuesta_se_verifica_sin_reescribir(bd):
    id_ = solicitud(bd["motor"])
    p = ProxyCongelable(bd["cierre"]["host"], bd["cierre"]["port"])
    try:
        p.tras(b"COMMIT", "perder_respuesta")
        t = time.monotonic()
        r = cierre.cerrar({**bd["cierre"], "host": "127.0.0.1", "port": p.puerto}, id_, estado="exitoso",
                          motivo="ok", escrito_por="cli", nombre_modelo=MODELO, escritura=2.0, total=6.0)
        assert p.disparado.is_set()
        assert (r.codigo, r.estado_persistido) == ("ya_registrado", "exitoso"), r
        assert "COMMIT sin respuesta" in r.detalle and time.monotonic() - t < 6.5
    finally:
        p.cerrar()


def test_commit_no_recibido_no_escribe(bd):
    id_ = solicitud(bd["motor"])
    p = ProxyCongelable(bd["cierre"]["host"], bd["cierre"]["port"])
    try:
        p.tras(b"COMMIT", "retener")
        r = cierre.cerrar({**bd["cierre"], "host": "127.0.0.1", "port": p.puerto}, id_, estado="exitoso",
                          motivo="ok", escrito_por="cli", nombre_modelo=MODELO, escritura=2.0, total=6.0)
        assert r.codigo == "no_escrito", r
        assert fila(bd["motor"], "resultados_reentrenamiento", id_) is None
    finally:
        p.cerrar()


def test_cierre_respeta_su_presupuesto_con_el_servidor_colgado(bd):
    id_ = solicitud(bd["motor"])
    p = ProxyCongelable(bd["cierre"]["host"], bd["cierre"]["port"])
    try:
        p.congelar()
        t = time.monotonic()
        r = cierre.cerrar({**bd["cierre"], "host": "127.0.0.1", "port": p.puerto}, id_, estado="fallido",
                          motivo="m", escrito_por="dag", nombre_modelo=MODELO, escritura=2.0, total=4.0)
        assert r.codigo == "indeterminado" and time.monotonic() - t < 4.0 + 1.6
    finally:
        p.cerrar()


# ------------------------------------------------------------------ vigilante
def test_vigilante_termina_al_escritor_colgado_y_registra_ausencia(bd):
    id_ = solicitud(bd["motor"])
    colgado = sesion_colgada(bd["cierre"], id_)
    try:
        t = time.monotonic()
        d = cierre.vigilar(bd["vigilante"], id_)
        dur = time.monotonic() - t
        assert d.tipo == "ausencia" and "terminados=1" in d.detalle, d
        assert 15 <= dur <= 34, d.fases
        assert fila(bd["motor"], "veredictos_reentrenamiento", id_) is not None
        with pytest.raises(psycopg2.Error), colgado.cursor() as cur:      # su sesión ya no existe
            cur.execute("SELECT 1")
    finally:
        colgado.close()
    r = cierre.cerrar(bd["cierre"], id_, estado="exitoso", motivo="tarde", escrito_por="cli", nombre_modelo=MODELO)
    assert r.codigo == "rechazado_por_veredicto"


def test_vigilante_sin_bloqueo_no_concluye_ausencia(bd):
    """El bloqueo lo retiene una sesión que NO es un escritor con el token (no se la puede
    terminar): el resultado es indeterminado y no se escribe veredicto."""
    id_ = solicitud(bd["motor"])
    ajeno = sesion_colgada(bd["admin"], id_, app="otra-cosa")
    try:
        d = cierre.vigilar(bd["vigilante"], id_)
        assert d.tipo == "indeterminado", d
        assert sum(d.fases.values()) <= 34 + 1, d.fases
    finally:
        ajeno.close()
    assert fila(bd["motor"], "veredictos_reentrenamiento", id_) is None


def test_vigilante_escritor_que_termina_durante_la_espera(bd):
    """El escritor confirma su resultado mientras el vigilante espera: el vigilante ve el
    resultado persistido (nunca un veredicto contradictorio)."""
    id_ = solicitud(bd["motor"])
    c = psycopg2.connect(**bd["cierre"], application_name=cierre.app_cierre(id_))
    cur = c.cursor()
    cur.execute(cierre.SQL_BLOQUEAR, {"id": id_})
    cur.execute(cierre.SQL_INSERTAR, {"id": id_, "estado": "exitoso", "promovido": False, "motivo": "ok",
                                      "escrito_por": "cli", "nombre": MODELO})

    def confirmar():
        time.sleep(3)
        c.commit()
    h = threading.Thread(target=confirmar)
    h.start()
    try:
        d = cierre.vigilar(bd["vigilante"], id_)
    finally:
        h.join()
        c.close()
    assert (d.tipo, d.estado) == ("persistido", "exitoso")
    assert fila(bd["motor"], "veredictos_reentrenamiento", id_) is None


def test_vigilante_con_postgres_colgado_es_indeterminado_dentro_de_38s(bd):
    id_ = solicitud(bd["motor"])
    p = ProxyCongelable(bd["vigilante"]["host"], bd["vigilante"]["port"])
    try:
        params = {**bd["vigilante"], "host": "127.0.0.1", "port": p.puerto}
        conn = psycopg2.connect(**params)       # calienta el proxy
        conn.close()
        p.congelar()
        t = time.monotonic()
        d = cierre.vigilar(params, id_)
        assert d.tipo == "indeterminado" and time.monotonic() - t <= cierre.VIG_RESPALDO_S + 1
    finally:
        p.cerrar()
    assert fila(bd["motor"], "veredictos_reentrenamiento", id_) is None


def test_url_a_parametros():
    u = make_url("postgresql+psycopg2://nova:c%2Fl%40v@db:5433/nm")
    p = cierre.parametros_desde_url(u.render_as_string(hide_password=False))
    assert p == {"host": "db", "port": 5433, "user": "nova", "password": "c/l@v", "dbname": "nm"}
