"""Secuencia de resoluciones (plan §5.3, criterio 15): una transacción no puede obtener su
``secuencia`` mientras otra que ya incrementó el contador no termine en COMMIT o
ROLLBACK; una transacción revertida no consume secuencia.

"Transacción confirmada" = terminó en COMMIT; "resultado" = ``confirmada``/``anulada``
(valor de negocio). Se usa el código de producción: ``SQL_FILA_CONCESION`` (bloquea la
fila de la concesión) y ``asignar_secuencia_e_insertar``."""

import multiprocessing as mp
import random
import time

import psycopg2
import pytest
from sqlalchemy.engine import make_url

from app.modelo.bloqueo import SQL_FILA_CONCESION, asignar_secuencia_e_insertar
from tests.integracion.apoyo_mlflow import url_de


def conectar(url):
    u = make_url(url)
    c = psycopg2.connect(host=u.host, port=u.port or 5432, user=u.username, password=u.password, dbname=u.database)
    c.autocommit = True
    return c


def nueva_decision(cur, epoca=1) -> int:
    cur.execute("INSERT INTO decisiones_promocion (nombre_modelo, version, run_id, ruta_artefacto, sha256_artefacto, "
                "epoca, origen) VALUES ('m', '1', 'r', 'x', %s, %s, 'prueba') RETURNING decision_id", ("a" * 64, epoca))
    return cur.fetchone()[0]


def bloquear(cur) -> int:
    cur.execute(SQL_FILA_CONCESION, (60,))
    return int(cur.fetchone()[2])


def _p1(url, cola_aviso, cola_orden, terminar_con):
    """Incrementa e inserta, avisa su pid y espera la orden de terminar."""
    c = conectar(url)
    cur = c.cursor()
    cur.execute("BEGIN")
    d = nueva_decision(cur)
    s = asignar_secuencia_e_insertar(cur, bloquear(cur), d, "confirmada", 1, "p1")
    cur.execute("SELECT pg_backend_pid()")
    cola_aviso.put((cur.fetchone()[0], s))
    cola_orden.get(timeout=60)
    cur.execute(terminar_con)
    c.close()


def _p2(url, cola_resultado):
    c = conectar(url)
    cur = c.cursor()
    cur.execute("SELECT pg_backend_pid()")
    cola_resultado.put(("pid", cur.fetchone()[0]))
    cur.execute("BEGIN")
    d = nueva_decision(cur)
    ultima = bloquear(cur)                                    # aquí queda bloqueada mientras P1 no termine
    s = asignar_secuencia_e_insertar(cur, ultima, d, "anulada", 1, "p2")
    cur.execute("COMMIT")
    cola_resultado.put(("leida", ultima, s))
    c.close()


@pytest.fixture
def ctx():
    return mp.get_context("spawn")


def ultima_secuencia(url):
    c = conectar(url)
    with c.cursor() as cur:
        cur.execute("SELECT ultima_secuencia FROM bloqueo_entrenamiento WHERE id = 1")
        v = cur.fetchone()[0]
    c.close()
    return v


@pytest.mark.parametrize("terminar_con", ["COMMIT", "ROLLBACK"])
def test_la_segunda_transaccion_espera_a_que_la_primera_termine(motor, ctx, terminar_con):
    url = url_de(motor)
    aviso, orden, res = ctx.Queue(), ctx.Queue(), ctx.Queue()
    p1 = ctx.Process(target=_p1, args=(url, aviso, orden, terminar_con))
    p1.start()
    pid1, s1 = aviso.get(timeout=60)
    assert s1 == 1
    p2 = ctx.Process(target=_p2, args=(url, res))
    p2.start()
    pid2 = res.get(timeout=60)[1]
    adm = conectar(url)
    with adm.cursor() as cur:                                 # P2 bloqueada por P1, esperando un lock
        t = time.monotonic()
        while True:
            cur.execute("SELECT pg_blocking_pids(%s), wait_event_type FROM pg_stat_activity WHERE pid = %s",
                        (pid2, pid2))
            bloqueadores, espera = cur.fetchone()
            if pid1 in (bloqueadores or []):
                break
            assert time.monotonic() - t < 30, "P2 no quedó bloqueada por P1"
            time.sleep(0.1)
        assert espera == "Lock"
        time.sleep(2)
        assert res.empty()                                    # P2 no devolvió nada en 2 s
        assert ultima_secuencia(url) == 0                     # el incremento de P1 no está confirmado
        # NOWAIT: ni siquiera puede adquirir la fila
        cur.execute("BEGIN")
        with pytest.raises(psycopg2.errors.LockNotAvailable):
            cur.execute("SELECT ultima_secuencia FROM bloqueo_entrenamiento WHERE id = 1 FOR UPDATE NOWAIT")
        cur.execute("ROLLBACK")
    orden.put("terminar")
    p1.join(30)
    _, leida, s2 = res.get(timeout=60)
    p2.join(30)
    if terminar_con == "COMMIT":
        assert (leida, s2) == (1, 2)                          # lee el incremento confirmado de P1
    else:
        assert (leida, s2) == (0, 1)                          # el incremento revertido nunca fue visible
    with adm.cursor() as cur:
        cur.execute("SELECT motivo, secuencia FROM resoluciones_promocion ORDER BY secuencia")
        filas = cur.fetchall()
    assert filas == ([("p1", 1), ("p2", 2)] if terminar_con == "COMMIT" else [("p2", 1)])
    adm.close()


def _trabajador(url, n, semilla, cola):
    rnd = random.Random(semilla)
    c = conectar(url)
    cur = c.cursor()
    confirmadas, fallas = [], []
    for _ in range(n):
        cur.execute("BEGIN")
        d = nueva_decision(cur)
        ultima = bloquear(cur)
        cur.execute("SELECT coalesce(max(secuencia), 0) FROM resoluciones_promocion")   # todas, sin filtrar
        maximo = cur.fetchone()[0]
        if maximo != ultima:                                   # la predecesora debe estar confirmada
            fallas.append((ultima, maximo))
        resultado = rnd.choice(["confirmada", "anulada"])
        s = asignar_secuencia_e_insertar(cur, ultima, d, resultado, 1, "carga")
        time.sleep(rnd.uniform(0, 0.01))                      # pausa con el bloqueo tomado
        if rnd.random() < 0.2:
            cur.execute("ROLLBACK")
        else:
            cur.execute("COMMIT")
            confirmadas.append((s, resultado))
    c.close()
    cola.put((confirmadas, fallas))


def _lector(url, detener, cola):
    c = conectar(url)
    cur = c.cursor()
    huecos, lecturas = [], 0
    while not detener.is_set():
        cur.execute("SELECT secuencia FROM resoluciones_promocion ORDER BY secuencia")
        vistas = [x[0] for x in cur.fetchall()]
        lecturas += 1
        if vistas != list(range(1, len(vistas) + 1)):
            huecos.append(vistas[-5:])
    c.close()
    cola.put((huecos, lecturas))


def test_carga_concurrente_sin_huecos_ni_duplicados(motor, ctx):
    url = url_de(motor)
    cola, cola_lector, detener = ctx.Queue(), ctx.Queue(), ctx.Event()
    lector = ctx.Process(target=_lector, args=(url, detener, cola_lector))
    lector.start()
    trabajadores = [ctx.Process(target=_trabajador, args=(url, 50, i, cola)) for i in range(8)]
    for t in trabajadores:
        t.start()
    resultados = [cola.get(timeout=300) for _ in trabajadores]
    for t in trabajadores:
        t.join(30)
    detener.set()
    huecos, lecturas = cola_lector.get(timeout=60)
    lector.join(30)
    confirmadas = sorted(x for r, _ in resultados for x in r)
    fallas = [f for _, f in resultados for f in f]
    k = len(confirmadas)                       # inserciones cuya transacción terminó en COMMIT
    assert fallas == []                        # cada transacción encontró confirmada a su predecesora
    assert [s for s, _ in confirmadas] == list(range(1, k + 1)) and 300 < k < 400
    assert huecos == [] and lecturas > 10      # el lector nunca vio s + 1 sin s
    c = conectar(url)
    with c.cursor() as cur:
        cur.execute("SELECT secuencia, resultado FROM resoluciones_promocion ORDER BY secuencia")
        assert cur.fetchall() == confirmadas
        cur.execute("SELECT ultima_secuencia FROM bloqueo_entrenamiento")
        assert cur.fetchone()[0] == k
        # el vigente calculado = la última inserción confirmada con resultado 'confirmada'
        cur.execute("SELECT max(secuencia) FROM resoluciones_promocion WHERE resultado = 'confirmada'")
        assert cur.fetchone()[0] == max(s for s, r in confirmadas if r == "confirmada")
        # UNIQUE (secuencia): una inserción repetida, saltándose el contador, se rechaza
        cur.execute("BEGIN")
        d = nueva_decision(cur)
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute("INSERT INTO resoluciones_promocion (decision_id, resultado, epoca, motivo, secuencia) "
                        "VALUES (%s, 'anulada', 1, 'repetida', 1)", (d,))
        cur.execute("ROLLBACK")
    c.close()
