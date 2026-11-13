"""Proceso hijo del entrenamiento (etapa 6, plan §4.2). Lo lanza solo el supervisor.

Uso interno:  python -m app.modelo.trabajo_entrenamiento <entrada.json> <salida.json>

Lo primero, antes de importar nada pesado: ``prctl(PR_SET_PDEATHSIG, SIGKILL)`` y
**después** comprobar que ``getppid()`` es el supervisor (``NM_PID_SUPERVISOR``). Si
el padre murió antes de que el mecanismo quedara activo, el hijo sale con código 6.
Fuera de Linux, un hilo compara ``getppid()`` cada segundo.

Luego arranca su **vigilante de la concesión** (conexión propia): si la época cambia,
el latido tiene más de 30 s o la base no responde, ``os._exit(6)`` termina el proceso
aunque el hilo principal esté dentro de código C. El hijo registra versiones solo en
``pendiente``/``invalido``; nunca escribe ``valido`` ni el alias.
"""

import os
import sys
import threading
import time

CODIGO_CONCESION_PERDIDA = 6


def _salir(motivo: str) -> None:
    try:
        sys.stderr.write(f"[trabajo_entrenamiento] se detiene: {motivo}\n")
        sys.stderr.flush()
    finally:
        os._exit(CODIGO_CONCESION_PERDIDA)


def morir_con_el_padre(pid_esperado: int) -> None:
    if sys.platform.startswith("linux"):
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
            _salir("no se pudo configurar PR_SET_PDEATHSIG")
        if os.getppid() != pid_esperado:        # el padre murió antes del prctl
            _salir("el supervisor ya no existe")
        return

    def vigilar_padre() -> None:
        while True:
            if os.getppid() != pid_esperado:
                _salir("el supervisor ya no existe")
            time.sleep(1)
    if os.getppid() != pid_esperado:
        _salir("el supervisor ya no existe")
    threading.Thread(target=vigilar_padre, name="vigilante-padre", daemon=True).start()


def main(argv: list[str]) -> int:
    pid = os.environ.get("NM_PID_SUPERVISOR", "")
    if not pid.isdigit():
        _salir("falta NM_PID_SUPERVISOR")
    morir_con_el_padre(int(pid))

    import json
    import logging

    from app.db.conexion import abrir_conexion_dedicada, crear_motor
    from app.modelo.bloqueo import LATIDO_MAX_HIJO_S, ConcesionPerdida, estado_concesion, vigilar_concesion
    from app.modelo.evaluacion import ConfigEvaluacion
    from app.modelo.registro import registrar_respaldo, registrar_version
    from ml.pronosticador import ConfigEntrenamiento

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    entrada = json.loads(open(argv[0], encoding="utf-8").read())
    url = os.environ["NM_URL_BD_ENTRENAMIENTO"]
    epoca = int(entrada["epoca"])

    threading.Thread(target=vigilar_concesion, args=(url, epoca, _salir), name="vigilante-concesion",
                     daemon=True).start()

    conn = abrir_conexion_dedicada(url)

    def comprobar() -> None:
        actual, edad = estado_concesion(conn)
        if actual != epoca or edad is None or edad > LATIDO_MAX_HIJO_S:
            raise ConcesionPerdida(f"la concesión de la época {epoca} ya no es vigente")

    comunes = {"tracking_uri": entrada["tracking_uri"], "nombre": entrada["nombre"],
               "git_commit": entrada["git_commit"], "epoca": epoca, "comprobar": comprobar,
               "etiquetas": entrada.get("etiquetas") or {}}
    if entrada["tarea"] == "candidato":
        motor = crear_motor(url, timeout_consulta=300)        # proceso por lotes: límite amplio
        try:
            kw = {}
            if entrada.get("config"):
                kw["config"] = ConfigEntrenamiento(**entrada["config"])
            if entrada.get("evaluacion"):
                kw["evaluacion"] = ConfigEvaluacion(**entrada["evaluacion"])
            r = registrar_version(motor, origen=entrada.get("origen", "manual"), **comunes, **kw)
        finally:
            motor.dispose()
    elif entrada["tarea"] == "respaldo":
        r = registrar_respaldo(run_candidato=entrada["run_candidato"], **comunes)
    else:
        raise ValueError(f"tarea desconocida: {entrada['tarea']!r}")
    with open(argv[1], "w", encoding="utf-8") as fh:
        json.dump(r.como_dict(), fh)
    return 0


if __name__ == "__main__":
    try:
        codigo = main(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        nombre = type(exc).__name__
        codigo = CODIGO_CONCESION_PERDIDA if nombre == "ConcesionPerdida" else 1
    sys.exit(codigo)
