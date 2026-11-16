"""Cierre: con algún criterio incumplido no se limpia el entorno y se guardan los logs de
los servicios en ``trabajo/logs``; solo una ejecución sin fallos limpia. Sin Docker: un
ejecutor simulado registra los comandos y responde por servicio."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from simulador import costos, datos, evidencia, simulacion
from simulador import entorno as ent
from simulador import estado as est
from simulador.escenario import CORTO
from tests.ml.sinteticos import escribir_carpeta, generar_contrato

SERVICIOS = ["db", "airflow-db", "mlflow", "preparar", "entrenamiento-inicial", "api", "airflow-init",
             "airflow-scheduler", "airflow-webserver"]
ESC = replace(CORTO, dias=3, corte="2026-04-05")


class Ejecutor:
    def __init__(self):
        self.comandos = []

    def __call__(self, cmd):
        self.comandos.append(cmd)
        if "config" in cmd and "--services" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "\n".join(SERVICIOS) + "\n", "")
        if "logs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, f"2026-11-14T09:00:00Z línea de {cmd[-1]}\n", "")
        if cmd[:3] == ["docker", "ps", "-a"]:
            bajado = any("down" in c for c in self.comandos)
            return subprocess.CompletedProcess(cmd, 0, "" if bajado else "nm-sim-abc12345-api-1\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def destructivos(self):
        return [c for c in self.comandos if any(x in c for x in ("down", "stop", "rm", "kill", "prune"))]


@pytest.fixture
def sim(tmp_path):
    fuente = tmp_path / "fuente"
    escribir_carpeta(generar_contrato(fin="2026-04-27", dias=120), fuente)
    carpeta = tmp_path / "ej"
    carpeta.mkdir()
    datos.preparar(ESC, fuente, carpeta, "a" * 40)
    trabajo = carpeta / "trabajo"
    dias = datos.leer_dias(trabajo)
    with open(trabajo / "respuestas.jsonl", "w") as fh:
        for i, f in enumerate(sorted(dias)):
            for t in sorted(dias[f]):
                preds = [{"familia_id": fam, "venta_esperada": float(v), "cantidad_sugerida": v}
                         for fam, (v, _) in dias[f][t].items()]
                fh.write(json.dumps({"fase": "resultado", "tipo": "prediccion", "codigo": 200, "seq": i * 10 + t,
                                     "fecha": f.isoformat(), "tienda": t,
                                     "respuesta": {"version_modelo": "1", "predicciones": preds}}) + "\n")
    estado = est.Estado(trabajo)
    estado.datos = {"fallos": [], "fase": "en_curso"}
    x = Ejecutor()
    entorno = ent.Entorno("nm-sim-abc12345", trabajo / "e.env", dict(ent.PUERTOS_POR_DEFECTO),
                          {"db_usuario": "u", "db_nombre": "d"}, ejecutar=x)
    ns = SimpleNamespace(estado=estado, entorno=entorno, evid=carpeta / "evidencia", trabajo=trabajo,
                         datos_dir=carpeta / "datos", esc=ESC, verificar_final=lambda: None, ejecutor=x)
    ns.calcular_costos = lambda: costos.calcular(*costos.leer_evidencia(ns.evid, trabajo / "respuestas.jsonl"))
    ns.fallo = lambda c, d, dia=None: estado["fallos"].append({"criterio": c, "detalle": d, "dia": dia})
    return ns


def test_con_fallo_no_limpia_y_guarda_logs(sim):
    sim.estado["fallos"].append({"criterio": 6, "detalle": "x", "dia": 71})
    simulacion.cerrar(sim)
    assert sim.ejecutor.destructivos() == []                      # contenedores y volúmenes intactos
    assert sim.datos_dir.is_dir()                                  # datos/ conservada
    logs = sim.trabajo / "logs"
    for s in ("airflow-scheduler", "api", "db"):
        assert f"línea de {s}" in (logs / f"{s}.log").read_text()
    assert {p.name for p in logs.iterdir()} == {f"{s}.log" for s in SERVICIOS} | {"ps.txt", "recursos.json"}
    assert all("--timestamps" in c for c in sim.ejecutor.comandos if "logs" in c)
    assert sim.estado["limpieza"]["omitida"] is True and evidencia.sellada(sim.evid)
    assert sim.estado["recalculo"]["filas"] == sim.estado["costos"]["filas"]     # criterio 10 sin limpiar
    assert sim.estado["fase"] == "cerrada"


def test_sin_fallos_limpia(sim):
    simulacion.cerrar(sim)
    assert [c for c in sim.ejecutor.comandos if "down" in c] == [sim.entorno.comando("down", "-v", "--remove-orphans")]
    assert not sim.datos_dir.exists() and not (sim.trabajo / "logs").exists()
    assert evidencia.sellada(sim.evid) and sim.estado["limpieza"]["datos_borrados"]


def test_detener_en_criterio_espera_un_punto_seguro():
    estado = type("E", (dict,), {"guardar": lambda self: None})(fallos=[])
    ns = SimpleNamespace(estado=estado, log=lambda m: None, detener_en=(6,), detencion=None)
    simulacion.Simulacion.fallo(ns, 4, "otro criterio")
    simulacion.Simulacion.fallo(ns, 6, "plazo")                    # no lanza aquí: termina de evaluar
    assert ns.detencion == 6 and len(ns.estado["fallos"]) == 2
    with pytest.raises(simulacion.Detener, match="criterio 6"):
        simulacion.Simulacion.revisar_detencion(ns)


def test_guardar_logs_no_toca_el_entorno(tmp_path):
    x = Ejecutor()
    e = ent.Entorno("nm-sim-abc12345", tmp_path / "e.env", dict(ent.PUERTOS_POR_DEFECTO), {}, ejecutar=x)
    e.guardar_logs(tmp_path / "logs")
    assert x.destructivos() == [] and oct((tmp_path / "logs").stat().st_mode & 0o777) == "0o700"
