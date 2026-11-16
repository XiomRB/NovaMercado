"""Ciclo de la simulación: preparar, levantar, operar los días, cerrar y verificar.

Las solicitudes se envían **de una en una** (el emparejamiento con PostgreSQL depende del
orden). Los criterios de aceptación 1–11 del plan v3 se registran como ``fallos`` con su
número; la ejecución sigue cuando es seguro (para reunir evidencia) y se detiene
(``Detener``) cuando no lo es.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from simulador import contrato, costos, datos, defectos, evidencia, imagenes
from simulador import entorno as ent
from simulador import estado as est
from simulador.cliente import Cliente
from simulador.escenario import INTERVALO_SONDEO, Escenario

MODELO = "novamercado-ventas"


class Detener(Exception):
    """No es seguro continuar: se cierra con el informe parcial."""


class Interrumpida(Exception):
    """Interrupción pedida con --interrumpir-tras-dia (pruebas de reanudación)."""


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ent.RAIZ, check=True).stdout.strip()


def arbol_limpio() -> bool:
    return git("status", "--porcelain", "--untracked-files=normal") == ""


class Memoria:
    """Lo que el simulador sabe que el producto debería tener, reconstruido de
    ``respuestas.jsonl`` (así sobrevive a una interrupción)."""

    def __init__(self, ultimas_iniciales: dict, maximos: dict):
        self.guardadas: dict[tuple[date, int, str], int] = {}
        self.ultima = dict(ultimas_iniciales)             # (tienda, familia) -> última fecha con venta guardada
        self.maximo_seguro = dict(maximos["reciente"])
        self.maximo_posible = dict(maximos["total"])

    def aplicar(self, r: dict) -> None:
        if r.get("fase") != "resultado" or r.get("codigo") != 200:
            return
        f, t, esp = date.fromisoformat(r["fecha"]), r["tienda"], r["esperado"]
        if r["tipo"] == "observacion":
            for fam, v in esp["valores"].items():
                self.guardadas[(f, t, fam)] = v
                if f > self.ultima.get((t, fam), date.min):
                    self.ultima[(t, fam)] = f
        else:
            for fam, v in esp["valores"].items():
                if fam not in esp["supuestas"]:
                    self.maximo_posible[fam] = max(self.maximo_posible.get(fam, 0), v)

    def atrasada(self, tienda: int, fecha: date, familias: list[str]) -> bool:
        return any(self.ultima.get((tienda, f), date.min) < fecha - timedelta(days=1) for f in familias)

    def registros_esperados(self, tienda: int, fecha: date, valores: dict[str, int]) -> int:
        return sum(1 for fam, v in valores.items() if self.guardadas.get((fecha, tienda, fam)) != v)


class Simulacion:
    def __init__(self, carpeta: Path, esc: Escenario, estado: est.Estado, entorno: ent.Entorno,
                 interrumpir_tras: int | None = None, salida=print, detener_en: tuple[int, ...] = ()):
        self.carpeta, self.esc, self.estado, self.entorno = Path(carpeta), esc, estado, entorno
        self.trabajo, self.evid = self.carpeta / "trabajo", self.carpeta / "evidencia"
        self.datos_dir = self.carpeta / "datos"
        self.api = Cliente(entorno.puertos["API_PUERTO_HOST"])
        self.mlflow = Cliente(entorno.puertos["MLFLOW_PUERTO_HOST"], timeout=10)
        self.interrumpir_tras = interrumpir_tras
        self.detener_en = tuple(detener_en)
        self.detencion: int | None = None
        self.salida = salida
        self.tiendas, fams = datos.leer_catalogo(self.datos_dir)
        self.familias = [f for f, _ in fams]
        self.catalogo = defectos.Catalogo(self.tiendas, self.familias, dict(fams))
        self.dias = datos.leer_dias(self.trabajo)
        self.mem = Memoria(datos.ultimas_ventas_iniciales(self.datos_dir), estado["s1"]["maximos_promocion"])
        self.registros = estado.respuestas.leer()
        for r in self.registros:
            self.mem.aplicar(r)
        self.ultimo_sondeo = 0.0
        self.del_dia: list[dict] = []

    # ------------------------------------------------------------- utilidades
    def log(self, msg: str) -> None:
        self.salida(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def fallo(self, criterio: int, detalle: str, dia: int | None = None) -> None:
        self.estado["fallos"].append({"criterio": criterio, "detalle": detalle[:600], "dia": dia, "t": time.time()})
        self.estado.guardar()
        self.log(f"FALLO criterio {criterio}: {detalle[:300]}")
        if criterio in self.detener_en and self.detencion is None:
            self.detencion = criterio            # se detiene en el siguiente punto seguro

    def revisar_detencion(self) -> None:
        if self.detencion is not None:
            raise Detener(f"falló el criterio {self.detencion} (--detener-en-criterio)")

    def medir(self, clave: str, valor) -> None:
        self.estado["mediciones"].setdefault(clave, []).append(valor)

    def salud(self) -> tuple[int | None, dict]:
        r = self.api.get("/health", timeout=10)
        return r.codigo, r.cuerpo if isinstance(r.cuerpo, dict) else {}

    def esperar_salud(self, objetivo: str, limite: float, paso: float = 1.0) -> float | None:
        t = time.monotonic()
        while time.monotonic() - t < limite:
            c, s = self.salud()
            if c == 200 and s.get("estado") == objetivo:
                return time.monotonic() - t
            time.sleep(paso)
        return None

    def mlflow_json(self, ruta: str, cuerpo=None) -> dict:
        r = (self.mlflow.post(ruta, json.dumps(cuerpo).encode()) if cuerpo is not None else self.mlflow.get(ruta))
        return r.cuerpo if r.codigo == 200 and isinstance(r.cuerpo, dict) else {}

    def versiones_mlflow(self) -> list[dict]:
        return self.mlflow_json(f"/api/2.0/mlflow/model-versions/search?filter=name%3D%27{MODELO}%27").get(
            "model_versions", [])

    def conteos(self) -> dict:
        exp = self.mlflow_json("/api/2.0/mlflow/experiments/get-by-name?experiment_name=novamercado-entrenamiento")
        runs = 0
        if "experiment" in exp:
            runs = len(self.mlflow_json("/api/2.0/mlflow/runs/search", {
                "experiment_ids": [exp["experiment"]["experiment_id"]], "max_results": 1000}).get("runs", []))
        pg = self.entorno.psql_json(
            "SELECT (SELECT count(*) FROM solicitudes_prediccion) s, (SELECT count(*) FROM envios_observacion) e, "
            "(SELECT count(*) FROM observaciones) o, (SELECT count(*) FROM resoluciones_promocion) r")[0]
        return {"ejecuciones_mlflow": runs, "versiones_mlflow": len(self.versiones_mlflow()), **pg}

    def desfase_pg(self) -> float:
        a = time.time()
        t = self.entorno.psql_json("SELECT extract(epoch FROM clock_timestamp())::float8 AS t")[0]["t"]
        b = time.time()
        return t - (a + b) / 2

    # ------------------------------------------------------------- versiones (criterio 4)
    def ver_version(self, v: str | None, cuando: float, origen: str, dia) -> None:
        if not v:
            return
        vistas = self.estado["versiones_vistas"]
        if not vistas or vistas[-1]["version"] != v:
            if any(x["version"] == v for x in vistas):
                self.fallo(4, f"retroceso: {origen} devolvió la versión {v} después de {vistas[-1]['version']}", dia)
                return
            vistas.append({"version": v, "t": cuando, "origen": origen})

    def modelo(self, dia) -> tuple[str | None, float]:
        r = self.api.get("/modelo")
        if r.codigo != 200 or contrato.modelo(r.cuerpo):
            self.fallo(2, f"/modelo {r.codigo}: {r.cuerpo!r}"[:300], dia)
            return None, r.fin
        self.ver_version(r.cuerpo["version"], r.fin, "/modelo", dia)
        return r.cuerpo["version"], r.fin

    # ------------------------------------------------------------- solicitudes
    def enviar(self, tipo: str, dia: int, tienda: int, fecha: date, sol: defectos.Solicitud, rol: str) -> dict:
        self.estado["seq"] += 1
        seq = self.estado["seq"]
        base = {"seq": seq, "tipo": tipo, "dia": dia, "tienda": tienda, "fecha": fecha.isoformat(), "rol": rol,
                "defectos": sol.defectos}
        self.estado.respuestas.agregar({**base, "fase": "intencion"})
        va = vd = None
        if tipo == "prediccion":
            va, _ = self.modelo(dia)
        r = self.api.post("/v1/prediccion" if tipo == "prediccion" else "/v1/observaciones", sol.bytes())
        if tipo == "prediccion":
            if r.codigo == 200 and isinstance(r.cuerpo, dict):      # en orden temporal: antes, respuesta, después
                self.ver_version(r.cuerpo.get("version_modelo"), r.fin, "prediccion", dia)
            vd, _ = self.modelo(dia)
        errores = self.validar(tipo, dia, tienda, fecha, sol, r, va, vd)
        reg = {**base, "fase": "resultado", "codigo": r.codigo, "respuesta": r.cuerpo,
               "esperado": sol.esperado.como_dict(),
               "cuerpo": sol.cuerpo, "texto": sol.texto, "va": va, "vd": vd, "inicio": r.inicio, "fin": r.fin,
               "errores": errores}
        self.estado.respuestas.agregar(reg)
        self.mem.aplicar(reg)
        self.registros.append(reg)
        self.del_dia.append(reg)
        self.sondear()
        return reg

    def validar(self, tipo, dia, tienda, fecha, sol, r, va, vd) -> list[str]:
        errores: list[tuple[int, str]] = []
        if r.codigo is None or r.codigo >= 500:
            errores.append((2, f"respuesta {r.codigo} {r.error or r.cuerpo!r}"))
        elif r.codigo == 422:
            errores += [(2, e) for e in contrato.rechazo(r.cuerpo)]
        elif r.codigo == 200:
            errores += [(2, e) for e in (contrato.prediccion(r.cuerpo, tienda, fecha.isoformat(), self.familias)
                                         if tipo == "prediccion" else contrato.observacion(r.cuerpo))]
        if r.codigo in (200, 422):
            if tipo == "prediccion":
                req, opc = defectos.contexto_prediccion(sol.esperado, self.mem.maximo_seguro, self.mem.maximo_posible,
                                                        self.mem.atrasada(tienda, fecha, self.familias))
            else:
                req, opc = [], []
            errores += [(3, e) for e in defectos.comparar(sol.esperado, r.codigo, r.cuerpo, req, opc)]
        if tipo == "observacion" and r.codigo == 200 == sol.esperado.codigo and isinstance(r.cuerpo, dict):
            esperado = self.mem.registros_esperados(tienda, fecha, sol.esperado.valores)
            if r.cuerpo.get("registros_guardados") != esperado:
                errores.append((5, f"registros_guardados {r.cuerpo.get('registros_guardados')} (esperado {esperado})"))
        if tipo == "prediccion" and r.codigo == 200 and isinstance(r.cuerpo, dict):
            vp = r.cuerpo.get("version_modelo")
            permitidas = {va, vd} - {None}
            if va == vd and vp != va or vp not in permitidas:
                errores.append((4, f"version_modelo {vp} con /modelo {va} antes y {vd} después"))
        for criterio, e in errores:
            self.fallo(criterio, f"{tipo} {fecha} tienda {tienda} ({','.join(sol.defectos) or 'limpia'}): {e}", dia)
        return [e for _, e in errores]

    def solicitud(self, tipo, tienda, fecha, rng, con_defectos: bool) -> defectos.Solicitud:
        valores = {f: (v if tipo == "observacion" else p) for f, (v, p) in self.dias[fecha][tienda].items()}
        nombres = defectos.elegir(rng, tipo) if con_defectos else []
        return defectos.generar(tipo, tienda, fecha, valores, self.catalogo, rng, nombres)

    def tienda_dia(self, tipo: str, dia: int, fecha: date, tienda: int) -> None:
        """Solicitud del día (con defectos en una fracción), reenvío limpio si se rechazó (S4)
        y, a veces, una repetición limpia."""
        rng = random.Random(f"{self.esc.semilla}-{tipo}-{fecha}-{tienda}")
        tasa = self.esc.tasa_defecto_prediccion if tipo == "prediccion" else self.esc.tasa_defecto_observacion
        r = self.enviar(tipo, dia, tienda, fecha, self.solicitud(tipo, tienda, fecha, rng, rng.random() < tasa),
                        "principal")
        if r["codigo"] == 422:
            self.enviar(tipo, dia, tienda, fecha, self.solicitud(tipo, tienda, fecha, rng, False), "reenvio_limpio")
        if rng.random() < self.esc.tasa_repeticion:
            self.enviar(tipo, dia, tienda, fecha, self.solicitud(tipo, tienda, fecha, rng, False), "repeticion")

    # ------------------------------------------------------------- reentrenamientos (criterio 6)
    def pendientes(self) -> list[dict]:
        return [e for e in self.estado["ejecuciones"] if e.get("id") and not e.get("terminal")]

    def sondear(self, forzar: bool = False) -> None:
        if not forzar and time.monotonic() - self.ultimo_sondeo < INTERVALO_SONDEO:
            return
        self.ultimo_sondeo = time.monotonic()
        for e in self.pendientes():
            r = self.api.get(f"/admin/reentrenar/{e['id']}")
            if r.codigo != 200 or contrato.estado_ejecucion(r.cuerpo):
                self.fallo(2, f"GET /admin/reentrenar/{e['id']}: {r.codigo} {r.cuerpo!r}", e["dia"])
                continue
            if r.cuerpo["estado"] != "en_curso":
                self.terminar(e, r.cuerpo, r.fin)
                self.revisar_detencion()
            elif time.time() - e["t_202"] > self.esc.plazos["reentrenamiento"] + INTERVALO_SONDEO:
                self.fallo(6, f"{e['id']} sigue en_curso más de {self.esc.plazos['reentrenamiento']} s", e["dia"])
                e["terminal"] = {"estado": "excedido"}
                self.estado.guardar()
                raise Detener(f"reentrenamiento {e['id']} excedió el plazo")

    def terminar(self, e: dict, cuerpo: dict, visto: float) -> None:
        e["terminal"], e["t_visto"] = cuerpo, visto
        esperado = "fallido" if e["tipo"] == "caida" else "exitoso"
        if cuerpo["estado"] != esperado:
            self.fallo(6, f"{e['tipo']} {e['id']}: {cuerpo['estado']} (se esperaba {esperado}): {cuerpo.get('motivo')}",
                       e["dia"])
        fin_pg = self.entorno.psql_json(
            "SELECT extract(epoch FROM coalesce((SELECT terminado_en FROM resultados_reentrenamiento WHERE id = "
            f"'{e['id']}'), (SELECT decidido_en FROM veredictos_reentrenamiento WHERE id = '{e['id']}')))::float8 AS t")
        t_fin = (fin_pg[0]["t"] - self.desfase_pg()) if fin_pg and fin_pg[0]["t"] else visto
        e["duracion"] = t_fin - e["t_202"]
        self.medir("reentrenamiento_s", round(e["duracion"], 1))
        if e["duracion"] > self.esc.plazos["reentrenamiento"]:
            self.fallo(6, f"{e['id']}: {e['duracion']:.0f} s desde el 202 "
                          f"(límite {self.esc.plazos['reentrenamiento']})",
                       e["dia"])
        if cuerpo["promovido"]:
            if e["tipo"] == "caida":
                self.fallo(7, f"{e['id']} promovió con MLflow detenido", e["dia"])
            self.esperar_promocion(e)
        else:
            v, _ = self.modelo(e["dia"])
            if v != cuerpo["version"]:
                self.fallo(6, f"{e['id']} no promovió (version {cuerpo['version']}) y /modelo sirve {v}", e["dia"])
        self.registrar_diagnostico_r2(e)
        self.log(f"reentrenamiento {e['tipo']} día {e['dia']}: {cuerpo['estado']} promovido={cuerpo['promovido']} "
                 f"versión {cuerpo['version']} en {e['duracion']:.0f} s")
        self.estado.guardar()

    def registrar_diagnostico_r2(self, e: dict) -> None:
        """Solo lectura (S8), sin efecto en los criterios: horas de la ejecución en PostgreSQL
        (a reloj del equipo), la última observación que el simulador había enviado en cada una,
        y las versiones registradas con su corte y R2 (tags y métricas de MLflow)."""
        try:
            d = self.desfase_pg()
            filas = self.entorno.psql_json(
                "SELECT extract(epoch FROM s.solicitado_en)::float8 AS solicitada, "
                "extract(epoch FROM t.iniciada_en)::float8 AS iniciada, "
                "extract(epoch FROM r.reclamada_en)::float8 AS reclamada, "
                "extract(epoch FROM x.terminado_en)::float8 AS terminada, x.escrito_por, r.epoca "
                "FROM solicitudes_reentrenamiento s LEFT JOIN tareas_reentrenamiento t USING (id) "
                "LEFT JOIN reclamaciones_reentrenamiento r USING (id) LEFT JOIN resultados_reentrenamiento x "
                f"USING (id) WHERE s.id = '{e['id']}'")
            tiempos = {k: (v - d if isinstance(v, float) else v) for k, v in (filas[0] if filas else {}).items()}

            def ultima_obs(t):
                envs = [r for r in self.registros if r["tipo"] == "observacion" and r["fase"] == "resultado"
                        and r["codigo"] == 200 and t is not None and r["fin"] <= t]
                return max((r["fecha"] for r in envs), default=None)
            versiones = []
            for v in self.versiones_mlflow():
                tags = {t["key"]: t["value"] for t in v.get("tags", [])}
                run = self.mlflow_json(f"/api/2.0/mlflow/runs/get?run_id={v.get('run_id')}").get("run", {})
                met = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])
                       if m["key"].startswith("r2")}
                versiones.append({"version": v["version"], "corte": tags.get("corte"), "estado": tags.get("estado"),
                                  "r2": tags.get("r2"), "r2_motivo": tags.get("r2_motivo"), "metricas": met})
            diag = {"id": e["id"], "dia": e["dia"], "tipo": e["tipo"], "t_202": e.get("t_202"), "pg": tiempos,
                    "ultima_observacion_enviada": {k: ultima_obs(tiempos.get(k)) for k in
                                                   ("solicitada", "iniciada", "reclamada", "terminada")},
                    "versiones": versiones}
            destino = self.trabajo / "r2"
            destino.mkdir(exist_ok=True)
            (destino / f"dia{e['dia']:02d}-{e['tipo']}.json").write_text(json.dumps(diag, ensure_ascii=False,
                                                                                   indent=1))
        except Exception as exc:                 # diagnóstico: nunca interrumpe la simulación
            self.estado.evento("diagnostico_r2_no_guardado", id=e["id"], error=str(exc)[:300])

    def esperar_promocion(self, e: dict) -> None:
        v = e["terminal"]["version"]
        info = self.mlflow_json(f"/api/2.0/mlflow/model-versions/get?name={MODELO}&version={v}").get(
            "model_version", {})
        run = self.mlflow_json(f"/api/2.0/mlflow/runs/get?run_id={info.get('run_id')}").get("run", {}).get("info", {})
        fin_entrenamiento = int(run["end_time"]) / 1000 if run.get("end_time") else e["t_visto"]
        limite = fin_entrenamiento + self.esc.plazos["promocion"]
        visto = next((x["t"] for x in self.estado["versiones_vistas"] if x["version"] == v), None)
        while visto is None and time.time() < limite + 1:
            self.modelo(e["dia"])
            visto = next((x["t"] for x in self.estado["versiones_vistas"] if x["version"] == v), None)
            time.sleep(0.5)
        if visto is None or visto - fin_entrenamiento > self.esc.plazos["promocion"]:
            self.fallo(6, f"la versión {v} no se sirvió en {self.esc.plazos['promocion']} s tras el fin del "
                          f"entrenamiento", e["dia"])
        else:
            e["promocion_s"] = round(visto - fin_entrenamiento, 1)
            self.medir("promocion_s", e["promocion_s"])

    def esperar_sin_pendientes(self) -> None:
        while self.pendientes():
            self.sondear(forzar=True)
            if self.pendientes():
                self.modelo(None)
                time.sleep(INTERVALO_SONDEO)

    def pedir_reentrenamiento(self, dia: int, tipo: str, clave: str | None) -> None:
        if any(x["dia"] == dia and x["tipo"] == tipo and x.get("id") for x in self.estado["ejecuciones"]):
            return                                                       # ya pedido (reanudación)
        self.esperar_sin_pendientes()
        e = next((x for x in self.estado["ejecuciones"] if x["dia"] == dia and x["tipo"] == tipo), None)
        if e is not None and clave is None:
            e["terminal"], e["id"] = {"estado": "no_verificable"}, "?"
            self.fallo(6, f"día {dia}: la solicitud sin clave quedó sin id tras una interrupción (no verificable)", dia)
            return
        if e is None:
            e = {"dia": dia, "tipo": tipo, "clave": clave, "id": None, "t_intencion": time.time()}
            self.estado["ejecuciones"].append(e)
            self.estado.guardar()
        r = self.api.post("/admin/reentrenar", cabeceras={"Idempotency-Key": clave} if clave else None)
        if r.codigo != 202 or contrato.aceptado(r.cuerpo):
            self.fallo(6, f"POST /admin/reentrenar día {dia}: {r.codigo} {r.cuerpo!r}", dia)
            e["id"], e["terminal"] = "?", {"estado": "no_iniciado"}
            self.estado.guardar()
            return
        e["id"], e["t_202"] = r.cuerpo["id"], r.fin
        self.estado.guardar()
        self.log(f"reentrenamiento {tipo} día {dia}: 202 {e['id']}")
        if dia == self.esc.dia_repetir_clave and clave:
            r2 = self.api.post("/admin/reentrenar", cabeceras={"Idempotency-Key": clave})
            if (r2.codigo, r2.cuerpo) != (202, {"id": e["id"]}):
                self.fallo(6, f"la misma Idempotency-Key devolvió {r2.codigo} {r2.cuerpo!r} (id {e['id']})", dia)
            e["repeticion"] = r2.cuerpo
            self.estado.guardar()

    # ------------------------------------------------------------- eventos de operación
    def instantanea(self) -> dict:
        return {"conteos": self.conteos(), "version": self.modelo(None)[0],
                "ejecuciones": {e["id"]: e["terminal"] for e in self.estado["ejecuciones"]
                                if e.get("terminal") and e["id"] != "?"}}

    def reinicio(self, dia: int, como: str) -> None:
        self.esperar_sin_pendientes()
        antes = self.instantanea()
        self.log(f"reinicio {como} (día {dia})")
        r = self.entorno.detener() if como == "stop/start" else self.entorno.bajar()
        if r.returncode != 0:
            raise Detener(f"{como}: {r.stderr[-400:]}")
        self.verificar_imagenes(dia)
        r = self.entorno.iniciar() if como == "stop/start" else self.entorno.levantar()
        if r.returncode != 0:
            raise Detener(f"{como}: {r.stderr[-400:]}")
        seg = self.esperar_salud("ok", self.esc.plazos["arranque"], 2)
        if seg is None:
            self.fallo(8, f"{como}: /health no volvió a ok en {self.esc.plazos['arranque']} s", dia)
            raise Detener("el producto no volvió a ok")
        self.medir(f"reinicio_{como}_s", round(seg, 1))
        for err in imagenes.verificar_contenedores(self.entorno, self.estado):
            self.fallo(11, err, dia)
        despues = self.instantanea()
        for clave in ("conteos", "version"):
            if antes[clave] != despues[clave]:
                self.fallo(8, f"{como}: {clave} antes {antes[clave]} y después {despues[clave]}", dia)
        for id_, terminal in antes["ejecuciones"].items():
            r = self.api.get(f"/admin/reentrenar/{id_}")
            if r.codigo != 200 or r.cuerpo != terminal:
                self.fallo(8, f"{como}: {id_} devuelve {r.codigo} {r.cuerpo!r}, antes {terminal!r}", dia)
        self.log(f"reinicio {como}: /health ok en {seg:.0f} s; versión {despues['version']}; datos conservados")

    def caida_mlflow(self, dia: int) -> None:
        self.esperar_sin_pendientes()
        self.log(f"MLflow detenido (día {dia})")
        self.estado["mlflow_detenido"] = True
        self.estado.guardar()
        self.entorno.detener_servicio("mlflow")
        seg = self.esperar_salud("degradado", self.esc.plazos["degradado"], 1)
        if seg is None:
            self.fallo(7, f"/health no pasó a degradado en {self.esc.plazos['degradado']} s", dia)
        else:
            self.medir("mlflow_degradado_s", round(seg, 1))

    def volver_mlflow(self, dia: int) -> None:
        self.esperar_sin_pendientes()                 # el reentrenamiento de la caída debe terminar antes
        self.entorno.iniciar_servicio("mlflow")
        seg = self.esperar_salud("ok", self.esc.plazos["recuperado"], 1)
        if seg is None:
            self.fallo(7, f"/health no volvió a ok en {self.esc.plazos['recuperado']} s tras volver MLflow", dia)
        else:
            self.medir("mlflow_recuperado_s", round(seg, 1))
        self.estado["mlflow_detenido"] = False
        self.estado.guardar()
        self.log(f"MLflow de vuelta: /health ok en {seg or 0:.0f} s")

    def verificar_imagenes(self, dia) -> None:
        try:
            ev = imagenes.preparar(self.entorno, self.estado, self.estado["commit"], git("rev-parse", "HEAD"),
                                   arbol_limpio())
        except imagenes.ImagenNoValida as exc:
            # se niega antes de levantar nada; no es un resultado de la simulación (exit 3)
            self.estado.evento("imagen_rechazada", detalle=str(exc), dia=dia)
            raise
        for x in ev:
            if x["accion"] != "reutilizada":
                self.estado.evento("imagen", **x)

    # ------------------------------------------------------------- PostgreSQL (S8)
    def verificar_pg(self, dia) -> None:
        ult = self.estado["verificado_pg"]
        preds = [r for r in self.registros if r["tipo"] == "prediccion" and r["fase"] == "resultado"
                 and r["seq"] > ult["seq"]]
        obs = [r for r in self.registros if r["tipo"] == "observacion" and r["fase"] == "resultado"
               and r["seq"] > ult["seq"]]
        sols = self.entorno.psql_json(
            "SELECT solicitud_id AS id, resultado, tienda_id, fecha::text AS fecha, version_modelo FROM "
            f"solicitudes_prediccion WHERE solicitud_id > {ult['solicitud']} ORDER BY solicitud_id")
        filas = self.entorno.psql_json(
            "SELECT solicitud_id AS id, familia_id, venta_esperada, cantidad_sugerida FROM predicciones "
            f"WHERE solicitud_id > {ult['solicitud']}")
        por_sol = defaultdict(dict)
        for f in filas:
            por_sol[f["id"]][f["familia_id"]] = f
        envios = self.entorno.psql_json(
            "SELECT envio_id AS id, resultado, tienda_id, fecha::text AS fecha, registros_nuevos + "
            f"registros_modificados AS guardados FROM envios_observacion WHERE envio_id > {ult['envio']} "
            "ORDER BY envio_id")
        extra = 1 if self.estado.get("intencion_sin_resultado") else 0
        for nombre, regs, db in (("predicciones", preds, sols), ("observaciones", obs, envios)):
            if not (len(regs) <= len(db) <= len(regs) + extra):
                self.fallo(4 if nombre == "predicciones" else 5,
                           f"PostgreSQL: {len(db)} {nombre} registradas, el simulador envió {len(regs)}", dia)
                continue
            db = db[len(db) - len(regs):]
            for r, fila in zip(regs, db):
                self.comparar_fila(nombre, r, fila, por_sol, dia)
        ult["seq"] = max([r["seq"] for r in preds + obs] + [ult["seq"]])
        ult["solicitud"] = max([s["id"] for s in sols] + [ult["solicitud"]])
        ult["envio"] = max([s["id"] for s in envios] + [ult["envio"]])
        self.estado["intencion_sin_resultado"] = False

    def comparar_fila(self, nombre, r, fila, por_sol, dia) -> None:
        ok_esperado = "respondida" if nombre == "predicciones" else "aceptado"
        if r["codigo"] == 200:
            errores = []
            if fila["resultado"] != ok_esperado or fila["tienda_id"] != r["tienda"] or fila["fecha"] != r["fecha"]:
                errores.append(f"fila {fila}")
            if nombre == "predicciones":
                resp = r["respuesta"]
                if fila["version_modelo"] != resp["version_modelo"]:
                    errores.append(f"versión registrada {fila['version_modelo']!r} ≠ devuelta "
                                   f"{resp['version_modelo']!r}")
                registradas = por_sol.get(fila["id"], {})
                for p in resp["predicciones"]:
                    g = registradas.get(p["familia_id"])
                    if (g is None or g["cantidad_sugerida"] != p["cantidad_sugerida"]
                            or abs(g["venta_esperada"] - p["venta_esperada"]) > 1e-9):
                        errores.append(f"{p['familia_id']}: registrada {g} ≠ devuelta {p}")
            elif fila["guardados"] != r["respuesta"]["registros_guardados"]:
                errores.append(f"guardados {fila['guardados']} ≠ {r['respuesta']['registros_guardados']}")
            for e in errores:
                self.fallo(4 if nombre == "predicciones" else 5, f"PostgreSQL seq {r['seq']}: {e}", dia)
        elif fila["resultado"] == ok_esperado:
            self.fallo(4 if nombre == "predicciones" else 5,
                       f"PostgreSQL seq {r['seq']}: respondió {r['codigo']} y quedó {fila['resultado']}", dia)

    # ------------------------------------------------------------- arranque y días
    def arrancar(self) -> None:
        e = self.estado
        e["protegidas_antes"] = imagenes.ids_protegidas(self.entorno.ejecutar)
        self.verificar_imagenes(None)
        t = time.monotonic()
        r = self.entorno.levantar()
        if r.returncode != 0:
            raise Detener(f"up falló: {r.stderr[-600:]}")
        seg = self.esperar_salud("ok", self.esc.plazos["arranque"], 2)
        if seg is None:
            self.fallo(1, f"el producto no llegó a /health ok en {self.esc.plazos['arranque']} s", 0)
            raise Detener("sin /health ok")
        self.medir("arranque_s", round(time.monotonic() - t, 1))
        for err in imagenes.verificar_contenedores(self.entorno, e):
            self.fallo(11, err, 0)
        self.comprobar_s1()
        v, _ = self.modelo(0)
        e["version_inicial"] = v
        e["fase"] = "en_curso"
        e.guardar()
        self.log(f"producto listo en {seg:.0f} s con la versión {v}")

    def comprobar_s1(self) -> None:
        s1 = self.estado["s1"]
        hist = self.entorno.psql_json("SELECT max(fecha)::text AS m, count(*) AS n FROM ventas_historicas")[0]
        nobs = self.entorno.psql_json("SELECT count(*) AS n FROM observaciones")[0]["n"]
        cargas = {c["archivo"]: c["sha256"] for c in self.entorno.psql_json("SELECT archivo, sha256 FROM cargas_datos")}
        tags = {}
        vs = self.versiones_mlflow()
        if vs:
            primera = min(vs, key=lambda x: int(x["version"]))
            tags = {t["key"]: t["value"] for t in primera.get("tags", [])}
        problemas = []
        if hist["m"] != self.esc.corte:
            problemas.append(f"ventas_historicas termina en {hist['m']}, no en {self.esc.corte}")
        if nobs != 0:
            problemas.append(f"observaciones tiene {nobs} filas antes del día 1")
        distintas = {a: s for a, s in cargas.items() if s1["sha256_datos"].get(a) != s}
        if set(cargas) != set(s1["sha256_datos"]) or distintas:
            problemas.append(f"cargas_datos no coincide con los SHA-256 de la carpeta: {sorted(cargas)} {distintas}")
        if tags.get("corte") != self.esc.inicio.isoformat():
            problemas.append(f"el primer modelo tiene corte {tags.get('corte')!r}, no {self.esc.inicio}")
        s1["verificacion"] = {"max_fecha": hist["m"], "filas": hist["n"], "observaciones": nobs,
                              "corte_modelo": tags.get("corte"), "problemas": problemas}
        for p in problemas:
            self.fallo(1, f"S1: {p}", 0)
        if problemas:
            raise Detener("las comprobaciones de S1 fallaron")

    def operar(self) -> None:
        e = self.estado
        for dia in range(e["dia_completo"] + 1, self.esc.dias + 2):
            cierre = dia == self.esc.dias + 1
            fecha = self.esc.fecha(dia) if not cierre else None
            self.del_dia = []
            if not cierre:
                self.log(f"día {dia} ({fecha})")
            if dia == self.esc.dia_stop_start and "stop_start" not in e["eventos_hechos"]:
                self.reinicio(dia, "stop/start")
                e["eventos_hechos"].append("stop_start")
            if dia == self.esc.dia_down_up and "down_up" not in e["eventos_hechos"]:
                self.reinicio(dia, "down/up")
                e["eventos_hechos"].append("down_up")
            if dia == self.esc.dia_mlflow and "mlflow" not in e["eventos_hechos"]:
                self.caida_mlflow(dia)
            if dia > 1:                                                    # observaciones del día anterior
                anterior = self.esc.fecha(dia - 1)
                for t in sorted(self.dias[anterior]):
                    self.tienda_dia("observacion", dia, anterior, t)
            if not cierre:
                if dia in self.esc.reentrenos:
                    k = self.esc.reentrenos.index(dia) + 1
                    self.pedir_reentrenamiento(dia, "programado", f"sim-{e['ejecucion']}-r{k}")
                if dia == self.esc.dia_mlflow and "mlflow" not in e["eventos_hechos"]:
                    self.pedir_reentrenamiento(dia, "caida", None)
                for t in sorted(self.dias[fecha]):
                    self.tienda_dia("prediccion", dia, fecha, t)
                if dia == self.esc.dia_mlflow and "mlflow" not in e["eventos_hechos"]:
                    self.volver_mlflow(dia)
                    e["eventos_hechos"].append("mlflow")
            self.sondear(forzar=True)
            self.verificar_pg(dia)
            e["dia_completo"] = dia
            e.guardar()
            self.revisar_detencion()
            if self.interrumpir_tras is not None and dia >= self.interrumpir_tras:
                raise Interrumpida(f"interrupción pedida tras el día {dia}")
        self.esperar_sin_pendientes()

    # ------------------------------------------------------------- cierre
    def verificar_final(self) -> None:
        e = self.estado
        obs = self.entorno.psql_json("SELECT fecha::text AS f, tienda_id AS t, familia_id AS fam, ventas AS v "
                                     "FROM observaciones")
        en_pg = {(date.fromisoformat(o["f"]), o["t"], o["fam"]): o["v"] for o in obs}
        if en_pg != self.mem.guardadas:
            dif = set(en_pg.items()) ^ set(self.mem.guardadas.items())
            self.fallo(5, f"observaciones en PostgreSQL ≠ lo guardado según el oráculo ({len(dif)} diferencias, "
                          f"p. ej. {sorted(dif)[:3]})", None)
        for x in e["ejecuciones"]:
            if x.get("terminal") and x["id"] != "?" and x["terminal"].get("estado") in ("exitoso", "fallido"):
                r = self.api.get(f"/admin/reentrenar/{x['id']}")
                if r.codigo != 200 or r.cuerpo != x["terminal"]:
                    self.fallo(6, f"{x['id']} ahora devuelve {r.codigo} {r.cuerpo!r}", None)
        conocidas = {e["version_inicial"]} | {x["terminal"].get("version") for x in e["ejecuciones"]
                                               if x.get("terminal")}
        for v in e["versiones_vistas"]:
            if v["version"] not in conocidas:
                self.fallo(4, f"se sirvió la versión {v['version']}, que no es la inicial ni la de una ejecución", None)
        esperadas = len(self.esc.reentrenos) + 1
        if len([x for x in e["ejecuciones"] if x.get("id") not in (None, "?")]) != esperadas:
            self.fallo(6, f"{len(e['ejecuciones'])} ejecuciones; se esperaban {esperadas}", None)
        e["protegidas_despues"] = imagenes.ids_protegidas(self.entorno.ejecutar)
        if e["protegidas_despues"] != e["protegidas_antes"]:
            self.fallo(11, f"imágenes :local cambiaron: {e['protegidas_antes']} → {e['protegidas_despues']}", None)

    def calcular_costos(self) -> dict:
        return costos.calcular(*costos.leer_evidencia(self.evid, self.trabajo / "respuestas.jsonl"))

    # ------------------------------------------------------------- reanudación
    def restablecer(self) -> None:
        """Deja el entorno como lo espera el día siguiente: imágenes verificadas, todos los
        servicios arriba (MLflow incluido, si la interrupción ocurrió durante su caída) y
        /health ok. Si quedó una intención sin resultado, se reconstruye solo esa clave."""
        e = self.estado
        regs = e.respuestas.leer()
        if regs and regs[-1]["fase"] == "intencion":
            x = regs[-1]
            e["intencion_sin_resultado"] = True
            self.log(f"intención sin resultado (seq {x['seq']}, {x['tipo']} {x['fecha']} tienda {x['tienda']})")
        self.verificar_imagenes(None)
        r = self.entorno.levantar()
        if r.returncode != 0:
            raise Detener(f"up falló: {r.stderr[-600:]}")
        if self.esperar_salud("ok", self.esc.plazos["arranque"], 2) is None:
            self.fallo(1, "al reanudar, /health no llegó a ok", None)
            raise Detener("sin /health ok al reanudar")
        for err in imagenes.verificar_contenedores(self.entorno, e):
            self.fallo(11, err, None)
        if e.get("intencion_sin_resultado") and regs[-1]["tipo"] == "observacion":
            x = regs[-1]
            filas = self.entorno.psql_json(
                f"SELECT familia_id AS fam, ventas AS v FROM observaciones WHERE fecha = '{x['fecha']}' "
                f"AND tienda_id = {int(x['tienda'])}")
            for f in filas:
                self.mem.guardadas[(date.fromisoformat(x["fecha"]), x["tienda"], f["fam"])] = f["v"]
        e["mlflow_detenido"] = False
        e.evento("reanudada", dia_completo=e["dia_completo"])
        e.guardar()


# ----------------------------------------------------------------- ciclo completo
def carpeta_base() -> Path:
    import os
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "novamercado-simulador"


def nueva(esc: Escenario, fuente: Path | None, base: Path, puertos: dict, salida=print) -> tuple[Path, est.Estado]:
    import secrets
    import shutil
    import tempfile
    if not arbol_limpio():
        raise ent.EntornoInvalido("el árbol de Git tiene cambios: el producto no podría entrenar (procedencia)")
    ocupados = ent.puertos_libres(puertos)
    if ocupados:
        raise ent.EntornoInvalido(f"puertos ocupados: {ocupados}")
    commit = git("rev-parse", "HEAD")
    ejecucion = secrets.token_hex(4)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    carpeta = base / f"{esc.nombre}-{ejecucion}"
    carpeta.mkdir(mode=0o700)
    tmp = None
    if esc.fuente == "sintetico":
        tmp = Path(tempfile.mkdtemp(prefix="nm_sim_fuente_"))
        datos.generar_fuente_sintetica(tmp)
        fuente = tmp
    try:
        s1 = datos.preparar(esc, fuente, carpeta, commit)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    trabajo = carpeta / "trabajo"
    acceso = ent.escribir_entorno(trabajo / "entorno.env", carpeta / "datos", puertos, imagenes.etiquetas(commit))
    estado = est.Estado(trabajo)
    estado.datos = {"ejecucion": ejecucion, "proyecto": f"nm-sim-{ejecucion}", "commit": commit,
                    "escenario": esc.nombre, "huella": esc.huella(), "puertos": puertos, "acceso": acceso,
                    "s1": s1, "dia_completo": 0, "seq": 0, "ejecuciones": [], "versiones_vistas": [],
                    "eventos_hechos": [], "fallos": [], "mediciones": {}, "fase": "preparada",
                    "verificado_pg": {"seq": 0, "solicitud": 0, "envio": 0}, "inicio": time.time()}
    estado.guardar()
    estado.evento("preparada", commit=commit, escenario=esc.nombre)
    salida(f"ejecución {ejecucion} en {carpeta}")
    return carpeta, estado


def abrir(carpeta: Path, esc: Escenario) -> tuple[est.Estado, ent.Entorno]:
    estado = est.Estado(carpeta / "trabajo")
    if not estado.datos:
        raise ent.EntornoInvalido(f"{carpeta} no tiene estado.json")
    if estado["huella"] != esc.huella():
        raise ent.EntornoInvalido("la huella del escenario no coincide con la de la ejecución")
    entorno = ent.Entorno(estado["proyecto"], carpeta / "trabajo" / "entorno.env", estado["puertos"], estado["acceso"])
    return estado, entorno


def conservar_diagnostico(sim) -> dict:
    """Guarda los logs de los servicios en ``trabajo/logs`` sin detener ni borrar nada."""
    destino = sim.trabajo / "logs"
    try:
        guardados = sim.entorno.guardar_logs(destino)
    except Exception as exc:                  # el diagnóstico nunca debe ocultar el fallo original
        sim.estado.evento("logs_no_guardados", error=str(exc)[:300])
        return {"error": str(exc)[:300]}
    sim.estado.evento("diagnostico_conservado", destino=str(destino), servicios=sorted(guardados))
    return {"destino": str(destino), "servicios": sorted(guardados)}


def cerrar(sim: Simulacion) -> dict:
    """Verificación final, costo, sellado y recálculo desde la evidencia. Limpia el entorno
    solo si no falló ningún criterio; si no, lo conserva y guarda sus logs."""
    import shutil
    e = sim.estado
    sim.verificar_final()
    resultado = sim.calcular_costos()
    e["costos"] = {k: v for k, v in resultado.items() if k != "desglose"}
    e["costos_desglose"] = resultado["desglose"]
    if sim.esc.criterio_costo and not resultado["costo_servicio"] < resultado["costo_referencia"]:
        sim.fallo(9, f"costo del servicio {resultado['costo_servicio']:.2f} no es menor que la referencia "
                     f"{resultado['costo_referencia']:.2f}", None)
    e["manifiesto"] = evidencia.sellar(sim.evid, sim.trabajo / "respuestas.jsonl")
    e["fase"] = "sellada"
    e.guardar()
    if e["fallos"]:
        # con algún criterio incumplido no se limpia: el entorno y sus logs quedan para el
        # diagnóstico (limpieza manual con ``python -m simulador limpiar``)
        e["limpieza"] = {"omitida": True, "logs": conservar_diagnostico(sim),
                         "evidencia_presente": evidencia.sellada(sim.evid)}
    else:
        borrados = sim.entorno.limpiar()
        shutil.rmtree(sim.datos_dir)
        e["limpieza"] = {"borrados": borrados, "datos_borrados": not sim.datos_dir.exists(),
                         "evidencia_presente": evidencia.sellada(sim.evid)}
    recalculo = costos.recalcular(sim.evid)
    e["recalculo"] = {k: recalculo[k] for k in ("costo_servicio", "costo_referencia", "filas")}
    if e["recalculo"] != {k: resultado[k] for k in ("costo_servicio", "costo_referencia", "filas")}:
        sim.fallo(10, f"costos --recalcular dio {e['recalculo']}, distinto de {e['costos']}", None)
    e["fase"] = "cerrada"
    e["fin"] = time.time()
    e.guardar()
    return resultado
