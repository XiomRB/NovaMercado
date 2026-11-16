"""Validación independiente del contrato, aislamiento de Compose (I3, I4) y estado
reanudable. Sin Docker: un ejecutor simulado registra los comandos."""

from __future__ import annotations

import subprocess

import pytest

from simulador import contrato, imagenes
from simulador import entorno as ent
from simulador import estado as est

FAMS = ["a", "b"]
PRED = {"tienda_id": 5, "fecha": "2026-01-01", "version_modelo": "3",
        "predicciones": [{"familia_id": "a", "venta_esperada": 1.5, "cantidad_sugerida": 2},
                         {"familia_id": "b", "venta_esperada": 0, "cantidad_sugerida": 0}],
        "calidad": {"estado": "ok", "observaciones": []}}


def test_contrato_prediccion_valida():
    assert contrato.prediccion(PRED, 5, "2026-01-01", FAMS) == []


@pytest.mark.parametrize("cambio", [
    {"version_modelo": 3}, {"tienda_id": "5"}, {"fecha": "01/01/2026"},
    {"predicciones": [{"familia_id": "a", "venta_esperada": 1, "cantidad_sugerida": 1.5}]},
    {"predicciones": [{"familia_id": "a", "venta_esperada": -1, "cantidad_sugerida": 1},
                      {"familia_id": "b", "venta_esperada": 0, "cantidad_sugerida": 0}]},
    {"calidad": {"estado": "ok", "observaciones": [{"campo": "x", "accion": "reparar", "detalle": "d"}]}},
    {"calidad": {"estado": "corregido", "observaciones": [{"campo": "x", "accion": "advertir", "detalle": "d"}]}},
    {"calidad": {"estado": "advertencia", "observaciones": [{"campo": "x", "accion": "rechazar", "detalle": "d"}]}},
])
def test_contrato_prediccion_invalida(cambio):
    assert contrato.prediccion({**PRED, **cambio}, 5, "2026-01-01", FAMS)


def test_contrato_otros():
    assert contrato.observacion({"registros_guardados": 0, "calidad": {"estado": "ok", "observaciones": []}}) == []
    assert contrato.observacion({"registros_guardados": True, "calidad": {"estado": "ok", "observaciones": []}})
    assert contrato.rechazo({"detail": "x"}) == [] and contrato.rechazo({"detalle": "x"})
    assert contrato.estado_ejecucion({"estado": "en_curso"}) == []
    assert contrato.estado_ejecucion({"estado": "exitoso", "promovido": True, "version": "4", "motivo": "m"}) == []
    assert contrato.estado_ejecucion({"estado": "exitoso", "promovido": True, "version": 4, "motivo": "m"})
    assert contrato.modelo({"nombre": "n", "alias": "vigente", "version": "1"}) == []
    assert contrato.salud(200, {"estado": "degradado"})


# ------------------------------------------------------------------ Compose (I3)
class Ejecutor:
    def __init__(self, respuestas=None):
        self.comandos: list[list[str]] = []
        self.respuestas = respuestas or {}

    def __call__(self, cmd):
        self.comandos.append(cmd)
        for clave, (codigo, salida) in self.respuestas.items():
            if clave in " ".join(cmd):
                return subprocess.CompletedProcess(cmd, codigo, salida, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _entorno(ejecutor, proyecto="nm-sim-abc12345"):
    return ent.Entorno(proyecto, "/tmp/x/entorno.env", dict(ent.PUERTOS_POR_DEFECTO),
                       {"db_usuario": "u", "db_nombre": "d"}, ejecutar=ejecutor)


def test_compose_siempre_con_proyecto_y_env_file():
    x = Ejecutor()
    e = _entorno(x)
    e.levantar(), e.detener(), e.iniciar(), e.bajar(), e.detener_servicio("mlflow"), e.iniciar_servicio("mlflow")
    for c in x.comandos:
        assert c[:4] == ["docker", "compose", "-p", "nm-sim-abc12345"]
        assert "--env-file" in c and c[c.index("--env-file") + 1] == "/tmp/x/entorno.env"
    assert all("--no-build" in c for c in x.comandos if "up" in c)
    with pytest.raises(ent.EntornoInvalido):
        e.compose("up", "-d")


def test_solo_proyectos_del_simulador():
    with pytest.raises(ent.EntornoInvalido):
        _entorno(Ejecutor(), proyecto="novamercado")


def test_limpieza_rechaza_recursos_ajenos():
    x = Ejecutor({"volume ls": (0, "nm-sim-abc12345_db_producto\nnovamercado_db_producto\n")})
    with pytest.raises(ent.EntornoInvalido, match="otro prefijo"):
        _entorno(x).limpiar()
    assert not any("down" in c for c in x.comandos)


def test_limpieza_solo_down_del_proyecto():
    x = Ejecutor()
    _entorno(x).limpiar()
    assert [c for c in x.comandos if "down" in c] == [_entorno(Ejecutor()).comando("down", "-v", "--remove-orphans")]
    assert not any(p in " ".join(c) for c in x.comandos for p in ("prune", "rmi", " tag "))


def test_entorno_minimo_sin_variables_del_usuario(monkeypatch):
    monkeypatch.setenv("NM_IMAGEN_APP", "novamercado-app:local")
    monkeypatch.setenv("POSTGRES_PASSWORD", "del-usuario")
    m = ent.entorno_minimo()
    assert "NM_IMAGEN_APP" not in m and "POSTGRES_PASSWORD" not in m


def test_archivo_de_entorno(tmp_path):
    acceso = ent.escribir_entorno(tmp_path / "e.env", tmp_path / "datos", dict(ent.PUERTOS_POR_DEFECTO),
                                  imagenes.etiquetas("b" * 40))
    texto = (tmp_path / "e.env").read_text()
    assert oct((tmp_path / "e.env").stat().st_mode & 0o777) == "0o600"
    assert "NM_IMAGEN_APP=novamercado-app:sim-bbbbbbbbbbbb" in texto
    assert "NM_IMAGEN_AIRFLOW=novamercado-airflow:sim-bbbbbbbbbbbb" in texto
    assert f"RAW_DIR={tmp_path / 'datos'}" in texto and "API_PUERTO_HOST=57000" in texto
    assert "cambie_esta_clave" not in texto and "reemplace_por" not in texto
    assert acceso["db_usuario"] == "novamercado"


# ------------------------------------------------------------------ imágenes (I3, I4)
COMMIT = "c" * 40


class EstadoFalso(dict):
    def guardar(self):
        pass


def _ejecutor_imagenes(ids, procedencia=(COMMIT, "True")):
    def ejecutar(cmd):
        texto = " ".join(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            i = ids.get(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0 if i else 1, i or "", "")
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, " ".join(procedencia), "")
        if "build" in cmd:
            servicio = cmd[-1]
            tipo = "app" if servicio == "api" else "airflow"
            ids[imagenes.etiquetas(COMMIT)[tipo]] = f"sha256:nueva-{tipo}"
        assert "tag" not in cmd[:3] and "rmi" not in texto
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return ejecutar


def test_etiquetas_por_commit():
    assert imagenes.etiquetas(COMMIT) == {"app": "novamercado-app:sim-cccccccccccc",
                                          "airflow": "novamercado-airflow:sim-cccccccccccc"}
    assert not set(imagenes.etiquetas(COMMIT).values()) & set(imagenes.PROTEGIDAS.values())
    with pytest.raises(imagenes.ImagenNoValida):
        imagenes.etiquetas("abc")


def test_construye_sin_tocar_las_locales():
    ids = {"novamercado-app:local": "sha256:local-app", "novamercado-airflow:local": "sha256:local-af"}
    x = Ejecutor()
    e = _entorno(x)
    e.ejecutar = _ejecutor_imagenes(ids)
    st = EstadoFalso()
    ev = imagenes.preparar(e, st, COMMIT, COMMIT, True)
    assert [x["accion"] for x in ev] == ["construida", "construida"]
    assert ids["novamercado-app:local"] == "sha256:local-app" and ids["novamercado-airflow:local"] == "sha256:local-af"
    ev = imagenes.preparar(e, st, COMMIT, COMMIT, True)                    # reanudación: sin construir
    assert [x["accion"] for x in ev] == ["reutilizada", "reutilizada"]


def test_no_construye_con_arbol_sucio_ni_otro_head():
    e = _entorno(Ejecutor())
    e.ejecutar = _ejecutor_imagenes({})
    with pytest.raises(imagenes.ImagenNoValida, match="cambios"):
        imagenes.preparar(e, EstadoFalso(), COMMIT, COMMIT, False)
    with pytest.raises(imagenes.ImagenNoValida, match="HEAD"):
        imagenes.preparar(e, EstadoFalso(), COMMIT, "d" * 40, True)


def test_reanudar_con_imagen_distinta_se_niega():
    ids = {}
    e = _entorno(Ejecutor())
    e.ejecutar = _ejecutor_imagenes(ids)
    st = EstadoFalso()
    imagenes.preparar(e, st, COMMIT, COMMIT, True)
    ids[imagenes.etiquetas(COMMIT)["app"]] = "sha256:otra"                  # reetiquetada por alguien
    with pytest.raises(imagenes.ImagenNoValida, match="no a la imagen registrada"):
        imagenes.preparar(e, st, COMMIT, COMMIT, True)


def test_imagen_de_otro_commit_se_niega():
    ids = {k: "sha256:x" for k in imagenes.etiquetas(COMMIT).values()}
    e = _entorno(Ejecutor())
    e.ejecutar = _ejecutor_imagenes(ids, procedencia=("e" * 40, "True"))
    with pytest.raises(imagenes.ImagenNoValida, match="procedencia"):
        imagenes.preparar(e, EstadoFalso(), COMMIT, COMMIT, True)


def test_etiqueta_borrada_se_reconstruye_del_mismo_commit():
    ids = {}
    e = _entorno(Ejecutor())
    e.ejecutar = _ejecutor_imagenes(ids)
    st = EstadoFalso()
    imagenes.preparar(e, st, COMMIT, COMMIT, True)
    del ids[imagenes.etiquetas(COMMIT)["airflow"]]
    ev = imagenes.preparar(e, st, COMMIT, COMMIT, True)
    assert [x["accion"] for x in ev] == ["reutilizada", "reconstruida"]


# ------------------------------------------------------------------ estado
def test_registro_tolera_linea_truncada(tmp_path):
    r = est.Registro(tmp_path / "r.jsonl")
    r.agregar({"a": 1})
    with open(tmp_path / "r.jsonl", "a") as fh:
        fh.write('{"a": 2')
    assert r.leer() == [{"a": 1}]


def test_estado_atomico(tmp_path):
    s = est.Estado(tmp_path)
    s["x"] = 1
    s.guardar()
    assert est.Estado(tmp_path)["x"] == 1 and not list(tmp_path.glob(".*.tmp"))
