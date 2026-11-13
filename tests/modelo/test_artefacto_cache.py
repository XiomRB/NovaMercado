"""modelo.joblib como única referencia: SHA antes de deserializar, humo y caché validada en cascada."""

import json

import joblib
import numpy as np
import pytest

import app.modelo.artefacto as art_mod
from app.modelo.artefacto import ArtefactoInvalido, cargar_artefacto, prueba_de_humo, sha256_bytes, validar_archivo
from app.modelo.cache import CacheModelos
from app.modelo.proveedores import ModeloNoDisponible


def meta(version, sha, **extra):
    return {"nombre": "novamercado-ventas", "alias": "vigente", "version": str(version), "run_id": f"r{version}",
            "ruta_artefacto": "modelo/artifacts/modelo.joblib", "sha256_artefacto": sha,
            "decision_id": int(version) * 10, "secuencia": int(version), **extra}


# ------------------------------------------------------------------- artefacto
def test_artefacto_valido(archivo):
    ruta, sha = archivo
    a = validar_archivo(ruta, sha)
    assert a["formato"] == 1


def test_sha_se_verifica_antes_de_deserializar(archivo, monkeypatch):
    ruta, _ = archivo
    llamado = []
    monkeypatch.setattr(art_mod.joblib, "load", lambda *a, **k: llamado.append(1))
    with pytest.raises(ArtefactoInvalido, match="SHA-256"):
        cargar_artefacto(ruta, "0" * 64)
    assert llamado == []                                   # nunca se deserializó


def test_prueba_de_humo_detecta_salidas_fuera_de_contrato(artefacto, monkeypatch):
    p = artefacto.pronosticador
    prueba_de_humo(p)
    original = p.predecir

    def con_nan(*a, **k):
        r = original(*a, **k)
        r.loc[0, "venta_esperada"] = np.nan
        return r
    monkeypatch.setattr(p, "predecir", con_nan)
    with pytest.raises(ArtefactoInvalido, match="humo"):
        prueba_de_humo(p)


@pytest.mark.parametrize("contenido, motivo", [({"formato": 9}, "formato"), (["x"], "formato")])
def test_formato_invalido(tmp_path, contenido, motivo):
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(contenido, ruta)
    with pytest.raises(ArtefactoInvalido, match=motivo):
        validar_archivo(ruta, None)


# ----------------------------------------------------------------------- caché
def test_guardar_y_validar(tmp_path, archivo):
    ruta, sha = archivo
    c = CacheModelos(tmp_path / "cache")
    c.guardar(ruta, meta(1, sha))
    assert c.versiones() == ["1"] and c.validar("1").metadatos["sha256_artefacto"] == sha
    assert not any(p.name.startswith(".tmp") for p in (tmp_path / "cache").iterdir())   # escritura atómica


def test_guardar_rechaza_sha_que_no_coincide(tmp_path, archivo):
    ruta, _ = archivo
    with pytest.raises(ArtefactoInvalido):
        CacheModelos(tmp_path / "cache").guardar(ruta, meta(1, "f" * 64))
    assert CacheModelos(tmp_path / "cache").versiones() == []


def test_conserva_las_tres_ultimas(tmp_path, archivo):
    ruta, sha = archivo
    c = CacheModelos(tmp_path / "cache", conservar=3)
    for v in (1, 2, 3, 10, 4):
        c.guardar(ruta, meta(v, sha))
    assert c.versiones() == ["10", "4", "3"]              # orden numérico, no alfabético


def _tres_versiones(tmp_path, artefacto, artefacto_b):
    c = CacheModelos(tmp_path / "cache")
    for v, a in ((1, artefacto_b), (2, artefacto_b), (3, artefacto)):
        r = tmp_path / f"m{v}.joblib"
        r.write_bytes(a.artefacto)
        c.guardar(r, meta(v, sha256_bytes(a.artefacto)))
    return c


def test_cascada_usa_la_mas_reciente_valida(tmp_path, artefacto, artefacto_b):
    c = _tres_versiones(tmp_path, artefacto, artefacto_b)
    assert c.primera_valida().metadatos["version"] == "3"


@pytest.mark.parametrize("dano", ["artefacto", "metadatos_sha", "metadatos_version", "metadatos_borrados",
                                  "metadatos_ilegibles"])
def test_version_corrupta_se_salta_y_se_usa_la_anterior(tmp_path, artefacto, artefacto_b, dano):
    c = _tres_versiones(tmp_path, artefacto, artefacto_b)
    v3 = tmp_path / "cache" / "3"
    m = json.loads((v3 / "metadatos.json").read_text())
    if dano == "artefacto":
        datos = bytearray((v3 / "modelo.joblib").read_bytes())
        datos[100] ^= 0xFF
        (v3 / "modelo.joblib").write_bytes(bytes(datos))
    elif dano == "metadatos_sha":
        m["sha256_artefacto"] = "0" * 64
        (v3 / "metadatos.json").write_text(json.dumps(m))
    elif dano == "metadatos_version":
        m["version"] = "2"
        (v3 / "metadatos.json").write_text(json.dumps(m))
    elif dano == "metadatos_borrados":
        (v3 / "metadatos.json").unlink()
    else:
        (v3 / "metadatos.json").write_text("{no es json")
    assert c.primera_valida().metadatos["version"] == "2"


def test_ninguna_valida(tmp_path, archivo):
    ruta, sha = archivo
    c = CacheModelos(tmp_path / "cache")
    c.guardar(ruta, meta(1, sha))
    (tmp_path / "cache" / "1" / "modelo.joblib").write_bytes(b"roto")
    with pytest.raises(ModeloNoDisponible, match="no hay una versión válida"):
        c.primera_valida()
    with pytest.raises(ModeloNoDisponible, match="vacía"):
        CacheModelos(tmp_path / "otra").primera_valida()


def test_modificacion_coherente_sin_mlflow_es_la_limitacion_documentada(tmp_path, artefacto, artefacto_b):
    """Si alguien reemplaza artefacto y metadatos de forma coherente, la caché sola no lo
    detecta (limitación aceptada). Se corrige al volver MLflow (prueba de integración)."""
    c = _tres_versiones(tmp_path, artefacto, artefacto_b)
    v3 = tmp_path / "cache" / "3"
    (v3 / "modelo.joblib").write_bytes(artefacto_b.artefacto)            # otro modelo válido
    m = json.loads((v3 / "metadatos.json").read_text())
    m["sha256_artefacto"] = sha256_bytes(artefacto_b.artefacto)           # metadatos coherentes
    (v3 / "metadatos.json").write_text(json.dumps(m))
    e = c.primera_valida()
    assert e.metadatos["version"] == "3"                                  # se acepta: no es detectable aquí


def test_archivo_si_coincide_recalcula_el_sha(tmp_path, archivo):
    ruta, sha = archivo
    c = CacheModelos(tmp_path / "cache")
    c.guardar(ruta, meta(1, sha))
    assert c.archivo_si_coincide("1", sha) is not None
    (tmp_path / "cache" / "1" / "modelo.joblib").write_bytes(b"otro contenido")
    assert c.archivo_si_coincide("1", sha) is None


# ------------------------------------------------- etapa 6: decisión y secuencia
def test_entrada_para_exige_la_misma_decision_y_sha(tmp_path, archivo):
    ruta, sha = archivo
    c = CacheModelos(tmp_path / "c")
    c.guardar(ruta, meta(3, sha))
    assert c.entrada_para(30, "3", sha) is not None
    assert c.entrada_para(31, "3", sha) is None                 # otra decisión, misma versión y SHA
    assert c.entrada_para(30, "3", "0" * 64) is None
    assert c.entrada_para(30, "4", sha) is None


def test_mejor_por_secuencia_y_metadatos_sin_decision(tmp_path, archivo):
    ruta, sha = archivo
    c = CacheModelos(tmp_path / "c")
    c.guardar(ruta, meta(1, sha, secuencia=9))
    c.guardar(ruta, meta(2, sha, secuencia=4))
    assert c.mejor_por_secuencia().metadatos["version"] == "1"     # por secuencia, no por número de versión
    m = json.loads((tmp_path / "c" / "1" / "metadatos.json").read_text())
    del m["decision_id"]                                            # entrada de la etapa 5: inválida
    (tmp_path / "c" / "1" / "metadatos.json").write_text(json.dumps(m))
    assert c.mejor_por_secuencia().metadatos["version"] == "2"
