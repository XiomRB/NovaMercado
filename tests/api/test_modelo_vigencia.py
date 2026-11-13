"""Serialización de GET /modelo en los cuatro estados de vigencia (revisión del PR #6).

``app/main.py`` serializa con ``model_dump(exclude_none=True)``; ``GestorVigente.servido``
entrega ``version_vigente=None`` solo en ``sin_verificar``. Estas pruebas fijan las dos
mitades: en ``sin_verificar`` la propiedad NO aparece en el JSON (ni como ``null``), y
en los otros tres estados siempre aparece con la versión observada."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db.promociones import Vigente
from app.main import create_app
from app.modelo.gestor import GestorVigente
from app.modelo.proveedor_mlflow import ConsultaIndeterminada, Obtenido
from app.modelo.proveedores import ModeloCargado, OrigenNoDisponible

CAIDA = ConsultaIndeterminada("PostgreSQL no responde")


def vig(d, version, sha):
    return Vigente(d, d, version, f"run{version}", "modelo/artifacts/modelo.joblib", sha, 1)


class Proveedor:
    def __init__(self):
        self.lecturas, self.fallo, self.al_obtener = [], None, None

    def consultar_vigente(self):
        r = self.lecturas.pop(0) if len(self.lecturas) > 1 else self.lecturas[0]
        if isinstance(r, Exception):
            raise r
        return r

    def alias_mlflow(self):
        return None

    def obtener(self, v):
        if self.al_obtener:
            self.al_obtener()
        if self.fallo:
            raise self.fallo
        return Obtenido(ModeloCargado(SimpleNamespace(), "novamercado-ventas", "vigente", v.version), v, None, False)

    def guardar_en_cache(self, o):
        pass

    def respaldo_sin_base(self):
        return (ModeloCargado(SimpleNamespace(), "novamercado-ventas", "vigente", "4"),
                {"decision_id": 4, "secuencia": 4, "version": "4", "run_id": "run4",
                 "ruta_artefacto": "modelo/artifacts/modelo.joblib", "sha256_artefacto": "d" * 64})


def get_modelo(gestor):
    deps = SimpleNamespace(gestor=gestor)
    with TestClient(create_app(deps, iniciar_hilos=False), raise_server_exceptions=False) as c:
        r = c.get("/modelo")
    assert r.status_code == 200
    return r.text, r.json()


def test_sin_verificar_omite_version_vigente_del_json():
    p = Proveedor()
    p.lecturas = [CAIDA]
    g = GestorVigente(p)
    g.refrescar()
    texto, m = get_modelo(g)
    assert "version_vigente" not in m and "version_vigente" not in texto       # ni la clave ni null
    assert m["estado_vigencia"] == "sin_verificar" and m["es_vigente"] is False and m["version"] == "4"
    assert set(m) == {"nombre", "alias", "version", "es_vigente", "estado_vigencia", "motivo"}


@pytest.mark.parametrize("estado", ["vigente", "actualizando", "anterior"])
def test_los_otros_estados_siempre_incluyen_version_vigente(estado):
    p = Proveedor()
    p.lecturas = [vig(1, "1", "a" * 64)]
    g = GestorVigente(p)
    g.refrescar()
    capturado = {}
    if estado == "actualizando":
        p.lecturas = [vig(2, "2", "b" * 64)]
        p.al_obtener = lambda: capturado.update(zip(("texto", "m"), get_modelo(g)))
        g.refrescar()
        texto, m = capturado["texto"], capturado["m"]
    else:
        if estado == "anterior":
            p.lecturas = [vig(2, "2", "b" * 64)]
            p.fallo = OrigenNoDisponible("MLflow no disponible")
            g.refrescar()
        texto, m = get_modelo(g)
    esperado = {"vigente": ("1", True, "1"), "actualizando": ("1", False, "2"), "anterior": ("1", False, "2")}[estado]
    assert m["estado_vigencia"] == estado
    assert (m["version"], m["es_vigente"], m["version_vigente"]) == esperado
    assert json.loads(texto)["version_vigente"] is not None
