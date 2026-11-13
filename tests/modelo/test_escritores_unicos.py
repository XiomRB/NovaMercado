"""Prueba estática (AST) de escritores únicos (plan §5.4–§5.5):

* ``set_registered_model_alias``/``delete_registered_model_alias`` solo dentro de
  ``promocion._escribir_alias``;
* ``_escribir_alias`` solo desde ``autorizar_y_promover`` (promoción) y ``reconciliar``
  (reflejo del vigente confirmado), decisión acordada durante la implementación;
* ``estado=valido`` solo lo escribe ``autorizar_y_promover``;
* la concesión (advisory lock y tabla ``bloqueo_entrenamiento``) solo la toca
  ``app/modelo/bloqueo.py``; las promociones solo se insertan desde ahí."""

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ARCHIVOS = sorted(p for d in ("app", "ml") for p in (RAIZ / d).rglob("*.py"))


def _funciones(arbol):
    """Mapa nodo → nombre de la función que lo contiene."""
    padre = {}
    for f in ast.walk(arbol):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(f):
                padre.setdefault(n, f.name) if n is not f else None
    return padre


def _llamadas(nombre_atributo):
    hallazgos = []
    for ruta in ARCHIVOS:
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        func = _funciones(arbol)
        for n in ast.walk(arbol):
            if isinstance(n, ast.Call):
                f = n.func
                nombre = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
                if nombre == nombre_atributo:
                    hallazgos.append((ruta.relative_to(RAIZ).as_posix(), func.get(n)))
    return hallazgos


def test_un_solo_escritor_del_alias():
    for metodo in ("set_registered_model_alias", "delete_registered_model_alias"):
        assert _llamadas(metodo) == [("app/modelo/promocion.py", "_escribir_alias")], metodo


def test_escribir_alias_solo_desde_promocion_y_reconciliacion():
    assert sorted(set(_llamadas("_escribir_alias"))) == [("app/modelo/promocion.py", "autorizar_y_promover"),
                                                         ("app/modelo/promocion.py", "reconciliar")]


def _escrituras_de_valido():
    hallazgos = []
    for ruta in ARCHIVOS:
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        func = _funciones(arbol)
        for n in ast.walk(arbol):
            es = False
            if isinstance(n, ast.Dict):          # {"estado": "valido", ...}
                es = any(isinstance(k, ast.Constant) and k.value == "estado" and isinstance(v, ast.Constant)
                         and v.value == "valido" for k, v in zip(n.keys, n.values))
            elif isinstance(n, ast.Call):        # set_model_version_tag(n, v, "estado", "valido")
                args = [a.value for a in n.args if isinstance(a, ast.Constant)]
                es = "estado" in args and "valido" in args
            if es:
                hallazgos.append((ruta.relative_to(RAIZ).as_posix(), func.get(n)))
    return hallazgos


def test_solo_autorizar_y_promover_escribe_valido():
    hallazgos = _escrituras_de_valido()
    assert hallazgos and set(hallazgos) == {("app/modelo/promocion.py", "autorizar_y_promover")}


def test_la_concesion_y_las_promociones_solo_desde_bloqueo():
    for ruta in ARCHIVOS:
        rel = ruta.relative_to(RAIZ).as_posix()
        texto = ruta.read_text(encoding="utf-8")
        if rel in ("app/modelo/bloqueo.py", "app/db/esquema.py", "app/db/promociones.py"):
            continue
        for patron in ("pg_try_advisory_lock", "pg_advisory_unlock", "UPDATE bloqueo_entrenamiento",
                       "INSERT INTO decisiones_promocion", "INSERT INTO resoluciones_promocion"):
            assert patron not in texto, f"{rel} usa {patron}"
