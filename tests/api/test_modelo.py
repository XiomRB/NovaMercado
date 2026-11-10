"""Proveedor de archivo, gestor del modelo, escritura atómica y advertencia de antigüedad."""

import threading
from datetime import date

import joblib
import pytest

from app.calidad.antiguedad import advertir_antiguedad
from app.calidad.resultado import RegistroCalidad
from app.modelo.entrenar_local import guardar_atomico
from app.modelo.gestor import GestorModelo
from app.modelo.proveedores import ModeloNoDisponible, ProveedorArchivo
from tests.api.conftest import ProveedorMemoria


# ------------------------------------------------------------ proveedor de archivo
def test_archivo_inexistente(tmp_path):
    p = ProveedorArchivo(tmp_path / "no.joblib")
    assert p.huella() is None
    with pytest.raises(ModeloNoDisponible, match="entrenar_local"):
        p.obtener()


def test_artefacto_valido(tmp_path, artefacto):
    ruta = tmp_path / "m.joblib"
    guardar_atomico(artefacto.artefacto, ruta)
    m = ProveedorArchivo(ruta).obtener()
    assert m.version == artefacto.version and m.alias == "local" and m.nombre == "novamercado-ventas"
    assert m.version.startswith("local-20260414-")


@pytest.mark.parametrize("contenido, motivo", [
    (b"esto no es joblib", "no se pudo leer"),
    ({"formato": 99}, "formato esperado"),
    ("texto", "formato esperado"),
])
def test_artefactos_invalidos(tmp_path, contenido, motivo):
    ruta = tmp_path / "m.joblib"
    if isinstance(contenido, bytes):
        ruta.write_bytes(contenido)
    else:
        joblib.dump(contenido, ruta)
    with pytest.raises(ModeloNoDisponible, match=motivo):
        ProveedorArchivo(ruta).obtener()


def test_artefacto_con_otras_librerias_se_rechaza(tmp_path, artefacto):
    import io
    a = joblib.load(io.BytesIO(artefacto.artefacto))
    a["librerias"] = {**a["librerias"], "scikit-learn": "0.0.1"}
    ruta = tmp_path / "m.joblib"
    joblib.dump(a, ruta)
    with pytest.raises(ModeloNoDisponible, match="se creó con"):
        ProveedorArchivo(ruta).obtener()


def test_guardar_atomico_no_deja_temporales_y_reemplaza(tmp_path):
    ruta = tmp_path / "sub" / "m.joblib"
    guardar_atomico(b"uno", ruta)
    guardar_atomico(b"dos", ruta)
    assert ruta.read_bytes() == b"dos"
    assert [p.name for p in ruta.parent.iterdir()] == ["m.joblib"]


# ----------------------------------------------------------------------- gestor
def test_gestor_sin_modelo():
    g = GestorModelo(ProveedorMemoria(None))
    assert g.refrescar() is False and not g.cargado()
    with pytest.raises(ModeloNoDisponible):
        g.actual()


def test_gestor_detecta_un_artefacto_nuevo_sin_reiniciar(tmp_path, artefacto):
    ruta = tmp_path / "m.joblib"
    g = GestorModelo(ProveedorArchivo(ruta))
    assert g.refrescar() is False                         # todavía no existe
    guardar_atomico(artefacto.artefacto, ruta)
    assert g.refrescar() is True and g.actual().version == artefacto.version
    assert g.refrescar() is False                         # misma huella: no recarga


def test_gestor_conserva_el_modelo_si_el_nuevo_no_carga(tmp_path, artefacto):
    ruta = tmp_path / "m.joblib"
    guardar_atomico(artefacto.artefacto, ruta)
    g = GestorModelo(ProveedorArchivo(ruta))
    g.refrescar()
    guardar_atomico(b"artefacto corrupto", ruta)
    assert g.refrescar() is False
    assert g.actual().version == artefacto.version and "no se pudo leer" in g.ultimo_error
    ruta.unlink()
    assert g.refrescar() is False and g.cargado()         # el origen desaparece: sigue el vigente


def test_gestor_proveedor_que_explota_no_afecta_al_vigente(artefacto):
    p = ProveedorMemoria(artefacto)
    g = GestorModelo(p)
    g.refrescar()

    def explota():
        raise ConnectionError("origen caído")
    p.huella = explota
    assert g.refrescar() is False and g.actual().version == artefacto.version


def test_lecturas_concurrentes_durante_recargas(artefacto):
    p = ProveedorMemoria(artefacto)
    g = GestorModelo(p)
    g.refrescar()
    errores, detener = [], threading.Event()

    def leer():
        while not detener.is_set():
            try:
                assert g.actual().version == artefacto.version
            except Exception as exc:  # noqa: BLE001
                errores.append(exc)

    hilos = [threading.Thread(target=leer) for _ in range(4)]
    for h in hilos:
        h.start()
    for i in range(20):
        p.huella = (lambda i=i: f"h{i}")                  # fuerza recargas
        g.refrescar()
    detener.set()
    for h in hilos:
        h.join()
    assert errores == []


# ------------------------------------------------------------------- antigüedad
F = date(2026, 4, 14)


def _adv(ultimas, familias=("a", "b")):
    r = RegistroCalidad()
    advertir_antiguedad(F, ultimas, list(familias), r)
    return r


def test_dia_previo_no_advierte():
    assert _adv({"a": date(2026, 4, 13), "b": date(2026, 4, 13)}).observaciones == []


def test_dias_faltantes_y_sin_nivel_y_sin_datos():
    r = _adv({"a": date(2026, 4, 10), "b": date(2026, 3, 1)}, familias=("a", "b", "c"))
    assert len(r.observaciones) == 1 and r.observaciones[0].accion == "advertir"
    d = r.observaciones[0].detalle
    assert "a (4 días)" in d and "b (44 días)" in d and "c" in d and "respaldo" in d


def test_ventas_de_la_fecha_o_posteriores_no_se_aceptan():
    with pytest.raises(ValueError, match="anteriores"):
        _adv({"a": F})
