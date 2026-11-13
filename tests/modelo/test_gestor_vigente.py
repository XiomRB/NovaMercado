"""GestorVigente sin servidores (plan §7): estados de vigencia, segunda lectura, adopción sin
descarga, E-ANT, PostgreSQL inaccesible o sin vigente y alias desincronizado."""

from types import SimpleNamespace

import pytest

from app.db.promociones import Vigente
from app.modelo.gestor import ACTUALIZANDO, ANTERIOR, VIGENTE, GestorVigente
from app.modelo.proveedor_mlflow import ConsultaIndeterminada, Obtenido
from app.modelo.proveedores import ModeloCargado, ModeloNoDisponible, OrigenNoDisponible, VersionNoUtilizable


def vig(decision, version, sha="a" * 64, secuencia=None):
    return Vigente(decision, secuencia or decision, str(version), f"run{version}", "modelo/artifacts/modelo.joblib",
                   sha, 1)


class ProveedorFalso:
    def __init__(self):
        self.lecturas = []          # respuestas sucesivas de consultar_vigente (la última se repite)
        self.alias = None           # None: el alias refleja el último vigente leído (sin desincronización)
        self._ultimo = None
        self.fallo_obtener = None
        self.obtenidos = []
        self.cache_guardada = []
        self.respaldo = None
        self.al_obtener = None

    def consultar_vigente(self):
        r = self.lecturas.pop(0) if len(self.lecturas) > 1 else self.lecturas[0]
        if isinstance(r, Exception):
            raise r
        self._ultimo = r
        return r

    def alias_mlflow(self):
        if isinstance(self.alias, Exception):
            raise self.alias
        if self.alias is None:
            return self._ultimo.version if self._ultimo else None
        return self.alias

    def obtener(self, v):
        if self.al_obtener:
            self.al_obtener()
        if self.fallo_obtener:
            raise self.fallo_obtener
        self.obtenidos.append(v.clave)
        return Obtenido(ModeloCargado(SimpleNamespace(nombre=f"modelo-{v.version}"), "m", "vigente", v.version),
                        v, None, False)

    def guardar_en_cache(self, o):
        self.cache_guardada.append(o.vigente.clave)

    def respaldo_sin_base(self):
        if self.respaldo is None:
            raise ModeloNoDisponible("la caché está vacía")
        return self.respaldo


CAIDA = ConsultaIndeterminada("PostgreSQL no responde")


@pytest.fixture
def p():
    return ProveedorFalso()


def cargar(p, v):
    p.lecturas = [v]
    g = GestorVigente(p)
    assert g.refrescar() is True
    return g


def test_activa_el_vigente_tras_la_segunda_lectura(p):
    g = cargar(p, vig(1, "1"))
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.es_vigente, s.version_vigente) == ("1", VIGENTE, True, "1")
    assert p.cache_guardada == [(1, 1)]
    assert g.refrescar() is False and p.obtenidos == [(1, 1)]          # misma decisión: sin cambios


def test_actualizando_informa_la_version_que_predice_y_la_observada(p):
    g = cargar(p, vig(1, "1"))
    vistos = []
    p.al_obtener = lambda: vistos.append(g.servido())
    p.lecturas = [vig(2, "2", "b" * 64)]
    assert g.refrescar() is True
    durante = vistos[0]
    assert (durante.modelo.version, durante.estado_vigencia, durante.es_vigente, durante.version_vigente) == (
        "1", ACTUALIZANDO, False, "2")
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.version_vigente) == ("2", VIGENTE, "2")


def test_actualizando_no_degrada_la_salud(p):
    g = cargar(p, vig(1, "1"))
    motivos = []
    p.al_obtener = lambda: motivos.append(g.motivos_degradacion())
    p.lecturas = [vig(2, "2", "b" * 64)]
    g.refrescar()
    assert motivos == [[]]


def test_promocion_tras_la_segunda_lectura_se_activa_en_el_siguiente_refresco(p):
    """Actualización pendiente: la API activa lo que leyó y descubre B en el siguiente refresco."""
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(2, "2", "b" * 64)]
    g._antes_del_intercambio = lambda: p.lecturas.__setitem__(0, vig(3, "3", "c" * 64))
    assert g.refrescar() is True
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.es_vigente) == ("2", VIGENTE, True)   # no conoce aún la 3
    g._antes_del_intercambio = None
    assert g.refrescar() is True and g.servido().modelo.version == "3"


def test_cambio_durante_la_carga_descarta_lo_descargado(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(2, "2", "b" * 64), vig(3, "3", "c" * 64), vig(3, "3", "c" * 64)]
    assert g.refrescar() is True
    assert g.servido().modelo.version == "3" and p.obtenidos == [(1, 1), (2, 2), (3, 3)]
    assert p.cache_guardada == [(1, 1), (3, 3)]                       # la 2 nunca se activó ni se guardó


def test_revalidacion_agotada_pasa_a_anterior(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(d, str(d), f"{d:064x}") for d in range(2, 12)]
    assert g.refrescar() is False
    s = g.servido()
    assert s.modelo.version == "1" and s.estado_vigencia == ANTERIOR and not s.es_vigente
    assert "revalidación" in g.motivos_degradacion()[0]


@pytest.mark.parametrize("fallo", [OrigenNoDisponible("MLflow no disponible"),
                                   VersionNoUtilizable("SHA-256 distinto")])
def test_e_ant_con_modelo_cargado(p, fallo):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(2, "2", "b" * 64)]
    p.fallo_obtener = fallo
    assert g.refrescar() is False
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.es_vigente, s.version_vigente) == ("1", ANTERIOR, False, "2")
    assert "se sirve la versión 1; la vigente 2" in g.motivos_degradacion()[0]
    p.fallo_obtener = None                                            # B obtenible: primer refresco la activa
    assert g.refrescar() is True and g.servido().es_vigente and g.servido().modelo.version == "2"


def test_vigente_no_obtenible_sin_modelo_no_carga_otra_version(p):
    p.lecturas = [vig(2, "2")]
    p.fallo_obtener = OrigenNoDisponible("MLflow no disponible")
    p.respaldo = (ModeloCargado(object(), "m", "vigente", "1"), {"sha256_artefacto": "a" * 64})
    g = GestorVigente(p)
    assert g.refrescar() is False and not g.cargado()                 # la caché no prevalece


def test_sin_vigente_retira_el_modelo(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [None]
    g.refrescar()
    assert not g.cargado()
    with pytest.raises(ModeloNoDisponible, match="no tiene un modelo vigente"):
        g.servido()


def test_sin_vigente_al_arrancar_no_usa_la_cache(p):
    p.lecturas = [None]
    p.respaldo = (ModeloCargado(object(), "m", "vigente", "1"), {"sha256_artefacto": "a" * 64})
    g = GestorVigente(p)
    g.refrescar()
    assert not g.cargado()


def test_base_inaccesible_conserva_modelo_estado_y_vigente_observado(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(2, "2", "b" * 64)]
    p.fallo_obtener = OrigenNoDisponible("MLflow no disponible")
    g.refrescar()                                                     # anterior, observado 2
    p.lecturas = [CAIDA]
    g.refrescar()
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.version_vigente) == ("1", ANTERIOR, "2")


def test_base_inaccesible_durante_actualizando_no_activa_b(p):
    """v9: si la segunda lectura no responde, no se declara B activada ni se pasa a vigente
    ni a anterior; se reintenta al volver la base."""
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(2, "2", "b" * 64), CAIDA]
    assert g.refrescar() is False
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.es_vigente, s.version_vigente) == ("1", ACTUALIZANDO, False, "2")
    assert g.motivos_degradacion() == []                              # solo degrada base_datos (regla general)
    p.lecturas = [CAIDA]
    g.refrescar()
    assert g.servido().estado_vigencia == ACTUALIZANDO and g.servido().version_vigente == "2"
    p.lecturas = [vig(2, "2", "b" * 64)]
    assert g.refrescar() is True and g.servido().modelo.version == "2" and g.servido().es_vigente


def test_base_inaccesible_al_arrancar_usa_la_cache_marcada_como_anterior(p):
    p.lecturas = [CAIDA]
    p.respaldo = (ModeloCargado(object(), "m", "vigente", "4"), {"sha256_artefacto": "a" * 64})
    g = GestorVigente(p)
    assert g.refrescar() is True
    s = g.servido()
    assert (s.modelo.version, s.estado_vigencia, s.es_vigente, s.version_vigente) == ("4", ANTERIOR, False, None)


# ------------------------------------------------------ adopción sin descarga (§7.3.4)
def test_misma_version_y_sha_otra_decision_se_adopta_sin_descargar(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(5, "1", secuencia=7)]                           # nueva decisión, mismo artefacto
    assert g.refrescar() is True
    assert p.obtenidos == [(1, 1)]                                    # no se descargó
    assert g._e.decision.clave == (5, 7) and g.servido().es_vigente   # nunca quedan los valores anteriores


def test_adopcion_exige_la_segunda_lectura(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(5, "1", secuencia=7), CAIDA]
    assert g.refrescar() is False
    assert g._e.decision.clave == (1, 1) and g.servido().estado_vigencia == ACTUALIZANDO
    p.lecturas = [vig(5, "1", secuencia=7), vig(6, "2", "b" * 64, 8), vig(6, "2", "b" * 64, 8)]
    assert g.refrescar() is True                                      # la segunda lectura cambió: no adopta
    assert g.servido().modelo.version == "2" and g._e.decision.clave == (6, 8)


def test_adopcion_exige_el_mismo_sha(p):
    g = cargar(p, vig(1, "1"))
    p.lecturas = [vig(5, "1", "f" * 64, secuencia=7)]
    assert g.refrescar() is True
    assert p.obtenidos == [(1, 1), (5, 7)]                            # SHA distinto: camino normal


def test_adopcion_exige_validaciones_registradas(p):
    g = cargar(p, vig(1, "1"))
    g._poner(validado=False)
    p.lecturas = [vig(5, "1", secuencia=7)]
    g.refrescar()
    assert p.obtenidos == [(1, 1), (5, 7)]


# --------------------------------------------------------------------- alias
def test_alias_desincronizado_degrada_solo_si_persiste_dos_refrescos(p):
    g = cargar(p, vig(1, "1"))
    p.alias = "9"
    g.refrescar()
    assert g.motivos_degradacion() == []
    g.refrescar()
    assert g.motivos_degradacion() == ["el alias de MLflow no coincide con la promoción confirmada"]
    p.alias = "1"
    g.refrescar()
    assert g.motivos_degradacion() == []
    p.alias = OrigenNoDisponible("MLflow caído")                     # MLflow caído no cuenta
    g.refrescar()
    assert g.alias_desincronizado == 0
