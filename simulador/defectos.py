"""Defectos en las solicitudes y su resultado esperado (oráculo).

El oráculo sale de la tabla de ``DECISIONES.md`` §3 (y §9, §13 D-D para las
advertencias que dependen del contexto), **no del código del producto**. Cada defecto
declara, al construirse, qué acción (campo, acción) debe aparecer en ``calidad``, si la
solicitud debe rechazarse (422) y qué valores quedan (promociones efectivas o ventas
guardadas). Si el servicio responde distinto, es una discrepancia: se informa, nunca se
ajusta el oráculo para que coincida.

Grupos combinables (a lo sumo un defecto por grupo): ``tienda``, ``fecha``, ``campo``
(nombres de campo de primer nivel), ``estructura`` (lista de ítems) y ``valor`` (un ítem).
``cuerpo`` va solo. Las combinaciones no están en el catálogo publicado.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

REPARAR, ADVERTIR = "reparar", "advertir"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]
ABREV = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


@dataclass
class Item:
    pares: list            # [[clave, valor], ...] o None si el ítem no es un objeto
    familia: str | None    # familia real que representa (None: ítem ignorado)
    crudo: object = None   # valor del ítem cuando no es un objeto
    reconocida: bool = True
    valor: int | None = None      # promoción efectiva / venta guardada (None: supuesta o descartada)
    marcado: bool = False         # ya tocado por un defecto
    duplicado: bool = False


@dataclass
class Esperado:
    codigo: int
    requeridas: list[tuple[str, str]] = field(default_factory=list)
    valores: dict[str, int] = field(default_factory=dict)      # familia -> promoción efectiva o venta guardada
    supuestas: list[str] = field(default_factory=list)          # familias con promoción supuesta (predicción)
    indices: dict[str, int] = field(default_factory=dict)       # familia -> índice del ítem usado

    def como_dict(self) -> dict:
        return {"codigo": self.codigo, "requeridas": [list(x) for x in self.requeridas], "valores": self.valores,
                "supuestas": self.supuestas, "indices": self.indices}


@dataclass
class Solicitud:
    cuerpo: object | None
    texto: str | None                  # cuerpo crudo (no JSON)
    defectos: list[str]
    esperado: Esperado

    def bytes(self) -> bytes:
        return self.texto.encode() if self.texto is not None else json.dumps(self.cuerpo, ensure_ascii=False).encode()


class Borrador:
    """Solicitud en construcción: claves de primer nivel ordenadas e ítems con su familia real."""

    def __init__(self, tipo: str, tienda: int, fecha: date, valores: dict[str, int], catalogo):
        self.tipo = tipo                                  # "prediccion" | "observacion"
        self.lista = "familias" if tipo == "prediccion" else "ventas"
        self.sub = "en_promocion" if tipo == "prediccion" else "ventas"
        self.tienda, self.fecha, self.catalogo = tienda, fecha, catalogo
        self.top = [["tienda_id", tienda], ["fecha", fecha.isoformat()], [self.lista, None]]
        self.items = [Item([["familia_id", f], [self.sub, v]], f, valor=v) for f, v in valores.items()]
        self.requeridas: list[tuple[str, str]] = []
        self.item_req: list[tuple[Item, str | None, str]] = []    # (ítem, subcampo, acción)
        self.rechazo = False
        self.como_objeto = False
        self.texto = None
        self.cuerpo_lista = False
        self.defectos: list[str] = []

    # -- utilidades
    def par(self, clave):
        return next(p for p in self.top if p[0] == clave)

    def tiene(self, clave) -> bool:
        return any(p[0] == clave for p in self.top)

    def libres(self) -> list[Item]:
        return [i for i in self.items if i.pares is not None and not i.marcado and not i.duplicado]

    def construir(self) -> Solicitud:
        if self.texto is not None:
            return Solicitud(None, self.texto, self.defectos, Esperado(422))
        if self.como_objeto:
            lista = {}
            for it in self.items:
                d = dict(it.pares)
                k = d.pop("familia_id")
                lista[k] = d[self.sub] if list(d) == [self.sub] else d
        else:
            lista = [dict(i.pares) if i.pares is not None else i.crudo for i in self.items]
        self.par_lista()[1] = lista
        cuerpo = {p[0]: p[1] for p in self.top}
        if self.cuerpo_lista:
            cuerpo = [cuerpo]
        if self.rechazo or self.cuerpo_lista:
            return Solicitud(cuerpo, None, self.defectos, Esperado(422))
        esp = Esperado(200, list(self.requeridas))
        vistos: dict[str, int] = {}
        for idx, it in enumerate(self.items):
            for item, sub, accion in self.item_req:
                if item is it:
                    esp.requeridas.append((f"{self.lista}[{idx}]" + (f".{sub}" if sub else ""), accion))
            if it.pares is not None and it.reconocida and it.familia not in vistos:
                vistos[it.familia] = idx
                if it.valor is not None:
                    esp.valores[it.familia] = it.valor
                elif self.tipo == "prediccion":
                    esp.valores[it.familia] = 0
                    esp.supuestas.append(it.familia)
        esp.indices = vistos
        faltan = [f for f in self.catalogo.familias if f not in vistos]
        if self.tipo == "prediccion":
            esp.requeridas += [("familias", ADVERTIR)] * len(faltan)
            for f in faltan:
                esp.valores[f] = 0
                esp.supuestas.append(f)
        else:
            if not esp.valores:
                return Solicitud(cuerpo, None, self.defectos, Esperado(422))
            if faltan:
                esp.requeridas.append(("ventas", ADVERTIR))
        return Solicitud(cuerpo, None, self.defectos, esp)

    def par_lista(self):
        return next(p for p in self.top if p[0] == self.lista or p[2:] == ["lista"])


@dataclass
class Catalogo:
    tiendas: dict[int, date]
    familias: list[str]
    nombres: dict[str, str]


# ------------------------------------------------------------------ defectos
DEFECTOS: dict[str, tuple[str, str, object]] = {}     # nombre -> (grupo, código del catálogo, función)


def defecto(nombre: str, grupo: str, codigo: str = "nuevo"):
    def registrar(fn):
        DEFECTOS[nombre] = (grupo, codigo, fn)
        return fn
    return registrar


def _tienda(b: Borrador, valor, accion):
    b.par("tienda_id")[1] = valor
    if accion == "rechazar":
        b.rechazo = True
    else:
        b.requeridas.append(("tienda_id", accion))


@defecto("tienda_texto", "tienda", "E01")
def _(b, r): _tienda(b, str(b.tienda), REPARAR)


@defecto("tienda_prefijo", "tienda", "E01")
def _(b, r): _tienda(b, f"T{b.tienda:02d}", REPARAR)


@defecto("tienda_palabra", "tienda", "E01")
def _(b, r): _tienda(b, f"tienda {b.tienda}", REPARAR)


@defecto("tienda_decimal_entero", "tienda", "E01")
def _(b, r): _tienda(b, float(b.tienda), REPARAR)


@defecto("tienda_espacios_unicode", "tienda")
def _(b, r): _tienda(b, f" {b.tienda} ", REPARAR)


@defecto("tienda_varios_numeros", "tienda")
def _(b, r): _tienda(b, f"T{b.tienda}-2", "rechazar")


@defecto("tienda_negativa", "tienda")
def _(b, r): _tienda(b, -b.tienda, "rechazar")


@defecto("tienda_booleana", "tienda")
def _(b, r): _tienda(b, True, "rechazar")


@defecto("tienda_inexistente", "tienda")
def _(b, r): _tienda(b, max(b.catalogo.tiendas) + 89, "rechazar")


@defecto("tienda_decimal", "tienda")
def _(b, r): _tienda(b, b.tienda + 0.5, "rechazar")


@defecto("tienda_ausente", "tienda")
def _(b, r):
    b.top = [p for p in b.top if p[0] != "tienda_id"]
    b.rechazo = True


def _fecha(b: Borrador, valor, accion):
    b.par("fecha")[1] = valor
    if accion == "rechazar":
        b.rechazo = True
    else:
        b.requeridas.append(("fecha", accion))


@defecto("fecha_barras", "fecha", "E04")
def _(b, r): _fecha(b, b.fecha.strftime("%Y/%m/%d"), REPARAR)


@defecto("fecha_puntos", "fecha")
def _(b, r): _fecha(b, b.fecha.strftime("%Y.%m.%d"), REPARAR)


@defecto("fecha_compacta", "fecha", "E04")
def _(b, r): _fecha(b, b.fecha.strftime("%Y%m%d"), REPARAR)


@defecto("fecha_con_hora", "fecha", "E04")
def _(b, r): _fecha(b, b.fecha.isoformat() + "T08:30:00-06:00", REPARAR)


@defecto("fecha_espacios", "fecha")
def _(b, r): _fecha(b, f"  {b.fecha.isoformat()} ", REPARAR)


@defecto("fecha_sin_ceros", "fecha", "E04")
def _(b, r):
    f = b.fecha
    if f.month >= 10 and f.day >= 10:
        return _fecha(b, f.strftime("%Y%m%d"), REPARAR)
    _fecha(b, f"{f.year}-{f.month}-{f.day}", REPARAR)


@defecto("fecha_dd_mm", "fecha", "E04")
def _(b, r):
    f = b.fecha
    ambigua = f.day <= 12 and f.day != f.month
    _fecha(b, f"{f.day:02d}/{f.month:02d}/{f.year}", ADVERTIR if ambigua else REPARAR)


@defecto("fecha_mm_dd", "fecha", "E04")
def _(b, r):
    f = b.fecha
    if f.day <= 12:                  # solo hay lectura única con el segundo número > 12
        return _fecha(b, f.strftime("%Y/%m/%d"), REPARAR)
    _fecha(b, f"{f.month:02d}/{f.day:02d}/{f.year}", REPARAR)


@defecto("fecha_anio_dos_digitos", "fecha")
def _(b, r):
    f = b.fecha
    _fecha(b, f"{f.day:02d}/{f.month:02d}/{f.year % 100:02d}", ADVERTIR)


@defecto("fecha_mes_nombre", "fecha", "E04")
def _(b, r):
    f = b.fecha
    texto = (f"{f.day} de {MESES[f.month - 1]} de {f.year}" if r.random() < 0.5
             else f"{f.day:02d}-{ABREV[f.month - 1]}-{f.year}")
    _fecha(b, texto, REPARAR)


@defecto("fecha_inexistente", "fecha")
def _(b, r): _fecha(b, f"{b.fecha.year}-02-30", "rechazar")


@defecto("fecha_ilegible", "fecha")
def _(b, r): _fecha(b, "pasado mañana", "rechazar")


@defecto("fecha_antes_apertura", "fecha")
def _(b, r): _fecha(b, (b.catalogo.tiendas[b.tienda] - timedelta(days=1)).isoformat(), "rechazar")


@defecto("fecha_ausente", "fecha")
def _(b, r):
    b.top = [p for p in b.top if p[0] != "fecha"]
    b.rechazo = True


# -- nombres de campo de primer nivel (E07 y generalizaciones)
def _renombrar(b: Borrador, actual: str, nuevo: str) -> bool:
    if not b.tiene(actual):          # quitado por otro defecto: la solicitud ya es un rechazo
        return False
    b.par(actual)[0] = nuevo
    if actual == b.lista:
        b.par(nuevo).append("lista")
    return True


@defecto("campo_mayusculas", "campo", "E07")
def _(b, r):
    if _renombrar(b, "tienda_id", "Tienda_ID"):
        b.requeridas.append(("tienda_id", REPARAR))


@defecto("campo_camel", "campo", "E07")
def _(b, r):
    if _renombrar(b, "tienda_id", "tiendaId"):
        b.requeridas.append(("tienda_id", REPARAR))


@defecto("campo_sinonimo", "campo", "E07")
def _(b, r):
    clave, nuevo = r.choice([("tienda_id", "store_id"), ("tienda_id", "tienda"), ("fecha", "date"),
                             (b.lista, "families" if b.tipo == "prediccion" else "sales")])
    if _renombrar(b, clave, nuevo):
        b.requeridas.append((clave, REPARAR))


@defecto("campo_errata", "campo")
def _(b, r):
    if _renombrar(b, "fecha", "fechaa"):
        b.requeridas.append(("fecha", ADVERTIR))


@defecto("campo_desconocido", "campo")
def _(b, r):
    b.top.append(["canal_envio", "pos-7"])
    b.requeridas.append(("canal_envio", ADVERTIR))


@defecto("campo_generico", "campo")
def _(b, r):
    _renombrar(b, "tienda_id", "id")          # genérico: se ignora y falta el obligatorio
    b.rechazo = True


@defecto("campo_repetido_igual", "campo")
def _(b, r):
    if b.tiene("tienda_id"):
        b.top.insert(1, ["tienda", b.par("tienda_id")[1]])
        b.requeridas.append(("tienda_id", REPARAR))


@defecto("campo_repetido_distinto", "campo")
def _(b, r):
    otra = next(t for t in sorted(b.catalogo.tiendas) if t != b.tienda)
    if b.tiene("tienda_id"):
        b.top.insert(1, ["tienda", otra])
    b.rechazo = True


# -- estructura de la lista
@defecto("faltan_familias", "estructura", "E05")
def _(b, r):
    for it in r.sample(b.libres(), r.randint(1, 3)):
        b.items.remove(it)


@defecto("familia_irreconocible", "estructura")
def _(b, r):
    it = r.choice(b.libres())
    it.pares[0][1], it.reconocida, it.marcado = "zz-desconocida", False, True
    b.item_req.append((it, "familia_id", ADVERTIR))


@defecto("familia_repetida_igual", "estructura", "E06")
def _(b, r):
    it = r.choice(b.libres())
    copia = Item(copy.deepcopy(it.pares), it.familia, valor=it.valor, duplicado=True)
    b.items.append(copia)
    b.item_req.append((copia, "familia_id", REPARAR))


@defecto("familia_repetida_distinta", "estructura", "E06")
def _(b, r):
    it = r.choice(b.libres())
    pares = copy.deepcopy(it.pares)
    pares[1][1] = it.valor + 1
    b.items.append(Item(pares, it.familia, duplicado=True))
    b.rechazo = True


@defecto("lista_como_objeto", "estructura")
def _(b, r):
    b.como_objeto = True
    b.requeridas.append((b.lista, REPARAR))


@defecto("lista_vacia", "estructura")
def _(b, r):
    b.items = []
    b.rechazo = True


@defecto("item_no_objeto", "estructura")
def _(b, r):
    it = Item(None, None, crudo=r.choice(b.catalogo.familias), reconocida=False)
    b.items.insert(r.randint(0, len(b.items)), it)
    b.item_req.append((it, None, ADVERTIR))


@defecto("ninguna_venta_valida", "estructura", "E08")
def _(b, r):
    if b.tipo != "observacion":
        return DEFECTOS["lista_vacia"][2](b, r)
    for it in b.items:
        it.pares[1][1] = -1 - it.valor
    b.rechazo = True


# -- valor de un ítem
def _valor(b: Borrador, r, nuevo, accion, efectivo="igual"):
    it = r.choice(b.libres())
    it.marcado = True
    v = it.valor
    it.pares[1][1] = nuevo(v)
    if efectivo == "descartado":
        it.valor = None
    elif callable(efectivo):
        it.valor = efectivo(v)
    b.item_req.append((it, b.sub, accion))


def _nombre_familia(b: Borrador, r, transformar, accion=REPARAR, solo=None):
    candidatos = [i for i in b.libres() if solo is None or i.familia == solo]
    if not candidatos:
        candidatos, transformar, accion = b.libres(), str.upper, REPARAR
    it = r.choice(candidatos)
    it.marcado = True
    it.pares[0][1] = transformar(it.familia)
    b.item_req.append((it, "familia_id", accion))


@defecto("familia_mayusculas", "valor", "E02")
def _(b, r): _nombre_familia(b, r, str.upper)


@defecto("familia_nombre_catalogo", "valor", "E02")
def _(b, r): _nombre_familia(b, r, lambda f: b.catalogo.nombres[f].upper() if b.catalogo.nombres[f] == f
                             else b.catalogo.nombres[f])


@defecto("familia_separadores", "valor", "E02")
def _(b, r): _nombre_familia(b, r, lambda f: f.replace("_", "-") if "_" in f else f"  {f} ")


@defecto("familia_parecida", "valor")
def _(b, r): _nombre_familia(b, r, lambda f: "lacteo", ADVERTIR, solo="lacteos")


@defecto("valor_texto", "valor", "E03/E10")
def _(b, r): _valor(b, r, lambda v: str(v), REPARAR)


@defecto("valor_decimal_entero", "valor", "E03/E10")
def _(b, r): _valor(b, r, lambda v: float(v), REPARAR)


@defecto("valor_texto_con_espacios", "valor")
def _(b, r): _valor(b, r, lambda v: f" {v} ", REPARAR)


@defecto("valor_nulo", "valor")
def _(b, r): _valor(b, r, lambda v: None, ADVERTIR, efectivo="descartado")


@defecto("valor_no_numerico", "valor")
def _(b, r): _valor(b, r, lambda v: "s/d", ADVERTIR, efectivo="descartado")


@defecto("valor_negativo", "valor", "E08")
def _(b, r): _valor(b, r, lambda v: -1 - v, ADVERTIR, efectivo="descartado")


@defecto("valor_booleano", "valor")
def _(b, r): _valor(b, r, lambda v: True, ADVERTIR, efectivo="descartado")


@defecto("valor_ausente", "valor")
def _(b, r):
    it = r.choice(b.libres())
    it.marcado = True
    it.pares = [p for p in it.pares if p[0] != b.sub]
    it.valor = None
    b.item_req.append((it, b.sub, ADVERTIR))


@defecto("valor_fraccion", "valor")
def _(b, r):
    if b.tipo == "prediccion":         # 2.5 -> advertir y redondear a 3 (§3)
        return _valor(b, r, lambda v: v + 0.5, ADVERTIR, efectivo=lambda v: v + 1)
    # "812,5" -> advertir y redondear (0,5 hacia arriba): se envía (v-1),5 para guardar v
    candidatos = [i for i in b.libres() if i.valor >= 1]
    if not candidatos:
        return DEFECTOS["valor_texto"][2](b, r)
    it = r.choice(candidatos)
    it.marcado = True
    it.pares[1][1] = f"{it.valor - 1},5"
    b.item_req.append((it, b.sub, ADVERTIR))


@defecto("valor_coma_decimal_cero", "valor", "E10")
def _(b, r): _valor(b, r, lambda v: f"{v},0", REPARAR)


@defecto("valor_miles", "valor", "E10")
def _(b, r):
    grandes = [i for i in b.libres() if i.valor >= 1000]
    if not grandes:
        return DEFECTOS["valor_texto"][2](b, r)
    it = r.choice(grandes)
    it.marcado = True
    it.pares[1][1] = r.choice([f"{it.valor:,}", f"{it.valor:,}".replace(",", ".") + ",0"])
    b.item_req.append((it, b.sub, REPARAR))


@defecto("campo_item_sinonimo", "valor", "E07")
def _(b, r):
    it = r.choice(b.libres())
    it.marcado = True
    nuevo = r.choice(["promo", "enPromocion", "Promocion"] if b.tipo == "prediccion"
                     else ["unidades_vendidas", "Ventas", "venta_real"])
    it.pares[1][0] = nuevo
    b.item_req.append((it, b.sub, REPARAR))


@defecto("campo_item_generico", "valor")
def _(b, r):
    it = r.choice(b.libres())
    it.marcado = True
    it.pares[1][0] = "cantidad"           # genérico: se ignora y falta el valor
    it.valor = None
    b.item_req.append((it, "cantidad", ADVERTIR))
    b.item_req.append((it, b.sub, ADVERTIR))


@defecto("cuerpo_no_json", "cuerpo")
def _(b, r): b.texto = '{"tienda_id": %d, "fecha": "%s", ' % (b.tienda, b.fecha.isoformat())


@defecto("cuerpo_lista", "cuerpo")
def _(b, r): b.cuerpo_lista = True


GRUPOS = ("tienda", "fecha", "campo", "estructura", "valor")
SOLO_PREDICCION = {"familia_parecida"}


def generar(tipo: str, tienda: int, fecha: date, valores: dict[str, int], catalogo: Catalogo,
            rng: random.Random, defectos: list[str] | None = None) -> Solicitud:
    """``defectos`` explícitos (pruebas) o ninguno: solicitud limpia."""
    b = Borrador(tipo, tienda, fecha, valores, catalogo)
    orden = {g: i for i, g in enumerate(("cuerpo", "estructura", "valor", "tienda", "fecha", "campo"))}
    for nombre in sorted(defectos or [], key=lambda n: orden[DEFECTOS[n][0]]):
        if DEFECTOS[nombre][0] in ("valor", "estructura") and not b.libres():
            continue                  # otro defecto ya vació la lista: este no aplica
        b.defectos.append(nombre)
        DEFECTOS[nombre][2](b, rng)
    return b.construir()


def elegir(rng: random.Random, tipo: str) -> list[str]:
    """Uno a tres defectos de grupos distintos (30 % combinaciones)."""
    nombres = [n for n in DEFECTOS if not (tipo == "observacion" and n in SOLO_PREDICCION)]
    if rng.random() < 0.06:
        return [rng.choice([n for n in nombres if DEFECTOS[n][0] == "cuerpo"])]
    grupos = rng.sample(GRUPOS, rng.choice([2, 3]) if rng.random() < 0.3 else 1)
    return [rng.choice([n for n in nombres if DEFECTOS[n][0] == g]) for g in grupos]


# ----------------------------------------------------- comparación con la respuesta
def contexto_prediccion(esp: Esperado, maximo_seguro: dict[str, int], maximo_posible: dict[str, int],
                        atrasada: bool) -> tuple[list, list]:
    """Advertencias que dependen del contexto (requeridas, opcionales):

    * §9: promoción mayor que el máximo de entrenamiento del modelo vigente. Ese máximo
      está entre ``maximo_seguro`` (incluido en todo entrenamiento) y ``maximo_posible``
      (todo lo que un modelo pudo ver): por encima de este último es obligatoria; entre
      ambos, admisible; hasta el seguro, prohibida.
    * §13 D-D: la última venta guardada de alguna familia es anterior a F−1 → obligatoria.
    """
    requeridas, opcionales = [], []
    if esp.codigo != 200:
        return requeridas, opcionales
    for fam, valor in esp.valores.items():
        if fam in esp.supuestas or fam not in esp.indices:
            continue
        campo = (f"familias[{esp.indices[fam]}].en_promocion", ADVERTIR)
        if valor > maximo_posible.get(fam, 0):
            requeridas.append(campo)
        elif valor > maximo_seguro.get(fam, 0):
            opcionales.append(campo)
    if atrasada:
        requeridas.append(("fecha", ADVERTIR))
    return requeridas, opcionales


def comparar(esp: Esperado, codigo, cuerpo, requeridas_extra=(), opcionales=()) -> list[str]:
    """Discrepancias entre la respuesta y el oráculo (vacía = coincide)."""
    if codigo != esp.codigo:
        return [f"código {codigo}, el oráculo esperaba {esp.codigo}"]
    if codigo != 200:
        return []
    obs = [(o.get("campo"), o.get("accion")) for o in (cuerpo or {}).get("calidad", {}).get("observaciones", [])]
    pendientes = list(obs)
    errores = []
    for req in list(esp.requeridas) + list(requeridas_extra):
        if req in pendientes:
            pendientes.remove(req)
        else:
            errores.append(f"falta la acción {req[1]} en {req[0]}")
    libres = list(opcionales)
    for o in pendientes:
        if o in libres:
            libres.remove(o)
        else:
            errores.append(f"acción no esperada: {o[1]} en {o[0]}")
    return errores
