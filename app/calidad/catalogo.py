"""Catálogos de referencia (tiendas y familias) usados por la capa de calidad."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.calidad.resultado import ADVERTIR, REPARAR
from app.calidad.texto import mejor_coincidencia, normalizar_clave, sin_palabras_vacias


@dataclass(frozen=True)
class Tienda:
    tienda_id: int
    ciudad: str
    tipo: str
    fecha_apertura: date


class Catalogo:
    def __init__(self, tiendas: dict[int, Tienda], familias: dict[str, str]):
        if not tiendas or not familias:
            raise ValueError("el catálogo requiere al menos una tienda y una familia")
        self.tiendas = dict(tiendas)
        self.familias = dict(familias)  # familia_id -> nombre legible
        # Claves normalizadas (id y nombre, con y sin palabras vacías) -> familia_id
        self._claves: dict[str, str] = {}
        for fid, nombre in self.familias.items():
            for texto in (fid, nombre):
                clave = normalizar_clave(texto)
                for variante in (clave, sin_palabras_vacias(clave), clave.replace("_", "")):
                    if variante:
                        self._claves.setdefault(variante, fid)

    @property
    def familia_ids(self) -> list[str]:
        return list(self.familias)

    def resolver_familia(self, valor: object) -> tuple[str, str | None, str | None] | None:
        """Devuelve ``(familia_id, accion, detalle)`` o ``None`` si no se reconoce.

        * igual al id                      -> sin acción
        * difiere en mayúsculas, tildes, espacios, separadores o palabras de
          enlace, o coincide con el nombre -> reparar (inequívoco)
        * parecido claro a una sola familia -> advertir (supuesto)
        """
        if not isinstance(valor, str):
            return None
        if valor in self.familias:
            return valor, None, None
        clave = normalizar_clave(valor)
        if not clave:
            return None
        for variante in (clave, sin_palabras_vacias(clave), clave.replace("_", "")):
            if variante in self._claves:
                fid = self._claves[variante]
                return fid, REPARAR, f"'{valor}' se interpretó como '{fid}'"
        fid = mejor_coincidencia(clave, self._claves)
        if fid is not None:
            return fid, ADVERTIR, f"'{valor}' no coincide exactamente; se asumió '{fid}' por similitud"
        return None

    @classmethod
    def desde_csv(cls, carpeta: str | Path) -> "Catalogo":
        carpeta = Path(carpeta)
        with open(carpeta / "tiendas.csv", encoding="utf-8") as fh:
            tiendas = {
                int(r["tienda_id"]): Tienda(
                    tienda_id=int(r["tienda_id"]),
                    ciudad=r["ciudad"],
                    tipo=r["tipo"],
                    fecha_apertura=date.fromisoformat(r["fecha_apertura"]),
                )
                for r in csv.DictReader(fh)
            }
        with open(carpeta / "familias.csv", encoding="utf-8") as fh:
            familias = {r["familia_id"]: r["nombre"] for r in csv.DictReader(fh)}
        return cls(tiendas, familias)
