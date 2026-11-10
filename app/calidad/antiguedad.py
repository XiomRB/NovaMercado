"""Advertencia por observaciones desactualizadas (DECISIONES.md §13, D-D).

En operación normal la venta real de F−1 llega antes de la solicitud de F, así
que la última venta conocida de cada familia es F−1. Si es anterior, el modelo
pronostica sin esos rezagos: se responde igual, con una advertencia.

Solo se usan ventas con fecha < F (las recibe ya filtradas) y la fecha F de la
solicitud: nunca el reloj del sistema.
"""

from __future__ import annotations

from datetime import date

from app.calidad.resultado import RegistroCalidad

DIAS_VENTANA_NIVEL = 28   # media_28: sin ventas en esa ventana el modelo usa el nivel de respaldo


def advertir_antiguedad(fecha: date, ultimas: dict[str, date | None], familias: list[str],
                        registro: RegistroCalidad) -> None:
    """``ultimas``: última fecha con venta real (< ``fecha``) por familia, o None."""
    atrasadas, sin_nivel, sin_datos = [], [], []
    for f in familias:
        u = ultimas.get(f)
        if u is None:
            sin_datos.append(f)
            continue
        if u >= fecha:
            raise ValueError("las ventas usadas para advertir deben ser anteriores a la fecha solicitada")
        dias = (fecha - u).days
        if dias > DIAS_VENTANA_NIVEL:
            sin_nivel.append(f"{f} ({dias} días)")
        elif dias > 1:
            atrasadas.append(f"{f} ({dias} días)")
    partes = []
    if atrasadas:
        partes.append("la última venta real conocida es anterior al día previo para: " + ", ".join(atrasadas)
                      + "; se pronosticó sin los días faltantes")
    if sin_nivel:
        partes.append(f"no hay ventas reales en los {DIAS_VENTANA_NIVEL} días previos para: " + ", ".join(sin_nivel)
                      + "; se usó el nivel de respaldo")
    if sin_datos:
        partes.append("no hay ventas reales anteriores para: " + ", ".join(sin_datos)
                      + "; se usó el nivel de respaldo")
    if partes:
        registro.advertir("fecha", "; ".join(partes))
