"""Datos sintéticos pequeños para probar el pipeline de ML sin depender de raw/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.datos import DatosBase

INICIO = pd.Timestamp("2025-01-01")
FAMILIAS = ["fam_a", "fam_b", "fam_c"]
# τ: fam_a 0.833, fam_b 0.286, fam_c 0.952
COSTOS = pd.DataFrame({"familia_id": FAMILIAS, "costo_faltante": [1.0, 1.0, 2.0],
                       "costo_sobrante": [0.2, 2.5, 0.1]})


def generar(dias: int = 160, semilla: int = 0, apertura_tienda_3: int = 130) -> DatosBase:
    rng = np.random.default_rng(semilla)
    tiendas = pd.DataFrame({
        "tienda_id": [1, 2, 3],
        "nombre": ["T1", "T2", "T3"],
        "ciudad": ["Guatemala", "Cobán", "Guatemala"],
        "departamento": ["Guatemala", "Alta Verapaz", "Guatemala"],
        "tipo": ["hipermercado", "supermercado", "express"],
        "fecha_apertura": [INICIO, INICIO, INICIO + pd.Timedelta(days=apertura_tienda_3)],
    })
    feriados = pd.DataFrame({
        "fecha": pd.to_datetime(["2025-02-14", "2025-03-10", "2025-04-17", "2025-05-01"]),
        "nombre": ["Cariño", "Fiesta Cobán", "Jueves Santo", "Trabajo"],
        "tipo": ["comercial", "feriado", "feriado", "feriado"],
        "ambito": ["nacional", "Cobán", "nacional", "nacional"],
    })
    nivel_tienda = {1: 300.0, 2: 150.0, 3: 40.0}
    nivel_fam = {"fam_a": 1.0, "fam_b": 0.5, "fam_c": 0.2}
    dow = np.array([0.85, 0.9, 0.9, 0.95, 1.1, 1.3, 1.0])
    filas = []
    for t, tienda in tiendas.iterrows():
        for d in range(dias):
            fecha = INICIO + pd.Timedelta(days=d)
            if fecha < tienda.fecha_apertura:
                continue
            for f in FAMILIAS:
                promo = int(rng.random() < 0.4) * int(rng.integers(1, 8))
                mu = nivel_tienda[tienda.tienda_id] * nivel_fam[f] * dow[fecha.dayofweek] * (1 + 0.05 * promo)
                filas.append((fecha, tienda.tienda_id, f, int(rng.poisson(mu)), promo))
    historico = pd.DataFrame(filas, columns=["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"])
    familias = pd.DataFrame({"familia_id": FAMILIAS, "nombre": ["A", "B", "C"]})
    return DatosBase(historico, tiendas, familias, COSTOS.copy(), feriados)
