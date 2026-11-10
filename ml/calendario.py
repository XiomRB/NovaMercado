"""Variables de calendario y eventos (feriados.csv).

Todas son conocidas de antemano para cualquier fecha: dependen solo de la fecha
y de ``feriados.csv`` (que cubre el periodo de evaluación). Un evento aplica a
una tienda si su ``ambito`` es ``nacional`` o igual a la ``ciudad`` de la tienda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIPOS_EVENTO = ("feriado", "comercial", "pago", "medio_dia")

COLUMNAS_CALENDARIO = [
    "dia_semana", "dia_mes", "mes", "dias_para_fin_mes",
    "ev_feriado", "ev_comercial", "ev_pago", "ev_medio_dia", "ev_codigo",
    "vispera_feriado", "post_feriado",
]


class Calendario:
    def __init__(self, feriados: pd.DataFrame):
        ev = feriados[["fecha", "nombre", "tipo", "ambito"]].copy()
        ev["fecha"] = pd.to_datetime(ev["fecha"])
        desconocidos = set(ev["tipo"]) - set(TIPOS_EVENTO)
        if desconocidos:
            raise ValueError(f"tipos de evento desconocidos en feriados: {desconocidos}")
        self.eventos = ev
        # Código estable por nombre de evento (orden alfabético): categoría para el modelo
        self.codigos = {n: i for i, n in enumerate(sorted(ev["nombre"].unique()))}

    def _eventos_de(self, claves: pd.DataFrame, desplazamiento_dias: int = 0) -> pd.DataFrame:
        """Eventos que aplican a (fecha + desplazamiento, ciudad) para cada clave."""
        c = claves[["fecha", "ciudad"]].drop_duplicates().copy()
        c["fecha_ev"] = c["fecha"] + pd.Timedelta(days=desplazamiento_dias)
        ev = self.eventos.rename(columns={"fecha": "fecha_ev"})
        nac = c.merge(ev[ev["ambito"] == "nacional"], on="fecha_ev")
        loc = c.merge(ev[ev["ambito"] != "nacional"], left_on=["fecha_ev", "ciudad"],
                      right_on=["fecha_ev", "ambito"])
        return pd.concat([nac, loc], ignore_index=True)[["fecha", "ciudad", "nombre", "tipo"]]

    def features(self, claves: pd.DataFrame) -> pd.DataFrame:
        """``claves``: DataFrame con ``fecha`` (datetime) y ``ciudad``. Devuelve una fila por clave."""
        out = claves[["fecha", "ciudad"]].copy().reset_index(drop=True)
        f = out["fecha"]
        out["dia_semana"] = f.dt.dayofweek
        out["dia_mes"] = f.dt.day
        out["mes"] = f.dt.month
        out["dias_para_fin_mes"] = (f + pd.offsets.MonthEnd(0) - f).dt.days

        hoy = self._eventos_de(out)
        base = out[["fecha", "ciudad"]].drop_duplicates()
        if len(hoy):
            flags = pd.crosstab([hoy["fecha"], hoy["ciudad"]], hoy["tipo"]).clip(upper=1)
            flags = flags.reindex(columns=TIPOS_EVENTO, fill_value=0).add_prefix("ev_").reset_index()
            codigo = (hoy.assign(cod=hoy["nombre"].map(self.codigos))
                      .groupby(["fecha", "ciudad"], as_index=False)["cod"].min()
                      .rename(columns={"cod": "ev_codigo"}))
            base = base.merge(flags, how="left", on=["fecha", "ciudad"]).merge(
                codigo, how="left", on=["fecha", "ciudad"])
        for t in TIPOS_EVENTO:
            if f"ev_{t}" not in base:
                base[f"ev_{t}"] = 0
        if "ev_codigo" not in base:
            base["ev_codigo"] = np.nan
        for t in TIPOS_EVENTO:
            base[f"ev_{t}"] = base[f"ev_{t}"].fillna(0).astype(int)

        def _marca(desp: int, tipos: tuple[str, ...], nombre: str) -> None:
            nonlocal base
            e = self._eventos_de(out, desp)
            e = e[e["tipo"].isin(tipos)][["fecha", "ciudad"]].drop_duplicates().assign(**{nombre: 1})
            base = base.merge(e, how="left", on=["fecha", "ciudad"])
            base[nombre] = base[nombre].fillna(0).astype(int)

        _marca(+1, ("feriado", "medio_dia"), "vispera_feriado")   # mañana es feriado
        _marca(-1, ("feriado",), "post_feriado")                   # ayer fue feriado

        return out.merge(base, how="left", on=["fecha", "ciudad"])[["fecha", "ciudad", *COLUMNAS_CALENDARIO]]
