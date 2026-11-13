"""Envoltorio ``mlflow.pyfunc`` del modelo de NovaMercado (M3).

El contenido real es ``modelo.joblib`` (el mismo archivo que valida y carga la
API, con su SHA-256 en el tag ``sha256_artefacto``). Este envoltorio solo
permite usar el modelo con ``mlflow.pyfunc.load_model``; la API no lo usa.

``predict`` recibe filas objetivo (``fecha``, ``tienda_id``, ``familia_id``,
``en_promocion``) y, opcionalmente, ventas previas en ``params["ventas"]``
(lista de registros ``fecha, tienda_id, familia_id, ventas``). Sin ventas previas
el modelo usa el nivel de respaldo.
"""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd


class ModeloNovaMercado(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.artefacto = joblib.load(context.artifacts["modelo"])

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        p = self.artefacto["pronosticador"]
        ventas = pd.DataFrame((params or {}).get("ventas") or [],
                              columns=["fecha", "tienda_id", "familia_id", "ventas"])
        ventas["fecha"] = pd.to_datetime(ventas["fecha"])
        objetivos = model_input.copy()
        objetivos["fecha"] = pd.to_datetime(objetivos["fecha"])
        return p.predecir(objetivos, ventas)
