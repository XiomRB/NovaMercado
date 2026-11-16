# Evidencia de MLflow

Modelo registrado `novamercado-ventas`; alias: `vigente` → versión 1.

`GET /modelo`: `{"nombre": "novamercado-ventas", "alias": "vigente", "version": "1", "es_vigente": true, "estado_vigencia": "vigente", "version_vigente": "1"}`

## Experimentos

| id | nombre | estado |
|---|---|---|
| 1 | novamercado-entrenamiento | active |
| 0 | Default | active |

## Ejecuciones de `novamercado-entrenamiento` (3)

| run_id | nombre | estado | inicio | verificación | versión | commit |
|---|---|---|---|---|---|---|
| `156b8eac8082` | traveling-colt-517 | FAILED | 2026-11-16T04:13:51+00:00 | — | — | `—` |
| `6807a13fb61d` | rare-ox-900 | FINISHED | 2026-11-16T04:16:45+00:00 | superada | 1 | `—` |
| `73457a30e036` | rogue-ram-136 | FAILED | 2026-11-16T04:22:49+00:00 | — | — | `—` |

Métricas y parámetros completos en `mlflow.json`.

## Versiones registradas

| versión | run_id | estado (tag) | sha256 del artefacto | creada |
|---|---|---|---|---|
| 1 | `6807a13fb61d` | valido | `d3bbef9291fee60a` | 2026-11-16T04:19:03+00:00 |

## Decisiones de promoción (PostgreSQL, fuente de verdad)

| versión | origen | resultado | secuencia | motivo |
|---|---|---|---|---|
| 1 | arranque | confirmada | 1 | costo 628,813 vs 879,479 del baseline (28.5 % menos); gana 7/7 cortes; peor corte -17.0 % |
