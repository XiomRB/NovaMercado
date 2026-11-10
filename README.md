# NovaMercado — servicio de pronóstico de ventas e inventario

Servicio de IA que pronostica las ventas diarias por tienda y familia de
producto, sugiere la cantidad a tener disponible según el costo de faltantes y
sobrantes, se defiende de datos defectuosos, registra la operación y se
reentrena automáticamente.

> **Estado:** etapas 0–1 completadas (repositorio, dependencias, capa de calidad
> de datos y esquemas del contrato); etapa 2 (modelos y backtesting) en revisión. La API, la base de datos, MLflow, Airflow y
> Docker Compose se agregan en las etapas siguientes; este README se actualizará
> con las instrucciones de `docker compose up` cuando existan.

## Datos publicados

Coloque los archivos publicados en la carpeta **`raw/`** en la raíz del
repositorio (está excluida de git):

```
raw/
├── tiendas.csv
├── familias.csv
├── historico_ventas.csv
├── feriados.csv
├── costos.csv
├── catalogo_errores.md
└── ejemplos/
    ├── prediccion_valida.json
    ├── observacion_valida.json
    └── errores/E01_*.json … E10_*.json
```

Si recibió `datos_lab2.zip`, descomprímalo dentro de `raw/`.

## Entorno de desarrollo

Requiere Python 3.11.

```bash
conda create -n lab2 python=3.11 -y && conda activate lab2
pip install -r requirements/dev.txt
python -m pytest -q
```

Las pruebas no requieren MLflow, la base de datos ni la carpeta `raw/`: usan
fixtures sintéticos de `tests/fixtures/` y datos generados, así que se ejecutan
completas desde un clon limpio.

## Experimento de modelos (etapa 2)

```bash
python -m scripts.experimento_etapa2 --raw raw --salida reportes/etapa2
```

Ejecuta el backtesting temporal (7 cortes de 28 días) del baseline y los
modelos de ML con varias estrategias de cantidad sugerida y escribe las tablas
en `reportes/etapa2/` (`resultados.md` y CSV). No registra nada en MLflow.

## Estructura actual

```
app/
  calidad/        limpieza de datos: reparar / advertir / rechazar
  schemas/        modelos Pydantic del contrato
ml/               variables, modelos, costo y cantidad sugerida, backtesting
scripts/          experimentos reproducibles
reportes/etapa2/  resultados del backtesting
requirements/     comun.txt, api.txt, dev.txt (versiones fijas)
tests/
  unit/           pruebas por módulo y de robustez
  ml/             sin fuga temporal, costo/cuantil, modelos y backtesting
  ejemplos/       un ejemplo sintético por caso del catálogo de errores
  fixtures/       catálogo y ejemplos sintéticos (sin datos originales)
AGENTS.md         contexto para agentes de programación
DECISIONES.md     decisiones técnicas y política de calidad de datos
```
