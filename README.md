# NovaMercado — servicio de pronóstico de ventas e inventario

Servicio de IA que pronostica las ventas diarias por tienda y familia de
producto, sugiere la cantidad a tener disponible según el costo de faltantes y
sobrantes, se defiende de datos defectuosos, registra la operación y se
reentrena automáticamente.

> **Estado:** etapas 0–1 completadas (repositorio, dependencias, capa de calidad
> de datos y esquemas del contrato). La API, la base de datos, MLflow, Airflow y
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

Las pruebas no requieren MLflow ni la base de datos. Las de `tests/ejemplos/`
usan `raw/ejemplos` y se omiten si esa carpeta no existe.

## Estructura actual

```
app/
  calidad/        limpieza de datos: reparar / advertir / rechazar
  schemas/        modelos Pydantic del contrato
requirements/     comun.txt, api.txt, dev.txt (versiones fijas)
tests/
  unit/           pruebas por módulo y de robustez
  ejemplos/       política sobre los ejemplos publicados
AGENTS.md         contexto para agentes de programación
DECISIONES.md     decisiones técnicas y política de calidad de datos
```
