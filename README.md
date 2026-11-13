# NovaMercado — servicio de pronóstico de ventas e inventario

[![CI](https://github.com/XiomRB/NovaMercado/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XiomRB/NovaMercado/actions/workflows/ci.yml)

Servicio de IA que pronostica las ventas diarias por tienda y familia de
producto, sugiere la cantidad a tener disponible según el costo de faltantes y
sobrantes, se defiende de datos defectuosos, registra la operación y se
reentrena automáticamente.

> **Estado:** etapas 0–4 integradas (calidad de datos, contrato, modelos,
> backtesting, persistencia, CI y API). Etapa 5 (MLflow) en revisión. El
> arranque completo con Docker y Airflow llegan en las etapas 6–7; este README
> se completará con `docker compose up` cuando existan.

## Arquitectura (etapa 5)

```
 tiendas ──HTTP──► API (FastAPI) ─────────────────────► PostgreSQL del producto (db)
                    │ GestorModelo ◄── ProveedorMLflow ◄── caché local validada (MODELO_CACHE_DIR)
                    │       ▲ cada 15 s: versión del alias "vigente"
                    │       │
                    │   MLflow (servicio "mlflow") ── mlflow_datos (SQLite: experimentos, registro, alias)
                    │       ▲                       └ mlflow_artefactos (modelo.joblib, copia de datos, backtest)
 entrenar_mlflow ───┘───────┘ lee db → copia inmutable → backtest + criterio → registra → valida → alias
 airflow-db (separada; etapa 7)
```

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

Requiere Python 3.11 y, para la base de datos, Docker con Compose v2.

```bash
conda create -n lab2 python=3.11 -y && conda activate lab2
pip install -r requirements/dev.txt
```

## Bases de datos (etapa 3)

`docker-compose.yml` define dos servicios PostgreSQL 16 **separados**, cada uno
con su volumen persistente:

| Servicio | Uso | Volumen |
|---|---|---|
| `db` | base del producto (solicitudes, predicciones, observaciones, datos publicados) | `db_producto` |
| `airflow-db` | base de Airflow (se usará en la etapa 7) | `db_airflow` |

```bash
cp .env.example .env              # y cambie las claves; .env no se versiona
docker compose up -d --wait db airflow-db
set -a && . ./.env && set +a      # exporta DATABASE_URL para los comandos siguientes
alembic upgrade head              # crea o actualiza el esquema
python -m app.db.carga --raw raw  # carga los datos publicados (idempotente)
```

* `alembic upgrade head` es reproducible: en una base nueva crea todo, en una
  existente no hace nada. `alembic downgrade base` revierte el esquema.
* La carga registra el SHA-256 de cada archivo: una segunda ejecución no repite
  el trabajo, y si un archivo publicado cambió se detiene con un error.
* `docker compose down` conserva los datos; solo `docker compose down -v` borra
  los volúmenes.

Diseño de tablas, claves, restricciones e idempotencia: `DECISIONES.md` §11.

## MLflow (etapa 5)

```bash
docker compose up -d --wait db airflow-db mlflow     # UI en http://localhost:5000
set -a && . ./.env && set +a
python -m app.modelo.entrenar_mlflow                 # entrena, evalúa, registra y (si es válido) asigna el alias
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

* **`entrenar_mlflow`** requiere un repositorio **sin cambios sin confirmar**
  (si no, termina con un mensaje claro y no crea nada). Lee la base, guarda una
  copia inmutable de los datos en la ejecución, evalúa el candidato frente al
  baseline en 7 cortes de 28 días (≥ 5 % menos costo, gana ≥ 5 de 7, ningún
  corte > 10 % peor), registra `modelo.joblib`, verifica la reproducción del
  backtest y la integridad del artefacto, y solo entonces mueve el alias.
  `--promover si_no_hay` (por defecto: solo si no hay alias), `siempre`
  (reemplazo manual explícito; nunca omite validaciones) o `nunca`.
* **`python -m app.modelo.reproducir --run-id <id>`** reconstruye un
  entrenamiento solo desde MLflow (aunque la base haya cambiado) e informa
  `identica`, `equivalente` o `no_reproduce`.
* La API carga la versión del alias `vigente`; un cambio de alias se detecta en
  ≤ 20 s sin reiniciar. Con MLflow caído sigue prediciendo con el modelo cargado
  (o, al arrancar, con la última versión válida de la caché) y `/health`
  responde `degradado`.
* Desarrollo sin MLflow: `MODELO_ORIGEN=archivo` y
  `python -m app.modelo.entrenar_local` (artefacto local de la etapa 4).

## API (etapa 4)

| Endpoint | Etapa 4 |
|---|---|
| `POST /v1/prediccion` | limpieza, advertencias, predicción y registro en PostgreSQL |
| `POST /v1/observaciones` | registro idempotente (`registros_guardados` = nuevos + modificados) |
| `GET /modelo` | modelo cargado: `nombre`, `alias` y `version` del Registry (con `MODELO_ORIGEN=archivo`: alias `local`) |
| `GET /health` | `ok` / `degradado` + `motivo` (200); 503 sin modelo |
| `POST /admin/reentrenar`, `GET /admin/reentrenar/{id}` | 503 / 404 hasta la etapa 7 (Airflow) |

* La API **no entrena al iniciar**: sin modelo responde 503 en
  `/v1/prediccion` y `/health`. Un modelo nuevo (alias movido o artefacto
  local nuevo) se carga solo (consulta cada `MODELO_INTERVALO_REFRESCO` s).
* Rechazos: 422 con `detail`; quedan registrados (si la base no responde, se
  responde 422 igual y el fallo se registra en el log).
* Base de datos no disponible o que no responde: 503 (nunca se entrega una
  predicción sin registrarla). Cada consulta tiene un límite (`DB_TIMEOUT_CONSULTA`,
  5 s, en el servidor; +`DB_MARGEN_CLIENTE` en el cliente): ninguna solicitud
  queda esperando indefinidamente.
* Documentación interactiva: `http://localhost:8000/docs`.

## Pruebas

```bash
python -m pytest -q -m "not integracion"     # sin base de datos (calidad, contrato, ML, API)
bash scripts/pruebas_integracion_local.sh     # integración con un PostgreSQL desechable en Docker
bash scripts/verificar_persistencia_docker.sh # reinicio y down/up sin perder datos (producto y MLflow)
```

* Ninguna prueba usa `raw/`: los datos son sintéticos (`tests/fixtures/`,
  `tests/ml/sinteticos.py`).
* Las pruebas de integración (`tests/integracion/`, marcador `integracion`)
  necesitan `TEST_DATABASE_URL` apuntando a un servidor **desechable**: cada
  prueba crea y borra su propia base. Sin esa variable se omiten con un motivo
  explícito. El script local la configura con un contenedor temporal. Las de
  MLflow levantan su propio servidor MLflow (proceso local con SQLite); tardan
  unos 10 minutos.
* `verificar_persistencia_docker.sh` usa un proyecto de Compose propio
  (`novamercado-verificacion`) con claves aleatorias y lo elimina al terminar:
  no toca los volúmenes reales.
* Con `PYTHON=/ruta/a/python` los scripts usan otro intérprete.

## Integración continua

`.github/workflows/ci.yml` se ejecuta en cada Pull Request hacia `main` (al
abrirlo y con cada push a su rama) y en cada push a `main`:

| Job | Qué verifica |
|---|---|
| Formato y calidad (ruff) | `ruff check .` (reglas en `pyproject.toml`) |
| Pruebas unitarias y de regresión | toda la suite sin base de datos (etapas 1–3), sin `raw/` |
| Pruebas de integración (PostgreSQL) | migraciones up/down/up y `pytest -m integracion` contra un PostgreSQL 16 de servicio |
| Persistencia con Docker Compose | `scripts/verificar_persistencia_docker.sh` |
| **CI resultado** | aprueba solo si todos los anteriores aprobaron; es el check obligatorio para fusionar |

El workflow no contiene credenciales: el PostgreSQL de servicio usa
autenticación `trust` (solo existe dentro del runner) y la verificación con
Compose genera claves aleatorias y las enmascara. Las acciones están fijadas
por SHA de commit. No despliega ni publica imágenes. Cada job publica un
resumen con el número de pruebas y los reportes JUnit como artefacto.

Ejecutar lo mismo localmente:

```bash
ruff check .
python -m pytest -q -m "not integracion"
bash scripts/pruebas_integracion_local.sh
bash scripts/verificar_persistencia_docker.sh
```

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
  db/             esquema, conexión/transacciones, carga de datos, repositorio
  modelo/         proveedores (archivo, MLflow), caché, gestor (recarga), entrenamiento,
                  copia de datos, evaluación y criterio, registro, reproducción, Git
  servicios/      predicción, observaciones, rechazos, interfaz de persistencia
  main.py         FastAPI (endpoints y manejo de errores)
  salud.py        monitor de salud en segundo plano
  orquestador.py  interfaz del reentrenamiento (Airflow en la etapa 7)
  config.py       configuración por variables de entorno
ml/               variables, modelos, costo y cantidad sugerida, backtesting
migraciones/      Alembic (versions/0001_esquema_inicial.py)
scripts/          experimento, verificación de persistencia, pruebas locales
reportes/etapa2/  resultados del backtesting
requirements/     comun.txt, api.txt, dev.txt (versiones fijas)
tests/
  unit/           pruebas por módulo y de robustez
  ml/             sin fuga temporal, costo/cuantil, modelos y backtesting
  modelo/         artefacto, caché, copia de datos, criterio, Git, reproducibilidad (sin servidores)
  ejemplos/       un ejemplo sintético por caso del catálogo de errores
  api/            contrato HTTP, salud y modelo (persistencia en memoria, modelo sintético)
  integracion/    PostgreSQL y MLflow reales: migraciones, carga, idempotencia, API, registro, alias
  fixtures/       catálogo y ejemplos sintéticos (sin datos originales)
.github/          workflow de CI y su script de resumen
docker-compose.yml, .env.example, alembic.ini
AGENTS.md         contexto para agentes de programación
DECISIONES.md     decisiones técnicas y de producto
```
