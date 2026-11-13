# NovaMercado — servicio de pronóstico de ventas e inventario

[![CI](https://github.com/XiomRB/NovaMercado/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XiomRB/NovaMercado/actions/workflows/ci.yml)

Servicio de IA que pronostica las ventas diarias por tienda y familia de
producto, sugiere la cantidad a tener disponible según el costo de faltantes y
sobrantes, se defiende de datos defectuosos, registra la operación y se
reentrena automáticamente.

> **Estado:** etapas 0–5 integradas (calidad de datos, contrato, modelos,
> backtesting, persistencia, CI, API y MLflow). Etapa 6 (producto con Docker
> Compose y arranque automático) en revisión. Airflow llega en la etapa 7.

## Levantar el producto desde cero (etapa 6)

Requiere Docker con Compose v2 (memoria recomendada para Docker: 4 GB o más).

```bash
git clone https://github.com/XiomRB/NovaMercado.git && cd NovaMercado   # clonar (no descargar un ZIP)
cp .env.example .env                  # y cambie las claves
# coloque los archivos publicados en raw/ (ver "Datos publicados")
docker compose up -d --build
```

* **`preparar`** aplica las migraciones y carga `raw/` (montado de solo lectura;
  los datos nunca entran en la imagen). En cada reinicio responde `ya_cargado`.
* **`entrenamiento-inicial`** entrena, valida y registra el primer modelo si
  PostgreSQL todavía no tiene un vigente. En cada reinicio termina en segundos
  sin entrenar. Progreso: `docker compose logs -f entrenamiento-inicial`.
* **`api`** (http://127.0.0.1:8000) arranca de inmediato con `/health` en 503
  (sin modelo) y carga el primer modelo sola cuando se confirma. Tiempo hasta
  `/health` `ok`: unos minutos (límite del enunciado: 10).
* `docker compose stop`/`start` o `down`/`up` **sin `-v`** conservan los datos,
  el registro de MLflow y el modelo (`docker compose down -v` los borra).
* Acceso desde otra máquina: `API_DIRECCION_PUBLICACION=0.0.0.0` en `.env`
  (la API no tiene autenticación; por defecto solo se publica en 127.0.0.1).
* La imagen registra el commit del que se construyó y solo puede entrenar si el
  árbol de Git estaba limpio (`procedencia.json`, verificado antes de entrenar).
  Por eso hay que construirla desde un clon, no desde un ZIP.

| Problema | Qué hacer |
|---|---|
| `preparar` termina con "faltan los archivos publicados" | coloque los CSV en `raw/` y vuelva a ejecutar `docker compose up -d` |
| `preparar` termina con "cambió desde que se cargó" | `raw/` tiene otro contenido que el ya cargado; no se mezclan versiones (restaure los archivos originales) |
| `entrenamiento-inicial` sale con código 2 | la imagen se construyó con cambios sin confirmar: confirme o descarte los cambios y `docker compose build` |
| `/health` sigue en 503 | `docker compose logs entrenamiento-inicial` (código 3: ningún modelo superó las validaciones) |

## Arquitectura (etapas 5 y 6)

```
 tiendas ──HTTP──► api (FastAPI) ───────────────────────────► db: PostgreSQL del producto
                    │ GestorVigente: cada 15 s lee el vigente     ├ operación y datos publicados
                    │ CONFIRMADO en PostgreSQL (fuente de verdad) ├ decisiones/resoluciones de promoción
                    │ y descarga el artefacto de MLflow           └ concesión de entrenamiento
                    ├ caché validada (volumen modelo_cache)
                    ▼
                  mlflow ── mlflow_datos (experimentos, registro, alias: reflejo del vigente)
                    ▲    └ mlflow_artefactos (modelo.joblib, copia de datos, backtest)
 entrenamiento-inicial / entrenar_mlflow:
   supervisor (concesión) ─► hijo: copia inmutable → backtest + criterio → registra "pendiente" → T3/T4a/T4b
   supervisor: decisión en PostgreSQL → tags y alias en MLflow → confirmación en PostgreSQL
 preparar: migraciones + carga de raw/          airflow-db (separada; etapa 7)
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

## MLflow (etapas 5 y 6)

```bash
# en el producto, dentro del contenedor (procedencia de la imagen):
docker compose run --rm entrenamiento-inicial python -m app.modelo.entrenar_mlflow --promover siempre
# o desde el equipo (repositorio limpio):
docker compose up -d --wait db airflow-db mlflow     # UI en http://localhost:5000
set -a && . ./.env && set +a
python -m app.modelo.entrenar_mlflow
```

* **`entrenar_mlflow`** exige que el código corresponda a un commit limpio: en el
  equipo, el árbol de Git sin cambios; en un contenedor, `procedencia.json`
  (código y dependencias verificados). Entrena bajo una **concesión** exclusiva
  (nunca dos entrenamientos a la vez) en un proceso hijo vigilado. Guarda una
  copia inmutable de los datos, evalúa el candidato frente al baseline en 7
  cortes de 28 días (≥ 5 % menos costo, gana ≥ 5 de 7, ningún corte > 10 % peor),
  registra `modelo.joblib` en estado `pendiente` y verifica T3 (artefacto
  descargado utilizable), T4a (reproducción del backtest) y T4b (identidad).
  Solo entonces el supervisor lo autoriza y, si corresponde, lo promueve:
  **PostgreSQL es la fuente de verdad del vigente**; el alias de MLflow es su
  reflejo. `--promover si_no_hay` (por defecto: solo si no hay vigente),
  `siempre` (reemplazo manual explícito; nunca omite validaciones) o `nunca`.
  Códigos de salida: 0 válido · 3 no válido · 2 procedencia · 4 resultado del
  alias indeterminado (promoción o reconciliación) · 5 otro entrenamiento en curso
  (`--esperar SEG`) · 6 concesión perdida · 1 error (incluido un alias que la
  reconciliación no pudo aplicar: no se entrena).
* **`python -m app.modelo.reproducir --run-id <id>`** reconstruye un
  entrenamiento solo desde MLflow (aunque la base haya cambiado) e informa
  `identica`, `equivalente` o `no_reproduce`.
* La API sirve el vigente **confirmado en PostgreSQL**; una promoción se activa
  sin reiniciar en ≤ 15 s más la descarga. Con MLflow caído sigue prediciendo con
  el modelo cargado (o, al arrancar, con la entrada de la caché del vigente) y
  `/health` responde `degradado`. Si una promoción confirmada no se puede
  descargar, sigue con el modelo anterior **marcado como tal** (E-ANT).
* Desarrollo sin MLflow: `MODELO_ORIGEN=archivo` y
  `python -m app.modelo.entrenar_local` (artefacto local de la etapa 4).

## API (etapa 4)

| Endpoint | Etapa 4 |
|---|---|
| `POST /v1/prediccion` | limpieza, advertencias, predicción y registro en PostgreSQL; `version_modelo` es la versión que produjo la respuesta y `modelo_es_vigente` (agregado, etapa 6) indica si coincide con el vigente observado |
| `POST /v1/observaciones` | registro idempotente (`registros_guardados` = nuevos + modificados) |
| `GET /modelo` | modelo que usa el servicio: `nombre`, `alias`, `version`; campos agregados (etapa 6): `es_vigente`, `estado_vigencia` (`vigente`, `actualizando`, `anterior`), `version_vigente` (vigente observado) y `motivo` |
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
| Producto con Docker Compose (extremo a extremo) | `scripts/verificar_producto_docker.sh`: imagen (sin datos ni bytecode, procedencia del commit, dependencias del lock), arranque desde cero con datos sintéticos hasta `ok`, contrato, reinicios sin entrenar de nuevo, MLflow caído y de vuelta, E-ANT y procedencia que no coincide |
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
bash scripts/verificar_producto_docker.sh     # requiere el árbol de Git limpio
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
  modelo/         proveedores (archivo, MLflow), caché, gestor (vigencia), entrenamiento,
                  copia de datos, evaluación y criterio, registro (hijo), promoción,
                  concesión (bloqueo), supervisor (ciclo_entrenamiento), arranque,
                  procedencia, dependencias, reproducción, Git
  servicios/      predicción, observaciones, rechazos, interfaz de persistencia
  main.py         FastAPI (endpoints y manejo de errores)
  salud.py        monitor de salud en segundo plano
  orquestador.py  interfaz del reentrenamiento (Airflow en la etapa 7)
  config.py       configuración por variables de entorno
ml/               variables, modelos, costo y cantidad sugerida, backtesting
migraciones/      Alembic (0001 esquema inicial, 0002 concesión y promociones)
scripts/          experimento, verificación de persistencia, pruebas locales
reportes/etapa2/  resultados del backtesting
requirements/     comun.txt, api.txt, dev.txt (versiones fijas) y api.lock.txt (todas, con hashes)
tests/
  unit/           pruebas por módulo y de robustez
  ml/             sin fuga temporal, costo/cuantil, modelos y backtesting
  modelo/         artefacto, caché, copia de datos, criterio, Git, reproducibilidad (sin servidores)
  ejemplos/       un ejemplo sintético por caso del catálogo de errores
  api/            contrato HTTP, salud y modelo (persistencia en memoria, modelo sintético)
  integracion/    PostgreSQL y MLflow reales: migraciones, carga, idempotencia, API, registro, alias
  fixtures/       catálogo y ejemplos sintéticos (sin datos originales)
.github/          workflow de CI y su script de resumen
Dockerfile, .dockerignore, .gitattributes, docker-compose.yml, .env.example, alembic.ini
AGENTS.md         contexto para agentes de programación
DECISIONES.md     decisiones técnicas y de producto
```
