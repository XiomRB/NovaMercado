#!/usr/bin/env bash
# Verifica con Docker Compose que los datos del producto sobreviven a:
#   1) docker compose restart db
#   2) docker compose down + up (sin -v)
# y que la base de Airflow es un servicio y un volumen separados.
#
# Usa un proyecto de Compose propio ("novamercado-verificacion") con claves
# aleatorias generadas en el momento: nunca toca los volúmenes reales.
#   Uso: bash scripts/verificar_persistencia_docker.sh   (PYTHON=... opcional)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
PROYECTO="novamercado-verificacion"
PUERTO="${PUERTO_VERIFICACION:-55434}"
TMP="$(mktemp -d)"
ENV_FILE="$TMP/verificacion.env"

clave() { "$PYTHON" -c "import secrets; print(secrets.token_hex(16))"; }
CLAVE_PRODUCTO="$(clave)"
CLAVE_AIRFLOW="$(clave)"
if [ -n "${GITHUB_ACTIONS:-}" ]; then
  echo "::add-mask::$CLAVE_PRODUCTO"; echo "::add-mask::$CLAVE_AIRFLOW"
fi
cat > "$ENV_FILE" <<ENV
POSTGRES_DB=novamercado
POSTGRES_USER=novamercado
POSTGRES_PASSWORD=$CLAVE_PRODUCTO
PRODUCTO_DB_PUERTO_HOST=$PUERTO
AIRFLOW_DB_NAME=airflow
AIRFLOW_DB_USER=airflow
AIRFLOW_DB_PASSWORD=$CLAVE_AIRFLOW
ENV

compose() { docker compose -p "$PROYECTO" --env-file "$ENV_FILE" "$@"; }
limpiar() { compose down -v --remove-orphans >/dev/null 2>&1 || true; rm -rf "$TMP"; }
trap limpiar EXIT

export DATABASE_URL="postgresql+psycopg2://novamercado:$CLAVE_PRODUCTO@127.0.0.1:$PUERTO/novamercado"

paso() { echo; echo "== $*"; }

paso "Levantar db y airflow-db"
compose up -d --wait db airflow-db

paso "Migrar y escribir datos sintéticos"
"$PYTHON" -m scripts.verificar_persistencia escribir --estado "$TMP/estado.json"

paso "Reinicio del contenedor (docker compose restart db)"
compose restart db
compose up -d --wait db
"$PYTHON" -m scripts.verificar_persistencia verificar --estado "$TMP/estado.json"

paso "Detener y volver a levantar sin borrar volúmenes (down + up)"
compose down
compose up -d --wait db airflow-db
"$PYTHON" -m scripts.verificar_persistencia verificar --estado "$TMP/estado.json"

paso "Separación de la base de Airflow"
TABLAS_AIRFLOW="$(compose exec -T airflow-db psql -U airflow -d airflow -tAc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")"
if [ "$TABLAS_AIRFLOW" != "0" ]; then
  echo "FALLO: la base de Airflow contiene $TABLAS_AIRFLOW tablas del producto"; exit 1
fi
VOL_DB="$(docker inspect -f '{{range .Mounts}}{{.Name}}{{end}}' "$(compose ps -q db)")"
VOL_AF="$(docker inspect -f '{{range .Mounts}}{{.Name}}{{end}}' "$(compose ps -q airflow-db)")"
echo "volumen producto: $VOL_DB | volumen airflow: $VOL_AF"
if [ "$VOL_DB" = "$VOL_AF" ] || [ -z "$VOL_DB" ] || [ -z "$VOL_AF" ]; then
  echo "FALLO: las bases no usan volúmenes separados"; exit 1
fi

echo; echo "OK: persistencia tras reinicio y down/up verificada; bases y volúmenes separados"
