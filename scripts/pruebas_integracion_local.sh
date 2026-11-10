#!/usr/bin/env bash
# Ejecuta las pruebas de integración contra un PostgreSQL desechable en Docker.
# El contenedor usa autenticación "trust" solo en 127.0.0.1 y se elimina al terminar.
#   Uso: bash scripts/pruebas_integracion_local.sh [argumentos extra de pytest]
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
NOMBRE="novamercado-pg-pruebas"
PUERTO="${PUERTO_PRUEBAS:-55432}"
IMAGEN="postgres:16.14-alpine"

docker rm -f "$NOMBRE" >/dev/null 2>&1 || true
docker run -d --rm --name "$NOMBRE" -e POSTGRES_HOST_AUTH_METHOD=trust \
  -p "127.0.0.1:$PUERTO:5432" "$IMAGEN" >/dev/null
trap 'docker rm -f "$NOMBRE" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 60); do
  docker exec "$NOMBRE" pg_isready -U postgres -q && break
  sleep 1
done

TEST_DATABASE_URL="postgresql+psycopg2://postgres@127.0.0.1:$PUERTO/postgres" \
  "$PYTHON" -m pytest -m integracion "$@"
