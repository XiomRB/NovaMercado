#!/bin/sh
# Procedencia del código (etapa 6, DECISIONES.md §15). Se ejecuta en la etapa
# "procedencia" del Dockerfile, con el contexto de construcción COMPLETO (incluido
# .git) en <repo>. Escribe en <salida> los datos crudos que la etapa final convierte
# en /app/procedencia.json (python -m app.modelo.procedencia --generar).
#
# Uso: generar_procedencia.sh <repo> <salida> [app|airflow]
#
# Etapa 7 (E7-14): con un perfil, la clase A incluye además el arranque verificado y el
# lanzador (docker/), y en la imagen de Airflow los DAG (dags/); se registran los
# directorios de clase A de esa imagen (archivos de más se rechazan). Sin perfil, lo
# mismo que en la etapa 6.
set -eu
REPO=$1
SALIDA=$2
PERFIL=${3:-}
mkdir -p "$SALIDA"
git config --global --add safe.directory "$REPO"
# El modo de los archivos no es contenido (un contexto desde Windows pierde el bit +x).
git config --global core.fileMode false
cd "$REPO"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "sin_git" > "$SALIDA/estado"
    exit 0
fi
git rev-parse HEAD > "$SALIDA/commit"
git status --porcelain --untracked-files=normal > "$SALIDA/cambios"
# Clase A: lo que se ejecuta o se instala en la imagen (se verifica antes de entrenar)
case "$PERFIL" in
    "") EXTRA_A=""; DIRS=""; DOCKERFILE=Dockerfile; BASE='^FROM python:' ;;
    app) EXTRA_A="docker/arranque_verificado.py docker/ejecutar"; DIRS="app ml migraciones requirements docker"
         DOCKERFILE=Dockerfile; BASE='^FROM python:' ;;
    airflow) EXTRA_A="docker/arranque_verificado.py docker/ejecutar dags"
             DIRS="app ml migraciones requirements docker dags"; DOCKERFILE=Dockerfile.airflow
             BASE='^FROM apache/airflow:' ;;
    *) echo "perfil desconocido: $PERFIL" >&2; exit 2 ;;
esac
# shellcheck disable=SC2086
git ls-files -s -- app ml migraciones alembic.ini \
    requirements/comun.txt requirements/api.txt requirements/api.lock.txt $EXTRA_A > "$SALIDA/clase_a"
# Clase B: lo que define la construcción y el arranque (se registra; garantía = árbol limpio)
git ls-files -s -- "$DOCKERFILE" .dockerignore .gitattributes scripts/generar_procedencia.sh \
    docker-compose.yml > "$SALIDA/clase_b"
grep -m1 -E "$BASE" "$DOCKERFILE" | awk '{print $2}' > "$SALIDA/imagen_base"
if [ -n "$DIRS" ]; then
    echo "$DIRS" > "$SALIDA/directorios_a"
fi
echo "con_git" > "$SALIDA/estado"
