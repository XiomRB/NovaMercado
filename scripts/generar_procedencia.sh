#!/bin/sh
# Procedencia del código (etapa 6, DECISIONES.md §15). Se ejecuta en la etapa
# "procedencia" del Dockerfile, con el contexto de construcción COMPLETO (incluido
# .git) en <repo>. Escribe en <salida> los datos crudos que la etapa final convierte
# en /app/procedencia.json (python -m app.modelo.procedencia --generar).
#
# Uso: generar_procedencia.sh <repo> <salida>
set -eu
REPO=$1
SALIDA=$2
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
git ls-files -s -- app ml migraciones alembic.ini \
    requirements/comun.txt requirements/api.txt requirements/api.lock.txt > "$SALIDA/clase_a"
# Clase B: lo que define la construcción y el arranque (se registra; garantía = árbol limpio)
git ls-files -s -- Dockerfile .dockerignore .gitattributes scripts/generar_procedencia.sh \
    docker-compose.yml > "$SALIDA/clase_b"
grep -m1 -E '^FROM python:' Dockerfile | awk '{print $2}' > "$SALIDA/imagen_base"
echo "con_git" > "$SALIDA/estado"
