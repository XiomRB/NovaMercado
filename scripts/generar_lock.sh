#!/bin/sh
# Genera requirements/api.lock.txt (etapa 6, E6-8) DENTRO de la imagen base fijada del
# Dockerfile (Linux, Python 3.11.18): las dependencias transitivas y sus marcadores
# corresponden a la plataforma de la imagen. Todas las versiones exactas y con hashes.
#
# Uso (desde la raíz del repositorio, con Docker):  sh scripts/generar_lock.sh
set -eu
IMAGEN=$(grep -m1 -E '^FROM python:' Dockerfile | awk '{print $2}')
docker run --rm -v "$(pwd)":/src -w /src -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -e DUENO="$(id -u):$(id -g)" "$IMAGEN" sh -c '
  pip install --quiet pip-tools==7.4.1 &&
  pip-compile --quiet --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url \
    --no-emit-trusted-host --resolver=backtracking --output-file=requirements/api.lock.txt requirements/api.txt &&
  chown "$DUENO" requirements/api.lock.txt'
echo "requirements/api.lock.txt generado con $IMAGEN"
