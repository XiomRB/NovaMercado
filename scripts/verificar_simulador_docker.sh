#!/usr/bin/env bash
# Simulación corta (sintética) e imágenes aisladas (I5–I7). Ver scripts/verificar_simulador_docker.py.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/verificar_simulador_docker.py "$@"
