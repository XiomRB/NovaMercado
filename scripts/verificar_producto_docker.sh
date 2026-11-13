#!/usr/bin/env bash
# Verificación de extremo a extremo del producto con Docker Compose (etapa 6).
# Datos sintéticos, proyecto y volúmenes propios (se borran al terminar). Requiere Docker,
# las dependencias de desarrollo (para generar los datos sintéticos) y el árbol de Git limpio.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/verificar_producto_docker.py "$@"
