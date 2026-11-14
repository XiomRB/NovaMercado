# Imagen de la aplicación NovaMercado (etapa 6, DECISIONES.md §15): la misma imagen sirve
# para la API (servicio "api"), la preparación de datos ("preparar") y el primer
# entrenamiento ("entrenamiento-inicial"), así las bibliotecas que entrenan son las que
# cargan el modelo. Los datos y las credenciales NO entran en la imagen (.dockerignore).

# ---------------------------------------------------------------- procedencia
# Recibe el contexto completo (incluido .git) y registra el commit, si el árbol estaba
# limpio y el blob de Git de cada archivo de las clases A y B. La imagen final solo
# recibe estos datos, nunca .git.
FROM alpine/git:v2.47.2@sha256:062a01ad7a0eb17cff382bc5e26086b4d710e56dfdfdf001109a49b6d9bd378c AS procedencia
COPY . /src
RUN sh /src/scripts/generar_procedencia.sh /src /out app

# --------------------------------------------------------------------- final
# Intérprete, biblioteca estándar y bibliotecas del sistema: solo el digest de la base.
FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencias: todas las del lock (transitivas incluidas) con --require-hashes --no-deps.
# Primero las herramientas de instalación (pip, setuptools, wheel) del lock, que
# reemplazan a las de la imagen base; después el resto. Sin bytecode (--no-compile).
COPY requirements/comun.txt requirements/api.txt requirements/api.lock.txt /app/requirements/
RUN awk 'BEGIN{e=0} /^[^ #]/{e=($0 ~ /^(pip|setuptools|wheel)==/)} e' requirements/api.lock.txt \
        > /tmp/herramientas.txt \
 && python -m pip install --require-hashes --no-deps --no-compile -r /tmp/herramientas.txt \
 && python -m pip install --require-hashes --no-deps --no-compile -r requirements/api.lock.txt \
 && python -m pip check \
 && rm -f /tmp/herramientas.txt

COPY alembic.ini /app/
COPY migraciones /app/migraciones
COPY ml /app/ml
COPY app /app/app
# Etapa 7 (E7-14): arranque verificado y lanzador de los comandos que entrenan
COPY docker/arranque_verificado.py docker/ejecutar /app/docker/
COPY --from=procedencia /out /tmp/procedencia_git

# procedencia.json y verificación final: dependencias siempre; código si el árbol
# estaba limpio. Se borra todo el bytecode (también el de la biblioteca estándar) y se
# comprueba que no queda ninguno: Python solo ejecuta las fuentes verificadas.
RUN chmod 0755 /app/docker/ejecutar \
 && python -m app.modelo.procedencia --generar /tmp/procedencia_git --salida /app/procedencia.json \
 && rm -rf /tmp/procedencia_git \
 && find /usr/local/lib/python3.11 /app -depth \( -name __pycache__ -o -name '*.pyc' -o -name '*.pyo' \) \
        -exec rm -rf {} + \
 && python -m app.modelo.procedencia --verificar-construccion \
 && test -z "$(find /usr/local/lib/python3.11 /app \( -name '*.pyc' -o -name '*.pyo' \) -print -quit)"

# Usuario sin privilegios: el código y las bibliotecas son de root y de solo lectura para
# él; solo puede escribir en la caché de modelos (volumen) y en /tmp.
RUN useradd --uid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
 && mkdir -p /app/modelos/cache && chown -R app:app /app/modelos
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
