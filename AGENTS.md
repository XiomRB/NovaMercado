# AGENTS.md — contexto para agentes de programación

## Proyecto

Servicio de IA para NovaMercado (10 tiendas, 12 familias): pronóstico diario de
ventas por tienda y familia, cantidad sugerida que minimiza el costo de
faltantes/sobrantes de `costos.csv`, defensa del contrato ante datos
defectuosos, registro persistente y reentrenamiento automático con Airflow +
MLflow. Todo se levanta con `docker compose up`.

## Reglas permanentes

* **El contrato del Anexo manda.** Endpoints: `POST /v1/prediccion`,
  `POST /v1/observaciones`, `POST /admin/reentrenar`,
  `GET /admin/reentrenar/{id}`, `GET /modelo`, `GET /health`. Puerto 8000.
  Rechazo = 422 con `detail`. Se pueden agregar campos, nunca omitir ni renombrar.
* **No subir a git** `instrucciones_lab2.md`, `Anexo_ContratoDeServicio.md` ni
  `raw/` (datos). Ya están en `.gitignore`.
* **Nunca usar el reloj del sistema** para validar fechas (fechas simuladas).
* **Sin fuga temporal:** al predecir la fecha F solo se usan datos con fecha < F.
* **No inventar datos:** una venta real nunca se supone; los supuestos en
  predicción se informan con `advertir`.
* Lógica de limpieza en `app/calidad/` (sin FastAPI), con pruebas. Pydantic
  estricto en `app/schemas/contrato.py` valida *después* de la limpieza.
* Toda política nueva debe documentarse en `DECISIONES.md` §3 y probarse.
* Trabajo por etapas: no declarar nada terminado sin ejecutar las pruebas.

## Entorno local

```bash
conda activate lab2            # Python 3.11
pip install -r requirements/dev.txt
ruff check .
python -m pytest -q -m "not integracion"
bash scripts/pruebas_integracion_local.sh      # requiere Docker
```

## Flujo de trabajo (obligatorio)

* Nunca commits en `main`; una rama por etapa desde el último `main` fusionado.
* Al terminar: publicar la rama, abrir PR hacia `main` y detenerse. El usuario
  revisa y fusiona; nunca aprobar, fusionar ni cerrar PR.
* El check obligatorio es **CI resultado** (`.github/workflows/ci.yml`).

Versiones fijadas alineadas con las constraints de Airflow 2.10.5 / Python 3.11
(ver `DECISIONES.md` §1). No cambiar numpy/pandas/pydantic/SQLAlchemy sin revisar
esas constraints.

## Datos (`raw/`)

* `historico_ventas.csv`: `fecha,tienda_id,familia_id,ventas,en_promocion`,
  2024-03-01 a 2026-02-28, cuadrícula completa (sin nulos ni duplicados).
* `tiendas.csv`: la tienda 10 abrió el 2025-03-15.
* `feriados.csv`: `fecha,nombre,tipo{feriado,comercial,pago,medio_dia},ambito`
  (`nacional` o ciudad de `tiendas.ciudad`), cubre hasta 2026-12-31.
* `costos.csv`: cuantil óptimo por familia τ = cf / (cf + cs).

## Estado de las etapas

| Etapa | Estado |
|---|---|
| 0 Repositorio y dependencias | hecho |
| 1 Calidad de datos + esquemas Pydantic | hecho (pruebas en `tests/unit`, `tests/ejemplos`) |
| 2 ML offline (baseline, modelo, backtest) | hecho (PR #1 y #2) |
| 3 Persistencia PostgreSQL + CI | hecho (PR #3) |
| 4 API FastAPI | hecho (PR #4) |
| 5 MLflow (registro, alias, recarga, health) | en revisión (`feat/etapa-5-mlflow`) |
| 6 Docker Compose + bootstrap | pendiente |
| 7 Airflow DAG + orquestador real de /admin (los endpoints ya existen) | pendiente |
| 8 Simulador de operación | pendiente |
| 9 Documentación final y tag `entrega-lab2` | pendiente |

## Mapa del código

```
app/calidad/   resultado.py (acciones, estado, rechazo), texto.py, numeros.py,
               fechas.py, campos.py (E07), catalogo.py (E02), comunes.py,
               prediccion.py, observaciones.py
app/schemas/   contrato.py (Pydantic del Anexo)
app/db/        esquema.py, conexion.py, carga.py, repositorio.py
app/modelo/    proveedores.py, proveedor_mlflow.py, cache.py, artefacto.py, gestor.py,
               entrenamiento.py, entrenar_local.py, copia_datos.py, evaluacion.py,
               registro.py, entrenar_mlflow.py, reproducir.py, git_estado.py
app/servicios/ persistencia.py, prediccion.py, observaciones.py, rechazos.py
app/main.py    FastAPI; app/salud.py monitor; app/orquestador.py; app/dependencias.py
migraciones/   Alembic
ml/           datos.py (carga raw/), calendario.py (feriados), features.py (variables
              sin fuga), modelos.py (baseline + HGB), costos.py (costo, τ, cantidad
              óptima, calibrador), pronosticador.py (entrenamiento compartido
              producción/backtest), backtest.py, metricas.py
scripts/      experimento_etapa2.py (backtesting -> reportes/etapa2/)
tests/unit/    pruebas por módulo + robustez (fuzzing determinista)
tests/ml/      sin fuga, costo/cuantil, baseline, backtest (datos sintéticos)
tests/api/     contrato HTTP con persistencia en memoria (tests/api/memoria.py) y modelo sintético
tests/modelo/  etapa 5 sin servidores (artefacto, caché, copia, criterio, Git, reproducibilidad)
tests/integracion/ PostgreSQL real (TEST_DATABASE_URL), base nueva por prueba
tests/ejemplos/ un ejemplo sintético por caso E01–E10 (tests/fixtures/ejemplos)
tests/fixtures/ catálogo y ejemplos sintéticos: las pruebas nunca leen raw/
```

## Reglas de persistencia

* Esquema en `app/db/esquema.py`; todo cambio requiere una migración nueva en
  `migraciones/versions/` (nunca editar una ya fusionada). Una prueba compara el
  esquema migrado con el modelo.
* Tablas de solo inserción (triggers): solicitudes, familias de la solicitud,
  predicciones, envíos, historial, calidad y ventas históricas.
  `observaciones` admite `UPDATE` (corrección) con fila en el historial.
  Una solicitud rechazada no admite familias/predicciones y un envío rechazado
  no crea ni modifica observaciones (triggers de coherencia).
* Escribir solo mediante `app/db/repositorio.py` dentro de
  `app.db.conexion.transaccion` (una unidad de trabajo por solicitud HTTP).
* Idempotencia de observaciones: clave (fecha, tienda, familia) + advisory lock
  por (tienda, fecha). `registros_guardados` = nuevos + modificados.
* Las pruebas de integración crean bases desechables a partir de
  `TEST_DATABASE_URL`; jamás apuntarla a la base del producto.

## Reglas de la API

* Endpoints delgados en `app/main.py`; la lógica va en `app/servicios/` y usa la
  interfaz `Persistencia` (nunca SQL en los servicios).
* El cuerpo se lee crudo: no declarar modelos Pydantic en la firma de los
  endpoints (FastAPI rechazaría antes de que la capa de calidad repare).
* Toda respuesta se valida con `app/schemas/contrato.py` antes de devolverse.
* La API nunca entrena al iniciar; el modelo llega por un `ProveedorModelo`
  (archivo local en la etapa 4, MLflow en la 5) a través de `GestorModelo`.
* `/health` solo lee el diagnóstico del `MonitorSalud`; nuevas dependencias se
  agregan como verificaciones en `app/dependencias.py:construir_monitor`.
* Para predecir F solo se leen ventas con fecha < F (`leer_ventas_previas`) y se
  vuelve a filtrar en el servicio.
* Toda conexión de la API usa `crear_motor` (límites de conexión, `statement_timeout`
  y vigilante del cliente). Los procesos por lotes pasan `timeout_consulta` mayor;
  no crear motores con `create_engine` directo fuera de pruebas y Alembic.

## Reglas de MLflow (etapa 5)

* `modelo.joblib` es la **única** referencia del modelo: su SHA-256 (tag
  `sha256_artefacto`) se verifica antes de deserializar al descargar, al leer la
  caché y en T4b. No validar carpetas ni otros archivos.
* El alias `vigente` solo se mueve en `app/modelo/registro.py`, como **última
  operación** (después de tags, métricas, `criterio.json` y el cierre de la
  ejecución; nada se escribe después) y con un candidato válido (T1–T4b, D1,
  C1–C3). Ningún modo omite validaciones.
* No definir `COMPONENTES_REQUERIDOS` en `.env.example`: se deriva de `MODELO_ORIGEN`.
* El entrenamiento y la evaluación leen la **copia releída** (datos y
  configuración del manifiesto) y fijan los hilos (`hilos_fijos`, 1 por
  defecto): sin eso los bytes de `modelo.joblib` no se reproducen.
* Descargar artefactos solo con `app/modelo/mlflow_util.descargar_de_ejecucion`
  (MLflow 2.22 ignora `tracking_uri` con `artifact_uri="runs:/…"`).
* `entrenar_mlflow` exige Git limpio (`app/modelo/git_estado.py`); no agregar
  opciones para omitirlo. Las pruebas llaman a `entrenar_y_registrar` con un commit
  de prueba.
* Toda llamada a MLflow pasa por `configurar_cliente_mlflow()` (5 s, 0 reintentos).
* El proveedor solo acepta versiones con `estado=valido`.

## Reglas de ML

* Toda variable de la fila con fecha F se deriva de `wide.shift(k)`, k >= 1, en
  `ml/features.py`. No agregar variables que lean ventas del día F ni promociones
  de días pasados (las observaciones no traen promociones).
* Producción y backtest usan las mismas funciones (`ml/pronosticador.py`); una
  prueba verifica que den resultados idénticos. No duplicar lógica.
* La cantidad sugerida la decide `CalibradorCuantil.sugerir` (óptimo entero del
  costo empírico con τ de la familia). No reemplazarla por un redondeo.
* Reentrenar = `Pronosticador.entrenar(ventas, promociones, ...)`: las ventas
  alimentan los rezagos; solo las filas con promoción **real** son ejemplos
  (`ml/datos.py: consolidar_promociones, filas_supervisadas`).
* Promociones: histórico > última solicitud real (`registrado_en`, desempate
  `solicitud_id`); las supuestas (`RegistroCalidad.promociones_supuestas`) nunca
  se entrenan; un conflicto histórico/solicitud se marca (`discrepancia`), no se
  descarta en silencio. La etapa 3 debe persistir `COLUMNAS_SOLICITUDES`
  (DECISIONES.md §10).
* `Pronosticador.maximos_promocion` (máximo por familia en entrenamiento) se pasa
  a `limpiar_solicitud_prediccion(..., maximos_promocion=...)` para advertir
  promociones atípicas sin modificarlas.
* Ninguna prueba debe leer `raw/`; usar `tests/fixtures/` o `tests/ml/sinteticos.py`.
* Con la CPU ocupada el backtesting completo tarda ~10–15 min; las pruebas de
  `tests/ml` usan datos sintéticos y `max_iter` reducido.
