# Decisiones técnicas y de producto

Cada decisión indica la alternativa considerada y su justificación. El contrato
del Anexo tiene prioridad sobre cualquier decisión de implementación.

## 1. Versiones y entorno (D4)

| Componente | Versión fijada |
|---|---|
| Python | 3.11 |
| Apache Airflow | 2.10.5 (pendiente, etapa 7) |
| MLflow | 2.22.1 (cliente) e imagen `ghcr.io/mlflow/mlflow:v2.22.1` (servidor) |
| PostgreSQL | imagen `postgres:16.14-alpine` |
| Alembic | 1.14.1 (igual que las constraints de Airflow 2.10.5) |
| joblib | 1.6.0 (formato del artefacto del modelo) |
| ruff, pytest-timeout (solo desarrollo y CI) | 0.17.2, 2.3.1 |
| numpy / pandas / scipy / scikit-learn | 1.26.4 / 2.1.4 / 1.15.2 / 1.6.1 |
| pydantic / SQLAlchemy / psycopg2-binary | 2.10.6 / 1.4.54 / 2.9.10 |
| FastAPI / uvicorn | 0.115.12 / 0.34.0 |

**Decisión:** las versiones de numpy, pandas, pydantic, SQLAlchemy, psycopg2 y
requests coinciden con las *constraints* oficiales de Airflow 2.10.5 para Python
3.11 (`constraints-2.10.5/constraints-3.11.txt`).

**Por qué:** el modelo se entrena dentro de Airflow y se carga en la API. Si las
bibliotecas difieren, la deserialización puede fallar o cambiar resultados. Con
una sola base de versiones el código compartido (`ml/`) corre igual en ambos.

**Alternativas:** (a) SQLAlchemy 2.x en la API y 1.4 en Airflow: obliga a que el
código compartido funcione con ambas; (b) `ExternalPythonOperator` con un
entorno virtual aparte dentro de la imagen de Airflow: más aislamiento, pero más
complejidad de imagen. Se reevaluará en la etapa 7 si aparece un conflicto.

**Verificación:** `pip install -r requirements/dev.txt` + `pip check` sin
conflictos en Python 3.11.18.

## 2. Calidad de datos: arquitectura

* La limpieza vive en `app/calidad/` (Python puro, sin FastAPI) y se ejecuta
  **antes** de Pydantic. Pydantic (`app/schemas/contrato.py`) valida en modo
  estricto el resultado ya normalizado y también cada respuesta.
* **Por qué:** un modelo Pydantic estricto rechazaría `"T05"` o `"17/11/2025"`
  antes de poder repararlos; uno permisivo haría conversiones implícitas sin
  informarlas en `calidad`. Separar ambas capas permite reparar *e informar*.
* El estado se deriva de las acciones: hay `advertir` → `advertencia`; solo
  `reparar` → `corregido`; ninguna → `ok`. El modelo `Calidad` valida esta
  coherencia, de modo que una respuesta incoherente no puede emitirse.
* El rechazo es una excepción (`SolicitudRechazada`) con `campo` y `motivo`; la
  API lo traducirá a `422 {"detail": "<campo>: <motivo>"}`.

## 3. Política ante cada tipo de problema

### Principios generales (aplican a problemas fuera del catálogo)

1. **Reparar** solo si existe una única interpretación posible del dato.
2. **Advertir** si se responde usando un supuesto (similitud, valor por
   defecto, orden de fecha ambiguo, redondeo, dato ignorado).
3. **Rechazar** si falta o es irreconocible lo que identifica la solicitud
   (tienda, fecha, lista de ítems), o si hay valores contradictorios para el
   mismo dato (no hay forma de saber cuál es el correcto).
4. **Una venta real nunca se supone**: un valor de venta no confiable se
   descarta (no se guarda) y se advierte. Una promoción sí admite supuesto (0),
   porque solo afecta la predicción de ese día y se informa.
5. El reloj del sistema nunca se usa para validar fechas.

### Tabla de políticas

| Código | Problema | Acción |
|---|---|---|
| E01 | `tienda_id` como texto `"5"`, con prefijo `"T05"`, `"tienda 5"`, o `5.0` | reparar |
| — | tienda con varios números (`"T5-2"`), negativa, decimal, booleana, ausente o inexistente | rechazar |
| E02 | familia con otras mayúsculas, tildes, espacios, separadores, o igual al nombre del catálogo (`"Frutas y Verduras"`) | reparar |
| — | familia parecida a una sola familia (similitud ≥ 0.8 y margen ≥ 0.08 sobre la segunda) p. ej. `"lacteo"` | advertir |
| — | familia irreconocible | advertir: se ignora el ítem (en predicción se completa como faltante) |
| E03 | `en_promocion` como texto `"3"`, decimal entero `2.0`, con miles `"1,000"` | reparar |
| — | `en_promocion` decimal no entero `2.5` | advertir: se redondea (3) |
| — | `en_promocion` negativo, nulo, ausente, no numérico o booleano | advertir: se asume 0 |
| E04 | fecha ISO con hora/zona, `AAAA/MM/DD`, `AAAAMMDD`, sin ceros, mes con nombre | reparar |
| E04 | `DD/MM/AAAA` con día > 12 o día = mes; `MM/DD/AAAA` con segundo número > 12 | reparar (única lectura) |
| — | `NN/NN/AAAA` con ambos ≤ 12 y distintos | advertir: se asume DD/MM (uso local) |
| — | año de dos dígitos | advertir: se asume 20AA |
| — | fecha inexistente, no interpretable o anterior a la apertura de la tienda | rechazar |
| E05 | faltan familias en la predicción (D1) | advertir: se pronostican las 12, con `en_promocion = 0` en las faltantes |
| E06 | familia repetida con valores idénticos (tras normalizar) | reparar: se elimina la repetida |
| E06 | familia repetida con valores distintos (D2) | rechazar la solicitud completa |
| E07 | nombre de campo distinto: mayúsculas/camelCase/tildes o sinónimo **inequívoco** (`tienda` → `tienda_id`, `store_id`, `promocion`) | reparar |
| — | nombre genérico que podría referirse a otro dato (`id`, `dia`, `cantidad`, `unidades`, `categoria`, `items`, `detalle`) | advertir: se ignora; si falta un obligatorio, rechazar |
| — | nombre de campo parecido a uno del contrato (errata, similitud ≥ 0.85) | advertir |
| — | campo desconocido | advertir: se ignora |
| — | mismo campo recibido dos veces con valores distintos | rechazar |
| — | cuerpo que no es objeto JSON; lista de ítems ausente, vacía o sin ítems reconocibles | rechazar |
| — | lista de ítems enviada como objeto `{"abarrotes": 3}` | reparar |
| — | ítem de la lista que no es objeto | advertir: se ignora |
| E08 | ventas negativas | advertir: ese registro no se guarda |
| — | ventas nulas, ausentes o no numéricas | advertir: ese registro no se guarda |
| — | ninguna venta válida en la observación | rechazar |
| — | faltan familias en la observación | advertir: no se registra nada para ellas |
| E09 | envío repetido de la misma observación | idempotente en la persistencia (etapa 3) |
| E10 | `"1,234"`, `"812,0"`, `812.0`, `"994"` | reparar |
| — | ventas decimales con fracción `"812,5"` | advertir: se redondea |

### Separadores numéricos

Se aplica la convención guatemalteca, que coincide con JSON: **coma = miles,
punto = decimal**. Solo se reinterpreta un separador cuando la convención local
es imposible: `"812,0"` no puede tener coma de miles (le sigue un solo dígito),
luego es coma decimal; `"1.234.567"` solo puede tener puntos de miles. Por eso
`"1.234"` se lee como 1.234 (y, al no ser entero, se advierte y redondea).

*Alternativa descartada:* tratar `"1.234"` como 1234 por simetría con la coma.
Se descartó porque contradice la convención local y JSON.

### Redondeo

Aritmético (0.5 hacia arriba), no el redondeo bancario de Python, para que
2.5 → 3 sea predecible para las tiendas.

### Decisiones del usuario incorporadas

* **D1:** se completan las 12 familias con `en_promocion = 0` y se advierte.
* **D2:** duplicados idénticos se eliminan (reparar); con valores distintos se
  rechaza la solicitud con 422, sin elegir automáticamente una aparición. La
  misma regla se aplica a las observaciones por consistencia.
* **Promociones atípicas:** si `en_promocion` supera el máximo de su familia en
  los datos de entrenamiento del modelo vigente → **advertir y conservar el
  valor** (nunca se recorta ni se convierte en 0). Ver §9.
* **D3:** se rechazan fechas anteriores a la apertura de la tienda; se aceptan
  fechas dentro y después del histórico. La advertencia por observaciones
  desactualizadas se implementó en la etapa 4 (§13, D-D).
* **D6:** los archivos publicados se colocan en `raw/` (excluida de git).

## 4. Pruebas sin datos del repositorio

**Ninguna prueba lee `raw/`**: toda la suite se ejecuta desde un clon limpio,
sin pruebas omitidas.

* `tests/fixtures/catalogo/`: catálogo sintético (3 tiendas inventadas, las 12
  familias del contrato), cargado con el mismo `Catalogo.desde_csv` del servicio.
* `tests/fixtures/ejemplos/`: un ejemplo por caso del catálogo de errores
  (E01–E10, más una solicitud y una observación válidas). Reproducen la
  estructura y el problema de cada caso publicado con **valores inventados**
  (fecha 2026-04-14, ventas y promociones distintas); no son copias de los
  archivos originales.
* `tests/ml/sinteticos.py`: generador de histórico, tiendas, feriados y costos
  para el pipeline de ML; `tests/ml/test_datos.py` escribe esa carpeta en un
  directorio temporal para probar `cargar_raw`.

*Alternativa descartada:* mantener pruebas sobre `raw/ejemplos` que se omiten
sin los datos; dejaban sin cubrir esos casos en cualquier clon limpio.
`tests/unit/test_robustez.py` aplica 800 mutaciones aleatorias deterministas y
verifica que el resultado sea siempre una solicitud válida o un rechazo
controlado, nunca otra excepción.

## 5. Información que usa el modelo al predecir

Para la fecha F de una tienda y familia (`ml/features.py`):

| Grupo | Variables | Disponible en producción porque… |
|---|---|---|
| Ventas reales previas | `lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28`, `media_dow_4` (mismo día de la semana, 4 semanas), `media_7`, `media_28`, `std_28`, `ultimo_valor` | histórico + observaciones registradas con fecha < F |
| Disponibilidad | `n_obs_28`, `n_dow_4`, `dias_desde_ultimo` | idem; permiten operar con observaciones faltantes |
| Promoción | `en_promocion` **del día F** | llega en la solicitud |
| Calendario | día de la semana, día del mes, mes, días para fin de mes | fecha |
| Eventos | tipo de evento (feriado, comercial, pago, medio día), código del evento, víspera de feriado, día posterior a feriado; nacional o de la ciudad de la tienda | `feriados.csv` cubre el periodo de evaluación |
| Atributos | tienda, tipo de tienda, familia (categóricas) | catálogos |

**Garantía anti-fuga:** toda variable de ventas sale de la matriz diaria
desplazada `shift(k)` con k ≥ 1. Se prueba de cuatro formas
(`tests/ml/test_features.py`): alterar el futuro no cambia las variables
(con objetivos de una y de varias fechas), calcular con todo el histórico da lo
mismo que truncar en F, y los valores coinciden con un cálculo directo. Se
verificó por mutación: con `shift(0)` fallan 8 pruebas.

**Promociones de días pasados no se usan:** las observaciones de ventas reales
no traen `en_promocion`, por lo que en operación no estarían disponibles de
forma confiable. *Alternativa descartada:* rezagos de promoción tomados de las
solicitudes de predicción; se puede reevaluar cuando existan registros.

## 6. Modelos y validación temporal

* **Baseline** (`baseline_dow4`): promedio de la venta del mismo día de la semana
  en las últimas 4 semanas. Respaldo con pocos datos: media de 28 días → último
  valor → media de la familia en tiendas del mismo tipo.
* **`hgb_directo`:** `HistGradientBoostingRegressor` global, pérdida Poisson.
* **`hgb_relativo`:** el mismo algoritmo, pero predice venta / media de 28 días y
  expresa las variables de ventas relativas a ese nivel. Es invariante a la
  escala de cada serie y se ajusta automáticamente a cambios de nivel.
* Hiperparámetros fijos (`ml/modelos.py`, `PARAMS_HGB`), sin búsqueda: ajustarlos
  sobre las mismas ventanas de prueba sesgaría la comparación. `early_stopping`
  desactivado (evita particiones aleatorias internas).
* **Backtesting con origen móvil:** 7 cortes, ventanas de prueba consecutivas de
  28 días (2025-08-17 a 2026-02-28, 196 días, 23 520 filas por combinación).
  En cada corte C se entrena con fechas < C y se predice [C, C+28) sin
  reentrenar; cada día usa las ventas reales de días anteriores, como en la
  operación. Todos los modelos se evalúan sobre las mismas filas (probado).
* Producción y backtesting usan las mismas funciones
  (`ml/pronosticador.py`); una prueba verifica que den resultados idénticos.

### Resultados (`reportes/etapa2/resultados.md`)

| Modelo + estrategia | Costo total | WAPE | MAE |
|---|---|---|---|
| **hgb_relativo + mult_familia** | **628 813** | 0.179 | 38.3 |
| hgb_directo + mult_familia | 631 415 | 0.180 | 38.6 |
| baseline_dow4 + mult_familia | 879 479 | 0.247 | 53.0 |
| hgb_relativo + redondeo | 821 651 | 0.179 | 38.3 |
| baseline_dow4 + redondeo | 1 124 054 | 0.247 | 53.0 |

* El ML reduce el costo un **28.5 %** frente al baseline con la misma estrategia
  de cantidad, y gana en **7 de 7** cortes (17–51 %; 18.3 % excluyendo los dos
  cortes de fin de año, donde el baseline no anticipa los feriados).
* Mejora en las 10 tiendas (24–33 %), en las 12 familias (20–53 %) y en las
  120 series tienda×familia (mínimo 8.9 %, en series pequeñas de tiendas express).
* `hgb_relativo` vs `hgb_directo`: diferencia de 0.4 % (gana en 5 de 7 cortes).
  Se elige `hgb_relativo` porque, además, su diseño se adapta a cambios de nivel
  de la demanda, que el enunciado anticipa para el periodo de evaluación.

## 7. Cantidad sugerida

* Minimizar `cf·(real − s)⁺ + cs·(s − real)⁺` equivale a minimizar la pérdida
  cuantílica con **τ = cf / (cf + cs)**: el óptimo es el cuantil τ de la
  demanda. τ va de 0.286 (panadería: sobrar cuesta 2.5) a 0.952 (licores:
  faltar cuesta 20× más que sobrar). Demostración en `ml/costos.py`.
* La demanda de cada fila se representa con muestras `pred · r_i`, donde r_i =
  real/pred son residuos **fuera de muestra** de la familia en los 56 días previos
  al corte (el modelo de calibración se entrenó antes de esa ventana).
* Como el costo es convexo, la cantidad **entera** óptima es floor(q) o ceil(q);
  se evalúan ambas y se elige la de menor costo empírico. (Una primera versión
  usaba siempre ceil(q); la prueba exhaustiva mostró que no es óptima con
  muestras continuas y se corrigió.)
* Pruebas: mínimo frente a búsqueda exhaustiva (300 casos y por fila del
  calibrador), costo esperado exacto con Poisson, τ propio mejor que otros τ
  fuera de muestra, enteros ≥ 0.
* **Evidencia en el backtesting:** la cobertura real P(real ≤ sugerido)
  coincide con τ en todas las familias (p. ej. licores 0.951 vs τ 0.952,
  panadería 0.290 vs 0.286). Frente a sugerir la venta esperada redondeada, el
  costo baja 23.5 % (821 651 → 628 813).
* Estrategias comparadas sobre el mismo modelo: multiplicativa o tipo Poisson
  (`raiz`), por familia o por tipo de tienda × familia, y cuantiles directos con
  HGB cuantílico. Las cinco quedan dentro del 1.4 % entre sí; se elige
  **`mult_familia`**: la más simple, la de menor costo total y primera en 4 de 7 cortes.

## 8. Tiempos

Entrenamiento de producción completo (modelo de calibración + modelo final +
calibrador) con todo el histórico: **14 s**. Predicción de una tienda: 0.12 s.
Holgado para el límite de 10 minutos por reentrenamiento.

## 9. Promociones atípicas

* Umbral: máximo de `en_promocion` por familia **en los ejemplos de
  entrenamiento del modelo** (fechas < corte, solo promociones conocidas).
  Se guarda en `Pronosticador.maximos_promocion`; con el histórico actual va
  de 7 (panadería) a 19 (abarrotes).
* Justificación: la distribución es discreta y acotada; el p99.9 está a 1–2
  unidades del máximo, así que superarlo es realmente inusual. Además los
  árboles no extrapolan: por encima del máximo el modelo responde como si fuera
  el máximo, y ese es el supuesto que la advertencia informa.
* Acción: `advertir`, conservando el valor. `limpiar_solicitud_prediccion`
  recibe los máximos del modelo vigente (parámetro `maximos_promocion`); la API
  se los pasa desde la etapa 4. Sin máximos (o familia sin máximo) no se advierte.
* Se prueba que el umbral no cambia si se alteran datos posteriores al corte.

## 10. Promociones al reentrenar

Las observaciones de ventas reales no traen `en_promocion`. Al reentrenar:

* **Ventas** (histórico + observaciones) → alimentan todas las variables
  históricas (rezagos, medias), tengan o no promoción conocida.
* **Ejemplos de entrenamiento** → solo filas con venta **y promoción real**
  conocidas (`ml.datos.filas_supervisadas`). Una venta sin promoción real (sin
  solicitud, o solo con promociones supuestas) no es ejemplo supervisado, pero
  su venta sí cuenta en los rezagos y medias de los días siguientes.

### Promoción real vs supuesta

* **Real:** la recibida de forma utilizable en la solicitud (aunque se haya
  reparado, p. ej. `"3"` → 3, o redondeado, 2.5 → 3: deriva del dato enviado).
* **Supuesta:** la que completó la capa de calidad porque faltaba la familia
  (D1), faltaba el campo o el valor era nulo, no numérico o negativo (se usó 0).
  `limpiar_solicitud_prediccion` las informa en
  `RegistroCalidad.promociones_supuestas`. Sirven para responder la predicción
  de ese día (con advertencia), pero **nunca** como dato de entrenamiento ni
  para detectar discrepancias.

### Origen de la promoción (`ml.datos.consolidar_promociones`)

1. **Histórico:** `en_promocion` de `historico_ventas.csv` es, según el
   enunciado, el número de productos en promoción registrado junto con la venta
   real de ese día (sin nulos ni marcas de imputación): se trata como la
   promoción **observada** y tiene prioridad.
2. **Solicitud:** si no hay valor histórico, la última solicitud con promoción
   real para esa tienda, fecha y familia.
   * **"Última"** = mayor `registrado_en`; empate → mayor `solicitud_id`. Es un
     orden total y determinista: no depende del orden de inserción ni del orden
     de las filas recibidas. Si faltan esas columnas o tienen nulos, se lanza un
     error en vez de suponer un orden.
3. **Conflicto** (histórico y solicitud existen y difieren): se usa el
   histórico, pero el valor de la solicitud **no se descarta en silencio**:
   queda en `en_promocion_solicitud`, la fila se marca con `discrepancia=True`
   (`ml.datos.discrepancias_promocion`), se registra una advertencia en el log
   y `Pronosticador.info["discrepancias_promocion"]` cuenta las del
   entrenamiento. Una promoción supuesta nunca genera discrepancia.

Pruebas (`tests/ml/test_reentrenamiento.py`): orden por `registrado_en` con
cualquier orden de entrada, desempate por `solicitud_id`, varias tiendas del
mismo día, columnas obligatorias, supuesta ignorada aunque sea la última,
coincidencia, conflicto (con log y conteo en el entrenamiento), ausencia de
histórico, supuesta excluida como ejemplo pero presente en `lag_1`, y
equivalencia con el histórico completo. Verificado por mutación: ignorar
`registrado_en` o no marcar discrepancias hace fallar las pruebas.

### Contrato de datos para la persistencia (etapa 3)

`consolidar_promociones` recibe una fila por familia de cada solicitud con
(`COLUMNAS_SOLICITUDES`):

| Columna | Tipo | Origen |
|---|---|---|
| `solicitud_id` | entero creciente, único por solicitud | clave primaria asignada por PostgreSQL |
| `registrado_en` | timestamp con zona | momento en que el servicio registró la solicitud |
| `fecha`, `tienda_id`, `familia_id` | fecha, entero, texto | solicitud ya limpia |
| `en_promocion` | entero ≥ 0 | valor usado para predecir |
| `promocion_supuesta` | booleano, no nulo | `familia_id in RegistroCalidad.promociones_supuestas` |

`registrado_en` solo ordena las solicitudes recibidas (auditoría); **nunca**
se usa para validar la `fecha` solicitada, que es simulada.

## 11. Persistencia en PostgreSQL (etapa 3)

### Decisiones del usuario

* Las solicitudes **rechazadas** (422) se guardan en la misma tabla que las
  respondidas (`resultado = 'rechazada'`), con el payload y el motivo.
* Si una observación coincide con una clave de `historico_ventas.csv`, se
  guardan ambas (tablas separadas, el histórico nunca se modifica); al
  entrenar **manda el histórico** y la diferencia se marca como discrepancia.
* Al modificar una observación se conserva el valor anterior en
  `observaciones_historial`.

### Bases de datos

* `db` (PostgreSQL 16.14): base **del producto**, volumen `db_producto`.
* `airflow-db` (PostgreSQL 16.14): base **de Airflow** (etapa 7), otro servicio,
  otro volumen (`db_airflow`) y otras credenciales. El producto nunca se conecta a ella.

### Tablas

Datos publicados (se cargan una vez desde `raw/`; nunca se modifican):

| Tabla | Clave primaria | Claves foráneas y restricciones |
|---|---|---|
| `cargas_datos` | `carga_id` (identity) | `archivo` único, `sha256` del archivo cargado |
| `tiendas` | `tienda_id` | `tienda_id > 0`; nombre, ciudad, tipo y apertura no nulos |
| `familias` | `familia_id` | nombre no nulo |
| `costos` | `familia_id` → `familias` | `costo_faltante > 0`, `costo_sobrante > 0` |
| `feriados` | `feriado_id` (identity) | único (`fecha`, `nombre`, `ambito`); `tipo` ∈ {feriado, comercial, pago, medio_dia} |
| `ventas_historicas` | (`fecha`, `tienda_id`, `familia_id`) | FK tiendas, familias, cargas_datos; `ventas ≥ 0`, `en_promocion ≥ 0` |

Operación:

| Tabla | Clave primaria | Relaciones y restricciones |
|---|---|---|
| `solicitudes_prediccion` | `solicitud_id` (identity, creciente) | `registrado_en` (timestamptz, `now()`), `resultado` ∈ {respondida, rechazada}, `payload` (jsonb, lo recibido tal cual); FK `tienda_id`. CHECK: respondida ⇒ tienda, fecha, modelo, versión y estado de calidad no nulos y sin motivo de rechazo; rechazada ⇒ motivo no nulo, sin versión ni estado. Índice (`tienda_id`, `fecha`, `registrado_en`, `solicitud_id`) |
| `solicitud_familias` | (`solicitud_id`, `familia_id`) | FK solicitud y familia; `en_promocion ≥ 0`; `promocion_supuesta` booleano no nulo (real vs supuesta) |
| `predicciones` | (`solicitud_id`, `familia_id`) | FK compuesta → `solicitud_familias` (solo se predice lo solicitado); `venta_esperada ≥ 0`, `cantidad_sugerida ≥ 0` entero. La versión del modelo está en la solicitud |
| `envios_observacion` | `envio_id` (identity) | cada `POST /v1/observaciones`, incluso los repetidos y rechazados; `registros_nuevos`, `registros_modificados`; CHECK análogo al de solicitudes |
| `observaciones` | (`fecha`, `tienda_id`, `familia_id`) | **clave de idempotencia**; `ventas ≥ 0`; FK `envio_id_creacion` y `envio_id_modificacion` → envíos |
| `observaciones_historial` | `historial_id` (identity) | FK compuesta → `observaciones`, FK `envio_id`; `ventas_anterior ≠ ventas_nueva` |
| `calidad_observaciones` | `calidad_id` (identity) | correcciones y advertencias (`accion` ∈ {reparar, advertir}) de una solicitud **o** de un envío (CHECK: exactamente uno); único (`solicitud_id`, `orden`) y (`envio_id`, `orden`) |

Relación predicción ↔ venta real: por la clave natural (`fecha`, `tienda_id`,
`familia_id`), porque la observación puede llegar sin predicción previa (y una
fecha puede tener varias solicitudes). La vista `v_predicciones_reales` las une.

### Integridad e inmutabilidad

* Tablas de **solo inserción**: `solicitudes_prediccion`, `solicitud_familias`,
  `predicciones`, `calidad_observaciones`, `envios_observacion`,
  `observaciones_historial` y `ventas_historicas`. Un trigger rechaza `UPDATE`,
  `DELETE` y `TRUNCATE`: una solicitud o predicción histórica no se puede sobrescribir.
* `observaciones` admite `UPDATE` (corrección de una venta) pero no `DELETE`
  ni `TRUNCATE`; cada cambio deja su fila en `observaciones_historial`.
* **Coherencia con los rechazos** (triggers `BEFORE INSERT`): una solicitud
  rechazada no admite familias (ni, por la FK compuesta, predicciones); un envío
  rechazado no puede crear ni modificar observaciones ni filas del historial.
  Basta comprobarlo al insertar porque el `resultado` del padre es inmutable.
  *Detectado en la verificación del PR #3:* sin estos triggers la base aceptaba
  una predicción colgada de una solicitud rechazada.
* Los triggers no interfieren con las migraciones: `DROP TABLE` no los
  dispara, así que `downgrade` funciona con datos cargados, y `downgrade`
  elimina también sus funciones.

### Idempotencia de observaciones

Por cada envío, en **una transacción** y con un *advisory lock* por (tienda, fecha)
que serializa envíos concurrentes del mismo día:

1. se compara cada venta limpia con la existente;
2. se inserta el envío con `registros_nuevos` y `registros_modificados`;
3. clave nueva → `INSERT`; valor distinto → `UPDATE` + fila en el historial;
   valor igual → nada.

`registros_guardados` = nuevos + modificados: reenviar lo mismo da 0 (E09).

### Transacciones

Cada operación de la API es una unidad de trabajo (`app.db.conexion.transaccion`):
la solicitud, sus familias, predicciones y observaciones de calidad se guardan
todas o ninguna. Los errores de conexión se traducen a `BaseDatosNoDisponible`
(la API responde 503, §13); los de integridad se propagan como
errores de programación. El motor usa `pool_pre_ping` (reconecta tras un corte)
y `connect_timeout`.

### Migraciones

Alembic 1.14.1 (misma versión que fijan las constraints de Airflow 2.10.5).
`migraciones/versions/0001_esquema_inicial.py` crea tablas, restricciones,
índices, triggers y la vista; `downgrade` la revierte. Las pruebas verifican
`upgrade → downgrade → upgrade` y que el esquema migrado coincida con
`app/db/esquema.py` (comparación de Alembic sin diferencias).

*Alternativas descartadas:* ORM declarativo (las operaciones son inserciones y
un *upsert* controlado; SQLAlchemy Core es más explícito); `ON CONFLICT DO UPDATE`
directo (no permite conocer el valor anterior para el historial ni contar
modificados sin una segunda lectura).

### Verificación

* `tests/integracion/` (60 pruebas, PostgreSQL real): migraciones
  up/down/up y esquema idéntico al modelo, carga idempotente y atómica con
  detección de archivos modificados, inserción y consulta de solicitudes con
  promociones reales y supuestas, rechazos, varias solicitudes del mismo día,
  inmutabilidad (`UPDATE`/`DELETE`/`TRUNCATE` rechazados), restricciones e
  integridad referencial, atomicidad ante un fallo a mitad de transacción,
  idempotencia y modificación de observaciones con historial, envíos
  concurrentes, observación de una fecha del histórico (manda el histórico) y
  relación predicción ↔ venta real. Además: historial con valor anterior,
  nuevo, envío y `registrado_en`, y atómico con la modificación (un fallo
  posterior revierte ambos); rechazos sin tienda ni fecha determinables, sin
  familias ni predicciones; modificar observaciones no altera el histórico;
  consolidación determinista; `downgrade`/`upgrade` con datos y `upgrade`
  repetido sin efecto.
* Errores de conexión: servidor inaccesible (prueba unitaria, falla en < 10 s),
  conexión cortada a mitad de transacción y recuperación del pool tras un corte.
* Mutaciones verificadas: quitar el *advisory lock* hace fallar la prueba de
  concurrencia (3 de 3 ejecuciones); contar como modificado un valor igual hace
  fallar 10 pruebas; quitar los triggers de coherencia hace fallar 5.
* `scripts/verificar_persistencia_docker.sh`: datos, migración y cargas intactos
  tras `docker compose restart db` y tras `down` + `up`; la base de Airflow no
  contiene tablas del producto y usa otro volumen.
* Con los datos reales de `raw/` (solo localmente): carga de 83 052 filas en
  13 s, segunda carga en 0.1 s, y lectura idéntica a los CSV.

## 12. Integración continua (GitHub Actions)

* **Disparadores:** `pull_request` hacia `main` (apertura y cada push a la
  rama) y `push` a `main`. `concurrency` cancela ejecuciones obsoletas de la
  misma rama.
* **Jobs independientes** (calidad, unitarias, integración, persistencia) y un
  job agregador **`CI resultado`**, que es el único check obligatorio: si se
  agregan jobs, basta con incluirlos en su `needs`.
* **Sin credenciales en el workflow:** el PostgreSQL de servicio usa `trust`
  (solo accesible dentro del runner efímero); la verificación con Compose
  genera claves aleatorias y las enmascara (`::add-mask::`). *Alternativa
  descartada:* una contraseña fija de prueba en el YAML, aunque sea de CI.
* **Reproducibilidad:** acciones fijadas por SHA de commit (con el tag como
  comentario), `ubuntu-24.04`, Python 3.11 y dependencias de `requirements/`.
* **Calidad:** ruff con reglas E, F, W e I. Las correcciones al código ya
  aprobado son solo de formato (orden de imports y una línea larga), en un
  commit propio y con la suite completa sin cambios.
* **Fuera de alcance:** no hay despliegue ni publicación de imágenes.

## 13. API (etapa 4)

### Decisiones aprobadas (D-A a D-G)

* **D-A · Modelo local:** `python -m app.modelo.entrenar_local` entrena con la
  base (configuración aprobada `hgb_relativo + mult_familia`; ventas
  consolidadas y promociones reales) y escribe `modelos/modelo_local.joblib` de
  forma atómica. Versión `local-AAAAMMDD-<sha256[:12]>`: depende del corte de
  los datos y del contenido, no del reloj. Alias `local`. **La API no entrena al
  iniciar**; sin artefacto (o si no carga) responde 503 y `/health` lo refleja.
  El artefacto no se versiona (`modelos/` en `.gitignore`). Se rechaza si fue
  creado con otras versiones de scikit-learn, numpy o pandas.
* **D-B · Componentes requeridos:** `COMPONENTES_REQUERIDOS` (por defecto
  `base_datos,modelo` en esta etapa). Un componente no requerido y sin
  verificación aparece como `no_configurado`; uno requerido sin verificación
  degrada el estado: `ok` nunca oculta una dependencia ausente.
* **D-C · Base caída al predecir:** 503; no se entrega una predicción sin registrar.
* **D-D · Antigüedad:** se advierte si la última venta real de una familia es
  anterior a F−1 (lo normal es que llegue la de F−1 antes de la solicitud de F).
  Un solo mensaje en `fecha` lista las familias y días; sin ventas en 28 días
  (o sin ventas previas) se informa que se usó el nivel de respaldo. Se calcula
  **solo con ventas de fecha < F** (`advertir_antiguedad` rechaza otra cosa) y
  nunca con el reloj del sistema.
* **D-E · Rechazo con la base caída:** se responde 422 igual; el log registra el
  fallo con el tipo de error, **sin el contenido de la solicitud**.
* **D-F · `/admin`:** `OrquestadorNoConfigurado` → `POST` 503 y `GET` 404 hasta la etapa 7.
* **D-G · `venta_esperada`:** redondeada a 2 decimales; la cantidad se calcula sin redondear.

### Arquitectura

* Endpoints delgados (`app/main.py`) que leen el cuerpo crudo: la validación de
  FastAPI no interviene, así que la capa de calidad puede reparar e informar.
  Cuerpo no JSON → 422 y se registra `{"cuerpo_no_json": <primeros 2000 caracteres>}`.
* Servicios sin HTTP (`app/servicios/`) sobre una interfaz `Persistencia`
  (PostgreSQL en producción; en memoria en las pruebas de contrato).
* Flujo de predicción: modelo vigente → limpieza con los **máximos de promoción
  del modelo cargado** → en una transacción: ventas previas (solo < F; ventana de
  60 días más la última venta de cada familia, para que `ultimo_valor` y
  `dias_desde_ultimo` coincidan con los del entrenamiento) → antigüedad →
  predicción → validación de la respuesta con Pydantic → registro.
* `GestorModelo`: referencia inmutable al modelo (cambio atómico), recarga en
  segundo plano cuando cambia la huella del proveedor y **conserva el modelo
  vigente** si la carga nueva falla. La etapa 5 agrega `ProveedorMLflow` sin
  cambiar la API.
* `MonitorSalud`: verificaciones **en paralelo** con límite global por ciclo
  (`SALUD_LIMITE`, 3 s) cada `SALUD_INTERVALO` (10 s); una verificación colgada
  cuenta como falla y no se relanza mientras siga colgada; `/health` solo lee el
  último resultado. Peor caso de transición: 13 s.
* Errores: `SolicitudRechazada` → 422; `ModeloNoDisponible` y
  `BaseDatosNoDisponible` → 503; cualquier otro → 500 genérico (registrado en el
  log, sin detalles internos en la respuesta).

### Verificación

* `tests/api/` (62, sin base): contrato de los seis endpoints, E01–E10 por HTTP,
  cuerpos inválidos, promociones supuestas y atípicas, antigüedad, sin fuga
  temporal (predecir F con datos posteriores en la base = sin ellos),
  solicitudes repetidas, 300 cuerpos aleatorios sin ningún 500, sin modelo, base
  caída y recuperación, rechazo con la base caída sin exponer datos en el log;
  monitor (paralelo, límite, colgadas, transiciones) y gestor (recarga, conserva
  el vigente, lecturas concurrentes).
* `tests/integracion/test_api_postgres.py` (9, PostgreSQL): registro completo,
  idempotencia y relación con la venta real, rechazos, sin fuga con la base real
  (igual al modelo con el histórico truncado en F), ventas previas estrictamente
  anteriores, base inaccesible, `entrenar_desde_bd` sin promociones supuestas y
  carga de un artefacto nuevo sin reiniciar.
* **Error encontrado:** SQLAlchemy entrega los nombres de columna como
  `quoted_name` y scikit-learn los rechazaba al predecir o entrenar con datos de
  la base; se convierten a `str` en el repositorio (prueba de regresión).
* Mutaciones: incluir ventas de F en la consulta, ejecutar las verificaciones en
  serie y descartar el modelo vigente al fallar una recarga hacen fallar pruebas.
* Con los datos reales (solo local): `entrenar_local` en 14 s
  (`local-20260301-…`, 80 028 filas); servidor uvicorn real con los ejemplos
  publicados (≈0.1 s por predicción); la predicción de la tienda 5 para
  2026-03-01 coincide con la de la etapa 2 (1837.13 / 2252); con la base
  congelada (`docker pause`) `/health` pasó a degradado en 3.7 s y volvió a ok
  en 4.9 s, y la predicción respondió 503 en 5 s.

### Límite de las consultas (revisión del PR #4)

La versión inicial solo limitaba el tiempo de **conexión**; una consulta ya
enviada a un servidor que dejaba de responder podía esperar indefinidamente. Se
agregaron dos capas en `app/db/conexion.py`:

| Capa | Mecanismo | Valor (API) | Variable |
|---|---|---|---|
| Conexión | `connect_timeout` de libpq | 5 s | `DB_TIMEOUT_CONEXION` |
| Servidor | `statement_timeout`: PostgreSQL cancela la consulta (también la espera de locks); la conexión sigue usable | 5 s | `DB_TIMEOUT_CONSULTA` |
| Servidor | `idle_in_transaction_session_timeout`: libera transacciones de un cliente que dejó de responder | máx(6 × consulta, 30 s) | (derivado) |
| Cliente | vigilante por operación (`execute`, `executemany`, `commit`, `rollback`) sobre una clase de conexión psycopg2: al vencer, cierra el socket; psycopg2 sale del bloqueo con error y la conexión se descarta | consulta + 2 s = 7 s | `DB_MARGEN_CLIENTE` |

* Ambas capas terminan en `BaseDatosNoDisponible` → 503. La transacción se
  revierte: si el servidor canceló, PostgreSQL la aborta; si el cliente cortó,
  nunca se envió el `COMMIT` y el servidor la revierte al cerrarse la conexión
  (o por `idle_in_transaction_session_timeout`).
* **Por qué 5 s:** una predicción usa unas pocas consultas de milisegundos
  (≈0.1 s en total con los datos reales); 5 s es 50 veces eso y deja margen
  para esperar el *advisory lock* de envíos concurrentes de observaciones.
  Peor caso de una solicitud con la base sin responder: 7 s (vigilante) + 5 s
  (reintento de conexión) ≈ 12 s, siempre acotado.
* Los procesos por lotes (`app.db.carga`, `entrenar_local`, verificación de
  persistencia) usan `timeout_consulta=300`. Las migraciones de Alembic usan su
  propio motor sin límite.
* *Alternativas descartadas:* solo `statement_timeout` (no actúa si el servidor
  no responde); TCP keepalive o `tcp_user_timeout` (un servidor congelado sigue
  confirmando paquetes a nivel del sistema operativo); un límite sobre toda la
  solicitud HTTP (no interrumpe la llamada bloqueada en psycopg2 ni revierte la
  transacción).

Verificación (`tests/integracion/test_limites_consulta.py`, 7 pruebas, con un
proxy TCP que se puede congelar para simular un servidor sin respuesta):
cancelación en el servidor (`pg_sleep`), corte en el cliente con el servidor
congelado, corte a mitad de una transacción sin confirmación, API con un
`INSERT` lento (503 en < 3 s, sin solicitud, familias ni calidad parciales),
API con el servidor congelado (503, `/health` responde en < 0.5 s y pasa a
degradado, `/modelo` responde, recuperación al volver), congelamiento a mitad
del registro (no se confirma nada) y monitor acotado por su límite. Estable en
3 de 3 repeticiones. Mutaciones: sin el vigilante del cliente la prueba queda
esperando indefinidamente (pytest-timeout la corta a los 60 s); sin
`statement_timeout` falla la prueba del servidor.

### Limitaciones que permanecen

* **Resultado incierto del `COMMIT`:** si el servidor deja de responder después
  de recibir el `COMMIT` pero antes de confirmarlo al cliente, la API responde
  503 aunque la predicción pudo haberse guardado. Es inherente a cualquier
  cliente de base de datos; queda una solicitud registrada sin respuesta
  entregada, nunca una respuesta sin registro.
* El corte del cliente descarta esa conexión; la siguiente operación abre otra
  (hasta `DB_TIMEOUT_CONEXION` si el servidor sigue sin responder).
* Una consulta legítima que tarde más de 5 s en la API se cancela (503); no
  ocurre con el volumen actual, pero si la base crece habría que revisar índices
  o el límite.

## 14. MLflow: registro, alias y proveedor del modelo (etapa 5)

Diseño aprobado en las versiones 1 a 5.1 del plan (M1–M13). Resumen y hallazgos
de la implementación.

### Servidor y almacenamiento (M1, M2, M4)

* Servicio `mlflow` (imagen oficial `v2.22.1`, la misma versión que el cliente)
  con backend **SQLite** en el volumen `mlflow_datos` y artefactos en
  `mlflow_artefactos`, servidos por el propio servidor (`--serve-artifacts`).
  No usa la base del producto ni la de Airflow. Puerto solo en `127.0.0.1`.
* Modelo registrado `novamercado-ventas`, alias `vigente`, experimento
  `novamercado-entrenamiento` (el de backtest previsto como opcional no se
  implementó: los resultados de la etapa 2 siguen en `reportes/etapa2/`).
* PostgreSQL no cambia (M11): `version_modelo` + `nombre_modelo` llevan a la
  versión del Registry y de ahí a su ejecución.

### Única referencia: `modelo.joblib` (M3)

El mismo archivo (mismos bytes) se calcula al entrenar (`sha256_artefacto`), se
registra como artefacto del modelo `pyfunc` (`modelo/artifacts/modelo.joblib`,
tag `ruta_artefacto`), se descarga **solo él** desde
`runs:/<run_id>/<ruta_artefacto>`, se guarda en la caché y se carga. El SHA se
verifica **antes** de `joblib.load`. El envoltorio `pyfunc`
(`ml/mlflow_modelo.py`) carga el mismo archivo; la API no lo usa.

### Evaluación, criterio y alias (M5, M6)

* Backtest de 7 cortes de 28 días (el de la etapa 2) desde la copia de datos;
  candidato `hgb_relativo + mult_familia` frente a `baseline_dow4 + mult_familia`.
* Válido solo si: T1 sin errores · T2 salidas dentro del contrato · D1 ≥ 3
  cortes evaluables · C1 ≥ 5 % menos costo total · C2 gana ≥ ⌈2n/3⌉ cortes ·
  C3 ningún corte > 10 % peor · T4a reproducción del backtest desde la copia
  descargada de MLflow · T4b integridad del modelo definitivo registrado.
* `--promover si_no_hay` (por defecto) / `siempre` (reemplazo manual explícito)
  / `nunca`. El alias **solo** se mueve con un candidato válido y como último
  paso; nunca se omiten validaciones. Una versión en proceso queda
  `estado=pendiente`; el proveedor solo acepta `estado=valido`.
* **Orden de la promoción:** primero todos los tags de la versión (`estado`,
  `motivo`, `promovido_por`, `alias_anterior`), las métricas finales,
  `criterio.json`, los tags de la ejecución y su cierre `FINISHED`; la
  asignación del alias es la **última** operación y después no se escribe nada
  en MLflow (corrección de la revisión del PR #5: antes se registraban métricas
  y tags después del alias, y un fallo ahí dejaba el alias en una versión
  marcada `invalido`).
* Fallos antes de la asignación: la versión queda `estado=invalido`, la
  ejecución `FAILED` y el alias intacto. Con datos insuficientes (D1) no se
  registra versión.
* **Fallo de la propia asignación del alias.** Que el cliente reciba un error no
  prueba que el servidor no aplicara el cambio (respuesta perdida, timeout). Por
  eso, tras la excepción se **consulta el alias una vez**
  (`resolver_asignacion_alias`):
  * apunta a la nueva versión → la promoción **se realizó**: se conservan
    `promovido_por`/`alias_anterior` y `promovido=true`; el comando termina bien;
  * apunta a la versión anterior, a otra o a ninguna → candidato **válido sin
    promover**: se retiran `promovido_por`/`alias_anterior`, `promovido=false`, y
    el error se propaga;
  * la consulta también falla → **indeterminado**: `PromocionIndeterminada`
    (código de salida 4 de `entrenar_mlflow`), tags `promocion=indeterminada` en
    la versión y `promovido=indeterminado` en la ejecución (mejor esfuerzo); se
    conservan las marcas de promoción porque pueden ser ciertas y **no** se
    afirma que el alias anterior siga vigente.
  * En los tres casos la versión ya es `valido`: el alias nunca puede quedar en
    una versión inválida.

### Copia inmutable de datos y reproducción (M12)

* `datos/` en cada ejecución: ventas y promociones consolidadas, observaciones,
  promociones de solicitudes (con las supuestas marcadas), catálogos y
  `manifiesto.json` (SHA-256 del contenido sin comprimir de cada archivo, corte,
  configuración, umbrales, commit, bibliotecas, entorno, hilos). `huella_datos`
  = SHA-256 del manifiesto. El entrenamiento y la evaluación **releen la copia**:
  desde ahí nada lee PostgreSQL.
* `python -m app.modelo.reproducir --run-id <id>`: `identica` (mismo SHA y
  backtest igual), `equivalente` (SHA distinto, predicciones y métricas dentro de
  tolerancia, cantidades iguales) o `no_reproduce`. Tolerancia de reales:
  `math.isclose(rel_tol=1e-9, abs_tol=1e-9)`; enteros exactos.

**Hallazgos de la implementación (determinismo de bytes):**

1. **Hilos de OpenMP.** El HGB paraleliza con OpenMP:
   * con distinto número de hilos (1, 2, 8) los bytes de `modelo.joblib` cambian;
   * incluso con un número **fijo** de hilos (4), el orden de las reducciones
     paralelas varía entre ejecuciones: con los datos reales, en ~1 de cada 3
     entrenamientos un valor de un nodo cambia en el último bit (≈1e-17), y en
     el límite podría cambiar el redondeo de una cantidad sugerida;
   * con **1 hilo** el entrenamiento es determinista byte a byte (5 de 5
     entrenamientos idénticos con los datos reales), a cambio de ~40 % más de
     tiempo (19 s frente a 14 s el modelo definitivo).

   Por eso `HILOS_ENTRENAMIENTO=1` por defecto, fijado **dentro del proceso** con
   `threadpoolctl` (no depende de `OMP_NUM_THREADS` ni del número de CPU) y
   registrado en el manifiesto; `reproducir` usa el valor registrado. Con los
   datos sintéticos (40 iteraciones) la variación no aparecía: se detectó al
   verificar con los datos reales.
2. **Identidad de objetos en `pickle`.** `pickle` reutiliza referencias a un
   mismo objeto: una configuración con cadenas literales y otra leída de JSON,
   iguales en valor, producen bytes distintos de forma determinista (medido con
   1 hilo). El registro toma la configuración **del manifiesto releído**, igual
   que `reproducir`.

Con ambos ajustes, dos entrenamientos en procesos independientes (con
`PYTHONHASHSEED` y `OMP_NUM_THREADS` distintos) dan el **mismo SHA-256**, y la
reproducción desde MLflow tras modificar PostgreSQL resulta `identica`. Si en
otro entorno los bytes difirieran, `reproducir` informa `equivalente` cuando las
predicciones y métricas están dentro de la tolerancia (diseño aprobado).

**Defecto encontrado en MLflow 2.22:** `mlflow.artifacts.download_artifacts`
resuelve los artefactos `mlflow-artifacts:/…` con la URI de tracking **global**
aunque se le pase `tracking_uri`. Sin `MLFLOW_TRACKING_URI` las descargas
fallaban o iban a un `./mlruns` local. Toda descarga pasa por
`app/modelo/mlflow_util.py:descargar_de_ejecucion`, que resuelve la URI de
artefactos de la ejecución contra el servidor indicado; una prueba lo verifica
en un proceso sin estado global.

### Proveedor, caché y recarga (M7–M10, M13)

* `ProveedorMLflow` (interfaz de la etapa 4): la huella es la versión del alias;
  `obtener()` usa la caché si su SHA **recalculado** coincide con el tag del
  Registry, o descarga y valida. Errores: alias o modelo inexistente →
  `VersionNoUtilizable`; sin respuesta → `OrigenNoDisponible`.
* Cliente de MLflow con `MLFLOW_HTTP_REQUEST_TIMEOUT=5` y
  `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0`: con los valores por defecto un servidor
  caído seguía reintentando a los 25 s; con estos falla en 0.01 s.
* `GestorModelo`: refresco cada 15 s (cambio de alias detectado en ≤ 20 s);
  conserva el modelo si falla una recarga; sin modelo y sin MLflow usa
  `respaldo()` (caché); `actualizado()` falso si MLflow responde pero el alias
  no ofrece una versión utilizable → `/health` `degradado` con motivo "no se
  pudo actualizar el modelo vigente" (M9), sin dejar de predecir.
* Componentes requeridos (M10): `.env.example` **no** define
  `COMPONENTES_REQUERIDOS`, porque definirla anula la selección automática:
  con `MODELO_ORIGEN=mlflow` se requiere `mlflow`; con `archivo`, no.
* Caché (`MODELO_CACHE_DIR/<versión>/modelo.joblib` + `metadatos.json`, 3
  versiones, escritura atómica) validada en cascada sin MLflow: metadatos →
  SHA recalculado → formato y bibliotecas → humo; primera válida o 503.
  **Limitación aceptada:** sin MLflow no se detecta una modificación coherente
  del artefacto y de sus metadatos a la vez; al volver MLflow se contrasta con
  el tag del Registry y se corrige (probado).
* `MODELO_ORIGEN=mlflow` (producto; exige `mlflow` en `/health`) o `archivo`
  (etapa 4, desarrollo y pruebas).

### Git limpio (v5, v5.1)

`entrenar_mlflow` verifica **primero** que el repositorio del código esté
limpio. La raíz se resuelve desde la ubicación del código (no del directorio
actual), `HEAD` se lee antes y después de `git status`, y los archivos
ignorados no cuentan. Si no, termina con código 2 y un mensaje claro, sin crear
nada en MLflow. No hay opción para omitirlo.

### Verificación

* `tests/modelo/` (73): resolución del fallo del alias con cliente simulado
  (8 respuestas posibles de la consulta) y código de salida 4; artefacto (SHA antes de deserializar, humo), caché
  (cascada, cinco tipos de corrupción, limitación coherente), copia de datos
  (determinista, alteraciones), criterio C1–C3/D1/T2, tolerancias, Git (13
  casos, incluido el comando real en un repositorio sucio), dos procesos
  independientes con el mismo SHA, proveedor y gestor con cliente simulado.
* `tests/integracion/test_mlflow_registro.py` (29) y `test_mlflow_api.py` (8)
  contra un **servidor MLflow real** y PostgreSQL: registro válido y alias;
  `si_no_hay`, `siempre` y `nunca`; candidato que no supera el costo (con
  `si_no_hay` y `siempre`) y datos insuficientes; fallo inyectado en 5 pasos;
  fallo inyectado en 6 operaciones previas a la promoción y en la propia
  asignación; fallo de la asignación resuelto consultando el alias (rechazada,
  aplicada con la respuesta perdida, y ambas con la consulta también caída →
  indeterminada); el alias es la última escritura;
  reproducción `identica` tras modificar la base; copia y backtest alterados;
  T4b; trazabilidad; `/modelo` y predicción con la versión real; cambio de alias
  detectado en < 30 s con el intervalo por defecto; MLflow detenido y
  reiniciado; arranque desde la caché, recuperación de la versión anterior y
  503 sin versiones válidas; caché corrupta con modelo cargado; manipulación
  coherente corregida al volver MLflow; alias a una versión inválida o borrado;
  8 hilos durante 10 cambios de alias entre dos modelos distintos.
* Mutaciones detectadas: promover un candidato no válido, caché sin verificar
  el SHA, usar la caché con un modelo cargado. La prueba de descargas sin estado
  global falla con la implementación anterior.
* `scripts/verificar_persistencia_docker.sh` con la imagen oficial de MLflow:
  experimento, versión, alias y artefacto intactos tras `restart` y `down`/`up`;
  volúmenes de MLflow separados de las dos bases.

### Limitaciones

* MLflow sin autenticación (solo red de Compose y `127.0.0.1`): entorno de laboratorio.
* La limitación de la caché descrita arriba.
* **Resolución del fallo del alias:** es una sola consulta inmediata. No se
  reintenta (el cliente usa 0 reintentos) y no se puede excluir que una solicitud
  que venció en el cliente se aplique en el servidor **después** de la consulta;
  en ese caso el resultado informado ("no promovida") quedaría desactualizado y
  el alias apuntaría a la nueva versión, que es válida. Con resultado
  indeterminado no hay resolución automática: hay que consultar el alias en
  MLflow (`get_model_version_by_alias`) antes de reintentar.
* Incidencia de pruebas sin identificar (revisión del PR #5): en una ejecución
  local de la suite completa con `34235e3` fallaron 4 pruebas que luego pasaron
  al repetirlas, en otra suite completa, en otra ejecución de la integración y
  en la CI. No se conservaron sus nombres ni la causa.
* La API en contenedor, el arranque automático (migraciones, carga y primer
  entrenamiento) y la procedencia del commit dentro de imágenes sin `.git` son
  de la etapa 6.

## Pendientes de decidir (etapas siguientes)

No quedan decisiones abiertas de las etapas 0–5. Por definir en las siguientes:

* Etapa 6: cómo verificar la procedencia del código (commit) al entrenar dentro
  de un contenedor sin `.git`; comportamiento si el primer candidato no es válido.
* Etapa 6: arranque automático (migraciones, carga y primer entrenamiento) y
  `COMPONENTES_REQUERIDOS` con los cinco componentes.
* Etapa 7: DAG, criterio de reemplazo con datos suficientes y orquestador real de `/admin`.
