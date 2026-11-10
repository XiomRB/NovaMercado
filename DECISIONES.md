# Decisiones técnicas y de producto

Cada decisión indica la alternativa considerada y su justificación. El contrato
del Anexo tiene prioridad sobre cualquier decisión de implementación.

## 1. Versiones y entorno (D4)

| Componente | Versión fijada |
|---|---|
| Python | 3.11 |
| Apache Airflow | 2.10.5 (pendiente, etapa 7) |
| MLflow | 2.22.1 |
| PostgreSQL | imagen `postgres:16.14-alpine` |
| Alembic | 1.14.1 (igual que las constraints de Airflow 2.10.5) |
| ruff (solo desarrollo y CI) | 0.17.2 |
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
  desactualizadas depende de la base de datos y se implementa en la etapa 4.
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
  los pasará en la etapa 4. Sin máximos (o familia sin máximo) no se advierte.
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
(la API responderá 503 en la etapa 4); los de integridad se propagan como
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

## Pendientes de decidir (etapas siguientes)


* Umbral de antigüedad de observaciones para advertir (D3), etapa 4.
