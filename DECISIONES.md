# Decisiones técnicas y de producto

Cada decisión indica la alternativa considerada y su justificación. El contrato
del Anexo tiene prioridad sobre cualquier decisión de implementación.

## 1. Versiones y entorno (D4)

| Componente | Versión fijada |
|---|---|
| Python | 3.11 |
| Apache Airflow | 2.10.5 (pendiente, etapa 7) |
| MLflow | 2.22.1 |
| PostgreSQL | 16 (pendiente, etapa 6) |
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
* **D3:** se rechazan fechas anteriores a la apertura de la tienda; se aceptan
  fechas dentro y después del histórico. La advertencia por observaciones
  desactualizadas depende de la base de datos y se implementa en la etapa 4.
* **D6:** los archivos publicados se colocan en `raw/` (excluida de git).

## 4. Pruebas sin datos del repositorio

Las pruebas unitarias usan un catálogo sintético (`tests/conftest.py`) para no
depender de `raw/`, que no se publica. `tests/ejemplos/` ejecuta la política
sobre los ejemplos publicados reales y se omite si `raw/ejemplos` no existe.
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

## Pendientes de decidir (etapas siguientes)

* Umbral de antigüedad de observaciones para advertir (D3), etapa 4.
* **Promociones atípicas (propuesta):** advertir cuando `en_promocion` supere el
  máximo histórico de su familia (7 en panadería a 19 en abarrotes,
  `Pronosticador.info["promocion_max_familia"]`), conservando el valor. Motivo:
  el p99.9 está a 1–2 unidades del máximo, de modo que superarlo es realmente
  inusual, y los árboles no extrapolan: más allá del máximo el modelo responde
  como si fuera el máximo, que es justamente el supuesto que debe informarse.
* Promoción de los días registrados por observaciones al reentrenar
  (las observaciones no la incluyen): tomarla de la solicitud de predicción de
  ese día; sin solicitud, excluir esa fila del entrenamiento pero usar su venta
  en los rezagos. Etapas 3 y 7.
