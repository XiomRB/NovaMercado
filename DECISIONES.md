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

## Pendientes de decidir (etapas siguientes)

* Umbral de antigüedad de observaciones para advertir (D3), etapa 4.
* Advertir `en_promocion` fuera del rango histórico (p. ej. > 3× el máximo
  observado). Requiere estadísticas del histórico; etapa 2/4.
