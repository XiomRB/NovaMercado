# Evidencia de la simulación completa (etapa 8)

Informe agregado de la reproducción de 90 días aceptada (`d14fdcfe`), exportado con
`python -m simulador exportar`. Diseño y criterios en `DECISIONES.md` §17.

| Archivo | Contenido |
|---|---|
| `resumen.md` | resultado, los 11 criterios, solicitudes, defectos, reentrenamientos, mediciones, costo |
| `metricas.json` | lo mismo en forma legible por máquina (escenario, plazos, fallos por criterio, ejecuciones) |
| `ejemplos.json` | un ejemplo por tipo de defecto: solicitudes y respuestas de predicción; de observaciones solo la respuesta, con los valores de ventas anonimizados |

En las respuestas de observaciones de `ejemplos.json`, los mensajes de calidad repiten el
valor de venta recibido. Al publicar, cada número del `detalle` de los campos
`ventas[i].ventas` y los valores `ventas[i].ventas=…` del `detail` de los 422 se
reemplazaron por `N`; se conservan el formato (`'N,N' se interpretó como N`), el campo,
la acción y el código. Las predicciones no se modificaron. La copia en `trabajo/informe/`
de la ejecución queda sin anonimizar y fuera del repositorio.

## Procedencia

* Commit del producto y del simulador: `1ff3243a77d8dde728a6e9479d3ce9edb9ee6965` (imágenes
  `novamercado-app:sim-1ff3243a77d8` → `sha256:f6facbe9fdfb`, `novamercado-airflow:sim-1ff3243a77d8`
  → `sha256:f10c001cd8a7`). Los commits posteriores de la rama solo agregan este informe y
  documentación.
* Escenario `completo`, semilla 20251201, huella
  `145ab0a71e5b1aeb71f4285f624fc2071b41907b38a7a80c50f319e7337dea01`; corte 2025-11-30, días
  2025-12-01 a 2026-02-28.
* Entorno: WSL 2 (kernel 6.6.87.2) con `memory=6GB` (5,9 GB visibles, swap 2 GB), Docker 29.1.3,
  8 CPU. 2026-11-14, 75 min, sin reanudaciones.

## Evidencia sellada (no publicada)

La carpeta `evidencia/` de la ejecución queda fuera del repositorio porque
`respuestas.jsonl` y `ventas_referencia.csv` contienen las ventas reales de los días
simulados. Aquí solo se publican sus huellas, para comprobar una copia:

```
0baee4424f183f0394aaf0e06f5ca633f3b5fe9a77c6985fc1d1a2b628ac00e4  costos.csv
029f3929e79aeca797cf962bf120ab0b22f46d0d5382982551b30576e531c91a  escenario.json
a3cbbaa498fcf67f7a4f9b6d7ae52de5e7a255827f19870fe5942e53377445b1  origen.json
3f86fa8b06447d1c4cdd378c68f1449b7a22e26c988b13a3a9a778d4725140dc  respuestas.jsonl
7c14ca2165fc5082499390a9de87b40480fa08fa9e73f7baca600d5edd54b8d1  ventas_referencia.csv
```

SHA-256 de `evidencia.sha256`: `482f9c14fa5e57283a06d8015c52261f7952621b09e472420485f946f372bb4f`.
`costos --recalcular` tras la limpieza: servicio 302 227,90, referencia 603 387,80, 10 800 filas.

## Reproducciones anteriores (no aceptadas)

Con el mismo escenario y criterios hubo dos reproducciones con 3,9 GB de memoria para WSL
(swap 1 GB). Ninguna se aceptó:

| | 1.ª `3e133fd9` | 2.ª `ebf48009` | 3.ª `d14fdcfe` (esta) |
|---|---|---|---|
| Commit | `7ef24c0` | `1ff3243` | `1ff3243` |
| Memoria de WSL | 3,9 GB, swap 1 GB | 3,9 GB, swap 1 GB | 5,9 GB, swap 2 GB |
| Resultado | criterio 6 incumplido | criterios 2 y 4 incumplidos | 11 de 11 |
| Fallo | día 71: concesión de entrenamiento perdida, `fallido` a los 670 s | día 20: un 503 tras 10,2 s con la solicitud registrada como respondida (COMMIT incierto, §13) | ninguno |
| Reentrenamientos (s) | 247–670 | 248–380 | 250–332 |
| Latencia p50 / p99 / máx. (s) | 0,031 / 0,65 / 2,98 | 0,031 / 0,82 / 14,3 | 0,032 / 0,56 / 2,55 |
| Costo del servicio | 302 313,90 | 302 227,90 | 302 227,90 |
| Memoria (máximo) | no medida | swap 1006 de 1024 MB, 207 MB libres; presión `full` ≈ 685 s | 3,8 de 5,9 GB, swap 114 MB; presión `full` ≈ 20 s |

En la 2.ª, los logs muestran el entorno detenido 10–12 s (heartbeats de Airflow recuperados
tras 12,55 s y 10,76 s) justo en ese momento. Con más memoria no se repitió ninguno de los dos
fallos. La conclusión se limita a este equipo: es la configuración con la que se obtuvo una
ejecución satisfactoria, no un mínimo garantizado.
