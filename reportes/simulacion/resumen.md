# Simulación de operación — escenario `completo`

Resultado: **APROBADA**

| | |
|---|---|
| Ejecución | `d14fdcfe` (proyecto `nm-sim-d14fdcfe`) |
| Commit | `1ff3243a77d8dde728a6e9479d3ce9edb9ee6965` |
| Fuente | raw; histórico inicial hasta 2025-11-30; días 2025-12-01 a 2026-02-28 (90) |
| Duración | 75 min |
| Reanudaciones | 0 |

## Criterios de aceptación

| # | Criterio | Resultado |
|---|---|---|
| 1 | Sin intervención manual, desde un solo comando | cumple |
| 2 | Respuestas conforme al Anexo; sin 5xx inesperados | cumple |
| 3 | Defectos con el resultado de DECISIONES.md §3 (oráculo) | cumple |
| 4 | version_modelo coherente con /modelo y registrada igual en PostgreSQL | cumple |
| 5 | Observaciones idempotentes y guardadas como se esperaba | cumple |
| 6 | Reentrenamientos: ≤ 10 min, promoción ≤ 120 s, vigente intacto si no promueve o falla | cumple |
| 7 | MLflow detenido: degradado < 60 s, 200, reentrenamiento fallido, ok < 120 s | cumple |
| 8 | Reinicios: ok ≤ 10 min, sin reentrenar, datos y ejecuciones conservados | cumple |
| 9 | Costo del servicio < referencia baseline_dow4 (90 días reales) | cumple |
| 10 | costos --recalcular reproduce el costo solo con la evidencia, tras la limpieza | cumple |
| 11 | Imágenes propias del commit; las :local no cambian | cumple |

Los criterios 10 y 11 se completan con las pruebas I1–I7 de la CI (ver README).

## Datos iniciales (S1, S10)

* Histórico preparado: 72252 filas, última fecha 2025-11-30.
* PostgreSQL antes del día 1: última fecha 2025-11-30, 0 observaciones; corte del primer modelo `2025-12-01`.
* `feriados.csv` completo: calendario conocido de antemano, única excepción declarada (S10).

## Solicitudes

| Tipo | Rol | Código | Cantidad |
|---|---|---|---|
| observacion | principal | 200 | 835 |
| observacion | principal | 422 | 65 |
| observacion | reenvio_limpio | 200 | 65 |
| observacion | repeticion | 200 | 39 |
| prediccion | principal | 200 | 788 |
| prediccion | principal | 422 | 112 |
| prediccion | reenvio_limpio | 200 | 112 |
| prediccion | repeticion | 200 | 40 |

Con defectos: 409 (111 combinadas); tipos distintos: 60.

| Defecto | Catálogo | Veces |
|---|---|---|
| campo_camel | E07 | 13 |
| campo_desconocido | nuevo | 13 |
| campo_errata | nuevo | 17 |
| campo_generico | nuevo | 11 |
| campo_item_generico | nuevo | 5 |
| campo_item_sinonimo | E07 | 7 |
| campo_mayusculas | E07 | 14 |
| campo_repetido_distinto | nuevo | 14 |
| campo_repetido_igual | nuevo | 15 |
| campo_sinonimo | E07 | 13 |
| cuerpo_lista | nuevo | 8 |
| cuerpo_no_json | nuevo | 13 |
| faltan_familias | E05 | 10 |
| familia_irreconocible | nuevo | 16 |
| familia_mayusculas | E02 | 4 |
| familia_nombre_catalogo | E02 | 4 |
| familia_parecida | nuevo | 4 |
| familia_repetida_distinta | E06 | 19 |
| familia_repetida_igual | E06 | 12 |
| familia_separadores | E02 | 8 |
| fecha_anio_dos_digitos | nuevo | 3 |
| fecha_antes_apertura | nuevo | 9 |
| fecha_ausente | nuevo | 11 |
| fecha_barras | E04 | 10 |
| fecha_compacta | E04 | 7 |
| fecha_con_hora | E04 | 5 |
| fecha_dd_mm | E04 | 9 |
| fecha_espacios | nuevo | 10 |
| fecha_ilegible | nuevo | 4 |
| fecha_inexistente | nuevo | 6 |
| fecha_mes_nombre | E04 | 3 |
| fecha_mm_dd | E04 | 11 |
| fecha_puntos | nuevo | 7 |
| fecha_sin_ceros | E04 | 6 |
| item_no_objeto | nuevo | 19 |
| lista_como_objeto | nuevo | 15 |
| lista_vacia | nuevo | 17 |
| ninguna_venta_valida | E08 | 16 |
| tienda_ausente | nuevo | 11 |
| tienda_booleana | nuevo | 9 |
| tienda_decimal | nuevo | 11 |
| tienda_decimal_entero | E01 | 13 |
| tienda_espacios_unicode | nuevo | 8 |
| tienda_inexistente | nuevo | 10 |
| tienda_negativa | nuevo | 13 |
| tienda_palabra | E01 | 12 |
| tienda_prefijo | E01 | 14 |
| tienda_texto | E01 | 10 |
| tienda_varios_numeros | nuevo | 10 |
| valor_ausente | nuevo | 8 |
| valor_booleano | nuevo | 2 |
| valor_coma_decimal_cero | E10 | 10 |
| valor_decimal_entero | E03/E10 | 9 |
| valor_fraccion | nuevo | 6 |
| valor_miles | E10 | 5 |
| valor_negativo | E08 | 6 |
| valor_no_numerico | nuevo | 5 |
| valor_nulo | nuevo | 4 |
| valor_texto | E03/E10 | 4 |
| valor_texto_con_espacios | nuevo | 4 |

## Reentrenamientos

| Día | Tipo | id | Estado | Promovido | Versión | Duración (s) | Promoción (s) |
|---|---|---|---|---|---|---|---|
| 8 | programado | `nm-2182a293-602a-4141-83d1-9d85c8c8b66a` | exitoso | False | 1 | 278 | — |
| 15 | programado | `nm-493bf0fb-3c5a-43f1-b2d7-86ae309706e6` | exitoso | False | 1 | 332 | — |
| 22 | programado | `nm-5f68c10e-40c5-4f89-b7d8-86c808efe0db` | exitoso | False | 1 | 296 | — |
| 29 | programado | `nm-96e53d35-5880-4db5-9d22-70a1ebfb0f0b` | exitoso | False | 1 | 290 | — |
| 36 | programado | `nm-5108973f-1b97-4682-bbec-513d9006a534` | exitoso | True | 6 | 325 | 9.9 |
| 43 | programado | `nm-03be514a-94cf-4291-ab45-840d66f3f143` | exitoso | False | 6 | 250 | — |
| 47 | caida | `nm-895d4ccb-7c76-414e-9d78-ab42a46c625d` | fallido | False | 6 | 44 | — |
| 50 | programado | `nm-0e6e237b-960a-49b6-9c24-dc25e914485e` | exitoso | True | 8 | 291 | 4.4 |
| 57 | programado | `nm-92f4c572-11f0-4e3c-8030-c455ba741248` | exitoso | False | 8 | 253 | — |
| 64 | programado | `nm-24fb9a64-1973-408c-a5aa-17a05e3aa38d` | exitoso | False | 8 | 309 | — |
| 71 | programado | `nm-1579b09a-9cc3-4bd7-990e-b6c60ad6b423` | exitoso | True | 11 | 315 | 16.4 |
| 78 | programado | `nm-f43351da-ded1-4b9a-b5ec-f194c0009be7` | exitoso | True | 12 | 292 | 10.0 |
| 85 | programado | `nm-00a86acb-4bd7-4aea-a2fa-1d49bba24b64` | exitoso | False | 12 | 275 | — |

## Mediciones

* `arranque_s`: [391.9]
* `mlflow_degradado_s`: [4.0]
* `mlflow_recuperado_s`: [23.2]
* `promocion_s`: [9.9, 4.4, 16.4, 10.0]
* `reentrenamiento_s`: [278.4, 331.9, 295.9, 290.2, 324.6, 250.1, 44.4, 291.4, 253.3, 309.3, 315.3, 291.8, 275.2]
* `reinicio_down/up_s`: [76.2]
* `reinicio_stop/start_s`: [84.7]

## Versiones servidas

* 1 (vista primero en /modelo)
* 6 (vista primero en /modelo)
* 8 (vista primero en /modelo)
* 11 (vista primero en /modelo)
* 12 (vista primero en /modelo)

## Costo (S9)

* Filas evaluadas: 10800.
* Servicio: **302 227.90**; referencia `baseline_dow4`: 603 387.80; reducción 49.9 %.

| Familia | Servicio | Referencia | Reducción |
|---|---|---|---|
| abarrotes | 38 600.00 | 87 563.00 | 55.9 % |
| bebe | 3 463.20 | 8 819.80 | 60.7 % |
| bebidas | 30 024.00 | 80 388.40 | 62.7 % |
| carnes | 46 876.00 | 79 568.00 | 41.1 % |
| cuidado_personal | 5 876.40 | 18 754.70 | 68.7 % |
| frutas_verduras | 48 095.00 | 77 910.00 | 38.3 % |
| lacteos | 45 317.00 | 65 079.60 | 30.4 % |
| licores | 6 055.70 | 33 471.90 | 81.9 % |
| limpieza | 7 504.90 | 24 342.50 | 69.2 % |
| mascotas | 3 029.60 | 5 344.40 | 43.3 % |
| panaderia | 48 290.00 | 85 630.50 | 43.6 % |
| snacks | 19 096.10 | 36 515.00 | 47.7 % |

| Tienda | Servicio | Referencia | Reducción |
|---|---|---|---|
| 1 | 50 179.60 | 97 533.20 | 48.6 % |
| 10 | 10 859.50 | 20 299.50 | 46.5 % |
| 2 | 28 710.80 | 54 476.90 | 47.3 % |
| 3 | 47 596.50 | 95 340.40 | 50.1 % |
| 4 | 10 632.00 | 19 879.00 | 46.5 % |
| 5 | 56 746.40 | 120 603.00 | 52.9 % |
| 6 | 29 762.10 | 55 685.70 | 46.6 % |
| 7 | 29 492.40 | 61 683.10 | 52.2 % |
| 8 | 28 435.60 | 58 051.50 | 51.0 % |
| 9 | 9 813.00 | 19 835.50 | 50.5 % |

| Semana | Servicio | Referencia | Reducción |
|---|---|---|---|
| 1 | 24 890.10 | 43 161.00 | 42.3 % |
| 10 | 22 380.70 | 35 780.50 | 37.5 % |
| 11 | 22 344.60 | 34 267.40 | 34.8 % |
| 12 | 21 302.90 | 35 996.60 | 40.8 % |
| 13 | 19 250.50 | 31 298.70 | 38.5 % |
| 2 | 24 686.70 | 42 182.30 | 41.5 % |
| 3 | 25 751.10 | 55 877.50 | 53.9 % |
| 4 | 31 202.10 | 75 007.10 | 58.4 % |
| 5 | 25 499.10 | 66 259.10 | 61.5 % |
| 6 | 20 496.20 | 51 384.40 | 60.1 % |
| 7 | 21 127.10 | 50 538.50 | 58.2 % |
| 8 | 20 709.90 | 42 928.80 | 51.8 % |
| 9 | 22 586.90 | 38 705.90 | 41.6 % |

| Version | Servicio | Referencia | Reducción |
|---|---|---|---|
| 1 | 152 525.30 | 333 871.40 | 54.3 % |
| 11 | 21 302.90 | 35 996.60 | 40.8 % |
| 12 | 19 250.50 | 31 298.70 | 38.5 % |
| 6 | 41 837.00 | 93 467.30 | 55.2 % |
| 8 | 67 312.20 | 108 753.80 | 38.1 % |

## Evidencia

* SHA-256 de `evidencia.sha256`: `482f9c14fa5e57283a06d8015c52261f7952621b09e472420485f946f372bb4f`.
* Limpieza: {'borrados': {'contenedores': ['nm-sim-d14fdcfe-airflow-db-1', 'nm-sim-d14fdcfe-airflow-init-1', 'nm-sim-d14fdcfe-airflow-scheduler-1', 'nm-sim-d14fdcfe-airflow-webserver-1', 'nm-sim-d14fdcfe-api-1', 'nm-sim-d14fdcfe-db-1', 'nm-sim-d14fdcfe-entrenamiento-inicial-1', 'nm-sim-d14fdcfe-mlflow-1', 'nm-sim-d14fdcfe-preparar-1'], 'volumenes': ['nm-sim-d14fdcfe_airflow_logs', 'nm-sim-d14fdcfe_db_airflow', 'nm-sim-d14fdcfe_db_producto', 'nm-sim-d14fdcfe_mlflow_artefactos', 'nm-sim-d14fdcfe_mlflow_datos', 'nm-sim-d14fdcfe_modelo_cache'], 'redes': ['nm-sim-d14fdcfe_default']}, 'datos_borrados': True, 'evidencia_presente': True}.
* `costos --recalcular` (solo con la evidencia): servicio 302 227.90, referencia 603 387.80, 10800 filas.

## Imágenes (S11)

* app: `novamercado-app:sim-1ff3243a77d8` → `sha256:f6facbe9fdfb`
* airflow: `novamercado-airflow:sim-1ff3243a77d8` → `sha256:f10c001cd8a7`
* `:local` antes: {'app': 'sha256:f77cb087cd3139e8aa188b6bc3738225ac3574fdf4c8a28081488f02c398ec96', 'airflow': 'sha256:e93fe671ab1420d167ddb3af98db03fd6e8979d59211312ac89f7ebabd5083ba'}; después: {'app': 'sha256:f77cb087cd3139e8aa188b6bc3738225ac3574fdf4c8a28081488f02c398ec96', 'airflow': 'sha256:e93fe671ab1420d167ddb3af98db03fd6e8979d59211312ac89f7ebabd5083ba'}

## Fallos

Ninguno.
