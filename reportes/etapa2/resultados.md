# Resultados del backtesting — etapa 2

Generado por `scripts/experimento_etapa2.py` en 680 s (Python 3.11.18, scikit-learn 1.6.1).

* Histórico: 2024-03-01 a 2026-02-28.
* 7 cortes con ventanas de prueba consecutivas de 28 días: 2025-08-17 a 2026-02-28 (196 días, 23,520 filas por combinación).
* En cada corte C: entrenamiento con fechas < C, calibración con residuos fuera de muestra de [C−56, C), prueba en [C, C+28) sin reentrenar.

## Resumen global (ordenado por costo de negocio)

| modelo | estrategia | costo_total | costo_por_dia_tienda | mae | rmse | wape | sesgo | tasa_quiebre | unidades_faltantes | unidades_sobrantes |
|---|---|---|---|---|---|---|---|---|---|---|
| hgb_relativo | mult_familia | 628,813 | 320.823 | 38.282 | 69.838 | 0.179 | -0.006 | 0.294 | 311,070 | 964,729 |
| hgb_relativo | mult_tipo_familia | 631,112 | 321.996 | 38.282 | 69.838 | 0.179 | -0.006 | 0.296 | 306,924 | 962,946 |
| hgb_directo | mult_familia | 631,415 | 322.151 | 38.553 | 71.494 | 0.180 | -0.020 | 0.293 | 314,268 | 958,284 |
| hgb_directo | mult_tipo_familia | 631,596 | 322.243 | 38.553 | 71.494 | 0.180 | -0.020 | 0.297 | 307,806 | 961,649 |
| hgb_relativo | cuantil_hgb | 631,698 | 322.295 | 38.282 | 69.838 | 0.179 | -0.006 | 0.295 | 310,315 | 918,212 |
| hgb_relativo | raiz_tipo_familia | 631,976 | 322.437 | 38.282 | 69.838 | 0.179 | -0.006 | 0.296 | 310,253 | 942,182 |
| hgb_directo | raiz_tipo_familia | 635,433 | 324.200 | 38.553 | 71.494 | 0.180 | -0.020 | 0.298 | 312,969 | 942,533 |
| hgb_relativo | raiz_familia | 637,472 | 325.241 | 38.282 | 69.838 | 0.179 | -0.006 | 0.298 | 315,619 | 887,801 |
| hgb_directo | raiz_familia | 643,378 | 328.254 | 38.553 | 71.494 | 0.180 | -0.020 | 0.298 | 324,803 | 886,990 |
| hgb_relativo | redondeo | 821,651 | 419.210 | 38.282 | 69.838 | 0.179 | -0.006 | 0.469 | 464,272 | 436,086 |
| hgb_directo | redondeo | 845,008 | 431.126 | 38.553 | 71.494 | 0.180 | -0.020 | 0.479 | 503,564 | 403,136 |
| baseline_dow4 | mult_familia | 879,479 | 448.714 | 52.990 | 100.098 | 0.247 | -0.004 | 0.298 | 395,582 | 1,455,671 |
| baseline_dow4 | raiz_tipo_familia | 880,326 | 449.146 | 52.990 | 100.098 | 0.247 | -0.004 | 0.300 | 401,569 | 1,378,006 |
| baseline_dow4 | mult_tipo_familia | 881,790 | 449.893 | 52.990 | 100.098 | 0.247 | -0.004 | 0.301 | 397,636 | 1,445,764 |
| baseline_dow4 | raiz_familia | 893,747 | 455.993 | 52.990 | 100.098 | 0.247 | -0.004 | 0.300 | 409,608 | 1,310,909 |
| baseline_dow4 | redondeo | 1,124,054 | 573.497 | 52.990 | 100.098 | 0.247 | -0.004 | 0.472 | 631,769 | 614,686 |

**Mejor combinación:** `hgb_relativo` + `mult_familia` — costo 628,813 vs 879,479 del mejor baseline (`mult_familia`): 28.5 % menos.

## Costo por corte: mejor combinación vs baseline

| corte | baseline | mejor | mejora_% |
|---|---|---|---|
| 2025-08-17 | 100,683 | 83,084 | 17.5 |
| 2025-09-14 | 104,176 | 86,451 | 17.0 |
| 2025-10-12 | 110,097 | 87,965 | 20.1 |
| 2025-11-09 | 110,273 | 90,030 | 18.4 |
| 2025-12-07 | 174,328 | 108,587 | 37.7 |
| 2026-01-04 | 171,129 | 84,040 | 50.9 |
| 2026-02-01 | 108,793 | 88,656 | 18.5 |

La mejor combinación gana en 7 de 7 cortes.

## Desglose por tienda

| tienda_id | costo_baseline | costo_mejor | wape_baseline | wape_mejor | mejora_% |
|---|---|---|---|---|---|
| 1 | 141,354 | 102,950 | 0.243 | 0.177 | 27.169 |
| 2 | 83,027 | 61,216 | 0.249 | 0.185 | 26.270 |
| 3 | 140,864 | 101,216 | 0.244 | 0.176 | 28.146 |
| 4 | 29,717 | 21,835 | 0.253 | 0.187 | 26.524 |
| 5 | 173,824 | 116,069 | 0.251 | 0.173 | 33.226 |
| 6 | 79,858 | 60,070 | 0.246 | 0.182 | 24.779 |
| 7 | 89,565 | 63,590 | 0.246 | 0.176 | 29.001 |
| 8 | 85,085 | 59,835 | 0.240 | 0.174 | 29.677 |
| 9 | 27,804 | 20,491 | 0.268 | 0.203 | 26.301 |
| 10 | 28,381 | 21,541 | 0.256 | 0.198 | 24.102 |

## Desglose por familia

| familia_id | costo_baseline | costo_mejor | wape_baseline | wape_mejor | mejora_% | tau |
|---|---|---|---|---|---|---|
| abarrotes | 118,484 | 83,948 | 0.246 | 0.178 | 29.149 | 0.833 |
| bebe | 10,768 | 7,386 | 0.264 | 0.204 | 31.406 | 0.882 |
| bebidas | 93,469 | 59,876 | 0.257 | 0.174 | 35.940 | 0.833 |
| carnes | 136,472 | 93,432 | 0.264 | 0.183 | 31.537 | 0.429 |
| cuidado_personal | 18,972 | 12,394 | 0.250 | 0.187 | 34.672 | 0.909 |
| frutas_verduras | 132,322 | 101,080 | 0.224 | 0.168 | 23.611 | 0.333 |
| lacteos | 126,369 | 96,915 | 0.227 | 0.176 | 23.307 | 0.455 |
| licores | 23,336 | 10,898 | 0.321 | 0.200 | 53.301 | 0.952 |
| limpieza | 23,806 | 15,719 | 0.261 | 0.183 | 33.973 | 0.909 |
| mascotas | 7,975 | 6,350 | 0.266 | 0.217 | 20.372 | 0.833 |
| panaderia | 130,883 | 101,703 | 0.227 | 0.179 | 22.295 | 0.286 |
| snacks | 56,624 | 39,112 | 0.240 | 0.172 | 30.927 | 0.769 |

## Series tienda×familia donde menos mejora (o empeora) el modelo

| tienda_id | familia_id | baseline_dow4 | hgb_relativo | mejora_pct |
|---|---|---|---|---|
| 10 | bebe | 301.4 | 274.7 | 8.9 |
| 9 | bebe | 306.1 | 274.4 | 10.4 |
| 10 | mascotas | 189.2 | 166.0 | 12.3 |
| 6 | panaderia | 10,925 | 9,538 | 12.7 |
| 2 | lacteos | 12,177 | 10,451 | 14.2 |
| 9 | frutas_verduras | 2,944 | 2,498 | 15.1 |
| 10 | panaderia | 5,969 | 5,018 | 15.9 |
| 4 | frutas_verduras | 2,563 | 2,148 | 16.2 |
| 6 | frutas_verduras | 15,572 | 12,888 | 17.2 |
| 10 | frutas_verduras | 2,455 | 2,025 | 17.5 |

## Cuantil objetivo por familia

| familia_id | tau | costo_faltante | costo_sobrante |
|---|---|---|---|
| abarrotes | 0.833 | 1.000 | 0.200 |
| bebidas | 0.833 | 1.000 | 0.200 |
| lacteos | 0.455 | 1.000 | 1.200 |
| carnes | 0.429 | 1.500 | 2.000 |
| panaderia | 0.286 | 1.000 | 2.500 |
| frutas_verduras | 0.333 | 1.000 | 2.000 |
| limpieza | 0.909 | 1.000 | 0.100 |
| cuidado_personal | 0.909 | 1.000 | 0.100 |
| licores | 0.952 | 2.000 | 0.100 |
| snacks | 0.769 | 1.000 | 0.300 |
| bebe | 0.882 | 1.500 | 0.200 |
| mascotas | 0.833 | 1.000 | 0.200 |

## Distribución de `en_promocion` en el histórico

| familia_id | maximo | p99 | p999 | frac_con_promo |
|---|---|---|---|---|
| abarrotes | 19 | 15.000 | 18.000 | 0.609 |
| bebe | 9 | 6.000 | 8.000 | 0.363 |
| bebidas | 14 | 11.000 | 13.000 | 0.579 |
| carnes | 11 | 6.000 | 10.000 | 0.286 |
| cuidado_personal | 16 | 11.000 | 14.000 | 0.502 |
| frutas_verduras | 12 | 8.000 | 11.000 | 0.403 |
| lacteos | 13 | 10.000 | 12.000 | 0.476 |
| licores | 10 | 7.000 | 9.000 | 0.383 |
| limpieza | 18 | 12.000 | 16.000 | 0.565 |
| mascotas | 9 | 6.000 | 8.000 | 0.363 |
| panaderia | 7 | 5.000 | 7.000 | 0.196 |
| snacks | 13 | 9.000 | 12.000 | 0.486 |
