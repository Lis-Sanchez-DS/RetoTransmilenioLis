# Hallazgos del análisis exploratorio de datos (EDA)

## Conjunto de datos

La tabla PostgreSQL `"Original Data"` contiene 51.840 observaciones de 12 estaciones, registradas cada 15 minutos. El periodo disponible va del 26 de julio al 9 de septiembre de 2026. Cada estación tiene 4.320 observaciones.

La variable `demand` es un conteo no negativo. El primer día contiene 1.152 observaciones (12 estaciones × 96 intervalos) sin una referencia del día anterior.

## Hallazgos principales

- La demanda presenta una estacionalidad diaria marcada, con picos recurrentes en la mañana y la tarde.
- Los días laborales y los fines de semana presentan patrones diferentes, lo que indica estacionalidad semanal.
- Banderas, Ricaurte - NQS, Portal Américas y Portal El Dorado generalmente tienen los rangos más altos.
- Universidades - CityU generalmente tiene el rango más bajo.
- La forma y magnitud de los picos varía por estación; conviene incluir variables específicas de estación.

## Desestacionalización

La estacionalidad diaria se removió por estación restando la mediana de demanda del intervalo de 15 minutos correspondiente. Hay 96 intervalos por día. Las gráficas desestacionalizadas todavía muestran secuencias de residuos positivos y negativos, especialmente durante periodos inusuales y fines de semana. Por lo tanto, remover la estacionalidad diaria no elimina toda la estructura predecible.

## Autocorrelación y ruido blanco

Se calculó la ACF de los residuos desestacionalizados hasta el rezago 96, donde 96 equivale a un día. La mayoría de estaciones conserva autocorrelación positiva significativa durante varios rezagos. Universidades - CityU y Universidad Nacional muestran una persistencia especialmente fuerte; Calle 100 - Marketmedios y Calle 72 también muestran estructura de mayor duración.

Los residuos no deben tratarse como ruido blanco solo porque estén dentro de un umbral de valores atípicos. La autocorrelación indica que los rezagos de demanda, estadísticas móviles o términos autorregresivos todavía pueden mejorar los pronósticos.

## Valores atípicos

Se usó una regla MAD robusta específica por estación:

```text
z_robusto = (residuo - mediana_estación) / (1,4826 × MAD_estación)
atípico   = abs(z_robusto) > 3,5
```

Las gráficas muestran los límites superior e inferior con líneas discontinuas y sombrean en gris las áreas que los superan. Un valor atípico no necesariamente es un error: puede representar un evento, un cambio operativo o una demanda genuinamente inusual.

## Modelo base

El modelo base usa la demanda de la misma estación exactamente un día antes. Como el muestreo es cada 15 minutos, esto corresponde a un rezago de 96 observaciones. Cuando el valor anterior no está disponible, la predicción es 0.

La métrica utilizada es:

```text
Accuracy = 100 × max(0, 1 − WAPE)
```

El WAPE se calcula por estación y luego se promedian las precisiones. El modelo base obtuvo 75,86 % de precisión general:

| Estación | Precisión |
| --- | ---: |
| Calle 100 - Marketmedios | 67,74 % |
| Portal Suba | 80,61 % |
| Portal Américas | 80,39 % |
| Banderas | 80,97 % |
| Portal El Dorado | 79,76 % |
| Universidades - CityU | 68,55 % |
| Movistar Arena | 76,37 % |
| Universidad Nacional | 69,29 % |
| Ricaurte - NQS | 80,68 % |
| Portal Usme | 79,85 % |
| Calle 72 | 69,82 % |
| Museo Nacional | 76,23 % |

## Resultados del modelo ARIMA

Se entrenó un modelo ARIMA(1,0,1) independiente para cada estación. Las
observaciones sin una predicción válida se omitieron, en lugar de asignarles
una predicción de 0. La tabla `arimas` contiene 51.840 predicciones y el
puntaje de su estación correspondiente.

| Estación | Precisión ARIMA |
| --- | ---: |
| Calle 100 - Marketmedios | 80,03 % |
| Portal Suba | 82,28 % |
| Portal Américas | 81,96 % |
| Banderas | 83,94 % |
| Portal El Dorado | 82,43 % |
| Universidades - CityU | 84,45 % |
| Movistar Arena | 83,54 % |
| Universidad Nacional | 85,02 % |
| Ricaurte - NQS | 83,30 % |
| Portal Usme | 81,16 % |
| Calle 72 | 82,17 % |
| Museo Nacional | 83,31 % |

La precisión promedio de ARIMA fue 82,80 %, superior al 77,23 % del modelo
base calculado sobre las mismas filas comparables.

## Evidencia gráfica

Las siguientes gráficas respaldan los hallazgos anteriores. Fueron generadas a
partir de `"Original Data"` y se encuentran en `docs/figures/`.

- [Demanda original de las 12 estaciones](figures/demand_by_station.png): muestra
  las diferencias de escala y los picos intradía.
- [Demanda original durante una semana](figures/demand_original_week.png): muestra
  la repetición diaria y las diferencias entre días laborales y fines de semana.
- [Demanda de un día](figures/demand_single_day.png): muestra la forma de los picos
  durante el 1 de septiembre de 2026 en hora local de Bogotá.
- [Demanda desestacionalizada durante una semana](figures/demand_deseasonalized_week.png):
  muestra los residuos después de remover el perfil diario típico.
- [Residuos con umbrales MAD](figures/demand_deseasonalized_thresholds_week.png):
  las líneas discontinuas representan los umbrales superior e inferior y el
  sombreado gris indica las regiones que los superan.
- [ACF de los residuos desestacionalizados](figures/acf_deseasonalized_by_station.png):
  muestra la autocorrelación persistente hasta un día de rezago.

## Modelo XGBoost

Se implementó un `XGBRegressor` independiente para cada estación. El modelo
utiliza los rezagos de demanda de 15, 30 y 60 minutos, además del mismo
intervalo del día anterior (`lag_96`) y de la semana anterior (`lag_672`).
También recibe variables cíclicas diarias y semanales mediante seno y coseno,
para representar correctamente la continuidad entre el final y el inicio de
un día o una semana.

La configuración seleccionada fue:

```text
n_estimators      = 400
learning_rate     = 0,05
max_leaves        = 40
max_depth         = 0
grow_policy       = lossguide
reg_lambda        = 1,0
objective         = reg:squarederror
random_state      = 42
```

En XGBoost, `max_leaves=40` con `grow_policy=lossguide` cumple el objetivo de
limitar el árbol a un máximo de 40 hojas. Se probó adicionalmente
`max_depth=3`, pero esa configuración redujo la precisión promedio de 95,17 %
a 89,62 %. Por esa razón se conservó la configuración sin límite de
profundidad explícito (`max_depth=0`).

La métrica utilizada es la Accuracy definida para el proyecto:

```text
Accuracy = 100 × max(0, 1 − WAPE)
```

Resultados finales de XGBoost:

| Estación | Accuracy |
| --- | ---: |
| 02300 | 94,91 % |
| 03000 | 95,34 % |
| 05000 | 95,34 % |
| 05100 | 95,50 % |
| 06000 | 95,07 % |
| 06111 | 95,02 % |
| 07105 | 95,10 % |
| 07107 | 95,35 % |
| 07111 | 95,38 % |
| 09000 | 94,88 % |
| 09122 | 95,34 % |
| 10009 | 94,84 % |

La Accuracy promedio fue **95,17 %**. Los modelos entrenados se guardan en
`models/xgboost/`, con un archivo independiente por estación.
