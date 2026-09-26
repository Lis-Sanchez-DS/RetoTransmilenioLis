# Pulso TransMi: pipeline de predicción de demanda

Sistema de MLOps para la competencia Pulso TransMi: cada ciclo de pronóstico
la API pide predicciones de demanda a +15/+30/+45/+60 minutos para 12
estaciones de transporte. El sistema **corre enteramente en GitHub Actions +
Supabase** - no hay ningún componente local que deba ejecutarse a mano para
mantener el pipeline vivo. Este documento resume la arquitectura, cómo están
distribuidas las piezas del repo, y el camino recorrido (con evidencia) hasta
los modelos finales en producción.

## Arquitectura general

Dos workflows de GitHub Actions de larga duración, cada uno con su propio
loop interno:

- **`.github/workflows/collector.yml`** corre `python -m app.collector` cada
  ~30 min: descarga observaciones nuevas de la API de Pulso hacia Supabase,
  predice cada punto nuevo con el modelo vigente (para monitorear drift) y
  evalúa si alguna estación necesita reentrenarse (`app/drift.py`).
- **`.github/workflows/submissions.yml`** corre `python -m app.submit_xgboost`
  cada ~5 min: descubre el ciclo de pronóstico abierto y envía las 4
  predicciones por estación.

Ambos workflows re-descargan los modelos desde Supabase Storage en cada
vuelta de su loop interno (`scripts/download_models.sh`), no solo al inicio
del job - así un reentrenamiento por drift llega al otro job en minutos, sin
esperar a que ese job se reinicie horas después.

`GET /v1/forecast-cycles/current` con 404 (`no_open_cycle`) es un estado
normal de "entre ventanas", no un error. Un modelo faltante o sin historia
suficiente para una estación **no aborta el ciclo completo**: esa estación se
omite y las demás se envían igual.

### Por qué los workflows hacen su propio loop en vez de depender de cron

El trigger `schedule` de GitHub Actions no tiene SLA: puede atrasarse horas
bajo carga. Por eso cada job hace polling interno en un `while` de bash
durante ~5.75h (`timeout-minutes: 355`), combinado con
`concurrency: {group: <nombre>, cancel-in-progress: false}`, que encola (no
descarta) cualquier disparo de `schedule` que llegue mientras el loop sigue
corriendo. Así, en cuanto un loop de ~5.75h termina, el siguiente ya está en
cola y arranca de inmediato - la cadena se sostiene sola una vez iniciada con
un `workflow_dispatch` manual. El collector duerme 30 min entre vueltas; las
submissions, 5 min.

Un fallo transitorio de red hacia la API de Pulso o Supabase Storage
(`app/net.py`, con reintentos exponenciales 5s/10s/20s) hace que el proceso
salga con código 75; el loop de bash interpreta ese código como "el servidor
está caído, no es un bug nuestro" y duerme 5 min sin contar como fallo real.
3 fallos reales consecutivos sí detienen el job para que quede visible en la
pestaña Actions.

### Confiabilidad del collector (heartbeat)

La detección de drift solo corre dentro de `collector.yml`. Si ese workflow
se deshabilita, se cancela o falla en silencio, nada más lo notaría por sí
solo - `submissions.yml` seguiría enviando contra un modelo cada vez más
viejo indefinidamente. Por eso cada corrida exitosa del collector escribe un
renglón en `ops.job_runs` (`app/health.py`); `submit_xgboost.py` revisa la
frescura de ese renglón *después* de que su propia submission ya salió (para
que un collector atrasado nunca bloquee un envío que de otro modo hubiera
funcionado), y lanza `CollectorStale` si la última corrida exitosa tiene más
de 90 minutos (3x el ciclo del collector) - eso hace fallar el job y se ve en
Actions.

## Distribución del repositorio

```
app/
  collector.py        recolecta observaciones nuevas, las predice y las guarda en "Temp"
  drift.py             detección de drift + reentrenamiento por estación y horizonte
  submit_xgboost.py     descubre el ciclo abierto y envía las 4 predicciones por estación
  features.py           codificación temporal compartida entre entrenamiento e inferencia
  health.py              heartbeat del collector (tabla ops.job_runs)
  net.py                  reintentos de red para fallos transitorios
  db.py                    conexión a Supabase Postgres
  load_original.py         carga única del histórico inicial (45 días) en "Original Data"
  scheduler.py / prediction_scheduler.py   loops usados para ejecución local/Docker (no usados en producción)
  train_comparison_models.py               script exploratorio para comparar variantes de modelo

.github/workflows/
  collector.yml            workflow de recolección + drift (ver arriba)
  submissions.yml           workflow de envío de pronósticos (ver arriba)
  restore_original_data.yml  workflow manual para restaurar el histórico original desde el seed

database/
  init/001_schema.sql        esquema completo, usado solo para instalaciones nuevas desde cero
  migrations/*.sql            migraciones aplicadas incrementalmente sobre la base real en Supabase

scripts/
  download_models.sh         descarga los 48 archivos de modelo (12 estaciones x 4 horizontes) desde Supabase Storage

models/xgboost/               modelos entrenados, excluidos de Git por tamaño; viven en Supabase Storage
docs/figures/                  gráficas del análisis exploratorio
tests/                          suite de pytest, con toda la conexión a la BD simulada (sin Supabase real)
```

### Modelo de datos (Supabase Postgres, vía `app/db.py`)

- **`"Original Data"`**: el histórico sembrado una sola vez por
  `app/load_original.py` (45 días, 12 estaciones, cada 15 min), más todo lo
  que un reentrenamiento por drift va incorporando desde `"Temp"`. Nunca se
  borra nada de aquí - la historia solo crece.
- **`"Temp"`**: tabla de aterrizaje donde cae *cada* observación nueva del
  stream, junto con la predicción que el modelo vigente le dio (usada para
  monitorear drift). Solo se vacía cuando una estación dispara un
  reentrenamiento, momento en el que sus filas se incorporan a
  `"Original Data"` (incluida la columna `prediction`, para que el chequeo de
  accuracy no pierda esos puntos al moverse de tabla).
- **`collector_state`**: cursor de paginación del stream de observaciones
  (una sola fila, `state_key='observations'`).
- **`ops.job_runs`**: bitácora de heartbeat del collector.

**Invariante importante**: cualquier código que arme historia de predicción
(búsquedas de rezagos) debe leer **ambas** tablas, nunca solo una. Los datos
más recientes de una estación que no ha tenido drift viven únicamente en
`"Temp"` - `"Original Data"` solo avanza vía reentrenamiento.

## Métrica de precisión

Definición propia del proyecto (no un MAPE genérico):

```
WAPE     = sum(|real - predicho|) / max(sum(|real|), 1)
Accuracy = max(0, 1 - WAPE)          # por estación
```

Las accuracies por estación se promedian para una cifra general. Esta misma
fórmula se usa en el entrenamiento, en la evaluación exploratoria y en la
detección de drift.

## Cómo se llegó a los modelos finales

### 1. Análisis exploratorio inicial

Sobre las 51.840 observaciones originales (12 estaciones x 4.320 puntos,
cada 15 min, del 26 de julio al 9 de septiembre de 2026) se encontró:

- Estacionalidad diaria marcada, con picos recurrentes en la mañana y la
  tarde, y un patrón distinto entre días laborales y fines de semana
  (estacionalidad semanal).
- Al desestacionalizar restando la mediana de demanda por intervalo de 15
  min (96 intervalos/día), los residuos seguían mostrando secuencias de
  signo, especialmente en periodos inusuales y fines de semana - la
  estacionalidad diaria por sí sola no agota la estructura predecible.
- La ACF de esos residuos hasta el rezago 96 (un día) mostró autocorrelación
  positiva significativa en la mayoría de estaciones, más fuerte en
  Universidades - CityU y Universidad Nacional. Esto motivó usar rezagos de
  demanda y no solo variables de calendario.
- Valores atípicos identificados con una regla MAD robusta por estación
  (`z = (residuo - mediana) / (1,4826 x MAD)`, atípico si `|z| > 3,5`) - un
  atípico no es necesariamente un error, puede ser un evento real.

Gráficas en `docs/figures/`.

### 2. Modelo base y ARIMA (referencia)

- **Modelo base** (naive): demanda de la misma estación exactamente un día
  antes (`lag_96`). Accuracy general: **75,86 %**.
- **ARIMA(1,0,1)** independiente por estación: **82,80 %** en promedio sobre
  las mismas filas comparables, superando claramente al modelo base pero por
  debajo de XGBoost.

### 3. XGBoost por estación, recursivo (primera versión en producción)

Un `XGBRegressor` independiente por estación, con rezagos de demanda de
15/30/60 min, un día antes (`lag_96`) y una semana antes (`lag_672`), más
variables cíclicas (seno/coseno) diarias y semanales para representar la
continuidad entre el final y el inicio de un día o semana.

Hiperparámetros iniciales (un solo config compartido por todas las
estaciones): `n_estimators=400, learning_rate=0.05, max_leaves=40,
max_depth=0, grow_policy=lossguide, reg_lambda=1.0`. Se probó también
`max_depth=3` explícito, que bajó la accuracy promedio de 95,17 % a 89,62 %
- por eso se mantuvo `max_depth=0` con el límite de hojas vía
`grow_policy=lossguide`.

Este modelo predecía los 4 horizontes de forma **recursiva**: la predicción
de +15 min se usaba como si fuera dato real para calcular los rezagos de
+30 min, y así sucesivamente - un enfoque con un problema de fondo: el error
se acumula de un horizonte al siguiente.

### 4. Ingeniería de features: qué se probó y qué no ayudó

Con el feed de la API ya estancado en `2026-09-13` (ver más abajo), varias
extensiones de features se evaluaron sobre datos históricos usando el mismo
split cronológico (nunca aleatorio) para no filtrar información del futuro:

- **Medias/desviaciones móviles y diferencias de 15 min** (`rolling_mean`,
  `rolling_std`, `diff_1`): no mostraron mejora consistente sobre el modelo
  base de rezagos + calendario.
- **Promedios de día/semana alrededor de `lag_96`/`lag_672` (±30 min) y
  diferencia horaria** en vez de diferencia de 15 min: tampoco superaron al
  set de features original de forma consistente.
- **Features de anomalía causales** (sin fuga de información): un z-score
  robusto por fila `(demanda - mediana_causal) / MAD_causal`, calculado por
  slot de 15 min del día usando solo observaciones estrictamente anteriores
  del mismo slot, leído en 3 rezagos (observación anterior, ~1 día antes,
  ~1 semana antes) como 3 features adicionales. Evaluado sobre el 20% final
  de cada estación como test: **+0,07 pp** de mejora promedio (86,02 % ->
  86,09 %), con 8/12 estaciones ganando pero ninguna por más de 0,32 pp -
  dentro del ruido, no se llevó a producción.
- **Bagging/ensembling**: habilitar `subsample=0.8` y `colsample_bytree=0.8`
  (por defecto ambos están en 1.0, lo que significa que sin esto no hay
  ninguna aleatoriedad real en el entrenamiento - `random_state` por sí solo
  no cambia nada) y promediar 8 semillas por estación: **+0,08 pp** de mejora
  promedio, 8/12 estaciones ganando. También dentro del ruido frente al
  costo de entrenar 8x más modelos - no se llevó a producción.
- **Búsqueda de semilla (`random_state`) sola**, sin tocar el resto de
  hiperparámetros: resultado plano, `delta=+0.00pp` en la gran mayoría de
  estaciones evaluadas - confirma que sin `subsample`/`colsample_bytree` < 1
  no existe aleatoriedad que una semilla distinta pueda aprovechar.

Conclusión de esta fase: dado el tamaño y la regularidad de los datos
disponibles, ni las features de anomalía ni el ensembling superan de forma
significativa al modelo de rezagos + calendario ya afinado. Lo que sí generó
una mejora real y reproducible fue afinar los hiperparámetros por estación
(punto 5) y separar los 4 horizontes en modelos independientes (punto 6).

### 5. Hiperparámetros por estación (no un config global)

Se hizo una búsqueda de hiperparámetros por estación con validación cruzada
walk-forward (ventana expansiva) **estrictamente dentro del split de
entrenamiento** - nunca tocando el conjunto de prueba reservado - para evitar
sobreajuste/fuga en un problema de series de tiempo. Cada estación conservó
su propia mejor combinación (`learning_rate`, `n_estimators`, `max_leaves`,
`reg_lambda`) en vez de forzar un config único, porque un config compartido
rindió peor que dejar que cada estación se quedara con su propio óptimo.
Resultado: accuracy media sobre datos de prueba retenidos subió de 85,07 % a
85,37 % (8/12 estaciones mejoraron). Los hiperparámetros vigentes por
estación están en `STATION_MODEL_PARAMS` (`app/drift.py`).

(Antes de esto se probó también una búsqueda de grid de hiperparámetros
*dentro* de cada reentrenamiento por drift, con `TimeSeriesSplit`. Se revirtió:
el ruido entre folds de una misma configuración, ~2 pp, superaba por mucho
la diferencia entre las 8 configuraciones candidatas, ~0,4 pp, así que el
modelo desplegado terminaba re-sorteándose cada 30 minutos a partir de ruido
en vez de converger. La búsqueda de hiperparámetros ahora es un proceso
aparte, offline, no algo que corre en cada reentrenamiento en producción.)

### 6. Modelos directos por horizonte (arquitectura final)

El enfoque recursivo del punto 3 se reemplazó por **4 modelos
independientes por estación**, uno por cada horizonte (+15/+30/+45/+60 min),
todos anclados al mismo punto de referencia real (`data_cutoff`): el feature
`lag_k` de un modelo de horizonte `h` siempre lee `h + k - 1` pasos atrás
desde el punto a predecir, lo que equivale exactamente al mismo timestamp
real y ya observado sin importar el horizonte (`data_cutoff - 15*(k-1)`
minutos) - nunca la predicción de otro horizonte. Así el error deja de
poder acumularse de un horizonte al siguiente.

Impacto medido específicamente en +60 min (`h4`, el horizonte más sensible a
la acumulación de error en el enfoque recursivo): mejora de +0,2 a +0,3 pp
frente al viejo modelo recursivo - una mejora real pero más modesta que la
de afinar hiperparámetros por estación.

Esto implica 12 estaciones x 4 horizontes = **48 modelos** en producción,
todos guardados en Supabase Storage como
`models/xgboost/xgboost_{estación}_h{horizonte}.joblib` y descargados en cada
vuelta de los loops de `collector.yml`/`submissions.yml`.

### Configuración final por estación

Todas las estaciones comparten la arquitectura base:

```
objective     = reg:squarederror
grow_policy   = lossguide
max_depth     = 0
random_state  = 42
```

Y cada una conserva su propio `learning_rate`, `n_estimators`, `max_leaves`
y `reg_lambda`, elegidos por la búsqueda walk-forward del punto 5
(`STATION_MODEL_PARAMS` en `app/drift.py`; una estación fuera de ese set cae
al default `learning_rate=0.05, n_estimators=400, max_leaves=40,
reg_lambda=1.0`).

Features de cada modelo: 5 rezagos de demanda (`lag_1`, `lag_2`, `lag_4`,
`lag_96`, `lag_672` - 15 min, 30 min, 1h, 1 día y 1 semana antes, ajustados
por el horizonte según la fórmula de anclaje de arriba) + 4 variables
cíclicas de calendario (seno/coseno diario y semanal, `app/features.py`).

## Detección de drift y reentrenamiento (`app/drift.py`)

- **Disparador**: la accuracy de una estación sobre sus últimos
  `RECENT_CHECKS=4` puntos con predicción real registrada (ventana por
  *conteo*, no por tiempo - así el disparador no se queda ciego si el stream
  se ralentiza o se detiene, y reacciona a cómo está funcionando el modelo
  ahora mismo en vez de diluirse con historia larga) cae por debajo de
  `ACCURACY_THRESHOLD=0.85`.
- **Gate previo**: el chequeo de drift solo corre si el collector insertó
  observaciones genuinamente nuevas en esa vuelta (no simplemente si la API
  devolvió registros, que pueden ser puramente duplicados durante un
  estancamiento del feed upstream). Esto evita reentrenar 48 modelos cada 30
  minutos sin necesidad cuando no hay datos nuevos.
- **Reentrenamiento**: siempre conserva todo el histórico - los puntos
  nuevos de `"Temp"` se incorporan a `"Original Data"` y nunca se descarta
  nada (una comparación directa mostró que un modelo con todo el histórico
  predice mejor que uno con una ventana recortada, en 12/12 estaciones; antes
  hubo una prueba de Kolmogorov-Smirnov para decidir esto caso por caso, y
  luego un recorte 1:1 fijo, ambos removidos el 2026-09-24).
- Al disparar, se reentrenan y republican **los 4 horizontes** de esa
  estación, no solo uno.

## Configuración local (solo para desarrollo/pruebas)

```env
SUPABASE_DATABASE_URL=postgresql://...
SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=...
PULSO_API_KEY=ptm_live_...
```

No subir `.env` al repositorio. En producción estos valores viven como
secretos del repositorio de GitHub Actions y nunca se imprimen ni registran
en logs.

```bash
pytest tests/
```

Toda la suite de pruebas simula el acceso a la base de datos (sin conexión
real a Supabase) - ver `tests/test_drift.py`, `tests/test_collector.py`,
`tests/test_submit_xgboost.py`, `tests/test_health.py`.

## Estado del feed de datos

Desde 2026-09-13 el stream de observaciones de la API de Pulso está
estancado (`observed_at` no avanza) aunque `released_at`/`server_time` sí lo
hacen con normalidad - un problema del lado del servidor de la competencia,
no del pipeline. El gate de "solo revisar drift si hay datos nuevos" existe
en parte para que este estancamiento no dispare reentrenamientos
innecesarios mientras persista.
