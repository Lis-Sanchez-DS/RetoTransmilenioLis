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

Ambos workflows verifican si hay modelos nuevos en Supabase Storage en cada
vuelta de su loop interno (`scripts/download_models.sh`), no solo al inicio
del job - así un reentrenamiento por drift llega al otro job en minutos, sin
esperar a que ese job se reinicie horas después. Esta verificación es
incremental: solo descarga los archivos cuyo `updated_at` cambió desde la
última vuelta (comparando contra un manifiesto local), no los 48 archivos
completos cada vez - antes sí lo hacía, lo que generaba ~184GB/mes de
egress y disparó un aviso de Fair Use Policy de Supabase (ver
`.claude/skills/pulso-transmi-ops/SKILL.md`).

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
  download_models.sh         wrapper que llama a download_models.py
  download_models.py         sincroniza (de forma incremental) los 48 archivos de modelo (12 estaciones x 4 horizontes) desde Supabase Storage - solo descarga los que cambiaron desde la última vuelta, comparando contra un manifiesto local

dashboard/app.py               panel local (Streamlit) de solo lectura sobre Supabase
web/                            panel estático (HTML/CSS/JS, sin build ni framework) desplegado en Vercel - ver "Panel de monitoreo" más abajo

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

Features de cada modelo: 4 rezagos de demanda (`lag_1`, `lag_2`, `lag_4`,
`lag_96` - 15 min, 30 min, 1h y 1 día antes, ajustados por el horizonte
según la fórmula de anclaje de arriba) + 4 variables cíclicas de calendario
(seno/coseno diario y semanal, `app/features.py`) = **8 features, iguales
para las 12 estaciones y los 4 horizontes, sin excepciones** (ver punto 7).

### 7. `lag_672` (una semana atrás): de una excepción por estación a eliminado en todas partes

`lag_672` ancla cada modelo a "qué pasó a esta misma hora la semana
pasada" - un buen default, pero activamente contraproducente mientras el
patrón de demanda de una estación está genuinamente cambiando (los colapsos
de varias horas de 05100 son el caso más claro). Se evaluó primero
(2026-09-26) sobre dos folds cronológicos independientes (70-85% del
histórico, y el último 15%), por estación y horizonte, comparando el
baseline de 9 features contra (a) quitar `lag_672` del todo y (b) restarle
peso suavemente (`feature_weights`, peso 0.2, con `colsample_bynode=0.8`).
Un candidato solo se adoptaba si superaba al baseline por >= 0,3 pp en
AMBOS folds - esa primera pasada dejó solo 7 pares estación/horizonte con
`lag_672` eliminado y 1 con peso suave, y el resto sin cambios.

**Re-test con más datos acumulados (2026-09-27):** repetir exactamente la
misma prueba de dos folds, ahora con más historia, mostró dos cosas:

1. El peso suave (probado con varios pesos: 0.2, 0.4, 0.6) **nunca superó a
   la eliminación total** en ningún par donde ambos se probaron - siempre
   quedaba por debajo, a veces por más de 1-2 pp (ej. 05100 h1: peso suave
   80,3-84,9% vs eliminación total 82,4-85,7%).
2. Con la historia más grande, **eliminar `lag_672` del todo ahora superaba
   el baseline en ~15 pares adicionales** a los 7 ya desplegados - no solo
   en el pequeño grupo original.

Dado que (a) el peso suave no ganó en ningún caso probado y (b) la lista de
pares que se benefician de eliminar `lag_672` seguía creciendo con más
datos, se simplificó la arquitectura del todo: **`lag_672` se quitó de
`LAGS` para las 12 estaciones y los 4 horizontes**, en vez de mantener una
lista de excepciones cada vez más larga. Verificado con un split held-out
del último 15% (el estándar del proyecto, no solo los dos folds del
re-test): **83,77% -> 84,37% (+0,60 pp), 11/12 estaciones mejoran** - 05100
es la que más gana (+2,62 pp promedio, hasta +2,93 pp en h2), reflejando su
colapso de demanda ya conocido; solo `09000` retrocede levemente (-0,11 pp
promedio) como costo aceptable de una sola arquitectura uniforme.

Esto también elimina una clase entera de bug de raíz: antes, `LAGS` variaba
por par estación/horizonte (`lags_for()` en `app/drift.py` decidía cuántas
features esperaba cada `.joblib` guardado), y ese código de inferencia tenía
que consultarse en cada lugar donde se llamaba a `.predict()`
(`collector.py` Y `submit_xgboost.py`) - olvidar uno de los dos es
exactamente lo que causó el incidente del punto 10. Con un solo `LAGS`
compartido por todos, `lags_for()` y toda la maquinaria de excepciones por
par (`NO_LAG672_MODELS`, `SOFT_DEEMPHASIZE_LAG672_MODELS`,
`_feature_weight_kwargs`) se eliminaron de `app/drift.py`; `collector.py` y
`submit_xgboost.py` ahora importan `LAGS` directamente de `app.drift` (una
sola fuente de verdad) en vez de mantener su propia copia local del tuple.

Antes de desplegar, se verificó localmente (no solo con la suite de tests
mockeados): los 48 modelos reentrenados con el nuevo `LAGS` de 4 rezagos se
confirmaron uno por uno en `n_features_in_ == 8`, y tanto
`predict_records()` como `predict_cycle_targets()` se corrieron de punta a
punta contra datos reales de la base de datos (sin mocks) antes de subir
ningún modelo a Supabase Storage.

### 8. Corrección de sesgo por EWMA en inferencia (no reentrenamiento)

Separado del reentrenamiento: un promedio móvil exponencial causal (sin
mirar al futuro) del residuo `(real - predicho)` propio de cada estación,
calculado en cada submission sobre todo el historial real de pares
predicción/real (`"Original Data"` + `"Temp"`) y sumado sobre la predicción
cruda justo antes de enviarla (`app/submit_xgboost.py`). Existe porque un
sesgo real y sostenido (el colapso de varias horas de 05100) puede tardar
horas en corregirse vía un reentrenamiento por drift - el EWMA reacciona
dentro de un solo ciclo del collector, sin necesitar reentrenar. Nunca toca
lo que se guarda como `prediction` en `"Temp"`/`"Original Data"` - eso sigue
siendo la salida cruda del modelo, para que la señal de drift y el propio
historial de residuos del EWMA no se retroalimenten entre sí.

`bias_t = alpha * residuo_t + (1 - alpha) * bias_{t-1}`, luego
`corrección = damping * bias_t`, sumada a la predicción cruda (con piso en
0). La configuración (`DEFAULT_EWMA_PARAMS = {alpha: 0.2, damping: 0.5}`) se
eligió igual que `STATION_MODEL_PARAMS`: una búsqueda de grid por estación
sobre `alpha x damping`, maximizando accuracy sobre **todo** el historial de
esa estación, no solo su tramo más reciente/ruidoso. Un candidato necesita
superar al default compartido por `MIN_EWMA_IMPROVEMENT_PP = 0.5` pp para
que una estación se quede con su propia configuración. En la búsqueda de
2026-09-26 (repetida dos veces, con datasets distintos y más grandes,
incluyendo el colapso real de 05100), **ninguna estación superó ese
margen** - ni siquiera 05100, cuyo colapso parecía necesitar una
configuración mucho más agresiva visto de forma aislada sobre esa ventana de
~24 filas, pero la ganancia casi desaparecía al juzgarla contra todo el
dataset. Por ahora las 12 estaciones usan el default compartido;
`STATION_EWMA_PARAMS` (vacío hoy) existe para que una futura excepción por
estación sea un cambio de una línea en cuanto aparezca un margen real.

### 9. `week_sin`/`week_cos`: misma pregunta que `lag_672`, resultado distinto

`week_sin`/`week_cos` (`app/features.py`) también codifican una periodicidad
de una semana, así que cabía preguntarse si sufrían el mismo problema que
`lag_672`. Estructuralmente no: `lag_672` es un **valor** de demanda de un
timestamp específico de hace una semana (un solo punto anómalo se filtra
directo a la predicción), mientras que `week_sin`/`week_cos` no cargan
ningún valor de demanda - solo dicen "es martes 8:15am", así que el modelo
aprende un patrón **agregado** sobre todos los martes 8:15am del histórico,
no un punto aislado. Se corrió la misma metodología de dos folds
independientes (baseline vs. quitar `week_sin`+`week_cos` del todo vs.
bajarles el peso suavemente, por estación y por horizonte): **ningún par
estación/horizonte fue robusto** - ninguno superó el margen de 0,3 pp en
ambos folds con el mismo candidato; las mejoras eran pequeñas y cambiaban
de fold a fold, la misma firma de ruido que el resto de resultados nulos de
este proyecto (peso por recencia, EWMA por estación). No se hizo ningún
cambio.

### 10. Incidente: `predict_records` no se actualizó junto con `lags_for`

Al desplegar el cambio del punto 7, `app/collector.py`'s `predict_records()`
(el nowcast de un paso usado para monitorear drift, no las submissions) se
quedó armando el vector de features con el tupla fija `LAGS` (9 rezagos) en
vez de consultar `lags_for(station_id, 1)`. En cuanto el modelo h1 de 09122
(uno de los que ahora usan 8 features) se redesplegó, cada llamada fallaba
con `ValueError: Feature shape mismatch, expected: 8, got 9` - y como 09122
aparece en cada tanda de registros, esto tumbó `submissions.yml` tras sus 3
fallos consecutivos permitidos. Corregido usando `lags_for()` también ahí;
confirmado en producción vía la tabla de heartbeat del collector
(`ops.job_runs`), que mostró una corrida exitosa después del fix. Lección:
`lags_for()` tiene más de un punto de uso (`submit_xgboost.py` Y
`collector.py`) - al agregar una arquitectura por `(estación, horizonte)`,
buscar con `grep -rn "\.predict(" app/` cada lugar donde se llama a un
modelo guardado, no solo el que ya se estaba editando.

**Actualización (2026-09-27):** con `lag_672` eliminado de `LAGS` para
absolutamente todos los pares (punto 7), esta clase de bug ya no puede
volver a ocurrir - no hay `lags_for()` ni ninguna otra fuente de un conteo
de features distinto entre modelos; los 48 comparten el mismo `LAGS`.

## Reducción de tráfico a Supabase (2026-09-27)

Varias consultas traían muchísimos más datos de los que en realidad usaban,
generando un consumo de egress muy por encima de lo necesario:

- `submit_current_cycle()` traía **toda** `"Original Data"` (~59.000 filas y
  creciendo) más toda `"Temp"`, cada ~5 min mientras hay un ciclo abierto,
  solo para resolver los rezagos de cada estación. Ahora filtra por
  `station_id = ANY(...) AND observed_at = ANY(...)` con los timestamps
  exactos que puede necesitar - verificado en vivo: 54 filas en vez de
  58.992 para el mismo cutoff real.
- `predict_records()` (`app/collector.py`) hacía lo mismo por estación -
  traía su historial completo para leer un puñado de rezagos. Mismo fix.
- `station_ewma_bias()` traía el historial completo de cada estación para
  una media móvil exponencial cuyo peso decae geométricamente - con
  `alpha=0.2` cualquier punto más allá de ~93 filas atrás no cambia el
  resultado en ninguna cifra significativa (`_ewma_lookback_rows()` deriva
  ese límite del propio `alpha` de la estación, no un número fijo).
  Verificado en vivo: coincide con el cálculo sobre el historial completo
  hasta ~9 cifras significativas.

Las tres consultas ahora escalan con el tamaño de la respuesta (una
estación, un puñado de timestamps), no con el tamaño del histórico
acumulado, que seguirá creciendo mientras dure la competencia.

Por separado, `check_and_retrain()` ahora exige `MIN_NEW_FOR_RETRAIN=20`
filas nuevas pendientes en `"Temp"` por estación antes de reentrenarla (ver
la sección de drift más abajo) - antes de este cambio, `has_pending_data()`
(un chequeo global) combinado con una estación ya por debajo del umbral
significaba que 1-2 filas nuevas bastaban para disparar un reentrenamiento
completo (4 horizontes, 4 subidas a Supabase) en casi cada vuelta del
collector, la causa directa del pico de egress que motivó esta sección.

## Detección de drift y reentrenamiento (`app/drift.py`)

- **Disparador**: la accuracy de una estación sobre sus últimos
  `RECENT_CHECKS=4` puntos con predicción real registrada (ventana por
  *conteo*, no por tiempo - así el disparador no se queda ciego si el stream
  se ralentiza o se detiene, y reacciona a cómo está funcionando el modelo
  ahora mismo en vez de diluirse con historia larga) cae por debajo de
  `ACCURACY_THRESHOLD=0.85`.
- **Gate previo**: el chequeo de drift solo corre si `"Temp"` tiene alguna
  fila pendiente (`has_pending_data()`), no si el collector insertó algo
  *en esa vuelta específica*. Con el gate anterior ("solo si esta vuelta
  insertó algo nuevo"), un feed estancado dejaba los datos de una estación
  ya drifteada esperando en `"Temp"` para siempre sin que ninguna vuelta
  futura volviera a insertar nada - bloqueando el reentrenamiento
  indefinidamente aunque los datos que lo dispararían ya estuvieran ahí.
  Revisar "¿hay algo en Temp?" en vez de "¿llegó algo nuevo ahora?" sigue
  evitando revisiones redundantes cuando Temp está genuinamente vacía (justo
  después de que todas las estaciones se reentrenaron y promovieron), pero
  ya no se queda ciego ante datos pendientes sin procesar.
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

## Panel de monitoreo (dashboard)

**Panel en vivo (no requiere nada local): https://web-sigma-lac-7j9zw6955x.vercel.app**

Hay dos paneles de solo lectura, ambos leyendo directamente de Supabase y
sin ningún efecto sobre el pipeline de collector/submissions:

- **`web/`** - panel estático (HTML/CSS/JS plano, sin build ni framework),
  desplegado en Vercel en la URL de arriba. Es el panel "de referencia":
  público, siempre disponible, sin depender de la máquina de nadie.
- **`dashboard/app.py`** - la misma información en una app de Streamlit,
  para correr localmente (`streamlit run dashboard/app.py`) durante
  desarrollo. Ninguno de los dos escribe en la base de datos.

Ambos muestran: accuracy por estación (histórico, últimos 20, últimos 4 -
la misma ventana que usa el gate de drift), un mapa interactivo de Bogotá
con las coordenadas reales de la tabla `stations` (clic en una estación la
selecciona), la serie real-vs-predicha y su tendencia de accuracy, el
heartbeat del collector y la tabla de estaciones elegibles para
reentrenamiento. La paleta de colores (rojo/amarillo/ámbar) se explica en
`dashboard/app.py` - tomada del logo oficial de TransMilenio en Wikimedia
Commons, no de un manual de marca con hex documentados (no se encontró
ninguno público).

### Arquitectura del panel web: por qué necesitó arreglar una vulnerabilidad primero

`web/` no tiene backend propio - el navegador llama a Supabase
directamente con la llave pública/`anon` (`web/config.js`, segura de
exponer por diseño, es el equivalente a la llave anónima de cualquier
proyecto Supabase). Al construir esto se encontró que esa llave pública
**ya tenía SELECT/INSERT/UPDATE/DELETE/TRUNCATE completos** sobre
`"Original Data"`, `"Temp"`, `stations` y otras tablas, con Row Level
Security desactivado - un problema previo y sin relación con el
dashboard: cualquiera con esa llave ya podía borrar la base de producción
antes de que existiera este panel.

Arreglado en `database/migrations/009_lock_down_anon_access.sql`:

- RLS activado en todas las tablas antes expuestas
  (`"Original Data"`, `"Temp"`, `stations`, `collector_state`, `baseline`,
  `dynamic_harmonic`, `sarimas`, `xgboost`), **sin ninguna política** sobre
  las tablas crudas - la llave pública ya no puede leer ni escribir nada
  por la vía normal `/rest/v1/<tabla>` de PostgREST, para ninguna tabla.
  Esto nunca afecta al pipeline en GitHub Actions: la conexión de
  `SUPABASE_DATABASE_URL` usa el rol `service_role`, que siempre ignora RLS.
- Todo el acceso de lectura pasa por **4 funciones SQL específicas**
  (`SECURITY DEFINER`, `search_path` fijo en `''`, cada identificador
  calificado por esquema - el endurecimiento estándar para este tipo de
  función), expuestas por PostgREST como endpoints propios en
  `/rest/v1/rpc/<función>`:
  - `api_station_list()` - metadata de estaciones (id, nombre, lat/lon).
  - `api_station_summary()` - accuracy agregada por estación (todo el
    historial, últimos 20, últimos 4); nunca expone filas crudas de
    demanda/predicción para esta vista, el agregado se calcula en SQL.
  - `api_station_series(p_station_id, p_limit)` - serie real-vs-predicha de
    una estación, con `p_limit` topado en 2000 filas del lado del servidor
    (no importa qué límite pida el cliente) para que no se pueda usar para
    extraer una tabla entera y cada vez más grande de una sola llamada.
  - `api_job_runs(p_limit)` - heartbeat del collector; evita exponer el
    esquema `ops` completo (no está en la lista de esquemas expuestos de
    Supabase por defecto).
- Verificado en vivo con la llave pública real antes de dar el cambio por
  bueno: lectura directa de tabla → `[]` vacío, escritura directa →
  rechazada explícitamente por RLS (`"new row violates row-level security
  policy"`), los 4 endpoints RPC → datos reales.

Si se agrega una nueva vista al panel que necesite otro dato de Supabase,
el patrón a seguir es este: una función `SECURITY DEFINER` nueva y
específica en una migración, nunca una política `SELECT`/`GRANT` amplia
sobre una tabla cruda para `anon`.

### Un bug real que vale la pena recordar: colisión de nombre con `window.supabase`

`web/app.js` declaraba `const supabase = window.supabase.createClient(...)`.
El script de `@supabase/supabase-js` (cargado por `<script>` antes que
`app.js`) ya crea su propio global `window.supabase` - redeclarar ese mismo
nombre con `const` en el scope superior de otro `<script>` lanza
`Identifier 'supabase' has already been declared` y mata el script
completo antes de que `main()` llegue a ejecutarse. El síntoma en el
navegador era el panel quedándose en "Cargando datos..." para siempre: ni
siquiera el manejo de errores de la propia página llegaba a correr, porque
el error ocurre antes de que cualquier función definida en el archivo
exista. Corregido renombrando la variable local a `supabaseClient`.

Este bug pasó una revisión manual del código y solo se detectó al probarlo
con un navegador real (`curl` nunca lo habría mostrado, ya que la página
HTML se sirve bien - el error es de ejecución de JavaScript, no de red).
Antes de dar por buena cualquier página que dependa de scripts de terceros
cargados por `<script>` global (no ES modules), conviene evitar nombres de
variable que puedan coincidir con lo que esa librería expone en `window`.

## Estado del feed de datos: un reloj virtual lento, no un estancamiento permanente

**Corrección (2026-09-26):** una versión anterior de este documento decía
que el feed estaba "estancado desde el 2026-09-13" - esa idea era
incorrecta. Al revisar la API en vivo durante varias horas, `data_cutoff`
sí avanzaba (por ejemplo, de 03:00 a 08:00 en una tarde) - el stream de la
competencia es un **reloj virtual que reproduce datos históricos a un ritmo
cercano al tiempo real, pero crónicamente ~12 días detrás del "ahora"
real**, no un feed que murió en una fecha fija. En la última revisión en
vivo (2026-09-26), el punto real más reciente `(demanda, predicción)` de
cada una de las 12 estaciones era `2026-09-14 11:30:00 UTC` - confirmando
que el desfase de ~12 días sigue siendo el modelo mental correcto.

Implicación práctica: un cambio de código o de modelo desplegado "hoy" no
se puede validar contra resultados reales revelados hasta que el reloj
virtual avance más allá de ese momento - no hay forma de comprobar en vivo
que un fix funcionó el mismo día que se despliega, solo mediante
reentrenamiento/pruebas offline como las de este documento.
