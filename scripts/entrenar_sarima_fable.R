library(DBI)
library(RPostgres)
library(dplyr)
library(tsibble)
library(fable)

con <- dbConnect(
  RPostgres::Postgres(),
  host = "127.0.0.1", port = 5433,
  dbname = Sys.getenv("POSTGRES_DB", "pulso_transmi"),
  user = Sys.getenv("POSTGRES_USER", "pulso_admin"),
  password = Sys.getenv("POSTGRES_PASSWORD")
)
on.exit(dbDisconnect(con), add = TRUE)
dir.create("models/sarimas", recursive = TRUE, showWarnings = FALSE)

datos <- dbGetQuery(con, 'SELECT station_id, observed_at, demand FROM "Original Data"') |>
  mutate(observed_at = as.POSIXct(observed_at, tz = "UTC")) |>
  as_tsibble(index = observed_at, key = station_id)

parametros <- tibble::tribble(
  ~station_id, ~p, ~d, ~q,
  "02300", 2L, 0L, 1L, "03000", 1L, 0L, 1L,
  "05000", 1L, 0L, 1L, "05100", 1L, 0L, 1L,
  "06000", 1L, 0L, 1L, "06111", 2L, 0L, 1L,
  "07105", 2L, 0L, 1L, "07107", 2L, 0L, 1L,
  "07111", 1L, 0L, 1L, "09000", 1L, 0L, 1L,
  "09122", 2L, 0L, 1L, "10009", 1L, 0L, 1L
)

ajustes <- purrr::map_dfr(split(datos, datos$station_id), function(serie) {
  id <- as.character(serie$station_id[[1]])
  par <- parametros |> filter(station_id == id)
  serie_ts <- serie |> as_tsibble(index = observed_at, key = station_id)
  modelo <- serie_ts |> model(sarima = ARIMA(
    demand ~ 1 + pdq(par$p, par$d, par$q) + PDQ(0, 0, 1, period = 96)
  ))
  saveRDS(modelo, file.path("models/sarimas", paste0("sarima_", id, ".rds")))
  augment(modelo) |>
    transmute(station_id, observed_at, actual_demand = demand,
              prediction = pmax(.fitted, 0)) |>
    filter(!is.na(prediction)) |>
    mutate(order_p = par$p, order_d = par$d, order_q = par$q,
           seasonal_period = 96L)
}) |>
  as_tibble()

puntajes <- ajustes |>
  group_by(station_id) |>
  summarise(score = 100 * max(0, 1 - sum(abs(actual_demand - prediction)) /
    sum(abs(actual_demand))), .groups = "drop")

resultado <- ajustes |>
  as_tibble() |>
  left_join(puntajes, by = "station_id") |>
  select(station_id, observed_at, actual_demand, prediction, score,
         order_p, order_d, order_q, seasonal_period)

dbExecute(con, "TRUNCATE TABLE sarimas")
dbWriteTable(con, "sarimas", resultado,
  append = TRUE, row.names = FALSE, overwrite = FALSE)

cat(sprintf("SARIMA automático entrenado para %d estaciones\n",
  n_distinct(resultado$station_id)))
cat(sprintf("Accuracy promedio: %.2f%%\n", mean(puntajes$score)))
