CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS "Original Data" (
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    demand INTEGER NOT NULL CHECK (demand >= 0),
    prediction NUMERIC,
    PRIMARY KEY (station_id, observed_at)
);

CREATE INDEX IF NOT EXISTS original_data_observed_at_idx
    ON "Original Data" (observed_at);

CREATE TABLE IF NOT EXISTS "Temp" (
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    demand INTEGER NOT NULL CHECK (demand >= 0),
    PRIMARY KEY (station_id, observed_at)
);

CREATE INDEX IF NOT EXISTS temp_observed_at_idx ON "Temp" (observed_at);

CREATE TABLE IF NOT EXISTS collector_state (
    state_key TEXT PRIMARY KEY,
    cursor_value TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ops.job_runs (
    id BIGSERIAL PRIMARY KEY,
    job_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
