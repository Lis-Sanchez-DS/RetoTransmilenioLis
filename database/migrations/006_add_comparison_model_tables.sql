BEGIN;
CREATE TABLE IF NOT EXISTS dynamic_harmonic (
    station_id TEXT NOT NULL REFERENCES stations(station_id), observed_at TIMESTAMPTZ NOT NULL,
    actual_demand INTEGER NOT NULL, prediction NUMERIC NOT NULL, score NUMERIC,
    PRIMARY KEY (station_id, observed_at)
);
CREATE TABLE IF NOT EXISTS xgboost (
    station_id TEXT NOT NULL REFERENCES stations(station_id), observed_at TIMESTAMPTZ NOT NULL,
    actual_demand INTEGER NOT NULL, prediction NUMERIC NOT NULL, score NUMERIC,
    PRIMARY KEY (station_id, observed_at)
);
GRANT SELECT, INSERT, UPDATE, DELETE ON dynamic_harmonic, xgboost TO pulso_api, pulso_scheduler;
COMMIT;
