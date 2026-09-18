BEGIN;

CREATE TABLE IF NOT EXISTS baseline (
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL,
    score NUMERIC,
    PRIMARY KEY (station_id, observed_at)
);

CREATE TABLE IF NOT EXISTS sarimas (
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL,
    score NUMERIC,
    order_p INTEGER NOT NULL DEFAULT 1,
    order_d INTEGER NOT NULL DEFAULT 0,
    order_q INTEGER NOT NULL DEFAULT 1,
    seasonal_period INTEGER NOT NULL DEFAULT 96,
    PRIMARY KEY (station_id, observed_at)
);

DROP TABLE IF EXISTS arimas;
GRANT SELECT, INSERT, UPDATE, DELETE ON baseline, sarimas TO pulso_api, pulso_scheduler;
COMMIT;
