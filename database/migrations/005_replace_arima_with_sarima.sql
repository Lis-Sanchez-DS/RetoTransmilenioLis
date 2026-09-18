BEGIN;
DROP TABLE IF EXISTS arimas;
CREATE TABLE IF NOT EXISTS sarimas (
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL,
    score NUMERIC,
    order_p INTEGER NOT NULL,
    order_d INTEGER NOT NULL DEFAULT 0,
    order_q INTEGER NOT NULL DEFAULT 1,
    seasonal_period INTEGER NOT NULL DEFAULT 96,
    PRIMARY KEY (station_id, observed_at)
);
GRANT SELECT, INSERT, UPDATE, DELETE ON sarimas TO pulso_api, pulso_scheduler;
COMMIT;
