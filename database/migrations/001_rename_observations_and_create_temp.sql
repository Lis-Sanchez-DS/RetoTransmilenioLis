BEGIN;

ALTER TABLE IF EXISTS observations RENAME TO "Original Data";

CREATE TABLE IF NOT EXISTS "Temp" (
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    demand INTEGER NOT NULL CHECK (demand >= 0),
    PRIMARY KEY (station_id, observed_at)
);

CREATE INDEX IF NOT EXISTS original_data_observed_at_idx
    ON "Original Data" (observed_at);
CREATE INDEX IF NOT EXISTS temp_observed_at_idx ON "Temp" (observed_at);

GRANT SELECT, INSERT, UPDATE, DELETE ON "Original Data", "Temp"
    TO pulso_api, pulso_scheduler;

COMMIT;
