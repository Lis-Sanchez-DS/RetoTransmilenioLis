BEGIN;

ALTER TABLE "Original Data"
    ADD COLUMN IF NOT EXISTS base_predictions NUMERIC NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS base_score NUMERIC;

-- Daily seasonal-naive baseline: 96 fifteen-minute intervals = one day.
UPDATE "Original Data" AS current_row
SET base_predictions = COALESCE(previous_day.demand, 0)
FROM "Original Data" AS previous_day
WHERE previous_day.station_id = current_row.station_id
  AND previous_day.observed_at = current_row.observed_at - INTERVAL '1 day';

-- Store the station-level Accuracy on every row belonging to that station.
WITH station_scores AS (
    SELECT station_id,
           100 * GREATEST(
               0,
               1 - SUM(ABS(demand - base_predictions)) / NULLIF(SUM(ABS(demand)), 0)
           ) AS accuracy
    FROM "Original Data"
    GROUP BY station_id
)
UPDATE "Original Data" AS current_row
SET base_score = station_scores.accuracy
FROM station_scores
WHERE station_scores.station_id = current_row.station_id;

COMMIT;
