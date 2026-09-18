BEGIN;

ALTER TABLE "Original Data"
    DROP COLUMN IF EXISTS base_predictions,
    DROP COLUMN IF EXISTS base_score;

COMMIT;
