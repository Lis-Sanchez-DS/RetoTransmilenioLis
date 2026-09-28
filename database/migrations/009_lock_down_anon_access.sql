BEGIN;

-- The `anon` role (used by the public/publishable API key - the one meant to
-- be safe to embed in a browser) had full SELECT/INSERT/UPDATE/DELETE/
-- TRUNCATE on these tables with Row Level Security disabled, i.e. no
-- restriction at all. That predates the web dashboard and is a real
-- vulnerability on its own: anyone with the anon key could already wipe
-- production data. Enabling RLS makes Postgres deny ALL access by default
-- for non-owner roles (service_role, used by the collector/submissions
-- pipeline via SUPABASE_DATABASE_URL, always bypasses RLS regardless - this
-- never affects that pipeline).
ALTER TABLE "Original Data" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "Temp" ENABLE ROW LEVEL SECURITY;
ALTER TABLE stations ENABLE ROW LEVEL SECURITY;
ALTER TABLE collector_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE baseline ENABLE ROW LEVEL SECURITY;
ALTER TABLE dynamic_harmonic ENABLE ROW LEVEL SECURITY;
ALTER TABLE sarimas ENABLE ROW LEVEL SECURITY;
ALTER TABLE xgboost ENABLE ROW LEVEL SECURITY;

-- Deliberately NO SELECT policy on any raw table for anon/authenticated -
-- RLS enabled with zero policies means the public key gets zero row-level
-- access through PostgREST's normal /rest/v1/<table> endpoints, for every
-- table, full stop. Public read access instead goes ONLY through the
-- narrow, purpose-built functions below - PostgREST exposes every function
-- as its own endpoint at /rest/v1/rpc/<function_name> - each returning
-- exactly the shape of data the dashboard needs and nothing more, none of
-- them accepting anything that could mutate data, and the row-returning
-- ones capped so a single call can't be used to pull an entire, ever-
-- growing table.
--
-- Each function is SECURITY DEFINER with search_path locked to '' and every
-- identifier schema-qualified (the standard Postgres hardening for
-- SECURITY DEFINER, which otherwise trusts the caller's search_path) - it
-- runs with the owning role's privileges, so it works regardless of the
-- RLS state on the underlying tables, while the tables themselves stay
-- completely inaccessible to anon/authenticated by any other path.

-- Station metadata (id, name, coordinates) - static reference data for the map.
CREATE FUNCTION public.api_station_list()
RETURNS TABLE (station_id text, name text, latitude double precision, longitude double precision)
LANGUAGE sql SECURITY DEFINER SET search_path = '' STABLE AS $$
    SELECT station_id, name, latitude, longitude FROM public.stations ORDER BY station_id;
$$;

-- Per-station accuracy summary (all-time / last 20 / last 4 - the drift
-- gate window), aggregated server-side so raw per-row demand/prediction
-- data is never sent to the client just to render this view.
CREATE FUNCTION public.api_station_summary()
RETURNS TABLE (
    station_id text,
    n_points bigint,
    last_observed_at timestamptz,
    accuracy_all_time double precision,
    accuracy_last_20 double precision,
    accuracy_last_4 double precision
)
LANGUAGE sql SECURITY DEFINER SET search_path = '' STABLE AS $$
    WITH combined AS (
        SELECT station_id, observed_at, demand, prediction
        FROM public."Original Data" WHERE prediction IS NOT NULL
        UNION ALL
        SELECT station_id, observed_at, demand, prediction
        FROM public."Temp" WHERE prediction IS NOT NULL
    ),
    ranked AS (
        SELECT *, row_number() OVER (PARTITION BY station_id ORDER BY observed_at DESC) AS rn
        FROM combined
    ),
    agg AS (
        SELECT
            station_id,
            count(*) AS n_points,
            max(observed_at) AS last_observed_at,
            1 - sum(abs(demand - prediction)) / greatest(sum(abs(demand)), 1) AS accuracy_all_time,
            1 - sum(abs(demand - prediction)) FILTER (WHERE rn <= 20)
                / greatest(sum(abs(demand)) FILTER (WHERE rn <= 20), 1) AS accuracy_last_20,
            1 - sum(abs(demand - prediction)) FILTER (WHERE rn <= 4)
                / greatest(sum(abs(demand)) FILTER (WHERE rn <= 4), 1) AS accuracy_last_4
        FROM ranked
        GROUP BY station_id
    )
    SELECT station_id, n_points, last_observed_at,
           greatest(accuracy_all_time, 0), greatest(accuracy_last_20, 0), greatest(accuracy_last_4, 0)
    FROM agg
    ORDER BY accuracy_all_time DESC;
$$;

-- One station's real-vs-predicted series, most recent p_limit points only -
-- hard-capped at 2000 server-side regardless of what's requested, so this
-- can't be turned into "give me the whole table" by passing a huge limit.
CREATE FUNCTION public.api_station_series(p_station_id text, p_limit int DEFAULT 500)
RETURNS TABLE (observed_at timestamptz, demand numeric, prediction numeric)
LANGUAGE sql SECURITY DEFINER SET search_path = '' STABLE AS $$
    SELECT observed_at, demand, prediction FROM (
        SELECT observed_at, demand, prediction
        FROM public."Original Data"
        WHERE station_id = p_station_id AND prediction IS NOT NULL
        UNION ALL
        SELECT observed_at, demand, prediction
        FROM public."Temp"
        WHERE station_id = p_station_id AND prediction IS NOT NULL
        ORDER BY observed_at DESC
        LIMIT LEAST(GREATEST(p_limit, 1), 2000)
    ) recent
    ORDER BY observed_at ASC;
$$;

-- Collector heartbeat (ops.job_runs, a non-public schema not exposed to
-- PostgREST by default) - most recent p_limit runs, capped. This function
-- is the only path anon/authenticated has to this data at all; the `ops`
-- schema itself is never exposed.
CREATE FUNCTION public.api_job_runs(p_limit int DEFAULT 20)
RETURNS TABLE (job_name text, status text, started_at timestamptz, finished_at timestamptz)
LANGUAGE sql SECURITY DEFINER SET search_path = '' STABLE AS $$
    SELECT job_name, status, started_at, finished_at
    FROM ops.job_runs
    ORDER BY started_at DESC
    LIMIT LEAST(GREATEST(p_limit, 1), 200);
$$;

REVOKE ALL ON FUNCTION public.api_station_list() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.api_station_summary() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.api_station_series(text, int) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.api_job_runs(int) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.api_station_list() TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.api_station_summary() TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.api_station_series(text, int) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.api_job_runs(int) TO anon, authenticated;

COMMIT;
