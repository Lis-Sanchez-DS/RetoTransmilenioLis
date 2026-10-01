"""Migra el proyecto a otra base de Supabase (otra cuenta).

Lee las credenciales NUEVAS desde `.env` (NEW_SUPABASE_*) y las ANTIGUAS desde
las SUPABASE_* de siempre. Pasos (cada uno es idempotente):

  schema   crea tablas, índices, RLS y las funciones api_* en la base nueva
  data     copia las filas (COPY binario, en orden de FK) y verifica conteos
  models   crea el bucket privado `models` y sube los .joblib de models/xgboost/
  config   reescribe web/config.js con la URL y la publishable key nuevas
  secrets  actualiza los secrets de GitHub Actions con `gh secret set`

Uso:  python -m scripts.migrate_supabase [schema data models config secrets]
Sin argumentos corre todo, en ese orden.

La base ANTIGUA solo se lee por Postgres directo: está restringida por Fair
Use (402 en API/Storage), pero la conexión directa sigue funcionando. Por eso
los modelos se suben desde la copia local en lugar de descargarlos.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import psycopg
import requests

ROOT = Path(__file__).resolve().parent.parent
STEPS = ("schema", "data", "models", "config", "secrets")
STATIONS = [
    "02300", "03000", "05000", "05100", "06000", "06111",
    "07105", "07107", "07111", "09000", "09122", "10009",
]

# (tabla calificada, columnas) en orden de dependencia de FK.
TABLES = [
    ("public.stations", ["station_id", "name", "latitude", "longitude"]),
    ("public.\"Original Data\"", ["station_id", "observed_at", "demand", "prediction"]),
    ("public.\"Temp\"", ["station_id", "observed_at", "demand", "prediction"]),
    ("public.baseline", ["station_id", "observed_at", "actual_demand", "prediction", "score"]),
    ("public.sarimas", ["station_id", "observed_at", "actual_demand", "prediction", "score",
                        "order_p", "order_d", "order_q", "seasonal_period"]),
    ("public.dynamic_harmonic", ["station_id", "observed_at", "actual_demand", "prediction", "score"]),
    ("public.xgboost", ["station_id", "observed_at", "actual_demand", "prediction", "score"]),
    ("public.collector_state", ["state_key", "cursor_value", "updated_at"]),
    ("ops.job_runs", ["id", "job_name", "status", "started_at", "finished_at"]),
]

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS public.stations (
    station_id TEXT PRIMARY KEY, name TEXT NOT NULL,
    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION);

CREATE TABLE IF NOT EXISTS public."Original Data" (
    station_id TEXT NOT NULL REFERENCES public.stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    demand INTEGER NOT NULL CHECK (demand >= 0),
    prediction NUMERIC,
    PRIMARY KEY (station_id, observed_at));
CREATE INDEX IF NOT EXISTS original_data_observed_at_idx ON public."Original Data" (observed_at);

CREATE TABLE IF NOT EXISTS public."Temp" (
    station_id TEXT NOT NULL REFERENCES public.stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL,
    demand INTEGER NOT NULL CHECK (demand >= 0),
    prediction NUMERIC,
    PRIMARY KEY (station_id, observed_at));
CREATE INDEX IF NOT EXISTS temp_observed_at_idx ON public."Temp" (observed_at);

CREATE TABLE IF NOT EXISTS public.baseline (
    station_id TEXT NOT NULL REFERENCES public.stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL, actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL, score NUMERIC,
    PRIMARY KEY (station_id, observed_at));

CREATE TABLE IF NOT EXISTS public.sarimas (
    station_id TEXT NOT NULL REFERENCES public.stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL, actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL, score NUMERIC,
    order_p INTEGER NOT NULL, order_d INTEGER NOT NULL DEFAULT 0,
    order_q INTEGER NOT NULL DEFAULT 1, seasonal_period INTEGER NOT NULL DEFAULT 96,
    PRIMARY KEY (station_id, observed_at));

CREATE TABLE IF NOT EXISTS public.dynamic_harmonic (
    station_id TEXT NOT NULL REFERENCES public.stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL, actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL, score NUMERIC,
    PRIMARY KEY (station_id, observed_at));

CREATE TABLE IF NOT EXISTS public.xgboost (
    station_id TEXT NOT NULL REFERENCES public.stations(station_id),
    observed_at TIMESTAMPTZ NOT NULL, actual_demand INTEGER NOT NULL,
    prediction NUMERIC NOT NULL, score NUMERIC,
    PRIMARY KEY (station_id, observed_at));

CREATE TABLE IF NOT EXISTS public.collector_state (
    state_key TEXT PRIMARY KEY, cursor_value TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now());

CREATE TABLE IF NOT EXISTS ops.job_runs (
    id BIGSERIAL PRIMARY KEY, job_name TEXT NOT NULL, status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ);
"""


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip("\"'")
    return env


def require(env: dict[str, str], *keys: str) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        sys.exit(f"Faltan en .env: {', '.join(missing)}")


def step_schema(env: dict[str, str]) -> None:
    require(env, "NEW_SUPABASE_DATABASE_URL")
    with psycopg.connect(env["NEW_SUPABASE_DATABASE_URL"], autocommit=True) as new:
        new.execute(SCHEMA_SQL)
        has_api = new.execute(
            "SELECT 1 FROM pg_proc WHERE proname = 'api_station_list' "
            "AND pronamespace = 'public'::regnamespace"
        ).fetchone()
        if has_api:
            print("schema: funciones api_* ya existen, se omite 009")
        else:
            # 009 = RLS en todas las tablas + funciones api_* para el panel web.
            new.execute((ROOT / "database/migrations/009_lock_down_anon_access.sql").read_text())
        # Cinturón extra: la key pública no debe tocar ninguna tabla directamente.
        new.execute(
            "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated; "
            "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon, authenticated;"
        )
    print("schema: OK")


def step_data(env: dict[str, str]) -> None:
    require(env, "SUPABASE_DATABASE_URL", "NEW_SUPABASE_DATABASE_URL")
    with psycopg.connect(env["SUPABASE_DATABASE_URL"]) as old, \
            psycopg.connect(env["NEW_SUPABASE_DATABASE_URL"]) as new:
        for table, cols in TABLES:
            collist = ", ".join(cols)
            n_old = old.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            n_new = new.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            if n_new == n_old:
                print(f"data: {table:28s} {n_new:>7} filas (ya copiada)")
                continue
            if n_new:
                sys.exit(f"{table} en la base nueva tiene {n_new} filas pero la vieja {n_old}; "
                         "vacíala a mano antes de reintentar.")
            with old.cursor().copy(f"COPY (SELECT {collist} FROM {table}) TO STDOUT (FORMAT binary)") as src, \
                    new.cursor().copy(f"COPY {table} ({collist}) FROM STDIN (FORMAT binary)") as dst:
                for chunk in src:
                    dst.write(chunk)
            new.commit()
            n_new = new.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            status = "OK" if n_new == n_old else "MISMATCH"
            print(f"data: {table:28s} {n_old:>7} -> {n_new:>7} {status}")
            if n_new != n_old:
                sys.exit("Conteos distintos, abortando.")
        new.execute(
            "SELECT setval(pg_get_serial_sequence('ops.job_runs', 'id'), "
            "COALESCE((SELECT max(id) FROM ops.job_runs), 1))"
        )
        new.commit()
    print("data: OK")


def step_models(env: dict[str, str]) -> None:
    require(env, "NEW_SUPABASE_URL", "NEW_SUPABASE_SERVICE_ROLE_KEY")
    url, key = env["NEW_SUPABASE_URL"].rstrip("/"), env["NEW_SUPABASE_SERVICE_ROLE_KEY"]
    auth = {"Authorization": f"Bearer {key}", "apikey": key}
    r = requests.post(f"{url}/storage/v1/bucket", headers=auth,
                      json={"id": "models", "name": "models", "public": False}, timeout=30)
    if r.status_code not in (200, 400, 409):  # 400/409 = el bucket ya existe
        sys.exit(f"No se pudo crear el bucket: {r.status_code} {r.text[:200]}")
    model_dir = ROOT / "models/xgboost"
    names = [f"xgboost_{s}_h{h}.joblib" for s in STATIONS for h in (1, 2, 3, 4)]
    missing = [n for n in names if not (model_dir / n).exists()]
    if missing:
        sys.exit(f"Faltan modelos locales: {missing}")
    for name in names:
        for attempt in range(5):
            try:
                r = requests.post(
                    f"{url}/storage/v1/object/models/xgboost/{name}",
                    headers={**auth, "Content-Type": "application/octet-stream", "x-upsert": "true"},
                    data=(model_dir / name).read_bytes(), timeout=120,
                )
                break
            except requests.exceptions.RequestException as exc:
                if attempt == 4:
                    sys.exit(f"Falló subir {name}: {exc}")
                time.sleep(3 * (attempt + 1))
        if r.status_code != 200:
            sys.exit(f"Falló subir {name}: {r.status_code} {r.text[:200]}")
    print(f"models: {len(names)} archivos subidos")


def step_config(env: dict[str, str]) -> None:
    require(env, "NEW_SUPABASE_URL", "NEW_SUPABASE_PUBLISHABLE_KEY")
    path = ROOT / "web/config.js"
    text = path.read_text()
    text = re.sub(r'(const SUPABASE_URL = )"[^"]*"', rf'\1"{env["NEW_SUPABASE_URL"]}"', text)
    text = re.sub(r'(const SUPABASE_PUBLISHABLE_KEY = )"[^"]*"',
                  rf'\1"{env["NEW_SUPABASE_PUBLISHABLE_KEY"]}"', text)
    path.write_text(text)
    print("config: web/config.js actualizado (falta commit + deploy en Vercel)")


def step_secrets(env: dict[str, str]) -> None:
    require(env, "NEW_SUPABASE_DATABASE_URL", "NEW_SUPABASE_URL", "NEW_SUPABASE_SERVICE_ROLE_KEY")
    pairs = {
        "SUPABASE_DATABASE_URL": env["NEW_SUPABASE_DATABASE_URL"],
        "SUPABASE_URL": env["NEW_SUPABASE_URL"],
        "SUPABASE_SERVICE_ROLE_KEY": env["NEW_SUPABASE_SERVICE_ROLE_KEY"],
    }
    for name, value in pairs.items():
        subprocess.run(["gh", "secret", "set", name, "--body", value], check=True, cwd=ROOT)
        print(f"secrets: {name} actualizado")


def main() -> None:
    steps = sys.argv[1:] or list(STEPS)
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"Pasos desconocidos: {bad}. Válidos: {STEPS}")
    env = load_env()
    for s in STEPS:
        if s in steps:
            globals()[f"step_{s}"](env)


if __name__ == "__main__":
    main()
