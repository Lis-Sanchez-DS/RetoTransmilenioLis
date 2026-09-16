#!/bin/bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v api_user="$API_DB_USER" -v api_password="$API_DB_PASSWORD" \
  -v scheduler_user="$SCHEDULER_DB_USER" -v scheduler_password="$SCHEDULER_DB_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'api_user', :'api_password')
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = :'api_user')\gexec
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'scheduler_user', :'scheduler_password')
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = :'scheduler_user')\gexec
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v api_user="$API_DB_USER" -v scheduler_user="$SCHEDULER_DB_USER" <<'SQL'
CREATE SCHEMA IF NOT EXISTS ops;
GRANT USAGE ON SCHEMA ops TO :"api_user", :"scheduler_user";
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ops TO :"api_user", :"scheduler_user";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ops TO :"api_user", :"scheduler_user";
SQL
