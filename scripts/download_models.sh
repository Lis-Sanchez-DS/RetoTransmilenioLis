#!/usr/bin/env bash
# Descarga los modelos XGBoost desde Supabase Storage.
#
# La lógica real vive en scripts/download_models.py (solo re-descarga lo que
# cambió desde la última vuelta - ver ese archivo). Este wrapper existe para
# no tener que tocar collector.yml/submissions.yml, que invocan este .sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m scripts.download_models
