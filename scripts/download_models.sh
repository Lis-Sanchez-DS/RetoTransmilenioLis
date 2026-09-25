#!/usr/bin/env bash
# Descarga los modelos XGBoost desde Supabase Storage.
#
# Se reintenta cada descarga (curl --retry) para no matar el job entero por un
# blip transitorio de Supabase, y se verifica al final que las 12 estaciones
# llegaron completas. Pensado para llamarse tanto al inicio del job como en
# cada vuelta del loop interno, así un reentrenamiento por drift (ver
# app/drift.py) se recoge en minutos en vez de esperar a que el job se
# reinicie horas después.
set -euo pipefail

mkdir -p models/xgboost

stations=(02300 03000 05000 05100 06000 06111 07105 07107 07111 09000 09122 10009)
horizons=(1 2 3 4)

for station in "${stations[@]}"; do
  for horizon in "${horizons[@]}"; do
    curl --fail --silent --show-error \
      --retry 3 --retry-delay 5 --retry-all-errors --retry-connrefused \
      -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
      -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
      "$SUPABASE_URL/storage/v1/object/models/xgboost/xgboost_${station}_h${horizon}.joblib" \
      -o "models/xgboost/xgboost_${station}_h${horizon}.joblib"
  done
done

expected=$(( ${#stations[@]} * ${#horizons[@]} ))
found="$(find models/xgboost -name '*.joblib' | wc -l)"
if [ "$found" -ne "$expected" ]; then
  echo "Se esperaban $expected modelos y se encontraron $found." >&2
  exit 1
fi
