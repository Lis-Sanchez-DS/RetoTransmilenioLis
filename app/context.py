"""Variables del endpoint /v1/context de la API (evento, lluvia, temperatura).

APAGADO por defecto (USE_CONTEXT_FEATURES = False): con la bandera apagada
ningún modelo ni ningún call site cambia. Se dejó implementado "por si acaso"
porque la API expone estos campos, pero hoy no se pueden usar en vivo:
`/v1/context` solo publica historia (hasta 2026-09-09) y no devuelve ni una
fila para los timestamps que se están pronosticando.

Evidencia (backtest 2026-10-01, modelo de razón, 3 folds, stack completo):
  - contexto PUBLICADO en el test (26-ago a 8-sep): +0.16pp agrupado.
  - contexto AUSENTE en el test (lo que pasaría hoy con la bandera encendida,
    porque el modelo se entrena con contexto hasta el 8-sep y en vivo recibe NaN):
    -0.54pp en el fold de la última semana y +0.61pp en la ventana del shock.
  Los efectos del contexto en los datos son chicos: un evento sube la demanda de
  la red ~11% y la lluvia (>1mm) la baja ~5%. Encender la bandera solo vale la
  pena si la API empieza a publicar contexto para el futuro (revisar con
  `GET /v1/context?start=<virtual_now>`), y exige reentrenar los 12 modelos.

Las variables se leen en el timestamp OBJETIVO de cada fila (evento, pronóstico
de lluvia, pronóstico de temperatura): son las que se conocen por adelantado.
`rain_mm`/`temperature_c` observadas se excluyen a propósito - en el momento
del corte no se conocen para el instante que se está prediciendo (fuga).
"""

import os
from datetime import datetime, timezone

import requests

from app.net import with_retries

USE_CONTEXT_FEATURES = False

CONTEXT_FIELDS = ("event_intensity", "rain_forecast", "temperature_forecast")
PAGE_SIZE = 1000
DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"


def _parse_utc(value) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_context(start, end) -> dict[datetime, dict]:
    """timestamp (UTC) -> {campo: valor} para todos los timestamps publicados en
    [start, end]. Devuelve {} si la API no tiene filas en ese rango (el caso
    actual para cualquier fecha posterior a 2026-09-09)."""
    base = os.getenv("PULSO_API_URL", DEFAULT_API_URL).rstrip("/")
    api_key = os.getenv("PULSO_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    table: dict[datetime, dict] = {}
    cursor = None
    while True:
        params = {"start": _parse_utc(start).isoformat(), "end": _parse_utc(end).isoformat(), "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor

        def _get():
            response = requests.get(f"{base}/v1/context", headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        page = with_retries(_get)
        for row in page.get("data", []):
            table[_parse_utc(row["observed_at"])] = {field: row.get(field) for field in CONTEXT_FIELDS}
        cursor = page.get("next_cursor")
        if not cursor:
            return table


def context_features(timestamp, table: dict | None) -> list[float]:
    """Los CONTEXT_FIELDS en `timestamp`, o NaN por campo si no hay contexto
    publicado para ese instante (XGBoost trata NaN como dato faltante)."""
    row = (table or {}).get(_parse_utc(timestamp))
    values = []
    for field in CONTEXT_FIELDS:
        value = None if row is None else row.get(field)
        values.append(float("nan") if value is None else float(value))
    return values

