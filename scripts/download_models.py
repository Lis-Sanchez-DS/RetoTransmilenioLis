"""Descarga (solo lo necesario) los modelos XGBoost desde Supabase Storage.

Antes, `scripts/download_models.sh` re-descargaba los 48 archivos de modelo
(~19MB) en cada vuelta del loop interno de collector.yml (cada 30 min) y
submissions.yml (cada 5 min) - 336 descargas completas al día, ~184GB/mes de
egress, muy por encima de la cuota gratuita de Supabase (aviso de Fair Use
Policy recibido 2026-09-26).

Ahora se pide un solo listado de metadata (JSON, sin los bytes del modelo -
prácticamente gratis) y se compara el `updated_at` de cada archivo contra un
manifiesto local (`models/xgboost/.manifest.json`). Solo se descarga (y
cuenta contra egress) lo que realmente cambió desde la última vuelta -
típicamente los 4 horizontes de una sola estación tras un reentrenamiento por
drift, no los 48 archivos.

El manifiesto vive en el mismo directorio (gitignorado junto con el resto de
`models/`) y persiste dentro de las ~5.75h que dura un job de GitHub Actions
- sirve para todas las vueltas de ese job. Al reiniciar el job,
`actions/checkout` limpia el directorio y la primera vuelta vuelve a
descargar todo, exactamente igual que antes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

from app.net import UpstreamUnavailable, with_retries

STATIONS = [
    "02300", "03000", "05000", "05100", "06000", "06111",
    "07105", "07107", "07111", "09000", "09122", "10009",
]
HORIZONS = (1, 2, 3, 4)

MODEL_DIR = Path("models/xgboost")
MANIFEST_PATH = MODEL_DIR / ".manifest.json"


def expected_names() -> list[str]:
    return [
        f"xgboost_{station}_h{horizon}.joblib"
        for station in STATIONS
        for horizon in HORIZONS
    ]


def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except FileNotFoundError:
        return {}


def _list_remote(supabase_url: str, headers: dict) -> dict:
    def _call():
        r = requests.post(
            f"{supabase_url}/storage/v1/object/list/models",
            headers={**headers, "Content-Type": "application/json"},
            json={"prefix": "xgboost", "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    items = with_retries(_call)
    return {item["name"]: item.get("updated_at") for item in items}


def _download(supabase_url: str, headers: dict, name: str) -> bytes:
    def _call():
        r = requests.get(
            f"{supabase_url}/storage/v1/object/models/xgboost/{name}",
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        return r.content

    return with_retries(_call)


def sync_models(supabase_url: str, service_role_key: str) -> list[str]:
    """Downloads only the models whose remote version changed. Returns the
    list of filenames actually downloaded."""
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    remote = _list_remote(supabase_url, headers)

    downloaded = []
    for name in expected_names():
        local_path = MODEL_DIR / name
        remote_updated = remote.get(name)
        unchanged = (
            local_path.exists()
            and remote_updated is not None
            and manifest.get(name) == remote_updated
        )
        if unchanged:
            continue

        local_path.write_bytes(_download(supabase_url, headers, name))
        manifest[name] = remote_updated
        downloaded.append(name)

    MANIFEST_PATH.write_text(json.dumps(manifest))
    return downloaded


def main() -> int:
    supabase_url = os.environ["SUPABASE_URL"]
    service_role_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    try:
        downloaded = sync_models(supabase_url, service_role_key)
    except UpstreamUnavailable as exc:
        print(f"Supabase Storage no disponible: {exc}", file=sys.stderr)
        return 1

    print(f"{len(downloaded)} modelo(s) actualizado(s): {downloaded}")

    missing = [name for name in expected_names() if not (MODEL_DIR / name).exists()]
    if missing:
        print(f"Se esperaban {len(expected_names())} modelos y faltan {len(missing)}: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
