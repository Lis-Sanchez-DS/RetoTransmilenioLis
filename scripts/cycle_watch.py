"""Vigilante independiente de ciclos (2026-10-04).

Corre en OTRO runner y en OTRO workflow (.github/workflows/watchdog.yml) que el
submitter principal. Solo biblioteca estandar (corre en un runner sin pip). Decide
si hay un ciclo ABIERTO que nadie ha entregado todavia:

  - ciclo abierto hace mas de GRACE_MINUTES (el submitter principal entrega en
    ~1-4 min; este margen evita doble trabajo en operacion normal) y
  - el ledger (collector_state 'last_submission', escrito por app.submit_xgboost
    en cada entrega) no lista ese cycle_id.

Modos:
  (sin flags)       escribe needed=true|false (y cycle_id, closes_at) en $GITHUB_OUTPUT.
  --until-delivered sale 0 si el ciclo ya figura entregado (o ya cerro), 1 si no;
                    el workflow lo usa para reintentar hasta que se entregue.

Si la API no responde desde este runner no puede ayudar: avisa con ::warning:: y
needed=false (el siguiente cron lo reintenta desde un runner nuevo). Si no puede
leer el ledger asume NO entregado: reenviar es inocuo (el servidor responde 409).
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

GRACE_MINUTES = 5.0
API_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def decide(cycle: dict | None, delivered_cycle_id: str | None, now: datetime, grace: float = GRACE_MINUTES):
    """(needed, reason). Pure: no I/O."""
    if cycle is None:
        return False, "sin ciclo abierto"
    if cycle.get("state") != "open":
        return False, f"ciclo en estado {cycle.get('state')}"
    if now >= _parse(cycle["closes_at"]):
        return False, "la ventana ya cerro"
    if delivered_cycle_id == cycle["cycle_id"]:
        return False, "ya entregado"
    age = (now - _parse(cycle["opens_at"])).total_seconds() / 60
    if age < grace:
        return False, f"abierto hace {age:.1f} min; el submitter principal tiene prioridad"
    return True, f"ciclo {cycle['cycle_id']} abierto hace {age:.1f} min sin entrega registrada"


def _get_json(url: str, headers: dict, attempts: int = 2, timeout: int = 20):
    last = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # timeout / DNS / refused: reintenta
            last = exc
    raise last


def fetch_cycle():
    try:
        return _get_json(
            f"{API_URL}/v1/forecast-cycles/current", {"Authorization": f"Bearer {os.environ['PULSO_API_KEY']}"}
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_delivered_cycle_id():
    try:
        base = os.environ["SUPABASE_URL"].rstrip("/")
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        rows = _get_json(
            f"{base}/rest/v1/collector_state?state_key=eq.last_submission&select=cursor_value",
            {"apikey": key, "Authorization": f"Bearer {key}"},
        )
        return json.loads(rows[0]["cursor_value"]).get("cycle_id") if rows else None
    except Exception as exc:
        print(f"::warning::no se pudo leer el ledger ({exc!r}); se asume no entregado.", flush=True)
        return None


def _write_output(**values) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{key}={value}" for key, value in values.items()]
    if path:
        with open(path, "a") as handle:
            handle.write("\n".join(lines) + "\n")
    print(" ".join(lines), flush=True)


def main(argv: list[str]) -> int:
    now = datetime.now(timezone.utc)
    try:
        cycle = fetch_cycle()
    except Exception as exc:
        print(f"::warning::la API de Pulso no responde desde este runner ({exc!r}); no puedo vigilar este ciclo.", flush=True)
        if "--until-delivered" in argv:
            return 1
        _write_output(needed="false")
        return 0
    delivered = fetch_delivered_cycle_id()
    if "--until-delivered" in argv:
        done = cycle is None or delivered == cycle["cycle_id"] or now >= _parse(cycle["closes_at"])
        return 0 if done else 1
    needed, reason = decide(cycle, delivered, now)
    print(reason, flush=True)
    if needed:
        print(f"::warning::{reason}: el vigilante entrega este ciclo.", flush=True)
    _write_output(needed=str(needed).lower(), cycle_id=(cycle or {}).get("cycle_id", ""), closes_at=(cycle or {}).get("closes_at", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
