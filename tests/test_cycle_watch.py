import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cycle_watch as cw  # noqa: E402

OPENS = datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc)


def cycle(state="open", cycle_id="cyc_1"):
    return {
        "cycle_id": cycle_id,
        "state": state,
        "opens_at": OPENS.isoformat().replace("+00:00", "Z"),
        "closes_at": (OPENS + timedelta(minutes=25)).isoformat().replace("+00:00", "Z"),
    }


def at(minutes):
    return OPENS + timedelta(minutes=minutes)


def test_does_nothing_without_an_open_cycle_or_after_close():
    assert cw.decide(None, None, at(10))[0] is False
    assert cw.decide(cycle(state="closed"), None, at(10))[0] is False
    assert cw.decide(cycle(), None, at(26))[0] is False


def test_gives_the_main_submitter_priority_during_the_grace_period():
    needed, reason = cw.decide(cycle(), None, at(cw.GRACE_MINUTES - 1))
    assert needed is False and "prioridad" in reason


def test_steps_in_when_the_cycle_is_open_and_not_in_the_ledger():
    needed, reason = cw.decide(cycle(), "cyc_0", at(cw.GRACE_MINUTES + 1))
    assert needed is True and "cyc_1" in reason
    assert cw.decide(cycle(), None, at(12))[0] is True  # ledger vacio tambien


def test_does_nothing_when_the_ledger_already_lists_the_cycle():
    assert cw.decide(cycle(), "cyc_1", at(12)) == (False, "ya entregado")


def test_ledger_read_failure_assumes_not_delivered(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    monkeypatch.setattr(cw, "_get_json", lambda *a, **k: (_ for _ in ()).throw(OSError("sin red")))
    assert cw.fetch_delivered_cycle_id() is None


def test_ledger_row_is_parsed(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://x")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    monkeypatch.setattr(cw, "_get_json", lambda *a, **k: [{"cursor_value": json.dumps({"cycle_id": "cyc_9"})}])
    assert cw.fetch_delivered_cycle_id() == "cyc_9"


def test_api_unreachable_means_no_action_and_a_warning(monkeypatch, capsys, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(cw, "fetch_cycle", lambda: (_ for _ in ()).throw(OSError("timeout")))
    assert cw.main([]) == 0
    assert "needed=false" in out.read_text()
    assert "::warning::" in capsys.readouterr().out


def test_until_delivered_mode_exit_codes(monkeypatch):
    monkeypatch.setattr(cw, "fetch_cycle", lambda: cycle())
    monkeypatch.setattr(cw, "fetch_delivered_cycle_id", lambda: "cyc_1")
    assert cw.main(["--until-delivered"]) == 0
    monkeypatch.setattr(cw, "fetch_delivered_cycle_id", lambda: "cyc_0")

    class Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return at(10)

    monkeypatch.setattr(cw, "datetime", Now)
    assert cw.main(["--until-delivered"]) == 1
