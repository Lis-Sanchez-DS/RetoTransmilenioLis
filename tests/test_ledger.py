import json

from app import health


class Conn:
    def __init__(self, row=None):
        self.row = row
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params=None):
        self.calls.append((query, params))
        return self

    def fetchone(self):
        return self.row


def test_record_submission_upserts_one_json_row(monkeypatch):
    conn = Conn()
    monkeypatch.setattr("app.health.connection", lambda: conn)
    health.record_submission("cyc_1", {"submission_id": "sub_9"})
    query, params = conn.calls[0]
    assert "collector_state" in query and "ON CONFLICT (state_key) DO UPDATE" in query
    assert params[0] == health.LEDGER_KEY
    assert json.loads(params[1]) == {"cycle_id": "cyc_1", "submission_id": "sub_9"}
    assert params[2] is True


def test_record_submission_without_overwrite_protects_the_same_cycles_receipt(monkeypatch):
    conn = Conn()
    monkeypatch.setattr("app.health.connection", lambda: conn)
    health.record_submission("cyc_1", {"status": "already"}, overwrite=False)
    query, params = conn.calls[0]
    assert params[2] is False
    assert params[3] == '{"cycle_id": "cyc_1"%'  # LIKE contra el valor JSON existente
    assert "NOT LIKE" in query


def test_last_submission_cycle_reads_the_ledger(monkeypatch):
    monkeypatch.setattr("app.health.connection", lambda: Conn(row=(json.dumps({"cycle_id": "cyc_7"}),)))
    assert health.last_submission_cycle() == "cyc_7"
    monkeypatch.setattr("app.health.connection", lambda: Conn(row=None))
    assert health.last_submission_cycle() is None
    monkeypatch.setattr("app.health.connection", lambda: Conn(row=("no es json",)))
    assert health.last_submission_cycle() is None
