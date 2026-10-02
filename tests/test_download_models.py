import json

import pytest

from scripts import download_models


class FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json_data = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _all_names():
    return download_models.expected_names()


@pytest.fixture(autouse=True)
def _isolated_model_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download_models, "MODEL_DIR", tmp_path / "models" / "xgboost")
    monkeypatch.setattr(download_models, "MANIFEST_PATH", tmp_path / "models" / "xgboost" / ".manifest.json")


def test_first_run_downloads_everything(monkeypatch):
    remote_listing = [{"name": name, "updated_at": "v1"} for name in _all_names()]
    monkeypatch.setattr(
        download_models.requests, "post", lambda *a, **k: FakeResponse(json_data=remote_listing)
    )
    downloaded = []

    def fake_get(url, headers, timeout):
        name = url.rsplit("/", 1)[-1]
        downloaded.append(name)
        return FakeResponse(content=b"fake-model-bytes")

    monkeypatch.setattr(download_models.requests, "get", fake_get)

    result = download_models.sync_models("https://example.supabase.co", "fake-key")

    assert set(result) == set(_all_names())
    assert set(downloaded) == set(_all_names())
    manifest = json.loads(download_models.MANIFEST_PATH.read_text())
    assert all(manifest[name] == "v1" for name in _all_names())


def test_second_run_skips_unchanged_files(monkeypatch):
    remote_listing = [{"name": name, "updated_at": "v1"} for name in _all_names()]
    monkeypatch.setattr(
        download_models.requests, "post", lambda *a, **k: FakeResponse(json_data=remote_listing)
    )
    monkeypatch.setattr(
        download_models.requests, "get", lambda *a, **k: FakeResponse(content=b"fake-model-bytes")
    )

    download_models.sync_models("https://example.supabase.co", "fake-key")

    def fail_get(*a, **k):
        raise AssertionError("no debería re-descargar un archivo sin cambios")

    monkeypatch.setattr(download_models.requests, "get", fail_get)

    result = download_models.sync_models("https://example.supabase.co", "fake-key")

    assert result == []


def test_only_changed_file_is_redownloaded(monkeypatch):
    remote_listing = [{"name": name, "updated_at": "v1"} for name in _all_names()]
    monkeypatch.setattr(
        download_models.requests, "post", lambda *a, **k: FakeResponse(json_data=remote_listing)
    )
    monkeypatch.setattr(
        download_models.requests, "get", lambda *a, **k: FakeResponse(content=b"fake-model-bytes")
    )
    download_models.sync_models("https://example.supabase.co", "fake-key")

    changed_name = next(iter(_all_names()))
    remote_listing_v2 = [
        {"name": name, "updated_at": "v2" if name == changed_name else "v1"}
        for name in _all_names()
    ]
    monkeypatch.setattr(
        download_models.requests, "post", lambda *a, **k: FakeResponse(json_data=remote_listing_v2)
    )
    downloaded = []

    def fake_get(url, headers, timeout):
        name = url.rsplit("/", 1)[-1]
        downloaded.append(name)
        return FakeResponse(content=b"fake-model-bytes-v2")

    monkeypatch.setattr(download_models.requests, "get", fake_get)

    result = download_models.sync_models("https://example.supabase.co", "fake-key")

    assert result == [changed_name]
    assert downloaded == [changed_name]


def test_missing_local_file_forces_redownload(monkeypatch):
    remote_listing = [{"name": name, "updated_at": "v1"} for name in _all_names()]
    monkeypatch.setattr(
        download_models.requests, "post", lambda *a, **k: FakeResponse(json_data=remote_listing)
    )
    monkeypatch.setattr(
        download_models.requests, "get", lambda *a, **k: FakeResponse(content=b"fake-model-bytes")
    )
    download_models.sync_models("https://example.supabase.co", "fake-key")

    removed_name = next(iter(_all_names()))
    (download_models.MODEL_DIR / removed_name).unlink()

    result = download_models.sync_models("https://example.supabase.co", "fake-key")

    assert result == [removed_name]


class StatusResponse(FakeResponse):
    def __init__(self, status_code, **kwargs):
        super().__init__(**kwargs)
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)


def test_gateway_502_is_retried_and_download_succeeds(monkeypatch):
    """Un 502 intermitente de Supabase Storage (mato un job el 2026-10-02) se
    reintenta en vez de propagarse como error de la aplicacion."""
    monkeypatch.setattr("app.net.time.sleep", lambda s: None)
    remote_listing = [{"name": name, "updated_at": "v1"} for name in _all_names()]
    monkeypatch.setattr(
        download_models.requests, "post", lambda *a, **k: StatusResponse(200, json_data=remote_listing)
    )
    calls = {"n": 0}

    def flaky_get(url, headers, timeout):
        calls["n"] += 1
        if calls["n"] <= 2:
            return StatusResponse(502)
        return StatusResponse(200, content=b"model")

    monkeypatch.setattr(download_models.requests, "get", flaky_get)
    downloaded = download_models.sync_models("https://example.supabase.co", "key")
    assert len(downloaded) == len(_all_names())
    assert calls["n"] == len(_all_names()) + 2  # dos 502 reintentados, luego todo bien


def test_persistent_gateway_error_surfaces_as_upstream_unavailable(monkeypatch):
    from app.net import UpstreamUnavailable

    monkeypatch.setattr("app.net.time.sleep", lambda s: None)
    monkeypatch.setattr(download_models.requests, "post", lambda *a, **k: StatusResponse(503))
    with pytest.raises(UpstreamUnavailable):
        download_models.sync_models("https://example.supabase.co", "key")


def test_real_client_errors_are_not_retried(monkeypatch):
    import requests

    monkeypatch.setattr("app.net.time.sleep", lambda s: None)
    calls = {"n": 0}

    def post(*a, **k):
        calls["n"] += 1
        return StatusResponse(401)

    monkeypatch.setattr(download_models.requests, "post", post)
    with pytest.raises(requests.exceptions.HTTPError):
        download_models.sync_models("https://example.supabase.co", "key")
    assert calls["n"] == 1
