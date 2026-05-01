from fastapi.testclient import TestClient

from dprox.config import Config
from dprox.server import create_app
from dprox.version import IMAGE, __version__


def test_healthz_returns_ok(baseline_config: Config) -> None:
    client = TestClient(create_app(baseline_config))
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["org"] == baseline_config.org
    assert "checks" in body


def test_version_endpoint_returns_version_and_image(baseline_config: Config) -> None:
    client = TestClient(create_app(baseline_config))
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["image"] == IMAGE


def test_image_uses_ghcr_jobcpf() -> None:
    assert IMAGE.startswith("ghcr.io/jobcpf/dprox:")
