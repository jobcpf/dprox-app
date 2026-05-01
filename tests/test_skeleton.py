from fastapi.testclient import TestClient

from dprox.server import app
from dprox.version import IMAGE, __version__

client = TestClient(app)


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "checks" in body


def test_version_endpoint_returns_version_and_image() -> None:
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["image"] == IMAGE


def test_image_uses_ghcr_jobcpf() -> None:
    assert IMAGE.startswith("ghcr.io/jobcpf/dprox:")
