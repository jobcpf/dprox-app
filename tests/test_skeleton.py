from fastapi.testclient import TestClient

from dprox.config import Config
from dprox.plan import PlanCache
from dprox.server import create_app
from dprox.version import IMAGE, __version__


def test_healthz_returns_ok_with_plan_summary(
    baseline_config: Config, plan_cache: PlanCache
) -> None:
    client = TestClient(create_app(baseline_config, plan_cache))
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["org"] == baseline_config.org
    plan_check = body["checks"]["plan"]
    assert plan_check["loaded"] is True
    assert plan_check["agents"] == 3
    assert plan_check["admins"] == 1


def test_version_endpoint_returns_version_and_image(
    baseline_config: Config, plan_cache: PlanCache
) -> None:
    client = TestClient(create_app(baseline_config, plan_cache))
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["image"] == IMAGE


def test_image_uses_ghcr_jobcpf() -> None:
    assert IMAGE.startswith("ghcr.io/jobcpf/dprox:")
