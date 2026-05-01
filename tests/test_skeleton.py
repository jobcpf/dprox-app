import httpx
from fastapi.testclient import TestClient

from dprox.config import Config
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache
from dprox.server import create_app
from dprox.version import IMAGE, __version__


def test_healthz_returns_ok_with_plan_and_ollama(
    baseline_config: Config, plan_cache: PlanCache, mock_ollama: OllamaClient
) -> None:
    client = TestClient(create_app(baseline_config, plan_cache, ollama=mock_ollama))
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["org"] == baseline_config.org

    plan_check = body["checks"]["plan"]
    assert plan_check["loaded"] is True
    assert plan_check["agents"] == 3
    assert plan_check["admins"] == 1

    ollama_check = body["checks"]["ollama"]
    assert ollama_check["reachable"] is True
    assert ollama_check["model_present"] is True
    assert ollama_check["error"] is None


def test_healthz_returns_503_when_ollama_unreachable(
    baseline_config: Config, plan_cache: PlanCache
) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated ollama down")

    broken_ollama = OllamaClient(
        baseline_config.embedding, transport=httpx.MockTransport(fail)
    )
    client = TestClient(create_app(baseline_config, plan_cache, ollama=broken_ollama))
    response = client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["ollama"]["reachable"] is False


def test_healthz_returns_503_when_model_not_present(
    baseline_config: Config, plan_cache: PlanCache
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "different-model:latest"}]})

    ollama = OllamaClient(
        baseline_config.embedding, transport=httpx.MockTransport(handler)
    )
    client = TestClient(create_app(baseline_config, plan_cache, ollama=ollama))
    response = client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["ollama"]["reachable"] is True
    assert body["checks"]["ollama"]["model_present"] is False


def test_version_endpoint_returns_version_and_image(
    baseline_config: Config, plan_cache: PlanCache, mock_ollama: OllamaClient
) -> None:
    client = TestClient(create_app(baseline_config, plan_cache, ollama=mock_ollama))
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["image"] == IMAGE


def test_image_uses_ghcr_jobcpf() -> None:
    assert IMAGE.startswith("ghcr.io/jobcpf/dprox:")
