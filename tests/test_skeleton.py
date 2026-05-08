import httpx
from fastapi.testclient import TestClient

from dprox.config import Config
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache
from dprox.qdrant import QdrantClient
from dprox.server import create_app
from dprox.version import IMAGE, __version__


def test_healthz_returns_ok_with_all_checks(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    client = TestClient(
        create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    )
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

    qdrant_check = body["checks"]["qdrant"]
    assert qdrant_check["reachable"] is True
    assert qdrant_check["collection_exists"] is True
    assert qdrant_check["vector_dim_matches"] is True


def test_healthz_returns_503_when_ollama_unreachable(
    baseline_config: Config, plan_cache: PlanCache, mock_qdrant: QdrantClient
) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated ollama down")

    broken_ollama = OllamaClient(
        baseline_config.embedding, transport=httpx.MockTransport(fail)
    )
    client = TestClient(
        create_app(baseline_config, plan_cache, ollama=broken_ollama, qdrant=mock_qdrant)
    )
    response = client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["ollama"]["reachable"] is False


def test_healthz_returns_503_when_model_not_present(
    baseline_config: Config, plan_cache: PlanCache, mock_qdrant: QdrantClient
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "different-model:latest"}]})

    ollama = OllamaClient(
        baseline_config.embedding, transport=httpx.MockTransport(handler)
    )
    client = TestClient(
        create_app(baseline_config, plan_cache, ollama=ollama, qdrant=mock_qdrant)
    )
    response = client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["ollama"]["model_present"] is False


def test_healthz_returns_503_when_qdrant_unreachable(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant_backend,
) -> None:
    mock_qdrant_backend.get_collection.side_effect = ConnectionError("refused")
    qdrant = QdrantClient(
        baseline_config.qdrant,
        baseline_config.embedding.vector_dim,
        backend=mock_qdrant_backend,
    )
    client = TestClient(
        create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=qdrant)
    )
    response = client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"]["reachable"] is False


def test_healthz_returns_503_when_qdrant_dim_mismatches(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant_backend,
) -> None:
    from types import SimpleNamespace

    # Collection vector dim is 768 but config expects baseline (also 768),
    # so flip it: pretend Qdrant reports a different dim than configured.
    mock_qdrant_backend.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=384))
        )
    )
    qdrant = QdrantClient(
        baseline_config.qdrant,
        baseline_config.embedding.vector_dim,  # 768
        backend=mock_qdrant_backend,
    )
    client = TestClient(
        create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=qdrant)
    )
    response = client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"]["vector_dim_matches"] is False
    assert body["checks"]["qdrant"]["vector_dim"] == 384


def test_version_endpoint_returns_version_and_image(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    client = TestClient(
        create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    )
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["image"] == IMAGE


def test_image_uses_ghcr_jobcpf() -> None:
    assert IMAGE.startswith("ghcr.io/jobcpf/dprox:")
