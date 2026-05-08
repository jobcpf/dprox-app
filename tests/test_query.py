"""Integration tests for the stub POST /v1/query route.

Step 4 only validates the auth + plan-resolution path. The full pipeline
(embed + Qdrant + audit) lands in step 7. We exercise:

    cert validation (via require_mtls) → plan lookup → response

using FastAPI's dependency_overrides to short-circuit require_mtls so we
don't need to mount a real TLS stack inside the test client.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from dprox.config import Config
from dprox.mtls import AuthFailure, require_mtls
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache, PlanError
from dprox.qdrant import QdrantClient
from dprox.server import create_app


def _client(
    config: Config,
    plan_cache: PlanCache,
    ollama: OllamaClient,
    qdrant: QdrantClient,
) -> TestClient:
    app = create_app(config, plan_cache, ollama=ollama, qdrant=qdrant)
    return TestClient(app)


def test_query_with_known_agent_returns_resolved_groups(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "ignored", "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "stub": True,
        "agent": "agent_alice",
        "role": "agent",
        "groups": ["g_engineering"],
    }


def test_query_with_admin_cn_resolves_admin_role(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "admin_alice"

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "ignored", "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "admin"
    assert set(body["groups"]) == {"g_engineering", "g_finance", "g_admin"}


def test_query_with_unknown_cn_returns_403_unknown_agent(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_ghost"

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "ignored", "limit": 5})

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "unknown_agent"
    assert "agent_ghost" in body["message"]


def test_query_without_cert_returns_401_auth_required(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    """No dependency_override + no peer cert → AuthFailure → 401 with spec error shape."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "ignored"})

    assert response.status_code == 401
    body = response.json()
    assert body == {"error": "auth_required", "message": "no client cert presented"}


def test_query_when_plan_reload_fails_returns_502(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    """If PlanCache.lookup raises PlanError mid-request, /v1/query returns 502."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    def _raise(_cn: str):
        raise PlanError("simulated registry mount disappeared")

    plan_cache.lookup = _raise  # type: ignore[method-assign]

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "ignored"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "upstream_unavailable"
    assert "simulated" in body["message"]


def test_auth_failure_handler_returns_flat_error_shape(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    """Spec §4.3 demands {"error": ..., "message": ...} — not FastAPI's default {"detail": ...}."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)

    def _fail():
        raise AuthFailure("cert_invalid", "test message", status=401)

    app.dependency_overrides[require_mtls] = _fail

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "x"})

    assert response.status_code == 401
    body = response.json()
    assert "detail" not in body
    assert body == {"error": "cert_invalid", "message": "test message"}


def test_healthz_remains_public_after_query_route_added(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    """Public route shouldn't have been accidentally caught by global mTLS."""
    with _client(baseline_config, plan_cache, mock_ollama, mock_qdrant) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_version_remains_public_after_query_route_added(
    baseline_config: Config,
    plan_cache: PlanCache,
    mock_ollama: OllamaClient,
    mock_qdrant: QdrantClient,
) -> None:
    with _client(baseline_config, plan_cache, mock_ollama, mock_qdrant) as client:
        response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()
