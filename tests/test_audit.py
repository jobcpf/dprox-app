"""Audit-logging tests (spec §9).

Uses structlog.testing.capture_logs() which intercepts every audit call
regardless of the global processor chain — tests don't need to configure
structlog or read stdout.
"""

from __future__ import annotations

import httpx
import structlog
from fastapi.testclient import TestClient

from dprox.audit import audit_reason_for_auth_failure
from dprox.mtls import require_mtls
from dprox.ollama import OllamaClient
from dprox.plan import PlanError
from dprox.qdrant import QdrantClient
from dprox.server import create_app

# --- reason mapping (unit, no app) -------------------------------------------


def test_audit_reason_translates_auth_required_to_no_client_cert() -> None:
    """Spec §9.3 audit reason vs spec §4.3 HTTP code: distinct vocabularies."""
    assert audit_reason_for_auth_failure("auth_required") == "no_client_cert"


def test_audit_reason_passthrough_for_cert_invalid() -> None:
    assert audit_reason_for_auth_failure("cert_invalid") == "cert_invalid"


def test_audit_reason_passthrough_for_cn_unparseable() -> None:
    assert audit_reason_for_auth_failure("cn_unparseable") == "cn_unparseable"


def test_audit_reason_passes_through_unknown_codes() -> None:
    """Defensive — never crash on a code we don't know; emit it as-is."""
    assert audit_reason_for_auth_failure("future_code") == "future_code"


# --- helpers -----------------------------------------------------------------


def _events_of_type(logs: list[dict], event_name: str) -> list[dict]:
    return [e for e in logs if e.get("event") == event_name]


# --- query event (success) ----------------------------------------------------


def test_query_emits_audit_event_with_required_fields(
    baseline_config, plan_cache, mock_ollama, mock_qdrant
) -> None:
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            response = c.post("/v1/query", json={"query": "wage policy", "limit": 5})

    assert response.status_code == 200

    events = _events_of_type(logs, "query")
    assert len(events) == 1
    e = events[0]
    assert e["agent"] == "agent_alice"
    assert e["groups_applied"] == ["g_engineering"]
    assert e["query_hash"] == response.json()["metadata"]["query_hash"]
    assert isinstance(e["result_count"], int)
    assert isinstance(e["latency_ms"], int)
    assert e["latency_ms"] >= 0
    assert "ollama_ms" in e
    assert "qdrant_ms" in e


def test_query_does_not_log_query_text_at_info(
    baseline_config, plan_cache, mock_ollama, mock_qdrant
) -> None:
    """Spec §9.1: query text is logged only at DEBUG when log_query_text=true."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            c.post("/v1/query", json={"query": "secret-content"})

    e = _events_of_type(logs, "query")[0]
    assert "query_text" not in e
    assert "secret-content" not in str(e)


def test_query_logs_query_text_when_log_query_text_enabled(
    baseline_config_dict, plan_yaml, mock_ollama, mock_qdrant
) -> None:
    """log_query_text=true → query event includes the raw text at DEBUG level."""
    from dprox.config import Config
    from dprox.plan import PlanCache

    baseline_config_dict["logging"]["log_query_text"] = True
    config = Config.model_validate(baseline_config_dict)
    plan_cache = PlanCache(config.plan)
    plan_cache.initial_load()

    app = create_app(config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            c.post("/v1/query", json={"query": "verbose-query-text"})

    e = _events_of_type(logs, "query")[0]
    assert e["query_text"] == "verbose-query-text"
    assert e["log_level"] == "debug"


# --- query_failed event ------------------------------------------------------


def test_query_failed_emitted_on_ollama_timeout(
    baseline_config, plan_cache, mock_qdrant
) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated")

    ollama = OllamaClient(
        baseline_config.embedding, transport=httpx.MockTransport(fail)
    )
    app = create_app(baseline_config, plan_cache, ollama=ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            c.post("/v1/query", json={"query": "x"})

    events = _events_of_type(logs, "query_failed")
    assert len(events) == 1
    e = events[0]
    assert e["agent"] == "agent_alice"
    assert e["groups_applied"] == ["g_engineering"]
    assert e["error"] == "upstream_timeout"
    assert "Timeout" in e["error_class"]
    assert e["latency_ms"] >= 0


def test_query_failed_emitted_on_qdrant_unavailable(
    baseline_config, plan_cache, mock_ollama, mock_qdrant_backend
) -> None:
    mock_qdrant_backend.search.side_effect = ConnectionError("qdrant down")
    qdrant = QdrantClient(
        baseline_config.qdrant,
        baseline_config.embedding.vector_dim,
        backend=mock_qdrant_backend,
    )
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            c.post("/v1/query", json={"query": "x"})

    e = _events_of_type(logs, "query_failed")[0]
    assert e["error"] == "upstream_unavailable"
    # Once embedding succeeded, the failed-event includes ollama_ms.
    assert "ollama_ms" in e
    assert e["ollama_ms"] is not None


def test_query_failed_emitted_when_plan_reload_breaks_mid_request(
    baseline_config, plan_cache, mock_ollama, mock_qdrant
) -> None:
    def _raise(_cn: str):
        raise PlanError("registry mount disappeared")

    plan_cache.lookup = _raise  # type: ignore[method-assign]

    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            c.post("/v1/query", json={"query": "x"})

    e = _events_of_type(logs, "query_failed")[0]
    assert e["error"] == "upstream_unavailable"
    assert e["error_class"] == "PlanError"
    # Plan failure happens before group resolution — groups_applied is None.
    assert e["groups_applied"] is None


# --- auth_rejected event ------------------------------------------------------


def test_auth_rejected_emitted_when_no_client_cert(
    baseline_config, plan_cache, mock_ollama, mock_qdrant
) -> None:
    """No dependency_override + no peer cert → require_mtls raises AuthFailure
    with code='auth_required'; the audit reason should be 'no_client_cert'."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            response = c.post("/v1/query", json={"query": "x"})

    assert response.status_code == 401
    e = _events_of_type(logs, "auth_rejected")[0]
    assert e["reason"] == "no_client_cert"
    assert e["path"] == "/v1/query"
    # cn / cert_serial unavailable (no cert was presented)
    assert e["cn"] is None
    assert e["cert_serial"] is None


def test_auth_rejected_emitted_when_cn_not_in_plan(
    baseline_config, plan_cache, mock_ollama, mock_qdrant
) -> None:
    """Cert valid but unknown CN → 403 with a separate audit reason."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_ghost"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            c.post("/v1/query", json={"query": "x"})

    e = _events_of_type(logs, "auth_rejected")[0]
    assert e["reason"] == "unknown_cn"
    assert e["cn"] == "agent_ghost"
    assert e["path"] == "/v1/query"


def test_no_audit_event_for_400_bad_request(
    baseline_config, plan_cache, mock_ollama, mock_qdrant
) -> None:
    """Spec §9 lists query / query_failed / auth_rejected only — body
    validation failure is a client bug, not a security event, and is NOT
    audit-logged."""
    app = create_app(baseline_config, plan_cache, ollama=mock_ollama, qdrant=mock_qdrant)
    app.dependency_overrides[require_mtls] = lambda: "agent_alice"

    with structlog.testing.capture_logs() as logs:
        with TestClient(app) as c:
            response = c.post("/v1/query", json={"query": "x", "groups": ["mal"]})

    assert response.status_code == 400
    assert _events_of_type(logs, "query") == []
    assert _events_of_type(logs, "query_failed") == []
    assert _events_of_type(logs, "auth_rejected") == []
