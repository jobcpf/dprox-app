from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.models import FieldCondition, Filter, MatchAny

from dprox.config import QdrantConfig
from dprox.qdrant import (
    CLASSIFICATION_GROUP_KEY,
    QdrantClient,
    QdrantHit,
    QdrantTimeout,
    QdrantUnavailable,
    build_classification_filter,
)


def _config(**overrides) -> QdrantConfig:
    base: dict = {
        "url": "http://test-qdrant:6333",
        "api_key_env": "TEST_QDRANT_API_KEY",
        "collection": "documents",
        "default_limit": 10,
        "max_limit": 50,
        "timeout_seconds": 5,
    }
    base.update(overrides)
    return QdrantConfig(**base)


def _client(backend, *, config=None, dim: int = 4) -> QdrantClient:
    return QdrantClient(config or _config(), expected_vector_dim=dim, backend=backend)


def _point(payload: dict, score: float = 0.85, point_id: int = 1):
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def _response(points: list) -> SimpleNamespace:
    """Wrap a list of points in a QueryResponse-shaped object (.points)."""
    return SimpleNamespace(points=points)


def _collection_info(size: int):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=size))
        )
    )


# --- build_classification_filter (security boundary) -------------------------
# Spec §10.3: 100% test coverage on this code path is non-negotiable.


def test_filter_uses_classification_group_key() -> None:
    f = build_classification_filter(frozenset({"g1", "g2"}))
    assert isinstance(f, Filter)
    assert len(f.must) == 1
    cond = f.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == CLASSIFICATION_GROUP_KEY
    assert cond.key == "classification_group"


def test_filter_must_clause_uses_match_any() -> None:
    f = build_classification_filter(frozenset({"g1", "g2"}))
    cond = f.must[0]
    assert isinstance(cond.match, MatchAny)
    assert set(cond.match.any) == {"g1", "g2"}


def test_filter_groups_are_sorted_for_deterministic_audit() -> None:
    f = build_classification_filter(frozenset({"zeta", "alpha", "mu"}))
    assert f.must[0].match.any == ["alpha", "mu", "zeta"]


def test_filter_with_empty_groups_still_constructs() -> None:
    """An agent with no groups gets a filter that matches no points
    (zero results) — never None / never absent."""
    f = build_classification_filter(frozenset())
    assert isinstance(f, Filter)
    assert f.must[0].match.any == []


def test_filter_with_single_group() -> None:
    f = build_classification_filter(frozenset({"g_only"}))
    assert f.must[0].match.any == ["g_only"]


def test_filter_uses_must_not_should() -> None:
    """Spec §3.4 critical invariant: the filter is non-negotiable. 'must' clauses
    are required matches in Qdrant; 'should' is optional. Wrong clause → filter
    is bypassable. This test guards against that regression."""
    f = build_classification_filter(frozenset({"g1"}))
    assert f.must is not None and len(f.must) == 1
    assert f.should is None
    assert f.must_not is None


# --- QdrantClient.search (calls AsyncQdrantClient.query_points under the hood) ---


async def test_search_passes_built_filter_to_backend() -> None:
    backend = AsyncMock()
    backend.query_points.return_value = _response([])
    client = _client(backend)

    await client.search([0.1] * 4, frozenset({"g1", "g2"}), limit=5)

    backend.query_points.assert_awaited_once()
    kwargs = backend.query_points.await_args.kwargs
    assert kwargs["collection_name"] == "documents"
    # qdrant-client universal query API: vector is passed as `query=`,
    # not `query_vector=` (the old search()/1.13- kwarg).
    assert kwargs["query"] == [0.1] * 4
    assert kwargs["limit"] == 5
    assert kwargs["with_payload"] is True

    q_filter = kwargs["query_filter"]
    assert isinstance(q_filter, Filter)
    assert q_filter.must[0].key == "classification_group"
    assert set(q_filter.must[0].match.any) == {"g1", "g2"}


async def test_search_does_not_call_legacy_search_method() -> None:
    """Regression guard: dprox must use query_points (qdrant-client >=1.10),
    NOT search() (removed circa 1.13). See dprox-v0.1.0-qdrant-client-bug.md."""
    backend = AsyncMock()
    backend.query_points.return_value = _response([])
    client = _client(backend)

    await client.search([0.1] * 4, frozenset({"g"}), limit=5)

    backend.query_points.assert_awaited_once()
    backend.search.assert_not_called()


async def test_search_projects_payload_to_qdrant_hit() -> None:
    backend = AsyncMock()
    backend.query_points.return_value = _response(
        [
            _point(
                {
                    "text": "wage policy",
                    "classification_group": "arc_g0",
                    "source_path_rel": "drive/policy.docx",
                    "file_type": "docx",
                    "chunk_index": 3,
                    "chunk_total": 12,
                    "modified_at": "2026-04-28T08:00:00Z",
                    "indexed_at": "2026-04-28T08:15:00Z",
                },
                score=0.781,
                point_id=42,
            )
        ]
    )
    client = _client(backend)

    hits, elapsed_ms = await client.search([0.1] * 4, frozenset({"arc_g0"}), limit=10)

    assert len(hits) == 1
    h = hits[0]
    assert isinstance(h, QdrantHit)
    assert h.text == "wage policy"
    assert h.classification_group == "arc_g0"
    assert h.score == pytest.approx(0.781)
    assert h.source_path_rel == "drive/policy.docx"
    assert h.chunk_index == 3
    assert elapsed_ms >= 0


async def test_search_handles_minimal_payload() -> None:
    """Optional fields can be missing; required fields cannot."""
    backend = AsyncMock()
    backend.query_points.return_value = _response(
        [_point({"text": "minimal", "classification_group": "g_min"})]
    )
    client = _client(backend)

    hits, _ = await client.search([0.1] * 4, frozenset({"g_min"}), limit=10)

    assert hits[0].source_path_rel is None
    assert hits[0].file_type is None
    assert hits[0].chunk_index is None


async def test_search_rejects_point_missing_text() -> None:
    backend = AsyncMock()
    backend.query_points.return_value = _response(
        [_point({"classification_group": "g_eng"})]  # no 'text'
    )
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="missing 'text'"):
        await client.search([0.1] * 4, frozenset({"g_eng"}), limit=10)


async def test_search_rejects_point_missing_classification_group() -> None:
    backend = AsyncMock()
    backend.query_points.return_value = _response(
        [_point({"text": "orphan"})]  # no classification_group
    )
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="missing 'classification_group'"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_rejects_point_with_empty_text() -> None:
    backend = AsyncMock()
    backend.query_points.return_value = _response(
        [_point({"text": "", "classification_group": "g"})]
    )
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="missing 'text'"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_rejects_response_without_points_attribute() -> None:
    """Defence in depth: a future client returning a non-QueryResponse
    object surfaces as QdrantUnavailable rather than AttributeError."""
    backend = AsyncMock()
    backend.query_points.return_value = SimpleNamespace()  # no .points
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="missing 'points'"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_timeout_raises_qdrant_timeout() -> None:
    backend = AsyncMock()
    backend.query_points.side_effect = TimeoutError()
    client = _client(backend)

    with pytest.raises(QdrantTimeout, match="timed out"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_unexpected_response_raises_unavailable() -> None:
    backend = AsyncMock()
    backend.query_points.side_effect = UnexpectedResponse(
        status_code=503, reason_phrase="busy", content=b"", headers={}
    )
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="status 503"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_response_handling_error_raises_unavailable() -> None:
    backend = AsyncMock()
    backend.query_points.side_effect = ResponseHandlingException("malformed")
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="response error"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_unknown_exception_wrapped_as_unavailable() -> None:
    backend = AsyncMock()
    backend.query_points.side_effect = ConnectionError("refused")
    client = _client(backend)

    with pytest.raises(QdrantUnavailable, match="ConnectionError"):
        await client.search([0.1] * 4, frozenset({"g"}), limit=10)


async def test_search_empty_results_returns_empty_list() -> None:
    backend = AsyncMock()
    backend.query_points.return_value = _response([])
    client = _client(backend)

    hits, _ = await client.search([0.1] * 4, frozenset({"g"}), limit=10)
    assert hits == []


# --- QdrantClient.check_health -----------------------------------------------


async def test_check_health_collection_exists_with_matching_dim() -> None:
    backend = AsyncMock()
    backend.get_collection.return_value = _collection_info(size=4)
    client = _client(backend, dim=4)

    status = await client.check_health()

    assert status["reachable"] is True
    assert status["collection_exists"] is True
    assert status["vector_dim"] == 4
    assert status["vector_dim_matches"] is True
    assert status["error"] is None


async def test_check_health_dim_mismatch_marks_failed() -> None:
    backend = AsyncMock()
    backend.get_collection.return_value = _collection_info(size=768)
    client = _client(backend, dim=384)

    status = await client.check_health()

    assert status["collection_exists"] is True
    assert status["vector_dim"] == 768
    assert status["vector_dim_matches"] is False


async def test_check_health_collection_not_found() -> None:
    backend = AsyncMock()
    backend.get_collection.side_effect = UnexpectedResponse(
        status_code=404, reason_phrase="Not Found", content=b"", headers={}
    )
    client = _client(backend)

    status = await client.check_health()

    assert status["reachable"] is True
    assert status["collection_exists"] is False
    assert "not found" in (status["error"] or "")


async def test_check_health_other_status_marks_unreachable() -> None:
    backend = AsyncMock()
    backend.get_collection.side_effect = UnexpectedResponse(
        status_code=500, reason_phrase="ISE", content=b"", headers={}
    )
    client = _client(backend)

    status = await client.check_health()
    assert status["reachable"] is False
    assert "500" in (status["error"] or "")


async def test_check_health_timeout() -> None:
    backend = AsyncMock()
    backend.get_collection.side_effect = TimeoutError()
    client = _client(backend)

    status = await client.check_health()
    assert status["reachable"] is False
    assert status["error"] == "timeout"


async def test_check_health_connection_error() -> None:
    backend = AsyncMock()
    backend.get_collection.side_effect = ConnectionError("refused")
    client = _client(backend)

    status = await client.check_health()
    assert status["reachable"] is False
    assert "refused" in (status["error"] or "")


async def test_check_health_handles_named_vectors_config() -> None:
    """Some collections use named-vector config (a dict), not VectorParams.
    dprox's v0.1 collection is unnamed; the helper still tolerates the
    other shape."""
    backend = AsyncMock()
    backend.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"my_vec": SimpleNamespace(size=4)}
            )
        )
    )
    client = _client(backend, dim=4)

    status = await client.check_health()
    assert status["vector_dim"] == 4
    assert status["vector_dim_matches"] is True


# --- API key resolution ------------------------------------------------------


def test_api_key_resolved_from_env(monkeypatch) -> None:
    """The QdrantClient pulls the API key from the env var named in config."""
    monkeypatch.setenv("TEST_QDRANT_API_KEY", "secret-token")
    # We don't construct AsyncQdrantClient for real (would try to connect on
    # first call) — but we can verify the env var is read by inspecting the
    # behaviour indirectly. Smoke: construction succeeds.
    client = QdrantClient(_config(), expected_vector_dim=4)
    # Smoke — the real backend is built lazily
    assert client.url == "http://test-qdrant:6333"
    assert client.collection == "documents"


def test_missing_api_key_env_does_not_raise() -> None:
    """Missing env var → API key is None; Qdrant rejects later if it requires auth."""
    client = QdrantClient(_config(api_key_env="DEFINITELY_NOT_SET_XYZ"), expected_vector_dim=4)
    assert client.collection == "documents"


# --- aclose -------------------------------------------------------------------


async def test_aclose_calls_backend_close() -> None:
    backend = AsyncMock()
    backend.close = AsyncMock(return_value=None)
    client = _client(backend)
    await client.aclose()
    backend.close.assert_awaited_once()


async def test_aclose_tolerates_backend_without_close_method() -> None:
    backend = SimpleNamespace()  # no close() attribute
    client = _client(backend)
    await client.aclose()  # should not raise
