"""Qdrant client: RBAC-filtered vector search + health check.

Mirrors `dprox-design-spec-v0.2.md` §3.2 (filter is non-negotiable),
§3.4 (critical invariant), §4.2 (result shape), §4.4 (/healthz upstream
check), §9.1 (qdrant_ms in audit log).

`build_classification_filter` is the single point in the codebase that
constructs the Qdrant filter for /v1/query. Per spec §10.3, this code
path requires 100% test coverage — it is the security boundary.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.models import FieldCondition, Filter, MatchAny

from dprox.config import QdrantConfig

CLASSIFICATION_GROUP_KEY = "classification_group"


class QdrantError(Exception):
    """Base for Qdrant-related failures."""


class QdrantUnavailable(QdrantError):
    """Maps to HTTP 502 upstream_unavailable."""


class QdrantTimeout(QdrantError):
    """Maps to HTTP 504 upstream_timeout."""


@dataclass(frozen=True)
class QdrantHit:
    """One Qdrant search result projected to dprox response shape (spec §4.2)."""

    text: str
    classification_group: str
    score: float
    source_path_rel: str | None = None
    file_type: str | None = None
    chunk_index: int | None = None
    chunk_total: int | None = None
    modified_at: str | None = None
    indexed_at: str | None = None


def build_classification_filter(groups: frozenset[str]) -> Filter:
    """Build the RBAC filter for a Qdrant search.

    THE SECURITY BOUNDARY. Caller must pass groups derived only from the
    verified mTLS peer cert's CN, resolved through compiled_plan.yml.
    There must be no code path where a caller-supplied parameter
    contributes to this filter (spec §3.4 critical invariant).

    Sorting the group list makes audit log diffs deterministic.
    """
    return Filter(
        must=[
            FieldCondition(
                key=CLASSIFICATION_GROUP_KEY,
                match=MatchAny(any=sorted(groups)),
            )
        ]
    )


class QdrantClient:
    """Async Qdrant client with health probe + RBAC-filtered search.

    Wraps qdrant-client's AsyncQdrantClient. Caller is responsible for
    closing — use `async with` or call `aclose()`. In production,
    FastAPI lifespan manages this.
    """

    def __init__(
        self,
        config: QdrantConfig,
        expected_vector_dim: int,
        *,
        backend: Any | None = None,
    ) -> None:
        """
        Args:
            config: QdrantConfig (url, api_key_env, collection, timeouts).
            expected_vector_dim: dim from EmbeddingConfig — health check
                verifies the collection's vector size matches.
            backend: optional pre-built async client (test injection).
                When None, a real AsyncQdrantClient is constructed using
                the API key from env var named by `config.api_key_env`.
        """
        self._config = config
        self._expected_dim = expected_vector_dim
        if backend is not None:
            self._client: Any = backend
        else:
            api_key = os.environ.get(config.api_key_env) or None
            self._client = AsyncQdrantClient(
                url=config.url,
                api_key=api_key,
                timeout=config.timeout_seconds,
            )

    @property
    def url(self) -> str:
        return self._config.url

    @property
    def collection(self) -> str:
        return self._config.collection

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result

    async def __aenter__(self) -> QdrantClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def search(
        self,
        query_vector: list[float],
        groups: frozenset[str],
        limit: int,
    ) -> tuple[list[QdrantHit], int]:
        """Run RBAC-filtered vector search. Returns (hits, elapsed_ms).

        Raises QdrantTimeout on timeout, QdrantUnavailable on every other
        failure (network, non-200, malformed response, missing payload).
        """
        q_filter = build_classification_filter(groups)
        start = time.monotonic()
        try:
            # qdrant-client >=1.10 exposes the universal query API. Server
            # 1.13.x supports it; AsyncQdrantClient.search() was removed by
            # 1.18. Response is QueryResponse with .points: list[ScoredPoint]
            # (the ScoredPoint shape — id/score/payload — is unchanged from
            # the old search() return type, so _point_to_hit is the same).
            response = await self._client.query_points(
                collection_name=self._config.collection,
                query=query_vector,
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
        except TimeoutError as exc:
            raise QdrantTimeout(
                f"qdrant search timed out after {self._config.timeout_seconds}s"
            ) from exc
        except UnexpectedResponse as exc:
            raise QdrantUnavailable(
                f"qdrant returned status {exc.status_code}"
            ) from exc
        except ResponseHandlingException as exc:
            raise QdrantUnavailable(f"qdrant response error: {exc}") from exc
        except QdrantError:
            raise
        except Exception as exc:
            raise QdrantUnavailable(
                f"qdrant unreachable: {exc.__class__.__name__}: {exc}"
            ) from exc

        elapsed_ms = int((time.monotonic() - start) * 1000)
        points = getattr(response, "points", None)
        if points is None:
            raise QdrantUnavailable(
                "qdrant query_points response missing 'points' field"
            )
        hits = [self._point_to_hit(p) for p in points]
        return hits, elapsed_ms

    @staticmethod
    def _point_to_hit(point: Any) -> QdrantHit:
        payload = getattr(point, "payload", None) or {}
        text = payload.get("text")
        group = payload.get("classification_group")
        if not isinstance(text, str) or not text:
            raise QdrantUnavailable(
                f"qdrant point id={getattr(point, 'id', '?')!r} missing 'text' payload"
            )
        if not isinstance(group, str) or not group:
            point_id = getattr(point, "id", "?")
            raise QdrantUnavailable(
                f"qdrant point id={point_id!r} missing 'classification_group' payload"
            )
        return QdrantHit(
            text=text,
            classification_group=group,
            score=float(getattr(point, "score", 0.0)),
            source_path_rel=payload.get("source_path_rel"),
            file_type=payload.get("file_type"),
            chunk_index=payload.get("chunk_index"),
            chunk_total=payload.get("chunk_total"),
            modified_at=payload.get("modified_at"),
            indexed_at=payload.get("indexed_at"),
        )

    async def check_health(self) -> dict[str, Any]:
        """Return structured health status. Does not raise.

        Shape:
            {
                "url": str, "collection": str,
                "reachable": bool, "collection_exists": bool,
                "vector_dim": int | None,
                "vector_dim_matches": bool,
                "error": str | None,
            }
        """
        try:
            info = await self._client.get_collection(self._config.collection)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return self._status(
                    reachable=True,
                    collection_exists=False,
                    error="collection not found",
                )
            return self._status(reachable=False, error=f"status {exc.status_code}")
        except TimeoutError:
            return self._status(reachable=False, error="timeout")
        except Exception as exc:
            return self._status(
                reachable=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )

        vector_dim = self._extract_vector_dim(info)
        matches = vector_dim == self._expected_dim
        return self._status(
            reachable=True,
            collection_exists=True,
            vector_dim=vector_dim,
            vector_dim_matches=matches,
        )

    @staticmethod
    def _extract_vector_dim(info: Any) -> int | None:
        """Pull the vector size out of CollectionInfo, tolerating named-vector configs."""
        try:
            vectors = info.config.params.vectors
        except AttributeError:
            return None
        size = getattr(vectors, "size", None)
        if isinstance(size, int):
            return size
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            inner_size = getattr(first, "size", None)
            return inner_size if isinstance(inner_size, int) else None
        return None

    def _status(
        self,
        *,
        reachable: bool,
        collection_exists: bool = False,
        vector_dim: int | None = None,
        vector_dim_matches: bool = False,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "url": self._config.url,
            "collection": self._config.collection,
            "reachable": reachable,
            "collection_exists": collection_exists,
            "vector_dim": vector_dim,
            "vector_dim_matches": vector_dim_matches,
            "error": error,
        }
