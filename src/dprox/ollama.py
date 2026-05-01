"""Ollama client: query embedding + health check.

Mirrors `dprox-design-spec-v0.1.md` §4.2 (embedding step in /v1/query),
§4.4 (/healthz upstream check), §13.1 #17 (vector-dim mismatch maps to
502 upstream_unavailable), and §9.1 (ollama_ms in audit log).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from dprox.config import EmbeddingConfig


class OllamaError(Exception):
    """Base class for Ollama-related failures."""


class OllamaUnavailable(OllamaError):
    """Ollama unreachable, returned an unexpected response, or sent a
    malformed/wrong-dim vector. Maps to HTTP 502 upstream_unavailable."""


class OllamaTimeout(OllamaError):
    """Ollama exceeded the configured timeout. Maps to HTTP 504 upstream_timeout."""


class OllamaClient:
    """Async client for Ollama's embeddings + tag-list endpoints.

    Wraps a single httpx.AsyncClient so the connection pool is reused
    across calls. Caller is responsible for closing — use `async with`
    or call `aclose()`. In production, FastAPI lifespan manages this.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        kwargs: dict[str, Any] = {
            "base_url": config.endpoint,
            "timeout": httpx.Timeout(config.timeout_seconds),
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._http = httpx.AsyncClient(**kwargs)

    @property
    def endpoint(self) -> str:
        return self._config.endpoint

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def vector_dim(self) -> int:
        return self._config.vector_dim

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def embed(self, text: str) -> tuple[list[float], int]:
        """Return (vector, elapsed_ms). Validates dim before returning.

        Raises OllamaTimeout on timeout; OllamaUnavailable on every other
        failure (network, non-200, malformed body, dim mismatch).
        """
        start = time.monotonic()
        try:
            response = await self._http.post(
                "/api/embeddings",
                json={"model": self._config.model, "prompt": text},
            )
        except httpx.TimeoutException as exc:
            raise OllamaTimeout(
                f"ollama embed timed out after {self._config.timeout_seconds}s"
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaUnavailable(f"ollama unreachable: {exc}") from exc

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if response.status_code != 200:
            raise OllamaUnavailable(
                f"ollama returned status {response.status_code}: {response.text[:200]!r}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaUnavailable(f"ollama response not JSON: {exc}") from exc

        vector = payload.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise OllamaUnavailable(
                "ollama response missing or empty 'embedding' field"
            )
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vector):
            raise OllamaUnavailable("ollama 'embedding' contains non-numeric values")
        if len(vector) != self._config.vector_dim:
            raise OllamaUnavailable(
                f"vector dim mismatch from ollama: got {len(vector)}, "
                f"expected {self._config.vector_dim}"
            )

        return [float(v) for v in vector], elapsed_ms

    async def check_health(self) -> dict[str, Any]:
        """Return structured health status. Does not raise.

        Shape:
            {
                "endpoint": str,
                "model": str,
                "reachable": bool,
                "model_present": bool,
                "error": str | None,
            }
        """
        try:
            response = await self._http.get("/api/tags")
        except httpx.TimeoutException:
            return self._status(False, False, error="timeout")
        except httpx.RequestError as exc:
            return self._status(False, False, error=str(exc) or exc.__class__.__name__)

        if response.status_code != 200:
            return self._status(False, False, error=f"status {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            return self._status(True, False, error=f"non-JSON tags response: {exc}")

        models = payload.get("models", []) if isinstance(payload, dict) else []
        names: set[str] = set()
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.add(entry["name"])

        present = self._config.model in names or any(
            n.startswith(f"{self._config.model}:") for n in names
        )
        return self._status(True, present)

    def _status(
        self,
        reachable: bool,
        model_present: bool,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "endpoint": self._config.endpoint,
            "model": self._config.model,
            "reachable": reachable,
            "model_present": model_present,
            "error": error,
        }
