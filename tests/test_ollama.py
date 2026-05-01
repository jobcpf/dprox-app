from __future__ import annotations

import json

import httpx
import pytest

from dprox.config import EmbeddingConfig
from dprox.ollama import OllamaClient, OllamaTimeout, OllamaUnavailable


def _config(**overrides) -> EmbeddingConfig:
    base: dict = {
        "endpoint": "http://test-ollama:11434",
        "model": "test-model",
        "vector_dim": 4,
        "timeout_seconds": 5,
    }
    base.update(overrides)
    return EmbeddingConfig(**base)


def _client(handler, *, config: EmbeddingConfig | None = None) -> OllamaClient:
    return OllamaClient(config or _config(), transport=httpx.MockTransport(handler))


# -- embed: happy path ---------------------------------------------------------


async def test_embed_happy_path() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3, 0.4]})

    async with _client(handler) as client:
        vector, elapsed_ms = await client.embed("hello world")

    assert captured["path"] == "/api/embeddings"
    assert captured["body"] == {"model": "test-model", "prompt": "hello world"}
    assert vector == [0.1, 0.2, 0.3, 0.4]
    assert elapsed_ms >= 0


async def test_embed_returns_floats_even_for_int_inputs() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [1, 2, 3, 4]})

    async with _client(handler) as client:
        vector, _ = await client.embed("x")

    assert vector == [1.0, 2.0, 3.0, 4.0]
    assert all(isinstance(v, float) for v in vector)


# -- embed: timeouts and connection errors ------------------------------------


async def test_embed_timeout_raises_ollama_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated")

    async with _client(handler) as client:
        with pytest.raises(OllamaTimeout, match="timed out"):
            await client.embed("hello")


async def test_embed_connect_error_raises_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="unreachable"):
            await client.embed("hello")


# -- embed: server-side response problems --------------------------------------


async def test_embed_non_200_status_raises_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="kaboom")

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="status 500"):
            await client.embed("hello")


async def test_embed_non_json_response_raises_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="not JSON"):
            await client.embed("hello")


async def test_embed_missing_embedding_field_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": "bar"})

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="missing or empty"):
            await client.embed("hello")


async def test_embed_empty_embedding_field_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": []})

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="missing or empty"):
            await client.embed("hello")


async def test_embed_non_numeric_values_raise() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1, "x", 0.3, 0.4]})

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="non-numeric"):
            await client.embed("hello")


async def test_embed_dim_mismatch_raises_with_descriptive_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    async with _client(handler) as client:
        with pytest.raises(OllamaUnavailable, match="dim mismatch.*got 2.*expected 4"):
            await client.embed("hello")


# -- check_health: happy paths -------------------------------------------------


async def test_check_health_ok_with_tagged_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "test-model:latest"},
                    {"name": "other:latest"},
                ]
            },
        )

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is True
    assert status["model_present"] is True
    assert status["error"] is None
    assert status["endpoint"] == "http://test-ollama:11434"
    assert status["model"] == "test-model"


async def test_check_health_ok_with_bare_model_name() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "test-model"}]})

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is True
    assert status["model_present"] is True


async def test_check_health_model_not_in_tags() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "different:latest"}]})

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is True
    assert status["model_present"] is False
    assert status["error"] is None


# -- check_health: failure modes ----------------------------------------------


async def test_check_health_unreachable_returns_structured_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is False
    assert status["model_present"] is False
    assert "refused" in (status["error"] or "")


async def test_check_health_timeout_returns_timeout_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow ollama")

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is False
    assert status["error"] == "timeout"


async def test_check_health_non_200_status_marked_unreachable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is False
    assert "503" in (status["error"] or "")


async def test_check_health_non_json_marks_present_false_with_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"surprise html")

    async with _client(handler) as client:
        status = await client.check_health()

    assert status["reachable"] is True
    assert status["model_present"] is False
    assert "non-JSON" in (status["error"] or "")
