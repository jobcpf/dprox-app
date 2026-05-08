from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import yaml

from dprox.config import Config
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache


@pytest.fixture
def plan_yaml(tmp_path: Path) -> Path:
    """Write a small but realistic compiled plan to tmp_path and return its path."""
    target = tmp_path / "compiled_plan.yml"
    target.write_text(
        textwrap.dedent(
            """
            meta:
              compiled_at: '2026-01-01T00:00:00Z'
              compiler_version: 0.3.0
            required_groups:
            - g_engineering
            - g_finance
            - g_admin
            agent_users:
            - name: agent_alice
              description: Engineering test agent
              groups:
              - g_engineering
            - name: agent_bob
              description: Finance test agent
              groups:
              - g_finance
            - name: agent_oversight
              description: Cross-functional test agent
              groups:
              - g_engineering
              - g_finance
            admin_users:
            - name: admin_alice
              groups:
              - g_engineering
              - g_finance
              - g_admin
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return target


def _baseline_config_dict(tmp_path: Path, plan_path: Path) -> dict:
    return {
        "org": "test",
        "server": {
            "bind": "127.0.0.1:8443",
            "request_timeout_seconds": 30,
            "max_request_body_bytes": 65536,
        },
        "mtls": {
            # Default to "off" so tests can exercise non-mTLS paths without
            # creating cert files. Tests of cert validation use synthetic
            # certs from tests/cert_helpers.py via validate_client_cert
            # directly. Tests that exercise the full uvicorn TLS startup
            # would override these fields.
            "ca_cert_path": str(tmp_path / "ca.crt"),
            "server_cert_path": str(tmp_path / "server.crt"),
            "server_key_path": str(tmp_path / "server.key"),
            "client_cert_mode": "off",
            "cn_to_agent_strategy": "cn_equals_name",
            "tls_min_version": "TLSv1.3",
            "tls_pin_enabled": True,
        },
        "plan": {
            "compiled_plan_path": str(plan_path),
            "cache_enabled": True,
            "reload_on_mtime_change": True,
            "reload_on_unknown_cn": True,
            "reload_min_interval_seconds": 5,
        },
        "embedding": {
            "endpoint": "http://localhost:11434",
            "model": "nomic-embed-text",
            "vector_dim": 768,
            "timeout_seconds": 10,
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key_env": "QDRANT_RO_API_KEY",
            "collection": "documents",
            "default_limit": 10,
            "max_limit": 50,
            "timeout_seconds": 10,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "log_query_text": False,
        },
    }


@pytest.fixture
def baseline_config_dict(tmp_path: Path, plan_yaml: Path) -> dict:
    """A valid config dict with a usable plan path. Cert paths exist only as strings."""
    return _baseline_config_dict(tmp_path, plan_yaml)


@pytest.fixture
def baseline_config(baseline_config_dict: dict) -> Config:
    return Config.model_validate(baseline_config_dict)


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict], Path]:
    """Return a callable that writes a config dict to a YAML file under tmp_path."""

    def _write(data: dict, name: str = "config.yml") -> Path:
        target = tmp_path / name
        target.write_text(yaml.safe_dump(data), encoding="utf-8")
        return target

    return _write


@pytest.fixture
def plan_cache(baseline_config: Config) -> PlanCache:
    cache = PlanCache(baseline_config.plan)
    cache.initial_load()
    return cache


@pytest.fixture
def mock_ollama_handler(baseline_config: Config) -> Callable[[httpx.Request], httpx.Response]:
    """Default mock: model present in tags, embed returns a vector of the right dim."""
    expected_model = baseline_config.embedding.model
    expected_dim = baseline_config.embedding.vector_dim

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": f"{expected_model}:latest"}]},
            )
        if request.url.path == "/api/embeddings":
            return httpx.Response(200, json={"embedding": [0.1] * expected_dim})
        return httpx.Response(404, json={"error": f"unexpected path {request.url.path}"})

    return handler


@pytest.fixture
def mock_ollama(
    baseline_config: Config,
    mock_ollama_handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaClient:
    """An OllamaClient wired to a MockTransport. Pass into create_app()."""
    return OllamaClient(
        baseline_config.embedding,
        transport=httpx.MockTransport(mock_ollama_handler),
    )
