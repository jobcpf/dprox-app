from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from dprox.config import Config


def _baseline_config_dict(tmp_path: Path) -> dict:
    return {
        "org": "test",
        "server": {
            "bind": "127.0.0.1:8443",
            "request_timeout_seconds": 30,
            "max_request_body_bytes": 65536,
        },
        "mtls": {
            "ca_cert_path": str(tmp_path / "ca.crt"),
            "server_cert_path": str(tmp_path / "server.crt"),
            "server_key_path": str(tmp_path / "server.key"),
            "client_cert_mode": "optional",
            "cn_to_agent_strategy": "cn_equals_name",
            "tls_min_version": "TLSv1.3",
            "tls_pin_enabled": True,
        },
        "plan": {
            "compiled_plan_path": str(tmp_path / "compiled_plan.yml"),
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
def baseline_config_dict(tmp_path: Path) -> dict:
    """A valid config dict, with cert/plan paths under tmp_path (not required to exist)."""
    return _baseline_config_dict(tmp_path)


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
