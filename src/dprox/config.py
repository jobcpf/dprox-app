"""Config schema and loader for dprox.

Mirrors `dprox-design-spec-v0.1.md` §6.2. Loaded once at startup; downstream
components consume the validated `Config` object. No per-org defaults are
baked in here — every per-org value comes from config.yml or env vars
templated by the platform's Ansible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_PATH = "/etc/dprox/config.yml"
ENV_CONFIG_PATH = "DPROX_CONFIG"


class ConfigError(Exception):
    """Raised when config loading or validation fails. Maps to exit code 3."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ServerConfig(_Strict):
    bind: str = Field(min_length=3)
    request_timeout_seconds: int = Field(gt=0, le=600)
    max_request_body_bytes: int = Field(gt=0, le=10_000_000)

    @field_validator("bind")
    @classmethod
    def _validate_bind(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("bind must be 'host:port'")
        host, _, port_s = v.rpartition(":")
        if not host or not port_s:
            raise ValueError("bind must be 'host:port'")
        try:
            port = int(port_s)
        except ValueError as exc:
            raise ValueError("bind port must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("bind port must be in 1..65535")
        return v

    @property
    def host(self) -> str:
        return self.bind.rsplit(":", 1)[0]

    @property
    def port(self) -> int:
        return int(self.bind.rsplit(":", 1)[1])


class MTLSConfig(_Strict):
    ca_cert_path: Path
    server_cert_path: Path
    server_key_path: Path
    client_cert_mode: Literal["optional", "required", "off"] = "optional"
    cn_to_agent_strategy: Literal["cn_equals_name"] = "cn_equals_name"
    tls_min_version: Literal["TLSv1.2", "TLSv1.3"] = "TLSv1.3"
    tls_pin_enabled: bool = True


class PlanConfig(_Strict):
    compiled_plan_path: Path
    cache_enabled: bool = True
    reload_on_mtime_change: bool = True
    reload_on_unknown_cn: bool = True
    reload_min_interval_seconds: int = Field(default=5, ge=0, le=3600)


class EmbeddingConfig(_Strict):
    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    vector_dim: int = Field(gt=0, le=8192)
    timeout_seconds: int = Field(gt=0, le=300)


class QdrantConfig(_Strict):
    url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    default_limit: int = Field(gt=0)
    max_limit: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0, le=300)

    @field_validator("max_limit")
    @classmethod
    def _max_ge_default(cls, v: int, info) -> int:
        default = info.data.get("default_limit")
        if default is not None and v < default:
            raise ValueError("max_limit must be >= default_limit")
        return v


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    log_query_text: bool = False


class Config(_Strict):
    org: str = Field(min_length=1)
    server: ServerConfig
    mtls: MTLSConfig
    plan: PlanConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    logging: LoggingConfig = LoggingConfig()


def resolve_config_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_CONFIG_PATH)
    if env:
        return Path(env)
    return Path(DEFAULT_CONFIG_PATH)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    resolved = resolve_config_path(path)

    if not resolved.exists():
        raise ConfigError(f"config file not found: {resolved}")
    if not resolved.is_file():
        raise ConfigError(f"config path is not a file: {resolved}")

    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config {resolved}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {resolved}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {resolved}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(raw).__name__}: {resolved}"
        )

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"config validation failed for {resolved}:\n{exc}") from exc
