from __future__ import annotations

from pathlib import Path

import pytest

from dprox.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
    resolve_config_path,
)


def test_baseline_config_loads(baseline_config: Config) -> None:
    assert baseline_config.org == "test"
    assert baseline_config.server.host == "127.0.0.1"
    assert baseline_config.server.port == 8443
    assert baseline_config.qdrant.api_key_env == "QDRANT_RO_API_KEY"
    assert baseline_config.logging.format == "json"


def test_loads_from_file(write_config, baseline_config_dict) -> None:
    path = write_config(baseline_config_dict)
    config = load_config(path)
    assert config.org == "test"
    assert isinstance(config.mtls.ca_cert_path, Path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yml")


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a file"):
        load_config(tmp_path)


def test_empty_file_raises(tmp_path: Path) -> None:
    target = tmp_path / "empty.yml"
    target.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(target)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    target = tmp_path / "list.yml"
    target.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(target)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    target = tmp_path / "bad.yml"
    target.write_text("this: is: not: valid: yaml:\n  - {", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML parse error"):
        load_config(target)


def test_missing_required_top_level_raises(write_config, baseline_config_dict) -> None:
    del baseline_config_dict["embedding"]
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_unknown_top_level_field_rejected(write_config, baseline_config_dict) -> None:
    baseline_config_dict["surprise"] = True
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_unknown_nested_field_rejected(write_config, baseline_config_dict) -> None:
    baseline_config_dict["server"]["typo"] = 1
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_bind_must_have_port(write_config, baseline_config_dict) -> None:
    baseline_config_dict["server"]["bind"] = "127.0.0.1"
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="host:port"):
        load_config(path)


def test_bind_port_out_of_range(write_config, baseline_config_dict) -> None:
    baseline_config_dict["server"]["bind"] = "127.0.0.1:70000"
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="1..65535"):
        load_config(path)


def test_negative_timeout_rejected(write_config, baseline_config_dict) -> None:
    baseline_config_dict["embedding"]["timeout_seconds"] = -1
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_max_limit_below_default_rejected(write_config, baseline_config_dict) -> None:
    baseline_config_dict["qdrant"]["default_limit"] = 50
    baseline_config_dict["qdrant"]["max_limit"] = 10
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="max_limit"):
        load_config(path)


def test_invalid_log_level_rejected(write_config, baseline_config_dict) -> None:
    baseline_config_dict["logging"]["level"] = "VERBOSE"
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_invalid_tls_version_rejected(write_config, baseline_config_dict) -> None:
    baseline_config_dict["mtls"]["tls_min_version"] = "TLSv1.0"
    path = write_config(baseline_config_dict)
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_logging_section_optional(write_config, baseline_config_dict) -> None:
    del baseline_config_dict["logging"]
    path = write_config(baseline_config_dict)
    config = load_config(path)
    assert config.logging.level == "INFO"
    assert config.logging.format == "json"
    assert config.logging.log_query_text is False


def test_resolve_path_explicit_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DPROX_CONFIG", "/should/be/ignored.yml")
    explicit = tmp_path / "explicit.yml"
    assert resolve_config_path(explicit) == explicit


def test_resolve_path_uses_env(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "from-env.yml"
    monkeypatch.setenv("DPROX_CONFIG", str(target))
    assert resolve_config_path() == target


def test_resolve_path_default(monkeypatch) -> None:
    monkeypatch.delenv("DPROX_CONFIG", raising=False)
    assert resolve_config_path() == Path(DEFAULT_CONFIG_PATH)


def test_load_uses_env_path(monkeypatch, write_config, baseline_config_dict) -> None:
    path = write_config(baseline_config_dict, name="from-env.yml")
    monkeypatch.setenv("DPROX_CONFIG", str(path))
    config = load_config()
    assert config.org == "test"
