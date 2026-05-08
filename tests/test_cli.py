from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from dprox.cli import cli
from dprox.version import __version__


@pytest.fixture
def stub_ollama_ok(monkeypatch):
    """Make `dprox health`'s ollama probe succeed without real network."""

    async def fake(_config) -> dict:
        return {
            "endpoint": "http://stub:11434",
            "model": "nomic-embed-text",
            "reachable": True,
            "model_present": True,
            "error": None,
        }

    monkeypatch.setattr("dprox.cli._check_ollama", fake)


@pytest.fixture
def stub_ollama_unreachable(monkeypatch):
    async def fake(_config) -> dict:
        return {
            "endpoint": "http://stub:11434",
            "model": "nomic-embed-text",
            "reachable": False,
            "model_present": False,
            "error": "connection refused",
        }

    monkeypatch.setattr("dprox.cli._check_ollama", fake)


@pytest.fixture
def stub_ollama_model_missing(monkeypatch):
    async def fake(_config) -> dict:
        return {
            "endpoint": "http://stub:11434",
            "model": "nomic-embed-text",
            "reachable": True,
            "model_present": False,
            "error": None,
        }

    monkeypatch.setattr("dprox.cli._check_ollama", fake)


@pytest.fixture
def stub_qdrant_ok(monkeypatch):
    async def fake(_config) -> dict:
        return {
            "url": "http://stub-qdrant:6333",
            "collection": "documents",
            "reachable": True,
            "collection_exists": True,
            "vector_dim": 768,
            "vector_dim_matches": True,
            "error": None,
        }

    monkeypatch.setattr("dprox.cli._check_qdrant", fake)


@pytest.fixture
def stub_qdrant_unreachable(monkeypatch):
    async def fake(_config) -> dict:
        return {
            "url": "http://stub-qdrant:6333",
            "collection": "documents",
            "reachable": False,
            "collection_exists": False,
            "vector_dim": None,
            "vector_dim_matches": False,
            "error": "connection refused",
        }

    monkeypatch.setattr("dprox.cli._check_qdrant", fake)


@pytest.fixture
def stub_qdrant_collection_missing(monkeypatch):
    async def fake(_config) -> dict:
        return {
            "url": "http://stub-qdrant:6333",
            "collection": "documents",
            "reachable": True,
            "collection_exists": False,
            "vector_dim": None,
            "vector_dim_matches": False,
            "error": "collection not found",
        }

    monkeypatch.setattr("dprox.cli._check_qdrant", fake)


@pytest.fixture
def stub_qdrant_dim_mismatch(monkeypatch):
    async def fake(_config) -> dict:
        return {
            "url": "http://stub-qdrant:6333",
            "collection": "documents",
            "reachable": True,
            "collection_exists": True,
            "vector_dim": 384,
            "vector_dim_matches": False,
            "error": None,
        }

    monkeypatch.setattr("dprox.cli._check_qdrant", fake)


def test_version_subcommand_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_health_all_ok_exits_zero(
    write_config, baseline_config_dict, stub_ollama_ok, stub_qdrant_ok
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 0
    assert "[OK]   config" in result.output
    assert "org=test" in result.output
    assert "[OK]   plan" in result.output
    assert "3 agents, 1 admins" in result.output
    assert "[OK]   ollama" in result.output
    assert "[OK]   qdrant.connect" in result.output
    assert "[OK]   qdrant.collection" in result.output
    assert "dim=768" in result.output


def test_health_ollama_unreachable_exits_two(
    write_config, baseline_config_dict, stub_ollama_unreachable, stub_qdrant_ok
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[OK]   config" in result.output
    assert "[OK]   plan" in result.output
    assert "[FAIL] ollama" in result.output


def test_health_model_missing_exits_two(
    write_config, baseline_config_dict, stub_ollama_model_missing, stub_qdrant_ok
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[FAIL] ollama" in result.output
    assert "not in tags" in result.output


def test_health_qdrant_unreachable_exits_two(
    write_config, baseline_config_dict, stub_ollama_ok, stub_qdrant_unreachable
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[OK]   ollama" in result.output
    assert "[FAIL] qdrant.connect" in result.output
    assert "refused" in result.output


def test_health_qdrant_collection_missing_exits_two(
    write_config, baseline_config_dict, stub_ollama_ok, stub_qdrant_collection_missing
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[OK]   qdrant.connect" in result.output
    assert "[FAIL] qdrant.collection" in result.output
    assert "not found" in result.output


def test_health_qdrant_dim_mismatch_exits_two(
    write_config, baseline_config_dict, stub_ollama_ok, stub_qdrant_dim_mismatch
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[FAIL] qdrant.collection" in result.output
    assert "vector dim mismatch" in result.output
    assert "got 384" in result.output


def test_health_missing_config_exits_two() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", "/nope/missing.yml"])
    assert result.exit_code == 2
    assert "[FAIL] config" in result.output


def test_health_bad_plan_exits_two(
    write_config, baseline_config_dict, tmp_path: Path
) -> None:
    baseline_config_dict["plan"]["compiled_plan_path"] = str(tmp_path / "no-plan.yml")
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[OK]   config" in result.output
    assert "[FAIL] plan" in result.output


def test_serve_invalid_config_exits_three(write_config, baseline_config_dict) -> None:
    baseline_config_dict["embedding"]["vector_dim"] = -1
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(path)])
    assert result.exit_code == 3
    assert "[FAIL] config" in result.output


def test_serve_bad_plan_exits_three(
    write_config, baseline_config_dict, tmp_path: Path
) -> None:
    baseline_config_dict["plan"]["compiled_plan_path"] = str(tmp_path / "no-plan.yml")
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(path)])
    assert result.exit_code == 3
    assert "[FAIL] plan" in result.output


def test_serve_missing_config_exits_three(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(tmp_path / "nope.yml")])
    assert result.exit_code == 3
    assert "[FAIL]" in result.output


def test_serve_missing_cert_files_exits_three(
    write_config, baseline_config_dict, tmp_path: Path
) -> None:
    """When client_cert_mode != 'off', the three cert files must exist before serve."""
    baseline_config_dict["mtls"]["client_cert_mode"] = "optional"
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(path)])
    assert result.exit_code == 3
    assert "[FAIL] mtls" in result.output
    assert "missing" in result.output


def test_serve_with_cert_files_present_proceeds_past_cert_check(
    write_config, baseline_config_dict, tmp_path: Path, monkeypatch
) -> None:
    """Empty placeholder cert files satisfy the existence check; serve then trips on
    the bad plan path so we don't actually need uvicorn to start."""
    baseline_config_dict["mtls"]["client_cert_mode"] = "optional"
    for name in ("ca.crt", "server.crt", "server.key"):
        (tmp_path / name).write_bytes(b"placeholder")
    baseline_config_dict["plan"]["compiled_plan_path"] = str(tmp_path / "missing.yml")
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(path)])
    assert result.exit_code == 3
    assert "[FAIL] plan" in result.output  # cert check passed; plan check failed
