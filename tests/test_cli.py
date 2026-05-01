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


def test_version_subcommand_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_health_with_valid_config_plan_and_ollama_exits_zero(
    write_config, baseline_config_dict, stub_ollama_ok
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
    assert "model=nomic-embed-text" in result.output


def test_health_ollama_unreachable_exits_two(
    write_config, baseline_config_dict, stub_ollama_unreachable
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[OK]   config" in result.output
    assert "[OK]   plan" in result.output
    assert "[FAIL] ollama" in result.output
    assert "unreachable" in result.output


def test_health_model_missing_exits_two(
    write_config, baseline_config_dict, stub_ollama_model_missing
) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 2
    assert "[FAIL] ollama" in result.output
    assert "not in tags" in result.output


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
