from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from dprox.cli import cli
from dprox.version import __version__


def test_version_subcommand_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_health_with_valid_config_exits_zero(write_config, baseline_config_dict) -> None:
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", str(path)])
    assert result.exit_code == 0
    assert "[OK]" in result.output
    assert "config" in result.output
    assert "org=test" in result.output


def test_health_missing_config_exits_two() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["health", "--config", "/nope/missing.yml"])
    assert result.exit_code == 2
    assert "[FAIL]" in result.output


def test_serve_invalid_config_exits_three(write_config, baseline_config_dict) -> None:
    baseline_config_dict["embedding"]["vector_dim"] = -1
    path = write_config(baseline_config_dict)
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(path)])
    assert result.exit_code == 3
    assert "[FAIL]" in result.output


def test_serve_missing_config_exits_three(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--config", str(tmp_path / "nope.yml")])
    assert result.exit_code == 3
    assert "[FAIL]" in result.output
