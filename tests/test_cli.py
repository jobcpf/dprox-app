from click.testing import CliRunner

from dprox.cli import cli
from dprox.version import __version__


def test_version_subcommand_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_health_subcommand_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["health"])
    assert result.exit_code == 0
    assert "[OK]" in result.output
