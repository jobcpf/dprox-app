import sys

import click
import uvicorn

from dprox.config import Config, ConfigError, load_config, resolve_config_path
from dprox.version import __version__


def _load_or_exit(config_path: str | None, exit_code: int) -> Config:
    """Load config or print [FAIL] and exit. Used by serve (exit 3) and health (exit 2)."""
    try:
        return load_config(config_path)
    except ConfigError as exc:
        click.echo(f"[FAIL] config: {exc}", err=True)
        sys.exit(exit_code)


@click.group(help="dprox — RBAC-enforcing query proxy.")
def cli() -> None:
    pass


@cli.command(help="Start the HTTP(S) server.")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to config.yml. Falls back to DPROX_CONFIG, then /etc/dprox/config.yml.",
)
def serve(config_path: str | None) -> None:
    config = _load_or_exit(config_path, exit_code=3)

    # TLS / mTLS land in build step 4. For now: plain HTTP on the configured bind.
    from dprox.server import create_app

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
    )


@cli.command(help="Run startup + upstream checks and exit.")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to config.yml. Falls back to DPROX_CONFIG, then /etc/dprox/config.yml.",
)
def health(config_path: str | None) -> None:
    resolved = resolve_config_path(config_path)
    config = _load_or_exit(config_path, exit_code=2)
    click.echo(f"[OK]   config                   {resolved} (org={config.org})")
    # Upstream checks (Qdrant, Ollama, plan) land in build step 8.
    sys.exit(0)


@cli.command(help="Print the version and exit.")
def version() -> None:
    click.echo(__version__)
    sys.exit(0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
