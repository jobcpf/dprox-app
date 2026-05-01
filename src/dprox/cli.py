import asyncio
import sys
from typing import Any

import click
import uvicorn

from dprox.config import Config, ConfigError, load_config, resolve_config_path
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache, PlanError
from dprox.version import __version__


def _load_or_exit(config_path: str | None, exit_code: int) -> Config:
    """Load config or print [FAIL] and exit. Used by serve (3) and health (2)."""
    try:
        return load_config(config_path)
    except ConfigError as exc:
        click.echo(f"[FAIL] config: {exc}", err=True)
        sys.exit(exit_code)


def _build_plan_cache_or_exit(config: Config, exit_code: int) -> PlanCache:
    """Construct PlanCache and run initial_load, or print [FAIL] and exit."""
    cache = PlanCache(config.plan)
    try:
        cache.initial_load()
    except PlanError as exc:
        click.echo(f"[FAIL] plan: {exc}", err=True)
        sys.exit(exit_code)
    return cache


async def _check_ollama(config: Config) -> dict[str, Any]:
    """Run an Ollama health probe. Wrapped as a function so tests can patch it."""
    async with OllamaClient(config.embedding) as client:
        return await client.check_health()


def _format_ollama_line(status: dict[str, Any]) -> tuple[str, bool]:
    """Format the [OK]/[FAIL] line for the CLI. Returns (line, is_ok)."""
    endpoint = status["endpoint"]
    model = status["model"]
    if status["reachable"] and status["model_present"]:
        return f"[OK]   ollama                   {endpoint} model={model}", True
    if not status["reachable"]:
        reason = status.get("error") or "unreachable"
        return (
            f"[FAIL] ollama                   {endpoint} unreachable ({reason})",
            False,
        )
    return (
        f"[FAIL] ollama                   {endpoint} model={model} not in tags",
        False,
    )


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
    plan_cache = _build_plan_cache_or_exit(config, exit_code=3)

    # TLS / mTLS land in build step 4. For now: plain HTTP on the configured bind.
    from dprox.server import create_app

    app = create_app(config, plan_cache)
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

    plan_cache = _build_plan_cache_or_exit(config, exit_code=2)
    agents, admins = plan_cache.counts()
    click.echo(
        f"[OK]   plan                     {config.plan.compiled_plan_path} "
        f"({agents} agents, {admins} admins)"
    )

    ollama_status = asyncio.run(_check_ollama(config))
    line, ollama_ok = _format_ollama_line(ollama_status)
    click.echo(line, err=not ollama_ok)

    # Qdrant check lands in build step 6's CLI integration.
    sys.exit(0 if ollama_ok else 2)


@cli.command(help="Print the version and exit.")
def version() -> None:
    click.echo(__version__)
    sys.exit(0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
