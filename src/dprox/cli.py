import asyncio
import ssl
import sys
from typing import Any

import click
import uvicorn

from dprox.config import Config, ConfigError, MTLSConfig, load_config, resolve_config_path
from dprox.mtls import DproxHttpProtocol
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache, PlanError
from dprox.qdrant import QdrantClient
from dprox.version import __version__

_CERT_REQS = {
    "off": ssl.CERT_NONE,
    "optional": ssl.CERT_OPTIONAL,
    "required": ssl.CERT_REQUIRED,
}


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


def _verify_cert_files_or_exit(cfg: MTLSConfig) -> None:
    """If mTLS is enabled, verify the three cert files exist before uvicorn opens the listener."""
    if cfg.client_cert_mode == "off":
        return
    missing = [
        p for p in (cfg.ca_cert_path, cfg.server_cert_path, cfg.server_key_path) if not p.exists()
    ]
    if missing:
        click.echo(
            "[FAIL] mtls: cert file(s) missing: " + ", ".join(str(p) for p in missing),
            err=True,
        )
        click.echo(
            "       For local dev, run: python scripts/dev_certs.py",
            err=True,
        )
        sys.exit(3)


def _build_uvicorn_config(config: Config, app: Any) -> uvicorn.Config:
    """Build a uvicorn Config with TLS + custom protocol when mTLS is enabled."""
    kwargs: dict[str, Any] = {
        "host": config.server.host,
        "port": config.server.port,
        "log_level": config.logging.level.lower(),
    }

    if config.mtls.client_cert_mode != "off":
        kwargs.update(
            {
                "ssl_keyfile": str(config.mtls.server_key_path),
                "ssl_certfile": str(config.mtls.server_cert_path),
                "ssl_ca_certs": str(config.mtls.ca_cert_path),
                "ssl_cert_reqs": _CERT_REQS[config.mtls.client_cert_mode],
                "http": DproxHttpProtocol,
            }
        )

    uconfig = uvicorn.Config(app, **kwargs)

    # Materialize the SSLContext now so we can pin minimum_version before serving.
    # Config.load() is idempotent — uvicorn calls it again on Server.run if needed.
    if config.mtls.client_cert_mode != "off":
        uconfig.load()
        if config.mtls.tls_pin_enabled and uconfig.ssl is not None:
            uconfig.ssl.minimum_version = (
                ssl.TLSVersion.TLSv1_3
                if config.mtls.tls_min_version == "TLSv1.3"
                else ssl.TLSVersion.TLSv1_2
            )

    return uconfig


async def _check_ollama(config: Config) -> dict[str, Any]:
    """Run an Ollama health probe. Wrapped as a function so tests can patch it."""
    async with OllamaClient(config.embedding) as client:
        return await client.check_health()


async def _check_qdrant(config: Config) -> dict[str, Any]:
    """Run a Qdrant health probe. Wrapped as a function so tests can patch it."""
    async with QdrantClient(config.qdrant, config.embedding.vector_dim) as client:
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


def _format_qdrant_lines(status: dict[str, Any]) -> tuple[list[str], bool]:
    """Format the qdrant.connect + qdrant.collection [OK]/[FAIL] lines.

    Returns (lines, all_ok). Mirrors spec §5.2 sample output.
    """
    url = status["url"]
    collection = status["collection"]
    lines: list[str] = []

    if not status["reachable"]:
        reason = status.get("error") or "unreachable"
        lines.append(f"[FAIL] qdrant.connect            {url} ({reason})")
        return lines, False

    lines.append(f"[OK]   qdrant.connect            {url}")

    if not status["collection_exists"]:
        lines.append(
            f"[FAIL] qdrant.collection         {collection} (not found)"
        )
        return lines, False

    if not status["vector_dim_matches"]:
        dim = status.get("vector_dim")
        lines.append(
            f"[FAIL] qdrant.collection         {collection} "
            f"(vector dim mismatch: got {dim})"
        )
        return lines, False

    dim = status["vector_dim"]
    lines.append(f"[OK]   qdrant.collection         {collection} (dim={dim})")
    return lines, True


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
    _verify_cert_files_or_exit(config.mtls)
    plan_cache = _build_plan_cache_or_exit(config, exit_code=3)

    from dprox.server import create_app

    app = create_app(config, plan_cache)
    uconfig = _build_uvicorn_config(config, app)
    server = uvicorn.Server(uconfig)
    server.run()


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
    ollama_line, ollama_ok = _format_ollama_line(ollama_status)
    click.echo(ollama_line, err=not ollama_ok)

    qdrant_status = asyncio.run(_check_qdrant(config))
    qdrant_lines, qdrant_ok = _format_qdrant_lines(qdrant_status)
    for line in qdrant_lines:
        click.echo(line, err=not qdrant_ok)

    sys.exit(0 if ollama_ok and qdrant_ok else 2)


@cli.command(help="Print the version and exit.")
def version() -> None:
    click.echo(__version__)
    sys.exit(0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
