import sys

import click
import uvicorn

from dprox.version import __version__


@click.group(help="dprox — RBAC-enforcing query proxy.")
def cli() -> None:
    pass


@cli.command(help="Start the HTTP(S) server.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port.")
def serve(host: str, port: int) -> None:
    # Skeleton — TLS, mTLS, config-driven binding land in build steps 2–4.
    # Default to plain HTTP on :8000 for local skeleton testing; production
    # serves :8443 with TLS.
    uvicorn.run("dprox.server:app", host=host, port=port, log_level="info")


@cli.command(help="Run upstream checks and exit.")
def health() -> None:
    # Skeleton — real plan/Ollama/Qdrant checks land in build step 8.
    click.echo(f"[OK]   skeleton                 dprox v{__version__}")
    sys.exit(0)


@cli.command(help="Print the version and exit.")
def version() -> None:
    click.echo(__version__)
    sys.exit(0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
