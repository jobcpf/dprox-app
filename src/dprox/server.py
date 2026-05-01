from fastapi import FastAPI

from dprox.config import Config
from dprox.version import IMAGE, __version__


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI app for a given config.

    Module-level `app` is intentionally absent — the app is always
    constructed against an explicit config so production and tests share
    the same code path.
    """
    app = FastAPI(title="dprox", version=__version__)
    app.state.config = config

    @app.get("/healthz")
    def healthz() -> dict:
        # Real upstream checks (Qdrant, Ollama, plan) land in build step 9.
        return {
            "status": "ok",
            "org": config.org,
            "checks": {},
        }

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "image": IMAGE}

    return app
