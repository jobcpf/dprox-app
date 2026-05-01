from fastapi import FastAPI

from dprox.config import Config
from dprox.plan import PlanCache
from dprox.version import IMAGE, __version__


def create_app(config: Config, plan_cache: PlanCache) -> FastAPI:
    """Build the FastAPI app with a pre-loaded plan cache.

    The cache is constructed and `initial_load`-ed by the caller (the CLI's
    serve subcommand) so that startup-time plan errors can be reported with
    the right exit code before uvicorn even tries to bind.
    """
    app = FastAPI(title="dprox", version=__version__)
    app.state.config = config
    app.state.plan_cache = plan_cache

    @app.get("/healthz")
    def healthz() -> dict:
        # Real upstream checks (Qdrant, Ollama, plan-stat) land in build step 9.
        agent_count, admin_count = plan_cache.counts()
        return {
            "status": "ok",
            "org": config.org,
            "checks": {
                "plan": {
                    "loaded": plan_cache.loaded,
                    "agents": agent_count,
                    "admins": admin_count,
                },
            },
        }

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "image": IMAGE}

    return app
