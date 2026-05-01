from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from dprox.config import Config
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache
from dprox.version import IMAGE, __version__


def create_app(
    config: Config,
    plan_cache: PlanCache,
    ollama: OllamaClient | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `plan_cache` must already be `initial_load`ed (the CLI does this at
    startup so plan errors surface with the right exit code).

    `ollama` is optional. When omitted, a real OllamaClient is constructed
    here and closed at lifespan shutdown. When supplied (e.g. by tests
    with a MockTransport), the caller owns its lifecycle and lifespan
    leaves it alone.
    """
    own_ollama = ollama is None
    if ollama is None:
        ollama = OllamaClient(config.embedding)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if own_ollama:
                await ollama.aclose()

    app = FastAPI(title="dprox", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.plan_cache = plan_cache
    app.state.ollama = ollama

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        agent_count, admin_count = plan_cache.counts()
        plan_check = {
            "loaded": plan_cache.loaded,
            "agents": agent_count,
            "admins": admin_count,
        }

        # Each /healthz hit calls Ollama. Acceptable for v0.1 monitoring
        # cadence (>=10s); revisit with a TTL cache if the endpoint gets
        # hammered.
        ollama_check = await ollama.check_health()

        ok = (
            plan_check["loaded"]
            and ollama_check["reachable"]
            and ollama_check["model_present"]
        )

        body = {
            "status": "ok" if ok else "degraded",
            "org": config.org,
            "checks": {"plan": plan_check, "ollama": ollama_check},
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "image": IMAGE}

    return app
