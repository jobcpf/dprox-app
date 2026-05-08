from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from dprox.config import Config
from dprox.mtls import AuthFailure, auth_failure_to_dict, require_mtls
from dprox.ollama import OllamaClient
from dprox.plan import PlanCache, PlanError
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

    @app.exception_handler(AuthFailure)
    async def _handle_auth_failure(_request: Request, exc: AuthFailure) -> JSONResponse:
        # Spec §4.3: error body shape is {"error": code, "message": text}
        # — flat, not FastAPI's default {"detail": ...}.
        # TODO step 10: emit auth_rejected audit log line here.
        return JSONResponse(status_code=exc.status, content=auth_failure_to_dict(exc))

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

    @app.post("/v1/query")
    async def query(cn: str = Depends(require_mtls)) -> JSONResponse:
        """Stub for build step 4.

        Validates the peer cert and resolves the CN against the plan,
        but does not yet embed/search. Full pipeline (auth → plan →
        embed → search → respond) lands in build step 7.
        """
        try:
            entry = plan_cache.lookup(cn)
        except PlanError as exc:
            return JSONResponse(
                status_code=502,
                content={"error": "upstream_unavailable", "message": str(exc)},
            )

        if entry is None:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "unknown_agent",
                    "message": f"agent {cn!r} not found in compiled_plan.yml",
                },
            )

        # Step 4 stub — return the resolved identity + groups so we can
        # exercise the full mTLS + plan pipeline end-to-end with curl.
        return JSONResponse(
            status_code=200,
            content={
                "stub": True,
                "agent": entry.name,
                "role": entry.role,
                "groups": sorted(entry.groups),
            },
        )

    return app
