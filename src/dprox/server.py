import hashlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from dprox.config import Config
from dprox.mtls import AuthFailure, auth_failure_to_dict, require_mtls
from dprox.ollama import OllamaClient, OllamaTimeout, OllamaUnavailable
from dprox.plan import PlanCache, PlanError
from dprox.qdrant import QdrantClient, QdrantHit, QdrantTimeout, QdrantUnavailable
from dprox.version import IMAGE, __version__


class QueryRequest(BaseModel):
    """Request body for POST /v1/query.

    Spec §4.2:
        query: required, plain text, embedded server-side via Ollama.
        limit: optional, defaults to config.qdrant.default_limit, max
            config.qdrant.max_limit. Range checked in the handler.

    `extra="forbid"` rejects identity-looking fields (groups, agent, cn,
    classification_group, etc.) per the §3.4 critical invariant.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1)
    limit: int | None = None


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    """Spec §4.3 error body shape: {"error": <code>, "message": <text>}."""
    return JSONResponse(status_code=status, content={"error": code, "message": message})


def _format_validation_error(err: dict[str, Any]) -> str:
    """Map a single Pydantic v2 ValidationError dict to a human-readable message."""
    err_type = err.get("type", "")
    loc = err.get("loc", ())
    # Skip the implicit "body" prefix that FastAPI prepends.
    parts = [str(p) for p in loc if p != "body"]
    field_path = ".".join(parts) if parts else ""

    if err_type == "extra_forbidden":
        return f"unexpected field '{field_path}'" if field_path else "unexpected field"
    if err_type == "missing":
        return (
            f"missing required field '{field_path}'"
            if field_path
            else "missing required field"
        )
    if err_type in ("json_invalid", "value_error.json"):
        return "request body is not valid JSON"
    if err_type.startswith("string_too_short"):
        return f"field '{field_path}' is empty" if field_path else "empty field"

    msg = err.get("msg", "validation error")
    if field_path:
        return f"invalid value for '{field_path}': {msg}"
    return msg


def _query_hash(query: str) -> str:
    """First 16 hex chars of sha256(query). Used in audit logs and response metadata."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _hit_to_dict(h: QdrantHit) -> dict[str, Any]:
    """Project a QdrantHit to spec §4.2 result shape (all fields present, null when absent)."""
    return {
        "text": h.text,
        "classification_group": h.classification_group,
        "score": h.score,
        "source_path_rel": h.source_path_rel,
        "file_type": h.file_type,
        "chunk_index": h.chunk_index,
        "chunk_total": h.chunk_total,
        "modified_at": h.modified_at,
        "indexed_at": h.indexed_at,
    }


def create_app(
    config: Config,
    plan_cache: PlanCache,
    ollama: OllamaClient | None = None,
    qdrant: QdrantClient | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `plan_cache` must already be `initial_load`ed (the CLI does this at
    startup so plan errors surface with the right exit code).

    `ollama` and `qdrant` are optional. When omitted, real clients are
    constructed here and closed at lifespan shutdown. When supplied
    (e.g. by tests with a MockTransport / AsyncMock), the caller owns
    their lifecycle and lifespan leaves them alone.
    """
    own_ollama = ollama is None
    if ollama is None:
        ollama = OllamaClient(config.embedding)

    own_qdrant = qdrant is None
    if qdrant is None:
        qdrant = QdrantClient(config.qdrant, config.embedding.vector_dim)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if own_ollama:
                await ollama.aclose()
            if own_qdrant:
                await qdrant.aclose()

    app = FastAPI(title="dprox", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.plan_cache = plan_cache
    app.state.ollama = ollama
    app.state.qdrant = qdrant

    @app.exception_handler(AuthFailure)
    async def _handle_auth_failure(_request: Request, exc: AuthFailure) -> JSONResponse:
        # Spec §4.3 error body shape — flat, not FastAPI's default {"detail": ...}.
        # TODO step 10: emit auth_rejected audit log line here.
        return JSONResponse(status_code=exc.status, content=auth_failure_to_dict(exc))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Map Pydantic body validation failures to the spec §4.3 400 shape.
        errors = exc.errors()
        msg = (
            _format_validation_error(errors[0])
            if errors
            else "request validation failed"
        )
        return _error_response(400, "bad_request", msg)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        agent_count, admin_count = plan_cache.counts()
        plan_check = {
            "loaded": plan_cache.loaded,
            "agents": agent_count,
            "admins": admin_count,
        }

        # Each /healthz hit calls Ollama and Qdrant. Acceptable for v0.1
        # monitoring cadence (>=10s); revisit with a TTL cache if hammered.
        ollama_check = await ollama.check_health()
        qdrant_check = await qdrant.check_health()

        ok = (
            plan_check["loaded"]
            and ollama_check["reachable"]
            and ollama_check["model_present"]
            and qdrant_check["reachable"]
            and qdrant_check["collection_exists"]
            and qdrant_check["vector_dim_matches"]
        )

        body = {
            "status": "ok" if ok else "degraded",
            "org": config.org,
            "checks": {
                "plan": plan_check,
                "ollama": ollama_check,
                "qdrant": qdrant_check,
            },
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "image": IMAGE}

    @app.post("/v1/query")
    async def query(
        req: QueryRequest, cn: str = Depends(require_mtls)
    ) -> JSONResponse:
        """RBAC-filtered query: auth → plan → embed → search → respond.

        Validation order (spec §4.2):
            1. Auth (require_mtls dependency) → 401 / 403
            2. Body schema (FastAPI/Pydantic) → 400
            3. Plan lookup → 403 unknown_agent
            4. Embedding → 502/504
            5. Search → 502/504
        """
        # --- 3. Plan resolution -------------------------------------------------
        try:
            entry = plan_cache.lookup(cn)
        except PlanError as exc:
            return _error_response(502, "upstream_unavailable", str(exc))

        if entry is None:
            return _error_response(
                403,
                "unknown_agent",
                f"agent {cn!r} not found in compiled_plan.yml",
            )

        # --- 3b. Limit range check (config-aware) -------------------------------
        limit = req.limit if req.limit is not None else config.qdrant.default_limit
        if not 1 <= limit <= config.qdrant.max_limit:
            return _error_response(
                400,
                "bad_request",
                f"limit must be 1..{config.qdrant.max_limit}, got {limit}",
            )

        # --- 4. Embedding -------------------------------------------------------
        try:
            vector, _ollama_ms = await ollama.embed(req.query)
        except OllamaTimeout as exc:
            return _error_response(504, "upstream_timeout", str(exc))
        except OllamaUnavailable as exc:
            return _error_response(502, "upstream_unavailable", str(exc))

        # --- 5. RBAC-filtered search -------------------------------------------
        try:
            hits, _qdrant_ms = await qdrant.search(vector, entry.groups, limit)
        except QdrantTimeout as exc:
            return _error_response(504, "upstream_timeout", str(exc))
        except QdrantUnavailable as exc:
            return _error_response(502, "upstream_unavailable", str(exc))

        # --- 6. Response shape (spec §4.2) -------------------------------------
        # TODO step 10: emit one structured audit log line here with
        #   {agent, groups_applied, query_hash, result_count,
        #    latency_ms, qdrant_ms, ollama_ms}.
        return JSONResponse(
            status_code=200,
            content={
                "results": [_hit_to_dict(h) for h in hits],
                "metadata": {
                    "agent": entry.name,
                    "groups_applied": sorted(entry.groups),
                    "result_count": len(hits),
                    "query_hash": _query_hash(req.query),
                },
            },
        )

    return app
