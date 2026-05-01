from fastapi import FastAPI

from dprox.version import IMAGE, __version__


def create_app() -> FastAPI:
    app = FastAPI(title="dprox", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict:
        # Skeleton — real upstream checks (Qdrant, Ollama, plan) land in build step 9.
        return {"status": "ok", "checks": {}}

    @app.get("/version")
    def version() -> dict:
        return {"version": __version__, "image": IMAGE}

    return app


app = create_app()
