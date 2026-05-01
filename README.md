# dprox

RBAC-enforcing query proxy. Read-path service for ARC Power's RAG system.

> One dprox instance per org. Stateless, long-running, mTLS-fronted query
> proxy in front of a per-org Qdrant collection.

The canonical reference is **[../dprox-design-spec-v0.1.md](../dprox-design-spec-v0.1.md)**.
Inputs and history live in `../proxy-brief-input.md` and
`../dprox-build-brief-v0.1.md`.

## Status

Pre-MVP. Build order is documented in §7.7 of the design spec; this repo
currently implements **step 1 (skeleton)**:

- `dprox version` prints the version
- `dprox health` is a placeholder that exits 0
- `dprox serve` starts a FastAPI app with `/healthz` and `/version`

mTLS, config loading, plan resolution, embedding, and Qdrant search land in
subsequent build steps.

## Quick start (venv, Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Run the server (skeleton: plain HTTP on :8000)
dprox serve

# In another shell:
curl http://localhost:8000/healthz
curl http://localhost:8000/version

# Tests
pytest
```

## Layout

```
src/dprox/        Application code
  cli.py          CLI entry point — serve / health / version
  server.py       FastAPI app factory
  version.py      Version + image identity (single source of truth)
tests/            Test suite
examples/         Sample config / compose / secrets (filled in as features land)
```

## Versioning

`__version__` lives in `src/dprox/version.py`. The wheel and the GHCR image
tag both derive from it. Production images are pinned by tag — never use
`:latest`.

## License

TBD — pending alignment with the sibling Ingstr project.
