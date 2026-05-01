# dprox

RBAC-enforcing query proxy. Read-path service for ARC Power's RAG system.

> One dprox instance per org. Stateless, long-running, mTLS-fronted query
> proxy in front of a per-org Qdrant collection.

The canonical reference is **[../dprox-design-spec-v0.1.md](../dprox-design-spec-v0.1.md)**.
Cert formats and lifecycle live in **[../cert-provisioning-brief.md](../cert-provisioning-brief.md)**.
Earlier inputs are in `../proxy-brief-input.md` and `../dprox-build-brief-v0.1.md`.

## Status

Pre-MVP. Build order is documented in §7.7 of the design spec; this repo
currently implements **steps 1–2**:

- **Step 1 (skeleton)** — `dprox version` / `dprox health` / `dprox serve`,
  FastAPI app, `/healthz`, `/version`.
- **Step 2 (config)** — full Pydantic schema mirroring spec §6.2, YAML loader
  honouring `DPROX_CONFIG`, exit code 3 on invalid config, `--config` CLI flag.

mTLS, plan resolution, embedding, and Qdrant search land in subsequent steps.

## Quick start (venv, Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Run the server using the dev config (plain HTTP on :8000 for now;
# TLS / mTLS land in build step 4).
dprox serve --config examples\config.dev.yml

# In another shell:
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/version

# Tests
pytest
```

## Configuration

`dprox` reads `config.yml` resolved in this order:

1. `--config <path>` CLI flag, if given.
2. `DPROX_CONFIG` env var.
3. `/etc/dprox/config.yml` (the in-container default — populated by the
   platform's Ansible templating).

The schema is enforced with Pydantic v2 (`extra="forbid"` everywhere), so a
typo in any field name causes startup to fail with exit code 3 rather than a
silent ignore. See [`examples/config.example.yml`](examples/config.example.yml)
for the production shape and [`examples/config.dev.yml`](examples/config.dev.yml)
for a localhost-targeted variant.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | `dprox health` upstream check failed (config-load failure included) |
| 3 | `dprox serve` could not load or validate config |
| 4 | Runtime error during `dprox serve` |

## Layout

```
src/dprox/
  cli.py          CLI entry point — serve / health / version
  server.py       FastAPI app factory (create_app(config))
  config.py       Pydantic schema + loader + ConfigError
  version.py      Version + image identity (single source of truth)
tests/            Test suite (pytest, with shared fixtures in conftest.py)
examples/         Sample config files (filled out as features land)
```

## Versioning

`__version__` lives in [`src/dprox/version.py`](src/dprox/version.py). The
wheel version, GHCR image tag (`ghcr.io/jobcpf/dprox:<tag>`), and the value
served at `GET /version` all derive from it. Production deployments pin
explicitly — never use `:latest`.

## License

TBD — pending alignment with the sibling Ingstr project.
