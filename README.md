# dprox

RBAC-enforcing query proxy. Read-path service for ARC Power's RAG system.

> One dprox instance per org. Stateless, long-running, mTLS-fronted query
> proxy in front of a per-org Qdrant collection.

The canonical reference is **[../integrations/dprox-design-spec-v0.2.md](../integrations/dprox-design-spec-v0.2.md)**.
Runtime cert contract lives in **[../integrations/dprox-cert-integration-v0.2.md](../integrations/dprox-cert-integration-v0.2.md)**;
issuance/lifecycle in **[../integrations/cert-provisioning-brief.md](../integrations/cert-provisioning-brief.md)**.
Earlier inputs and superseded specs are under `../integrations/archive/`.

## Status

**v0.1.1 is deployed on the platform** (`ghcr.io/jobcpf/dprox:v0.1.1`, GHCR
public). All twelve build steps are complete; end-to-end `/v1/query`
against the real arc Qdrant + Ollama is passing on otter.

| Step | Coverage |
|---|---|
| 1. Skeleton | `dprox` CLI (`serve`, `health`, `version`), FastAPI app, `/healthz`, `/version` |
| 2. Config | Pydantic v2 schema (`extra="forbid"`), `--config`/`DPROX_CONFIG`, exit 3 on bad config |
| 3. Plan | `compiled_plan.yml` parser + `PlanCache` (mtime + unknown-CN reload, rate-limited, fail-closed) |
| 4. mTLS | uvicorn `CERT_OPTIONAL`, custom protocol, `require_mtls` dependency (EKU + single-CN), public `/healthz`/`/version` |
| 5. Ollama | `OllamaClient.embed` + `check_health`, error taxonomy (`OllamaTimeout` → 504, `OllamaUnavailable` → 502 incl. dim mismatch) |
| 6. Qdrant | RBAC-filtered search via `query_points` + collection health, error taxonomy (`QdrantTimeout` → 504, `QdrantUnavailable` → 502); filter-construction path 100% tested |
| 7. `/v1/query` | Full pipeline auth → plan → embed → search → respond, spec §4.2 response shape, validation order |
| 10. Audit logging | `event:"query"`, `event:"query_failed"`, `event:"auth_rejected"` via structlog; `query_text` only at DEBUG |
| 11. CI + GHCR | `Dockerfile` (slim runtime, drops to UID 10042); `test.yml` (Py 3.12 + 3.13 matrix); `release.yml` builds + pushes on tag |
| 12. Real-platform smoke | ✓ deployed on otter for `org=arc` (2026-05-13); `agent_arc_exec` end-to-end query succeeded |

### Release history

| Tag | Notes |
|---|---|
| `v0.1.0` (2026-05-12) | First public image. Smoke surfaced [`qdrant-client` API drift](../integrations/archive/dprox-v0.1.0-qdrant-client-bug.md) — `AsyncQdrantClient.search()` removed in 1.18. |
| `v0.1.1` (2026-05-12) | Migrated to `query_points` (universal query API). Pin tightened to `qdrant-client>=1.13,<2.0`. Two regression-guard tests added. |

## Quick start (venv, Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Generate a throwaway CA + server + agent certs (one-shot).
python scripts\dev_certs.py
# certs/{ca,server,agent_alice,agent_bob,agent_oversight}.{crt,key} written.

# Run the server with mTLS on :8443.
dprox serve --config examples\config.dev.yml

# In another shell:
# /healthz and /version are public — no client cert needed (--insecure
# only because the dev CA isn't in your system trust store).
curl --cacert certs\ca.crt https://127.0.0.1:8443/healthz
curl --cacert certs\ca.crt https://127.0.0.1:8443/version

# /v1/query requires a verified agent client cert.
curl --cacert certs\ca.crt `
     --cert certs\agent_alice.crt --key certs\agent_alice.key `
     -X POST https://127.0.0.1:8443/v1/query `
     -H 'Content-Type: application/json' `
     -d '{\"query\": \"test\", \"limit\": 5}'

# A request without a cert hits the same port, returns 401 auth_required.
curl --cacert certs\ca.crt `
     -X POST https://127.0.0.1:8443/v1/query `
     -H 'Content-Type: application/json' `
     -d '{\"query\": \"test\"}'

# Tests
pytest
```

## Smoke testing on Windows

`scripts/smoke_mtls.py` runs an end-to-end test against a running dprox using
the dev certs. **It will fail on machines with antivirus / endpoint security
that performs TLS interception** (AVG, Avast, ESET, Symantec etc) — these
products substitute their own cert in place of dprox's, which the test
client cannot validate. This is environmental, not a dprox bug.

If you see `unable to get local issuer certificate` from the smoke client,
check what cert the server is actually presenting:

```powershell
.\.venv\Scripts\python.exe -c "import socket, ssl; ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE; s = ctx.wrap_socket(socket.create_connection(('localhost', 8443)), server_hostname='localhost'); from cryptography import x509; c = x509.load_der_x509_certificate(s.getpeercert(binary_form=True)); print('Issuer:', c.issuer.rfc4514_string())"
```

If the issuer isn't `CN=dprox-dev-ca`, an interceptor is in the way.
Either add a localhost exclusion in the AV product, or run the smoke test
on a clean container / VM. The unit tests (`pytest`) don't go over the
wire and pass regardless.

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

## Versioning + release

`__version__` lives in [`src/dprox/version.py`](src/dprox/version.py).
[`pyproject.toml`](pyproject.toml) carries the same string. Production
deployments pin explicitly — never use `:latest`.

To cut a release:

```bash
# 1. Bump version in BOTH files (no automation in v0.1 — discipline).
#    Match exactly; release CI gates on it.
$EDITOR src/dprox/version.py pyproject.toml

# 2. Commit, tag, push.
git commit -am "release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

The `release.yml` workflow runs the test gate, verifies the tag matches
`dprox.__version__`, builds the image with Buildx, and pushes both
`ghcr.io/jobcpf/dprox:v0.1.0` and `ghcr.io/jobcpf/dprox:0.1.0`.

## Cert mount permissions (production note)

The platform's Ansible bind-mounts `~/docker/dprox/<org>/certs/` into the
container at `/etc/dprox/certs/`. The container runs as **UID 10042**
(see [`Dockerfile`](Dockerfile)). For dprox to read `server.key` (mode
0400, owner-only), the host file must be owned by UID 10042 — either
chown the certs at distribution time, or run the container with
`--user $(id -u <ansi-user>)` to override. Either is fine; document
the choice in the per-org compose template.

## License

TBD — pending alignment with the sibling Ingstr project.
