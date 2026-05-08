# dprox

RBAC-enforcing query proxy. Read-path service for ARC Power's RAG system.

> One dprox instance per org. Stateless, long-running, mTLS-fronted query
> proxy in front of a per-org Qdrant collection.

The canonical reference is **[../dprox-design-spec-v0.2.md](../dprox-design-spec-v0.2.md)**.
Runtime cert contract lives in **[../dprox-cert-integration-v0.2.md](../dprox-cert-integration-v0.2.md)**;
issuance/lifecycle in **[../cert-provisioning-brief.md](../cert-provisioning-brief.md)**.
Earlier inputs are in `../proxy-brief-input.md` and `../dprox-build-brief-v0.1.md`.

## Status

Pre-MVP. Build order is documented in §7.7 of the design spec; this repo
currently implements **steps 1, 2, 3, 5** (step 4 implemented out-of-order
because step 5 was unblocked while certs were in flight):

- **Step 1 (skeleton)** — `dprox version` / `dprox health` / `dprox serve`,
  FastAPI app, `/healthz`, `/version`.
- **Step 2 (config)** — Pydantic schema mirroring spec §6.2, YAML loader
  honouring `DPROX_CONFIG`, exit 3 on invalid config, `--config` CLI flag.
- **Step 3 (plan)** — `compiled_plan.yml` parser, `PlanCache` with mtime
  invalidation + unknown-CN reload (rate-limited), fail-closed on corrupt
  reload.
- **Step 5 (Ollama)** — `OllamaClient.embed` and `check_health`, error
  taxonomy (`OllamaTimeout` → 504, `OllamaUnavailable` → 502 incl. dim
  mismatch), wired into `/healthz` and `dprox health`.
- **Step 4 (mTLS)** — uvicorn TLS listener with `CERT_OPTIONAL`, custom
  `DproxHttpProtocol` injects peer cert into ASGI scope, `require_mtls`
  dependency enforces EKU=clientAuth + single-CN on `/v1/query` only,
  `/healthz` and `/version` stay public. Stub `POST /v1/query` returns
  the resolved agent's groups (full embed + search pipeline lands in
  step 7).

Step 6 (Qdrant), 7 (`/v1/query` end-to-end), 10 (audit logging), 11
(CI/GHCR), 12 (smoke) remain.

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

## Versioning

`__version__` lives in [`src/dprox/version.py`](src/dprox/version.py). The
wheel version, GHCR image tag (`ghcr.io/jobcpf/dprox:<tag>`), and the value
served at `GET /version` all derive from it. Production deployments pin
explicitly — never use `:latest`.

## License

TBD — pending alignment with the sibling Ingstr project.
