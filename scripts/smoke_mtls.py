"""End-to-end mTLS smoke test against a running dprox.

Uses httpx (Python's SSL stack honours our PEM CA, unlike Windows curl
which uses schannel and the system trust store).

Usage (with dprox already running on https://127.0.0.1:8443):

    python scripts/smoke_mtls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BASE = "https://127.0.0.1:8443"
CERTS = Path("certs")


def _show(label: str, response: httpx.Response) -> None:
    body = response.text
    try:
        body = json.dumps(response.json(), indent=2)
    except ValueError:
        pass
    print(f"\n=== {label} ===")
    print(f"status: {response.status_code}")
    print(body)


def main() -> int:
    ca = str(CERTS / "ca.crt")

    # Public routes — no client cert.
    with httpx.Client(verify=ca) as c:
        _show("GET /healthz (no cert)", c.get(f"{BASE}/healthz"))
        _show("GET /version (no cert)", c.get(f"{BASE}/version"))
        _show(
            "POST /v1/query (no cert) — expect 401 auth_required",
            c.post(f"{BASE}/v1/query", json={"query": "test"}),
        )

    # /v1/query with each agent cert.
    for agent in ("agent_alice", "agent_bob", "agent_oversight"):
        cert = (str(CERTS / f"{agent}.crt"), str(CERTS / f"{agent}.key"))
        with httpx.Client(verify=ca, cert=cert) as c:
            _show(
                f"POST /v1/query (cert={agent}) — expect 200 + groups",
                c.post(f"{BASE}/v1/query", json={"query": "test"}),
            )

    # Server cert presented as a client cert — expect 401 cert_invalid.
    bad_cert = (str(CERTS / "server.crt"), str(CERTS / "server.key"))
    with httpx.Client(verify=ca, cert=bad_cert) as c:
        _show(
            "POST /v1/query (server cert as client) — expect 401 cert_invalid",
            c.post(f"{BASE}/v1/query", json={"query": "test"}),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
