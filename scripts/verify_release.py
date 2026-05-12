"""Verify a dprox release landed: GHA workflow + GHCR public visibility.

Run after `git push origin v0.1.0` to confirm:
    1. The release workflow ran and succeeded.
    2. The image manifest is fetchable anonymously (= GHCR package is public).
    3. Multi-arch / config digest details for the image.

Usage:
    python scripts/verify_release.py [TAG]   (default: v0.1.0)
"""

from __future__ import annotations

import sys

import httpx

OWNER = "jobcpf"
REPO = "dprox-app"
PACKAGE = "dprox"


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v0.1.0"

    print(f"=== 1. Recent GitHub Actions runs for {OWNER}/{REPO} ===")
    r = httpx.get(
        f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs?per_page=5"
    )
    for run in r.json().get("workflow_runs", [])[:5]:
        name = run["name"]
        ref = run["head_branch"] or run["head_sha"][:7]
        status = run["status"]
        concl = run["conclusion"]
        event = run["event"]
        print(
            f"  {name:20s} | {ref:15s} | status={status:12s} "
            f"| conclusion={str(concl):12s} | event={event}"
        )

    print()
    print("=== 2. GHCR anonymous token (only granted if package is public) ===")
    r = httpx.get(
        "https://ghcr.io/token",
        params={"service": "ghcr.io", "scope": f"repository:{OWNER}/{PACKAGE}:pull"},
    )
    print(f"  status: {r.status_code}")
    if r.status_code != 200:
        print(f"  body: {r.text[:300]}")
        return 1

    token = r.json().get("token", "")
    print(f"  token tail: ...{token[-12:] if token else '(empty)'}")

    print()
    print(f"=== 3. Manifest GET for {tag} ===")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": (
            "application/vnd.oci.image.index.v1+json, "
            "application/vnd.docker.distribution.manifest.v2+json, "
            "application/vnd.oci.image.manifest.v1+json"
        ),
    }
    r = httpx.get(
        f"https://ghcr.io/v2/{OWNER}/{PACKAGE}/manifests/{tag}", headers=headers
    )
    print(f"  status: {r.status_code}")
    if r.status_code != 200:
        print(f"  body: {r.text[:300]}")
        return 1

    m = r.json()
    print(f"  mediaType: {m.get('mediaType')}")
    print(f"  schemaVersion: {m.get('schemaVersion')}")
    if "config" in m:
        print(f"  config.digest: {m['config']['digest']}")
        print(f"  layers: {len(m.get('layers', []))}")
    elif "manifests" in m:
        print(f"  manifests (multi-arch): {len(m['manifests'])}")
        for sm in m["manifests"]:
            plat = sm.get("platform", {})
            print(
                f"    - {plat.get('architecture')}/{plat.get('os')}: "
                f"{sm.get('digest', '')[:24]}..."
            )

    print()
    print(f"=== 4. List all tags on {OWNER}/{PACKAGE} ===")
    r = httpx.get(
        f"https://ghcr.io/v2/{OWNER}/{PACKAGE}/tags/list",
        headers={"Authorization": f"Bearer {token}"},
    )
    print(f"  status: {r.status_code}")
    if r.status_code == 200:
        tags = r.json().get("tags", [])
        for t in sorted(tags):
            print(f"    - {t}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
