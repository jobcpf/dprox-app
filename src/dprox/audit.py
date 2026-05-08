"""Audit logging — `event: query`, `query_failed`, `auth_rejected`.

Mirrors `dprox-design-spec-v0.2.md` §9 (audit logging is the legal record).
All audit lines go to stdout as JSON; journald collects from there
(spec §8.2). The audit log is the only durable trace of "who asked what,
was the RBAC filter applied, what came back" — treat it as such.
"""

from __future__ import annotations

import logging
import sys

import structlog

AUDIT_LOGGER_NAME = "dprox.audit"

# Map an AuthFailure HTTP short-code (spec §4.3) to the audit-event
# reason (spec §9.3). The split lets the wire-shape and the audit-shape
# evolve independently — e.g. "auth_required" reads better in HTTP
# responses, "no_client_cert" is more accurate for log analysis.
_AUTH_FAILURE_REASON: dict[str, str] = {
    "auth_required": "no_client_cert",
    "cert_invalid": "cert_invalid",
    "cn_unparseable": "cn_unparseable",
}


def audit_reason_for_auth_failure(code: str) -> str:
    """Translate AuthFailure.code → spec §9.3 audit reason."""
    return _AUTH_FAILURE_REASON.get(code, code)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure stdlib + structlog to emit JSON (or console) lines to stdout.

    Call once at startup (CLI's serve subcommand). Idempotent — safe to
    call again (subsequent calls overwrite global structlog config).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(message)s",
        force=True,
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def get_audit_logger() -> structlog.stdlib.BoundLogger:
    """Return the audit-event logger.

    Structured fields go via kwargs; the event name is the first
    positional arg to `logger.info()`:

        audit.info("query", agent="agent_alice", result_count=7, ...)
    """
    return structlog.get_logger(AUDIT_LOGGER_NAME)
