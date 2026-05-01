"""Compiled-plan parsing, agent resolution, and the in-memory plan cache.

Mirrors `dprox-design-spec-v0.1.md` §3 (trust model), §6.2 (config), §7.3
(plan caching strategy). The plan is the authoritative source of
identity → group-set mapping; the cache exists purely to avoid YAML-parse
overhead on the hot path. mtime-stat per query keeps staleness near-zero.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dprox.config import PlanConfig

Role = Literal["agent", "admin"]


class PlanError(Exception):
    """Raised when plan parsing, file IO, or validation fails."""


@dataclass(frozen=True)
class AgentEntry:
    """One resolvable identity from the compiled plan."""

    name: str
    role: Role
    groups: frozenset[str]
    description: str | None = None


class _Entry(BaseModel):
    """Loose Pydantic model for one user entry. Tolerates unknown fields."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    groups: list[str] = Field(default_factory=list)
    description: str | None = None


class _PlanModel(BaseModel):
    """Loose Pydantic model for the compiled plan's top level.

    `meta`, `required_groups`, `directory_classifications` are all
    informational and ignored by dprox. We only project agent_users and
    admin_users into the runtime model.
    """

    model_config = ConfigDict(extra="allow")
    agent_users: list[_Entry] = Field(default_factory=list)
    admin_users: list[_Entry] = Field(default_factory=list)


def parse_plan_dict(data: dict) -> dict[str, AgentEntry]:
    """Convert a plan dict into a CN-keyed AgentEntry map.

    Raises PlanError on validation failure, duplicate names across
    agent_users + admin_users, or an entirely empty plan.
    """
    try:
        plan = _PlanModel.model_validate(data)
    except ValidationError as exc:
        raise PlanError(f"plan validation failed:\n{exc}") from exc

    out: dict[str, AgentEntry] = {}

    for entry in plan.agent_users:
        if entry.name in out:
            raise PlanError(f"duplicate name in agent_users: {entry.name!r}")
        out[entry.name] = AgentEntry(
            name=entry.name,
            role="agent",
            groups=frozenset(entry.groups),
            description=entry.description,
        )

    for entry in plan.admin_users:
        if entry.name in out:
            raise PlanError(
                f"name {entry.name!r} appears in both agent_users and admin_users"
            )
        out[entry.name] = AgentEntry(
            name=entry.name,
            role="admin",
            groups=frozenset(entry.groups),
            description=entry.description,
        )

    if not out:
        raise PlanError("plan has no agent_users or admin_users entries")

    return out


def load_plan_from_file(path: Path) -> dict[str, AgentEntry]:
    """Read + parse a compiled plan from disk. Raises PlanError on any failure."""
    if not path.exists():
        raise PlanError(f"plan file not found: {path}")
    if not path.is_file():
        raise PlanError(f"plan path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanError(f"could not read plan {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PlanError(f"YAML parse error in {path}: {exc}") from exc

    if raw is None:
        raise PlanError(f"plan file is empty: {path}")
    if not isinstance(raw, dict):
        raise PlanError(
            f"plan root must be a mapping, got {type(raw).__name__}: {path}"
        )

    return parse_plan_dict(raw)


class PlanCache:
    """In-memory CN→AgentEntry cache with mtime-driven and unknown-CN-driven reloads.

    Thread-safe via a single lock. Multi-worker deployments end up with one
    cache per worker — they converge via the mtime stat on each lookup.

    Failure-mode contract (per spec §7.3): a reload that fails (file gone,
    parse error, etc.) raises PlanError. We never return stale data when
    reload fails — that's the fail-closed posture.
    """

    def __init__(self, config: PlanConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._last_unknown_cn_reload_mono: float = 0.0
        self._agents: dict[str, AgentEntry] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._config.compiled_plan_path

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    def initial_load(self) -> None:
        """Load the plan once at startup. Raises PlanError on any failure."""
        with self._lock:
            self._reload_locked()

    def lookup(self, cn: str) -> AgentEntry | None:
        """Resolve a CN to its AgentEntry, applying caching strategy.

        Returns None if the CN is not in the plan after exhausting reload
        opportunities. Raises PlanError if reload fails — caller maps that
        to 502 at request time.
        """
        with self._lock:
            if not self._config.cache_enabled:
                self._reload_locked()
                return self._agents.get(cn)

            if self._config.reload_on_mtime_change and self._mtime_changed_locked():
                self._reload_locked()

            entry = self._agents.get(cn)
            if entry is not None:
                return entry

            if self._config.reload_on_unknown_cn:
                now = time.monotonic()
                interval = float(self._config.reload_min_interval_seconds)
                if (now - self._last_unknown_cn_reload_mono) >= interval:
                    self._last_unknown_cn_reload_mono = now
                    self._reload_locked()
                    return self._agents.get(cn)

            return None

    def counts(self) -> tuple[int, int]:
        """Return (agent_count, admin_count) for human-readable status output."""
        with self._lock:
            agents = sum(1 for e in self._agents.values() if e.role == "agent")
            admins = sum(1 for e in self._agents.values() if e.role == "admin")
            return agents, admins

    # --- internals ----------------------------------------------------------

    def _mtime_changed_locked(self) -> bool:
        try:
            current = self.path.stat().st_mtime
        except OSError:
            # File gone or unreadable — let _reload_locked surface the real error.
            return True
        return current != self._mtime

    def _reload_locked(self) -> None:
        agents = load_plan_from_file(self.path)  # raises PlanError
        try:
            mtime = self.path.stat().st_mtime
        except OSError as exc:
            raise PlanError(f"could not stat plan after reload {self.path}: {exc}") from exc
        self._agents = agents
        self._mtime = mtime
        self._loaded = True
