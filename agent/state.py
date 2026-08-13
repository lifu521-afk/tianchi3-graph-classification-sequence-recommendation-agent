from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentState:
    round_index: int = 0
    no_improve_rounds: int = 0
    budget_seconds_used: float = 0.0
    accepted_rounds: int = 0
    rejected_rounds: int = 0
    best: dict[str, float] = field(default_factory=dict)
    observed_best: dict[str, float] = field(default_factory=dict)
    best_by_protocol: dict[str, float] = field(default_factory=dict)
    observed_best_by_protocol: dict[str, float] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)
    disabled: dict[str, str] = field(default_factory=dict)
    experiment_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_run_id: str | None = None
    lifecycle_status: str = "ready"
    stop_reason: str | None = None
    action_required: str | None = None
    status_updated_utc: str | None = None

    @classmethod
    def load(cls, path: Path, official_reference: dict[str, float] | None = None) -> "AgentState":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            round_index=int(payload.get("round_index", 0)),
            no_improve_rounds=int(payload.get("no_improve_rounds", 0)),
            budget_seconds_used=float(payload.get("budget_seconds_used", 0.0)),
            accepted_rounds=int(payload.get("accepted_rounds", 0)),
            rejected_rounds=int(payload.get("rejected_rounds", 0)),
            best={str(k): float(v) for k, v in payload.get("best", {}).items()},
            observed_best={
                str(k): float(v)
                for k, v in payload.get("observed_best", payload.get("best", {})).items()
            },
            best_by_protocol={
                str(k): float(v) for k, v in payload.get("best_by_protocol", {}).items()
            },
            observed_best_by_protocol={
                str(k): float(v)
                for k, v in payload.get(
                    "observed_best_by_protocol",
                    payload.get("best_by_protocol", {}),
                ).items()
            },
            seen=[str(value) for value in payload.get("seen", [])],
            disabled={str(k): str(v) for k, v in payload.get("disabled", {}).items()},
            experiment_stats={
                str(k): dict(v)
                for k, v in payload.get("experiment_stats", {}).items()
                if isinstance(v, dict)
            },
            last_run_id=payload.get("last_run_id"),
            lifecycle_status=str(payload.get("lifecycle_status", "ready")),
            stop_reason=payload.get("stop_reason"),
            action_required=payload.get("action_required"),
            status_updated_utc=payload.get("status_updated_utc"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
