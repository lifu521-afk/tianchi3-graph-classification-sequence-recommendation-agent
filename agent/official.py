from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_reference(config: dict[str, Any], path: Path) -> dict[str, Any]:
    reference = dict(config.get("official_reference", {}))
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            reference.update(payload)
            component_bests = payload.get("confirmed_component_bests", {})
            if isinstance(component_bests, dict):
                for key in ("A1_accuracy", "A2_ndcg_at_10"):
                    if key not in component_bests:
                        continue
                    confirmed = float(component_bests[key])
                    current = float(reference.get(key, float("-inf")))
                    reference[key] = max(current, confirmed)
                reference["reference_policy"] = (
                    "confirmed component bests override a lower latest submission"
                )
    return reference


def record_feedback(
    path: Path,
    values: dict[str, float],
    submitted_at: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    current: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            current.update(payload)
    current.update(values)
    current["source"] = "user_confirmed_official_feedback"
    current["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if submitted_at:
        current["submitted_at"] = submitted_at
    if note:
        current["note"] = note
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current
