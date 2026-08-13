from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(payload: dict[str, Any], path: str) -> float | None:
    value = get_value(payload, path)
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def get_deployment_gate(payload: dict[str, Any]) -> dict[str, Any] | None:
    gate = payload.get("deployment_gate")
    return gate if isinstance(gate, dict) else None


def evaluate(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / str(spec["result_file"])
    if not result_path.exists():
        return {
            "status": "missing_result",
            "metrics": {},
            "result_file": str(result_path),
        }
    payload = load_json(result_path)
    metrics: dict[str, float] = {}
    baselines: dict[str, float] = {}
    if "metric_path" in spec:
        metric = get_path(payload, str(spec["metric_path"]))
        baseline = get_path(payload, str(spec.get("baseline_path", "")))
        if metric is not None:
            metrics[str(spec["task"])] = metric
        if baseline is not None:
            baselines[str(spec["task"])] = baseline
    else:
        for task, metric_path in spec.get("metric_paths", {}).items():
            metric = get_path(payload, str(metric_path))
            if metric is not None:
                metrics[str(task)] = metric
    return {
        "status": "ok" if metrics else "no_metric",
        "metrics": metrics,
        "baselines": baselines,
        "deployment_gate": get_deployment_gate(payload),
        "raw_payload": payload,
        "result_file": str(result_path),
        "raw_keys": sorted(payload.keys()),
    }


def audit_stability(
    spec: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    promotion = config.get("promotion", {})
    if not promotion.get("require_agent_stability_audit", True):
        return {"passed": True, "reason": "audit_disabled"}

    kind = str(spec.get("kind", "")).lower()
    if kind not in {"oof", "multisplit", "crossfit"}:
        return {
            "passed": False,
            "reason": "experiment_kind_is_not_multisplit",
            "kind": kind,
        }
    payload = result.get("raw_payload", {})
    if not isinstance(payload, dict):
        return {"passed": False, "reason": "missing_raw_payload"}

    stability = spec.get("stability", {})
    paths = stability.get(
        "fold_lifts_paths",
        [
            "best.fold_lifts",
            "fold_lifts",
            "validation.five_oof_fold_lifts",
            "validation.five_oof_fold_nets",
            "best.fold_nets",
        ],
    )
    fold_values: list[float] | None = None
    selected_path = ""
    for path in paths:
        value = get_value(payload, str(path))
        if isinstance(value, list):
            try:
                fold_values = [float(item) for item in value]
            except (TypeError, ValueError):
                continue
            selected_path = str(path)
            break
    min_folds = int(stability.get("min_folds", 3))
    if fold_values is None or len(fold_values) < min_folds:
        return {
            "passed": False,
            "reason": "missing_or_insufficient_fold_evidence",
            "minimum_folds": min_folds,
        }

    epsilon = float(promotion.get("stability_epsilon", 1e-12))
    negative = [value for value in fold_values if value < -epsilon]
    positive = [value for value in fold_values if value > epsilon]
    reasons: list[str] = []
    if promotion.get("require_zero_negative_folds", True) and negative:
        reasons.append("negative_fold_detected")
    if stability.get("require_positive_fold", True) and not positive:
        reasons.append("no_positive_fold")

    required_positive_paths = stability.get("required_positive_paths", [])
    path_values: dict[str, float | None] = {}
    for path in required_positive_paths:
        value = get_path(payload, str(path))
        path_values[str(path)] = value
        if value is None or value <= epsilon:
            reasons.append(f"nonpositive_required_path:{path}")

    return {
        "passed": not reasons,
        "reason": "ok" if not reasons else ",".join(reasons),
        "fold_evidence_path": selected_path,
        "fold_values": fold_values,
        "fold_count": len(fold_values),
        "positive_folds": len(positive),
        "negative_folds": len(negative),
        "minimum_fold_value": min(fold_values),
        "required_positive_paths": path_values,
    }
