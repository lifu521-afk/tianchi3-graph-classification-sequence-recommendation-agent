from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm_planner import choose_with_llm, is_configured
from .state import AgentState


def _task_gaps(
    official_reference: dict[str, float],
    targets: dict[str, float],
) -> dict[str, float]:
    gaps: dict[str, float] = {}
    for task, target_key in (
        ("A1", "A1_accuracy"),
        ("A2", "A2_ndcg_at_10"),
    ):
        target = max(float(targets[target_key]), 1e-12)
        observed = float(official_reference.get(target_key, 0.0))
        gaps[task] = max(0.0, float(targets[target_key]) - observed) / target
    return gaps


def score_experiment(
    item: dict[str, Any],
    state: AgentState,
    official_reference: dict[str, float],
    targets: dict[str, float],
) -> tuple[float, float, float, float, str]:
    gaps = _task_gaps(official_reference, targets)
    task = str(item.get("task", "A1"))
    task_gap = max((gaps.get(part, 0.0) for part in task.split("+")), default=0.0)
    priority_hint = float(item.get("priority_hint", 0.0))
    kind_bonus = {
        "diagnostic": 0.40,
        "oof": 0.30,
        "submission": 0.15,
    }.get(str(item.get("kind", "")), 0.0)
    stats = state.experiment_stats.get(str(item.get("id")), {})
    trials = int(stats.get("trials", 0))
    max_trials = int(item.get("max_trials", 1))
    if trials >= max_trials:
        return (-1.0, task_gap, priority_hint, kind_bonus, str(item["id"]))
    cost = max(float(item.get("cost_seconds", 300.0)), 1.0)
    novelty = 1.0 / (1.0 + trials)
    transfer = float(item.get("transfer_hint", 0.0))
    resource_score = 1.0 / (1.0 + cost / 1800.0)
    family = str(item.get("family", item.get("id", "")))
    family_failures = sum(
        1
        for stats_item in state.experiment_stats.values()
        if str(stats_item.get("family", "")) == family
        and not bool(stats_item.get("accepted", False))
    )
    total = task_gap + priority_hint * 0.05 + kind_bonus * 0.02
    total += novelty * 0.03 + transfer * 0.02 + resource_score * 0.01
    total -= min(family_failures, 5) * 0.02
    return total, task_gap, novelty, transfer, str(item["id"])


def available_experiments(
    experiments: list[dict[str, Any]],
    state: AgentState,
) -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for item in experiments:
        experiment_id = str(item["id"])
        if experiment_id in state.disabled:
            continue
        stats = state.experiment_stats.get(experiment_id, {})
        if "trials" in stats:
            trials = int(stats["trials"])
        else:
            trials = 1 if experiment_id in state.seen else 0
        if trials < int(item.get("max_trials", 1)):
            available.append(item)
    return available


def choose_experiment(
    experiments: list[dict[str, Any]],
    state: AgentState,
    official_reference: dict[str, float],
    targets: dict[str, float],
) -> dict[str, Any] | None:
    available = available_experiments(experiments, state)
    if not available:
        return None

    return sorted(
        available,
        key=lambda item: score_experiment(item, state, official_reference, targets),
        reverse=True,
    )[0]


def plan_experiment(
    config: dict[str, Any],
    state: AgentState,
    official_reference: dict[str, Any],
    targets: dict[str, Any],
    memory_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    experiments = config.get("experiments", [])
    available = available_experiments(experiments, state)
    if not available:
        return None, {"source": "none", "reason": "experiment_pool_exhausted"}

    mode = str(config.get("agent_policy", {}).get("planner", "auto")).lower()
    fallback = bool(
        config.get("llm_planner", {}).get("fallback_to_deterministic", True)
    )
    if mode in {"auto", "llm"} and is_configured(config):
        try:
            decision = choose_with_llm(
                config=config,
                available=available,
                official=official_reference,
                targets=targets,
                state_summary={
                    "round_index": state.round_index,
                    "budget_seconds_used": state.budget_seconds_used,
                    "accepted_rounds": state.accepted_rounds,
                    "rejected_rounds": state.rejected_rounds,
                    "best_local_by_protocol": state.best_by_protocol,
                    "experiment_stats": state.experiment_stats,
                },
                memory_path=memory_path,
            )
            selected = next(
                item
                for item in available
                if str(item["id"]) == decision["experiment_id"]
            )
            return selected, decision
        except (RuntimeError, ValueError, KeyError, StopIteration) as exc:
            if mode == "llm" and not fallback:
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = (
            "planner_mode_deterministic"
            if mode == "deterministic"
            else "llm_environment_not_configured"
        )

    selected = sorted(
        available,
        key=lambda item: score_experiment(
            item,
            state,
            official_reference,
            targets,
        ),
        reverse=True,
    )[0]
    return selected, {
        "source": "deterministic",
        "reason": "feedback_weighted_serial_selection",
        "fallback_reason": fallback_reason,
    }
