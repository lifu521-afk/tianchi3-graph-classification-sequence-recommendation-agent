from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluators import audit_stability, evaluate
from .llm_planner import is_configured as llm_planner_is_configured
from .official import load_reference, record_feedback
from .memory import append_event
from .planner import available_experiments, plan_experiment, score_experiment
from .runner import run_experiment
from .state import AgentState


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
CONFIG_PATH = AGENT_DIR / "config.json"
STATE_PATH = AGENT_DIR / "state.json"
MEMORY_PATH = AGENT_DIR / "memory.jsonl"
RUNS_DIR = AGENT_DIR / "runs"
LOCK_PATH = AGENT_DIR / ".agent.lock"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def make_run_id(experiment_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{experiment_id}_{uuid.uuid4().hex[:6]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_tasks(spec: dict[str, Any]) -> set[str]:
    value = str(spec.get("task", ""))
    return {part for part in value.split("+") if part in {"A1", "A2"}}


def protocol_id(spec: dict[str, Any], task: str) -> str:
    mapped = spec.get("protocol_ids", {})
    if isinstance(mapped, dict) and task in mapped:
        return str(mapped[task])
    return str(spec.get("protocol_id", f"{task}:unspecified"))


def should_accept_candidate(
    execution_status: str,
    result_status: str,
    gate_passed: bool,
    improvements: dict[str, float],
    min_lift: float,
    stability_passed: bool = True,
) -> bool:
    comparable = [value for value in improvements.values() if value == value]
    return bool(
        execution_status == "completed"
        and result_status == "ok"
        and gate_passed
        and stability_passed
        and comparable
        and any(value >= min_lift for value in comparable)
        and all(value >= 0.0 for value in comparable)
    )


@contextmanager
def agent_lock() -> Any:
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = LOCK_PATH.open("x", encoding="ascii")
        handle.write(json.dumps({
            "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }))
        handle.flush()
        yield
    except FileExistsError as exc:
        raise RuntimeError(
            f"agent lock exists at {LOCK_PATH}; only one experiment may run at a time"
        ) from exc
    finally:
        if handle is not None:
            handle.close()
            LOCK_PATH.unlink(missing_ok=True)


def _zip_payload(path: Path, required: set[str]) -> tuple[set[str], dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        members = {name for name in archive.namelist() if not name.endswith("/")}
        return members, {name: archive.read(name) for name in required if name in members}


def _closed_temp_path(prefix: str, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=ROOT)
    os.close(fd)
    return Path(name)


def has_multi_split_evidence(
    spec: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    if not config.get("budget", {}).get("require_multi_split_for_promotion", False):
        return True
    if str(spec.get("kind", "")).lower() in {"oof", "multisplit", "crossfit"}:
        return True
    payload = result.get("raw_payload", {})
    for key in ("fold_lifts", "folds", "split_scores", "oof"):
        value = payload.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return True
    return False


def official_targets_met(
    official: dict[str, Any],
    targets: dict[str, Any],
) -> bool:
    return (
        float(official.get("A1_accuracy", 0.0)) >= float(targets["A1_accuracy"])
        and float(official.get("A2_ndcg_at_10", 0.0)) >= float(targets["A2_ndcg_at_10"])
    )


def remaining_round_budget(
    config: dict[str, Any],
    state: AgentState,
    requested_rounds: int | None = None,
) -> int:
    configured = max(0, int(config.get("budget", {}).get("max_rounds", 0)))
    remaining = max(0, configured - state.round_index)
    if requested_rounds is None:
        return remaining
    return min(remaining, max(0, int(requested_rounds)))


def control_status(
    config: dict[str, Any],
    state: AgentState,
    official: dict[str, Any],
) -> dict[str, Any]:
    targets = config.get("targets", {})
    experiments = config.get("experiments", [])
    available = available_experiments(experiments, state)
    configured_rounds = max(0, int(config.get("budget", {}).get("max_rounds", 0)))
    rounds_remaining = max(0, configured_rounds - state.round_index)
    max_seconds = float(
        config.get("budget", {}).get("max_total_seconds", float("inf"))
    )
    seconds_remaining = max(0.0, max_seconds - state.budget_seconds_used)

    if official_targets_met(official, targets):
        lifecycle = "complete"
        stop_reason = "official_targets_confirmed"
        action_required = None
    elif rounds_remaining == 0:
        lifecycle = "budget_exhausted"
        stop_reason = "max_rounds_exhausted"
        action_required = "review_results_before_increasing_max_rounds"
    elif seconds_remaining == 0:
        lifecycle = "budget_exhausted"
        stop_reason = "total_time_budget_exhausted"
        action_required = "review_results_before_increasing_time_budget"
    elif not available:
        lifecycle = "awaiting_experiments"
        stop_reason = "experiment_pool_exhausted"
        action_required = "register_a_new_experiment_with_a_new_hypothesis"
    else:
        lifecycle = "ready"
        stop_reason = None
        action_required = None

    enabled_count = sum(
        1 for item in experiments if str(item.get("id")) not in state.disabled
    )
    return {
        "lifecycle_status": lifecycle,
        "stop_reason": stop_reason,
        "action_required": action_required,
        "available_experiment_count": len(available),
        "available_experiment_ids": [str(item["id"]) for item in available],
        "registry": {
            "registered": len(experiments),
            "enabled": enabled_count,
            "disabled": len(experiments) - enabled_count,
            "trial_exhausted_or_seen": enabled_count - len(available),
        },
        "round_budget": {
            "configured": configured_rounds,
            "completed": state.round_index,
            "remaining": rounds_remaining,
        },
        "time_budget_seconds": {
            "configured": max_seconds,
            "used": state.budget_seconds_used,
            "remaining": seconds_remaining,
        },
        "persisted_status": {
            "lifecycle_status": state.lifecycle_status,
            "stop_reason": state.stop_reason,
            "action_required": state.action_required,
            "updated_utc": state.status_updated_utc,
        },
    }


def update_lifecycle(
    state: AgentState,
    lifecycle_status: str,
    stop_reason: str | None = None,
    action_required: str | None = None,
) -> None:
    state.lifecycle_status = lifecycle_status
    state.stop_reason = stop_reason
    state.action_required = action_required
    state.status_updated_utc = datetime.now(timezone.utc).isoformat()


def promote_candidate(spec: dict[str, Any], run_dir: Path, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    promotion = config.get("promotion", {})
    if not promotion.get("write_root_prediction_zip", False):
        return {"promoted": False, "reason": "promotion_disabled"}
    candidate_zip = run_dir / str(spec.get("candidate_zip", "prediction.zip"))
    if not candidate_zip.exists():
        return {"promoted": False, "reason": "candidate_zip_missing"}
    required = set(promotion.get("required_zip_members", ["A1.csv", "A2.csv"]))
    members, extracted = _zip_payload(candidate_zip, required)
    if members != required:
        return {"promoted": False, "reason": "zip_members_mismatch", "members": sorted(members)}

    root_zip = ROOT / "prediction.zip"
    changed_tasks = experiment_tasks(spec)
    if root_zip.exists():
        root_members, root_bytes = _zip_payload(root_zip, required)
        if root_members != required:
            return {
                "promoted": False,
                "reason": "root_zip_members_mismatch",
                "members": sorted(root_members),
            }
        unchanged_members = {
            "A1.csv" if task == "A1" else "A2.csv"
            for task in {"A1", "A2"} - changed_tasks
        }
        mismatched = sorted(
            name for name in unchanged_members if extracted[name] != root_bytes[name]
        )
        if mismatched:
            return {
                "promoted": False,
                "reason": "unchanged_task_bytes_mismatch",
                "members": mismatched,
            }
        declared_changed_members = {
            "A1.csv" if task == "A1" else "A2.csv"
            for task in changed_tasks
        }
        unchanged_declared = sorted(
            name
            for name in declared_changed_members
            if extracted[name] == root_bytes[name]
        )
        if unchanged_declared:
            return {
                "promoted": False,
                "reason": "declared_task_bytes_unchanged",
                "members": unchanged_declared,
            }

    backup = ROOT / f"prediction_backup_before_{run_id}.zip"
    if root_zip.exists():
        shutil.copy2(root_zip, backup)
    staged_zip = _closed_temp_path(".prediction.", ".zip")
    try:
        shutil.copy2(candidate_zip, staged_zip)
        os.replace(staged_zip, root_zip)
        for name, content in extracted.items():
            staged_csv = _closed_temp_path(f".{name}.", ".csv")
            try:
                staged_csv.write_bytes(content)
                os.replace(staged_csv, ROOT / name)
            finally:
                staged_csv.unlink(missing_ok=True)
    finally:
        staged_zip.unlink(missing_ok=True)
    return {
        "promoted": True,
        "backup": str(backup),
        "candidate_zip": str(candidate_zip),
        "root_zip": str(root_zip),
        "changed_tasks": sorted(changed_tasks),
        "sha256": {name: file_sha256(ROOT / name) for name in sorted(required)},
    }


def run_agent(max_rounds: int | None, dry_run: bool) -> int:
    config = load_config()
    budget = config["budget"]
    official = load_reference(config, AGENT_DIR / "official_feedback.json")
    targets = config["targets"]
    state = AgentState.load(STATE_PATH, official)
    rounds = remaining_round_budget(config, state, max_rounds)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with agent_lock():
        initial_status = control_status(config, state, official)
        if initial_status["lifecycle_status"] != "ready":
            update_lifecycle(
                state,
                str(initial_status["lifecycle_status"]),
                str(initial_status["stop_reason"]),
                initial_status["action_required"],
            )
            if not dry_run:
                state.save(STATE_PATH)
            append_event(MEMORY_PATH, {
                "event": "stop",
                "reason": initial_status["stop_reason"],
                "action_required": initial_status["action_required"],
                "control_status": initial_status,
                "state": state.__dict__,
            })
            print(json.dumps({
                "event": "stop",
                **initial_status,
            }, ensure_ascii=False), flush=True)
            return 0

        update_lifecycle(state, "running")
        if not dry_run:
            state.save(STATE_PATH)

        for round_offset in range(rounds):
            if official_targets_met(official, targets):
                update_lifecycle(state, "complete", "official_targets_confirmed")
                if not dry_run:
                    state.save(STATE_PATH)
                append_event(MEMORY_PATH, {
                    "event": "stop",
                    "reason": "official_targets_confirmed",
                    "state": state.__dict__,
                })
                break
            if state.budget_seconds_used >= float(budget.get("max_total_seconds", float("inf"))):
                update_lifecycle(
                    state,
                    "budget_exhausted",
                    "total_time_budget_exhausted",
                    "review_results_before_increasing_time_budget",
                )
                if not dry_run:
                    state.save(STATE_PATH)
                append_event(MEMORY_PATH, {
                    "event": "stop",
                    "reason": "total_time_budget_exhausted",
                    "state": state.__dict__,
                })
                break
            spec, planning = plan_experiment(
                config,
                state,
                official,
                targets,
                MEMORY_PATH,
            )
            if spec is None:
                update_lifecycle(
                    state,
                    "awaiting_experiments",
                    "experiment_pool_exhausted",
                    "register_a_new_experiment_with_a_new_hypothesis",
                )
                if not dry_run:
                    state.save(STATE_PATH)
                append_event(MEMORY_PATH, {
                    "event": "stop",
                    "reason": "experiment_pool_exhausted",
                    "action_required": state.action_required,
                    "state": state.__dict__,
                })
                print(json.dumps({
                    "event": "stop",
                    "reason": "experiment_pool_exhausted",
                    "action_required": state.action_required,
                }, ensure_ascii=False), flush=True)
                break
            run_id = make_run_id(str(spec["id"]))
            run_dir = RUNS_DIR / run_id
            plan_event = {
                "event": "plan",
                "run_id": run_id,
                "experiment_id": spec["id"],
                "task": spec.get("task"),
                "round_index": state.round_index + 1,
                "budget_remaining_rounds": rounds - round_offset,
                "budget_seconds_used": round(state.budget_seconds_used, 3),
                "decision_score": score_experiment(spec, state, official, targets),
                "planner": planning,
                "hypothesis": planning.get(
                    "hypothesis",
                    spec.get("hypothesis", spec.get("description", "")),
                ),
                "reason": planning.get(
                    "reason",
                    "feedback_weighted_serial_selection",
                ),
            }
            append_event(MEMORY_PATH, plan_event)
            print(json.dumps(plan_event, ensure_ascii=False), flush=True)
            if dry_run:
                state.seen.append(str(spec["id"]))
                state.round_index += 1
                continue

            execution = run_experiment(
                spec,
                ROOT,
                run_dir,
                min(
                    int(budget["max_seconds_per_experiment"]),
                    int(max(1, float(budget.get("max_total_seconds", float("inf")) - state.budget_seconds_used))),
                ),
            )
            state.budget_seconds_used += float(execution.get("duration_seconds", 0.0))
            external_result = ROOT / str(spec["result_file"])
            isolated_result = run_dir / str(spec["result_file"])
            if external_result.exists() and not isolated_result.exists():
                isolated_result.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(external_result, isolated_result)
            result = evaluate(spec, run_dir) if execution["status"] == "completed" else {
                "status": execution["status"],
                "metrics": {},
                "baselines": {},
            }
            gate = result.get("deployment_gate")
            gate_passed = gate is None or gate.get("passed") is True
            if gate is not None and not gate_passed:
                result["status"] = "rejected_deployment_gate"
            stability_audit = audit_stability(spec, result, config)
            stability_passed = bool(stability_audit.get("passed", False))
            if not stability_passed:
                result["status"] = "rejected_stability_gate"
            previous_best = dict(state.best)
            previous_protocol_best = dict(state.best_by_protocol)
            improvements: dict[str, float] = {}
            for task, metric in result.get("metrics", {}).items():
                comparison = result.get("baselines", {}).get(task)
                if comparison is None:
                    comparison = state.best_by_protocol.get(protocol_id(spec, task))
                improvements[task] = (
                    metric - comparison if comparison is not None else float("nan")
                )
            accepted = should_accept_candidate(
                execution_status=str(execution["status"]),
                result_status=str(result.get("status")),
                gate_passed=gate_passed,
                improvements=improvements,
                min_lift=float(budget["min_lift"]),
                stability_passed=stability_passed,
            )
            for task, metric in result.get("metrics", {}).items():
                if metric > state.observed_best.get(task, float("-inf")):
                    state.observed_best[task] = metric
                key = protocol_id(spec, task)
                if metric > state.observed_best_by_protocol.get(key, float("-inf")):
                    state.observed_best_by_protocol[key] = metric
                if accepted and metric > state.best_by_protocol.get(key, float("-inf")):
                    state.best_by_protocol[key] = metric
                if accepted and metric > state.best.get(task, float("-inf")):
                    state.best[task] = metric
            if accepted:
                state.no_improve_rounds = 0
                state.accepted_rounds += 1
            else:
                state.no_improve_rounds += 1
                state.rejected_rounds += 1
            if execution["status"] in {"failed", "timeout"}:
                state.disabled[str(spec["id"])] = execution["status"]
            state.seen.append(str(spec["id"]))
            stats = state.experiment_stats.setdefault(str(spec["id"]), {})
            stats.update({
                "trials": int(stats.get("trials", 0)) + 1,
                "last_status": str(execution["status"]),
                "last_lift": {
                    task: value for task, value in improvements.items() if value == value
                },
                "last_duration_seconds": execution.get("duration_seconds", 0.0),
                "accepted": bool(accepted),
                "family": str(spec.get("family", spec.get("id", ""))),
                "round_index": state.round_index + 1,
            })
            state.round_index += 1
            state.last_run_id = run_id
            state.save(STATE_PATH)
            promotion_result = (
                promote_candidate(spec, run_dir, run_id, config)
                if accepted
                else {"promoted": False, "reason": "candidate_not_accepted"}
            )
            event = {
                "event": "result",
                "run_id": run_id,
                "experiment_id": spec["id"],
                "task": spec.get("task"),
                "execution": execution,
                "evaluation": result,
                "stability_gate_passed": stability_passed,
                "stability_audit": stability_audit,
                "previous_best": previous_best,
                "current_best": state.best,
                "observed_best": state.observed_best,
                "previous_protocol_best": previous_protocol_best,
                "current_protocol_best": state.best_by_protocol,
                "protocols": {
                    task: protocol_id(spec, task)
                    for task in result.get("metrics", {})
                },
                "improvements": improvements,
                "accepted_local_candidate": accepted,
                "promotion": promotion_result,
                "official_scores_unchanged": True,
                "candidate_artifact_dir": str(run_dir),
            }
            append_event(MEMORY_PATH, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)

        if not dry_run:
            final_status = control_status(config, state, official)
            if final_status["lifecycle_status"] == "ready":
                update_lifecycle(
                    state,
                    "ready",
                    "invocation_round_limit_reached",
                )
            else:
                update_lifecycle(
                    state,
                    str(final_status["lifecycle_status"]),
                    str(final_status["stop_reason"]),
                    final_status["action_required"],
                )
            state.save(STATE_PATH)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the serial experiment-control agent.")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--record-official",
        nargs="+",
        metavar="KEY=VALUE",
        help="Record user-confirmed official feedback, e.g. A1_accuracy=0.87 A2_ndcg_at_10=0.512",
    )
    parser.add_argument("--submitted-at", default=None)
    parser.add_argument("--feedback-note", default="")
    args = parser.parse_args()
    if args.record_official:
        values: dict[str, float] = {}
        for item in args.record_official:
            key, separator, value = item.partition("=")
            if not separator:
                parser.error(f"invalid official feedback: {item!r}; expected KEY=VALUE")
            try:
                values[key] = float(value)
            except ValueError:
                parser.error(f"invalid official feedback value: {item!r}")
        payload = record_feedback(
            AGENT_DIR / "official_feedback.json",
            values,
            submitted_at=args.submitted_at,
            note=args.feedback_note,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.status:
        config = load_config()
        official = load_reference(config, AGENT_DIR / "official_feedback.json")
        state = AgentState.load(STATE_PATH, official)
        print(json.dumps({
            "official_reference": official,
            "targets": config.get("targets", {}),
            "control": control_status(config, state, official),
            "local_experiment_state": state.__dict__,
            "llm_planner_configured": llm_planner_is_configured(config),
        }, ensure_ascii=False, indent=2))
        return
    raise SystemExit(run_agent(args.max_rounds, args.dry_run))


if __name__ == "__main__":
    main()
