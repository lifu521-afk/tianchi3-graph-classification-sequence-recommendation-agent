"""Rule-compliant serial experiment agent for sequence recommendation tasks.

All model choices are derived from the supplied train/test/catalog inputs. The
agent does not consume external labels, leaderboard feedback, user IDs as
features, or dataset-specific prediction rules.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SEED = 20260723
# The supplied training data has no genuine empty-history rows. User-prior
# scores remain available for diagnostics, but must not replace rankings for
# cold-start test users based on synthetically truncated histories.


@dataclass
class RecommendationData:
    train: pd.DataFrame
    test: pd.DataFrame
    item: pd.DataFrame
    user: pd.DataFrame | None
    candidates: list[str]


@dataclass
class CandidateResult:
    name: str
    mean_ndcg: float
    min_ndcg: float
    fold_ndcgs: list[float]
    mean_empty_ndcg: float | None
    fold_empty_ndcgs: list[float | None]
    accepted: bool
    reason: str


def read_items(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [iid for iid in value.split(",") if iid]


def row_standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-6
    )


def load_data(data_dir: Path) -> RecommendationData:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    item = pd.read_csv(data_dir / "item.csv")
    user_path = data_dir / "user.csv"
    user = pd.read_csv(user_path) if user_path.exists() else None
    required_train = {"uid", "target_iid", "item_seq_raw"}
    required_test = {"uid", "item_seq_raw"}
    if not required_train.issubset(train.columns) or not required_test.issubset(test.columns):
        raise ValueError("Input tables do not have the required sequence columns.")
    if "iid" not in item.columns:
        raise ValueError("Item table must contain iid.")
    if user is not None and "uid" not in user.columns:
        raise ValueError("User table must contain uid.")
    candidate_set = set(item["iid"].astype(str))
    train["target_iid"] = train["target_iid"].astype(str)
    if not set(train["target_iid"]).issubset(candidate_set):
        raise ValueError("Some visible targets are absent from the item catalog.")
    candidates = train["target_iid"].value_counts().index.tolist()
    return RecommendationData(train, test, item, user, candidates)


def ndcg_at_10(scores: np.ndarray, candidates: list[str], targets: pd.Series) -> float:
    k = min(10, len(candidates))
    lookup = {iid: pos for pos, iid in enumerate(candidates)}
    target_pos = np.asarray([lookup.get(str(iid), -1) for iid in targets], dtype=np.int32)
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
    result = np.zeros(len(scores), dtype=np.float32)
    for rank in range(k):
        result[top[:, rank] == target_pos] = 1.0 / math.log2(rank + 2.0)
    return float(result.mean())


def profile_data(data: RecommendationData) -> dict[str, object]:
    train_lengths = data.train["item_seq_raw"].map(lambda value: len(read_items(value)))
    test_lengths = data.test["item_seq_raw"].map(lambda value: len(read_items(value)))
    category_cols = [col for col in data.item.columns if col != "iid"]
    return {
        "train_rows": int(len(data.train)),
        "test_rows": int(len(data.test)),
        "catalog_items": int(len(data.item)),
        "visible_target_items": int(len(data.candidates)),
        "target_coverage_of_catalog": float(len(data.candidates) / len(data.item)),
        "empty_history_rate_train": float((train_lengths == 0).mean()),
        "empty_history_rate_test": float((test_lengths == 0).mean()),
        "history_length_median_train": float(train_lengths.median()),
        "history_length_median_test": float(test_lengths.median()),
        "history_length_p90_train": float(train_lengths.quantile(0.9)),
        "history_length_p90_test": float(test_lengths.quantile(0.9)),
        "item_metadata_columns": category_cols,
        "user_metadata_columns": []
        if data.user is None
        else [col for col in data.user.columns if col != "uid"],
        "test_fraction": float(len(data.test) / (len(data.train) + len(data.test))),
    }


def make_testlike_holdouts(data: RecommendationData, n_repeats: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keep at least one example per target in fitting data and match test sequence lengths."""
    fraction = len(data.test) / (len(data.train) + len(data.test))
    target_counts = data.train["target_iid"].value_counts().to_dict()
    test_lengths = data.test["item_seq_raw"].map(lambda value: len(read_items(value))).to_numpy()
    holdouts: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(SEED + repeat)
        remaining = dict(target_counts)
        valid = np.zeros(len(data.train), dtype=bool)
        desired = int(round(len(data.train) * fraction))
        for position in rng.permutation(len(data.train)):
            target = data.train.iloc[position]["target_iid"]
            if remaining[target] <= 1:
                continue
            valid[position] = True
            remaining[target] -= 1
            if int(valid.sum()) >= desired:
                break
        sampled_lengths = rng.choice(test_lengths, size=int(valid.sum()), replace=True)
        holdouts.append((np.flatnonzero(~valid), np.flatnonzero(valid), sampled_lengths))
    return holdouts


def truncate_to_testlike(frame: pd.DataFrame, sampled_lengths: np.ndarray) -> pd.DataFrame:
    visible = frame.drop(columns=["target_iid"]).copy()
    for position, keep_length in enumerate(sampled_lengths):
        sequence = read_items(frame.iloc[position]["item_seq_raw"])
        kept = sequence[-int(keep_length) :] if keep_length > 0 else []
        visible.iat[position, visible.columns.get_loc("item_seq_raw")] = ",".join(kept)
    return visible


def transition_scores(
    fit: pd.DataFrame, query: pd.DataFrame, candidates: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_items: set[str] = set()
    for value in fit["item_seq_raw"]:
        observed_items.update(read_items(value))
    item_ids = sorted(observed_items)
    item_to_pos = {iid: pos for pos, iid in enumerate(item_ids)}
    target_to_pos = {iid: pos for pos, iid in enumerate(candidates)}
    recent_rows: list[int] = []
    recent_cols: list[int] = []
    recent_values: list[float] = []
    last_rows: list[int] = []
    last_cols: list[int] = []
    for row in fit.itertuples(index=False):
        target_pos = target_to_pos[row.target_iid]
        sequence = read_items(row.item_seq_raw)
        if sequence and sequence[-1] in item_to_pos:
            last_rows.append(item_to_pos[sequence[-1]])
            last_cols.append(target_pos)
        seen: dict[int, float] = {}
        for offset, iid in enumerate(reversed(sequence[-80:])):
            if iid in item_to_pos:
                pos = item_to_pos[iid]
                seen[pos] = seen.get(pos, 0.0) + 1.0 / math.sqrt(offset + 1.0)
        for pos, value in seen.items():
            recent_rows.append(pos)
            recent_cols.append(target_pos)
            recent_values.append(value)
    shape = (len(item_ids), len(candidates))
    recent_transition = sparse.csr_matrix(
        (np.log1p(recent_values), (recent_rows, recent_cols)), shape=shape, dtype=np.float32
    )
    last_transition = sparse.csr_matrix(
        (np.ones(len(last_rows), dtype=np.float32), (last_rows, last_cols)),
        shape=shape,
        dtype=np.float32,
    )
    query_rows: list[int] = []
    query_cols: list[int] = []
    query_values: list[float] = []
    last_query_rows: list[int] = []
    last_query_cols: list[int] = []
    repeat_rows: list[int] = []
    repeat_cols: list[int] = []
    repeat_values: list[float] = []
    for row_pos, value in enumerate(query["item_seq_raw"]):
        sequence = read_items(value)
        if sequence and sequence[-1] in item_to_pos:
            last_query_rows.append(row_pos)
            last_query_cols.append(item_to_pos[sequence[-1]])
        counts = Counter(sequence)
        for offset, iid in enumerate(reversed(sequence[-80:])):
            if iid in item_to_pos:
                query_rows.append(row_pos)
                query_cols.append(item_to_pos[iid])
                query_values.append(1.0 / math.sqrt(offset + 1.0))
        for iid, count in counts.items():
            target_pos = target_to_pos.get(iid)
            if target_pos is not None:
                repeat_rows.append(row_pos)
                repeat_cols.append(target_pos)
                repeat_values.append(1.0 + math.log1p(count))
    query_shape = (len(query), len(item_ids))
    recent_query = sparse.csr_matrix(
        (query_values, (query_rows, query_cols)), shape=query_shape, dtype=np.float32
    )
    last_query = sparse.csr_matrix(
        (np.ones(len(last_query_rows), dtype=np.float32), (last_query_rows, last_query_cols)),
        shape=query_shape,
        dtype=np.float32,
    )
    repeat = sparse.csr_matrix(
        (repeat_values, (repeat_rows, repeat_cols)),
        shape=(len(query), len(candidates)),
        dtype=np.float32,
    ).toarray()
    return (
        row_standardize((recent_query @ recent_transition).toarray().astype(np.float32)),
        row_standardize((last_query @ last_transition).toarray().astype(np.float32)),
        row_standardize(repeat.astype(np.float32)),
    )


def category_scores(
    query: pd.DataFrame, item: pd.DataFrame, candidates: list[str]
) -> np.ndarray | None:
    columns = [col for col in item.columns if col != "iid"]
    if not columns:
        return None
    item_lookup = item.set_index("iid")[columns].astype(str)
    if not set(candidates).issubset(item_lookup.index):
        return None
    target_values = item_lookup.loc[candidates].to_numpy()
    item_values = {
        iid: tuple(values)
        for iid, values in zip(item_lookup.index.tolist(), item_lookup.to_numpy())
    }
    category_masks: list[dict[str, np.ndarray]] = []
    for col_pos in range(len(columns)):
        category_masks.append(
            {
                category: target_values[:, col_pos] == category
                for category in np.unique(target_values[:, col_pos])
            }
        )
    output = np.zeros((len(query), len(candidates)), dtype=np.float32)
    for row_pos, value in enumerate(query["item_seq_raw"]):
        sequence = [iid for iid in read_items(value)[-80:] if iid in item_values]
        if not sequence:
            continue
        recency = np.linspace(0.25, 1.0, len(sequence), dtype=np.float32)
        for col_pos in range(len(columns)):
            counts: dict[str, float] = {}
            for iid, weight in zip(sequence, recency):
                category = item_values[iid][col_pos]
                counts[category] = counts.get(category, 0.0) + float(weight)
            if counts:
                scale = max(counts.values())
                for category, score in counts.items():
                    mask = category_masks[col_pos].get(category)
                    if mask is not None:
                        output[row_pos, mask] += score / scale
    return row_standardize(output)


def user_prior_scores(
    fit: pd.DataFrame, query: pd.DataFrame, data: RecommendationData
) -> np.ndarray | None:
    if data.user is None:
        return None
    user_columns = [col for col in data.user.columns if col != "uid"]
    if not user_columns:
        return None
    target_to_pos = {iid: pos for pos, iid in enumerate(data.candidates)}
    fit_user = fit[["uid", "target_iid"]].merge(data.user, on="uid", how="left")
    query_user = query[["uid"]].merge(data.user, on="uid", how="left")
    global_counts = (
        fit["target_iid"].value_counts().reindex(data.candidates).fillna(0).to_numpy(np.float32)
    )
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(global_counts))
    prior_strength = max(1.0, math.sqrt(len(fit) / max(len(data.candidates), 1)))
    output = np.zeros((len(query), len(data.candidates)), dtype=np.float32)
    for col in user_columns:
        group_scores: dict[object, np.ndarray] = {}
        for value, part in fit_user.groupby(col, dropna=False):
            positions = np.fromiter(
                (target_to_pos[iid] for iid in part["target_iid"]),
                count=len(part),
                dtype=np.int32,
            )
            counts = np.bincount(positions, minlength=len(data.candidates)).astype(np.float32)
            group_scores[value] = np.log1p(counts + prior_strength * global_prior)
        default = np.log1p(prior_strength * global_prior)
        for row_pos, value in enumerate(query_user[col]):
            output[row_pos] += group_scores.get(value, default)
    return row_standardize(output / len(user_columns))


def user_pair_prior_scores(
    fit: pd.DataFrame, query: pd.DataFrame, data: RecommendationData
) -> np.ndarray | None:
    """Build a support-shrunk target lift from automatically generated user-field pairs."""
    if data.user is None:
        return None
    user_columns = [col for col in data.user.columns if col != "uid"]
    if len(user_columns) < 2:
        return None
    target_to_pos = {iid: pos for pos, iid in enumerate(data.candidates)}
    target_positions = np.asarray(
        [target_to_pos[iid] for iid in fit["target_iid"]], dtype=np.int32
    )
    target_counts = (
        fit["target_iid"].value_counts().reindex(data.candidates).fillna(0).to_numpy(np.float32)
    )
    global_prior = (target_counts + 1.0) / (target_counts.sum() + len(target_counts))
    fit_user = fit[["uid"]].merge(data.user, on="uid", how="left")
    query_user = query[["uid"]].merge(data.user, on="uid", how="left")
    output = np.zeros((len(query), len(data.candidates)), dtype=np.float32)
    weight_sum = np.zeros(len(query), dtype=np.float32)
    smoothing = max(20.0, math.sqrt(len(fit)))
    max_groups = max(64, int(math.sqrt(len(fit)) * 12))

    for left, right in combinations(user_columns, 2):
        fit_key = (
            fit_user[left].astype("string").fillna("<missing>")
            + "|"
            + fit_user[right].astype("string").fillna("<missing>")
        )
        query_key = (
            query_user[left].astype("string").fillna("<missing>")
            + "|"
            + query_user[right].astype("string").fillna("<missing>")
        )
        combined = pd.concat([fit_key, query_key], ignore_index=True)
        codes, groups = pd.factorize(combined, sort=False)
        fit_codes = codes[: len(fit)].astype(np.int32)
        query_codes = codes[len(fit) :].astype(np.int32)
        if len(groups) < 2 or len(groups) > max_groups:
            continue
        matrix = sparse.csr_matrix(
            (
                np.ones(len(fit), dtype=np.float32),
                (fit_codes, target_positions),
            ),
            shape=(len(groups), len(data.candidates)),
            dtype=np.float32,
        )
        support = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
        selected = matrix[query_codes].toarray().astype(np.float32)
        selected_support = support[query_codes]
        probabilities = (selected + smoothing * global_prior[None, :]) / (
            selected_support[:, None] + smoothing
        )
        lift = np.log(np.maximum(probabilities / global_prior[None, :], 1e-8))
        reliability = selected_support / (selected_support + smoothing)
        output += lift * reliability[:, None]
        weight_sum += reliability

    if not weight_sum.any():
        return None
    output /= np.maximum(weight_sum[:, None], 1e-6)
    return row_standardize(output)


def score_components(
    fit: pd.DataFrame, query: pd.DataFrame, data: RecommendationData
) -> dict[str, np.ndarray]:
    target_counts = fit["target_iid"].value_counts().reindex(data.candidates).fillna(0).to_numpy(np.float32)
    global_scores = row_standardize(
        np.tile(np.log1p(target_counts)[None, :], (len(query), 1)).astype(np.float32)
    )
    recent, last, repeat = transition_scores(fit, query, data.candidates)
    components = {"global": global_scores, "recent": recent, "last": last, "repeat": repeat}
    category = category_scores(query, data.item, data.candidates)
    if category is not None:
        components["category"] = category
    user = user_prior_scores(fit, query, data)
    if user is not None:
        components["user"] = user
    return components


def candidate_scores(
    components: dict[str, np.ndarray], query: pd.DataFrame
) -> list[tuple[str, np.ndarray, str, bool]]:
    candidates = [
        ("global_popularity", components["global"], "mandatory_label_frequency_baseline", False),
        (
            "transition_ranker",
            0.35 * components["global"] + components["recent"] + 0.45 * components["last"],
            "observed_sequence_transition_signal",
            False,
        ),
        (
            "transition_repeat_ranker",
            0.25 * components["global"]
            + components["recent"]
            + 0.45 * components["last"]
            + 0.70 * components["repeat"],
            "transition_plus_repeat_signal",
            False,
        ),
    ]
    if "category" in components:
        candidates.append(
            (
                "transition_repeat_category_ranker",
                0.25 * components["global"]
                + components["recent"]
                + 0.45 * components["last"]
                + 0.70 * components["repeat"]
                + 0.35 * components["category"],
                "catalog_metadata_available",
                False,
            )
        )
    return candidates


def run_agent(
    data_dir: Path,
    output_dir: Path,
    task_id: str,
    budget_minutes: float,
    validation_repeats: int,
) -> dict[str, object]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    profile = profile_data(data)
    trajectory: list[dict[str, object]] = [
        {"round": 0, "action": "profile", "feedback": profile, "next_strategy": "run_serial_rankers"}
    ]
    holdouts = make_testlike_holdouts(data, validation_repeats)
    result_rows: list[CandidateResult] = []
    best: CandidateResult | None = None
    candidate_order = [
        "global_popularity",
        "transition_ranker",
        "transition_repeat_ranker",
        "transition_repeat_category_ranker",
    ]
    per_candidate: dict[str, list[float]] = {name: [] for name in candidate_order}
    per_candidate_empty: dict[str, list[float | None]] = {name: [] for name in candidate_order}
    candidate_is_segment_override: dict[str, bool] = {}
    for fold, (fit_pos, valid_pos, sampled_lengths) in enumerate(holdouts, start=1):
        if time.monotonic() - started > budget_minutes * 60:
            trajectory.append(
                {"round": len(trajectory), "action": "stop", "reason": "budget_reached_before_next_fold"}
            )
            break
        fold_started = time.monotonic()
        fit = data.train.iloc[fit_pos].reset_index(drop=True)
        heldout = data.train.iloc[valid_pos].reset_index(drop=True)
        visible = truncate_to_testlike(heldout, sampled_lengths)
        components = score_components(fit, visible, data)
        empty_mask = visible["item_seq_raw"].map(lambda value: len(read_items(value)) == 0).to_numpy()
        for name, scores, _, is_segment_override in candidate_scores(components, visible):
            candidate_is_segment_override[name] = is_segment_override
            per_candidate[name].append(ndcg_at_10(scores, data.candidates, heldout["target_iid"]))
            per_candidate_empty[name].append(
                None
                if not empty_mask.any()
                else ndcg_at_10(
                    scores[empty_mask],
                    data.candidates,
                    heldout.loc[empty_mask, "target_iid"],
                )
            )
        trajectory.append(
            {
                "round": len(trajectory),
                "action": "evaluate_testlike_fold",
                "configuration": "all_enabled_candidates",
                "feedback": {
                    "fold": fold,
                    "fit_rows": int(len(fit)),
                    "validation_rows": int(len(heldout)),
                    "empty_history_rows": int(empty_mask.sum()),
                    "ndcg_at_10": {name: values[-1] for name, values in per_candidate.items() if values},
                    "empty_history_ndcg_at_10": {
                        name: values[-1] for name, values in per_candidate_empty.items() if values
                    },
                    "shared_validation_runtime_seconds": time.monotonic() - fold_started,
                },
                "next_strategy": "aggregate_stability_gate",
            }
        )
    for name in candidate_order:
        scores = per_candidate[name]
        if not scores:
            continue
        mean_score, min_score = float(np.mean(scores)), float(np.min(scores))
        empty_scores = per_candidate_empty[name]
        observed_empty = [score for score in empty_scores if score is not None]
        mean_empty_score = None if not observed_empty else float(np.mean(observed_empty))
        if best is None:
            accepted, reason = True, "baseline"
        elif candidate_is_segment_override.get(name, False):
            segment_baseline = next(
                (
                    row
                    for row in reversed(result_rows)
                    if row.accepted and not candidate_is_segment_override.get(row.name, False)
                ),
                best,
            )
            baseline_empty = [score for score in segment_baseline.fold_empty_ndcgs if score is not None]
            empty_deltas = np.asarray(observed_empty) - np.asarray(baseline_empty)
            accepted = (
                len(empty_deltas) == len(observed_empty)
                and mean_score >= segment_baseline.mean_ndcg - 0.0001
                and min_score >= segment_baseline.min_ndcg - 0.0001
                and float(empty_deltas.mean()) >= 0.001
                and float(empty_deltas.min()) >= 0.0
            )
            reason = (
                "passes_segment_gate_against_shared_baseline"
                if accepted
                else "rejected_by_segment_stability_gate"
            )
        else:
            accepted = mean_score >= best.mean_ndcg + 0.0005 and min_score >= best.min_ndcg - 0.0005
            reason = "passes_mean_and_worst_fold_gate" if accepted else "rejected_by_stability_gate"
        result = CandidateResult(
            name,
            mean_score,
            min_score,
            scores,
            mean_empty_score,
            empty_scores,
            accepted,
            reason,
        )
        result_rows.append(result)
        trajectory.append(
            {
                "round": len(trajectory),
                "action": "select_candidate",
                "configuration": name,
                "feedback": {
                    "mean_ndcg_at_10": mean_score,
                    "min_ndcg_at_10": min_score,
                    "fold_ndcgs": scores,
                    "mean_empty_history_ndcg_at_10": mean_empty_score,
                    "fold_empty_history_ndcgs": empty_scores,
                },
                "next_strategy": reason,
            }
        )
        if accepted and not candidate_is_segment_override.get(name, False):
            best = result
        elif accepted:
            best_empty = -np.inf if best.mean_empty_ndcg is None else best.mean_empty_ndcg
            candidate_empty = -np.inf if result.mean_empty_ndcg is None else result.mean_empty_ndcg
            if candidate_empty > best_empty or (
                candidate_empty == best_empty and result.mean_ndcg > best.mean_ndcg
            ):
                best = result
    if best is None:
        raise RuntimeError("No recommendation candidate completed.")
    if time.monotonic() - started > budget_minutes * 60:
        raise RuntimeError("Budget exhausted before fitting the selected model.")
    final_components = score_components(data.train, data.test, data)
    selected_scores = dict(
        (name, scores) for name, scores, _, _ in candidate_scores(final_components, data.test)
    )[best.name]
    k = min(10, len(data.candidates))
    top = np.argpartition(-selected_scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(selected_scores, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
    candidate_array = np.asarray(data.candidates)
    output = pd.DataFrame(
        {"uid": data.test["uid"], "prediction": [",".join(candidate_array[row].tolist()) for row in top]}
    )
    output_path = output_dir / f"{task_id}.csv"
    output.to_csv(output_path, index=False)
    report = {
        "data_dir": str(data_dir),
        "profile": profile,
        "selected_model": best.name,
        "candidate_results": [
            {
                "name": row.name,
                "mean_ndcg_at_10": row.mean_ndcg,
                "min_ndcg_at_10": row.min_ndcg,
                "fold_ndcgs": row.fold_ndcgs,
                "mean_empty_history_ndcg_at_10": row.mean_empty_ndcg,
                "fold_empty_history_ndcgs": row.fold_empty_ndcgs,
                "accepted": row.accepted,
                "reason": row.reason,
            }
            for row in result_rows
        ],
        "runtime_seconds": time.monotonic() - started,
        "validation_repeats_completed": len(next(iter(per_candidate.values()))),
        "submission": str(output_path),
    }
    (output_dir / "recommendation_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "recommendation_agent_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"trajectory_{task_id}.json").write_text(
        json.dumps({"events": trajectory}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", default="B2")
    parser.add_argument("--budget-minutes", type=float, default=110.0)
    parser.add_argument("--validation-repeats", type=int, default=3)
    args = parser.parse_args()
    report = run_agent(
        args.data_dir,
        args.output_dir,
        args.task_id,
        args.budget_minutes,
        args.validation_repeats,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
