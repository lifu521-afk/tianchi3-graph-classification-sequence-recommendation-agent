from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir

REC_DIR = task_dir("B_recommendation")
SEED = 20260722


def read_items(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [part for part in value.split(",") if part]


def format_counts(items: list[str]) -> str | float:
    if not items:
        return np.nan
    return ",".join(f"{iid}:{count}" for iid, count in Counter(items).items())


def make_holdout(targets: pd.Series, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    counts = targets.value_counts().to_dict()
    remaining = counts.copy()
    valid = np.zeros(len(targets), dtype=bool)
    desired = int(round(len(targets) * fraction))
    for row in rng.permutation(len(targets)):
        target = targets.iloc[row]
        if counts[target] < 2 or remaining[target] <= 1:
            continue
        valid[row] = True
        remaining[target] -= 1
        if valid.sum() >= desired:
            break
    return np.flatnonzero(~valid), np.flatnonzero(valid)


def make_test_like(validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    test_lengths = test["item_seq_raw"].map(lambda value: len(read_items(value))).to_numpy()
    sampled_lengths = rng.choice(test_lengths, size=len(validation), replace=True)
    visible = validation.drop(columns=["target_iid"]).copy()
    for row_pos, keep_length in enumerate(sampled_lengths):
        sequence = read_items(validation.iloc[row_pos]["item_seq_raw"])
        kept = sequence[-int(keep_length) :] if keep_length > 0 else []
        if kept:
            visible.iat[row_pos, visible.columns.get_loc("item_seq_raw")] = ",".join(kept)
            visible.iat[row_pos, visible.columns.get_loc("item_seq_dedup")] = ",".join(dict.fromkeys(kept))
            visible.iat[row_pos, visible.columns.get_loc("item_seq_counts")] = format_counts(kept)
        else:
            visible.iat[row_pos, visible.columns.get_loc("item_seq_raw")] = np.nan
            visible.iat[row_pos, visible.columns.get_loc("item_seq_dedup")] = np.nan
            visible.iat[row_pos, visible.columns.get_loc("item_seq_counts")] = np.nan
    return visible


def row_standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def ndcg_rows(scores: np.ndarray, class_items: list[str], targets: pd.Series) -> tuple[float, np.ndarray]:
    lookup = {iid: idx for idx, iid in enumerate(class_items)}
    target_cols = np.asarray([lookup.get(iid, -1) for iid in targets], dtype=np.int32)
    top = np.argpartition(-scores, kth=9, axis=1)[:, :10]
    top_scores = np.take_along_axis(scores, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
    values = np.zeros(len(scores), dtype=np.float32)
    for rank in range(10):
        values[top[:, rank] == target_cols] = 1.0 / math.log2(rank + 2.0)
    return float(values.mean()), values


def history_query_matrix(
    frame: pd.DataFrame,
    item_to_idx: dict[str, int],
    recent_limit: int = 80,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    recent_rows: list[int] = []
    recent_cols: list[int] = []
    recent_values: list[float] = []
    last_rows: list[int] = []
    last_cols: list[int] = []
    repeat_rows: list[int] = []
    repeat_cols: list[int] = []
    repeat_values: list[float] = []
    for row_pos, value in enumerate(frame["item_seq_raw"]):
        sequence = read_items(value)
        if sequence:
            last_idx = item_to_idx.get(sequence[-1])
            if last_idx is not None:
                last_rows.append(row_pos)
                last_cols.append(last_idx)
        counts = Counter(sequence)
        for offset, iid in enumerate(reversed(sequence[-recent_limit:])):
            item_idx = item_to_idx.get(iid)
            if item_idx is None:
                continue
            recent_rows.append(row_pos)
            recent_cols.append(item_idx)
            recent_values.append(1.0 / math.sqrt(offset + 1.0))
        for iid, count in counts.items():
            item_idx = item_to_idx.get(iid)
            if item_idx is None:
                continue
            repeat_rows.append(row_pos)
            repeat_cols.append(item_idx)
            repeat_values.append(1.0 + math.log1p(count))
    shape = (len(frame), len(item_to_idx))
    recent = sparse.csr_matrix((recent_values, (recent_rows, recent_cols)), shape=shape, dtype=np.float32)
    last = sparse.csr_matrix(
        (np.ones(len(last_rows), dtype=np.float32), (last_rows, last_cols)),
        shape=shape,
    )
    repeat = sparse.csr_matrix((repeat_values, (repeat_rows, repeat_cols)), shape=shape, dtype=np.float32)
    return recent, last, repeat


def fit_transition_matrices(
    train: pd.DataFrame,
    item_to_idx: dict[str, int],
    target_to_col: dict[str, int],
    recent_limit: int = 80,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    recent_rows: list[int] = []
    recent_cols: list[int] = []
    recent_values: list[float] = []
    last_rows: list[int] = []
    last_cols: list[int] = []
    for row in train.itertuples(index=False):
        target_col = target_to_col[row.target_iid]
        sequence = read_items(row.item_seq_raw)
        if sequence:
            last_idx = item_to_idx.get(sequence[-1])
            if last_idx is not None:
                last_rows.append(last_idx)
                last_cols.append(target_col)
        unique_recent: dict[int, float] = {}
        for offset, iid in enumerate(reversed(sequence[-recent_limit:])):
            item_idx = item_to_idx.get(iid)
            if item_idx is None:
                continue
            unique_recent[item_idx] = unique_recent.get(item_idx, 0.0) + 1.0 / math.sqrt(offset + 1.0)
        for item_idx, value in unique_recent.items():
            recent_rows.append(item_idx)
            recent_cols.append(target_col)
            recent_values.append(value)
    shape = (len(item_to_idx), len(target_to_col))
    recent = sparse.csr_matrix((recent_values, (recent_rows, recent_cols)), shape=shape, dtype=np.float32)
    last = sparse.csr_matrix(
        (np.ones(len(last_rows), dtype=np.float32), (last_rows, last_cols)),
        shape=shape,
        dtype=np.float32,
    )
    recent.data = np.log1p(recent.data)
    last.data = np.log1p(last.data)
    return recent, last


def category_scores(
    frame: pd.DataFrame,
    item: pd.DataFrame,
    class_items: list[str],
) -> np.ndarray:
    item_lookup = item.set_index("iid")
    columns = ["i_cat_01", "i_cat_02", "i_cat_03", "i_bucket_01"]
    target_values = item_lookup.loc[class_items, columns].to_numpy(dtype=np.int32)
    output = np.zeros((len(frame), len(class_items)), dtype=np.float32)
    weights = np.asarray([0.35, 0.55, 1.15, 0.45], dtype=np.float32)
    valid_items = set(item_lookup.index)
    for row_pos, value in enumerate(frame["item_seq_raw"]):
        sequence = [iid for iid in read_items(value) if iid in valid_items]
        if not sequence:
            continue
        recent = sequence[-80:]
        metadata = item_lookup.loc[recent, columns].to_numpy(dtype=np.int32)
        recency = np.linspace(0.25, 1.0, len(metadata), dtype=np.float32)
        for col_pos, weight in enumerate(weights):
            values = metadata[:, col_pos]
            counts: dict[int, float] = {}
            for category, row_weight in zip(values, recency):
                counts[int(category)] = counts.get(int(category), 0.0) + float(row_weight)
            maximum = max(counts.values())
            for category, score in counts.items():
                output[row_pos, target_values[:, col_pos] == category] += weight * score / maximum
    return output


def user_prior_scores(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    user: pd.DataFrame,
    class_items: list[str],
) -> np.ndarray:
    target_to_col = {iid: idx for idx, iid in enumerate(class_items)}
    user_cols = [col for col in user.columns if col != "uid"]
    merged = train[["uid", "target_iid"]].merge(user, on="uid", how="left")
    frame_user = frame[["uid"]].merge(user, on="uid", how="left")
    global_counts = train["target_iid"].value_counts().reindex(class_items).fillna(0).to_numpy(np.float32)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(global_counts))
    output = np.zeros((len(frame), len(class_items)), dtype=np.float32)
    for col in user_cols:
        group_arrays: dict[object, np.ndarray] = {}
        for value, part in merged.groupby(col, dropna=False):
            counts = np.zeros(len(class_items), dtype=np.float32)
            for iid, count in part["target_iid"].value_counts().items():
                counts[target_to_col[iid]] = float(count)
            group_arrays[value] = np.log1p((counts + 20.0 * global_prior) * 100.0)
        for row_pos, value in enumerate(frame_user[col]):
            output[row_pos] += group_arrays.get(value, np.log1p(global_prior * 100.0))
    return output / max(len(user_cols), 1)


def rule_components(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    user: pd.DataFrame,
    item: pd.DataFrame,
    class_items: list[str],
) -> dict[str, np.ndarray]:
    item_ids = item["iid"].tolist()
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    target_to_col = {iid: idx for idx, iid in enumerate(class_items)}
    class_item_indices = np.asarray([item_to_idx[iid] for iid in class_items], dtype=np.int32)
    recent_transition, last_transition = fit_transition_matrices(train, item_to_idx, target_to_col)
    recent_query, last_query, repeat_query = history_query_matrix(frame, item_to_idx)

    global_counts = train["target_iid"].value_counts().reindex(class_items).fillna(0).to_numpy(np.float32)
    global_scores = np.tile(np.log1p(global_counts)[None, :], (len(frame), 1))
    recent_scores = (recent_query @ recent_transition).toarray().astype(np.float32)
    last_scores = (last_query @ last_transition).toarray().astype(np.float32)
    repeat_scores = repeat_query[:, class_item_indices].toarray().astype(np.float32)
    categories = category_scores(frame, item, class_items)
    users = user_prior_scores(train, frame, user, class_items)
    return {
        "global": row_standardize(global_scores),
        "recent": row_standardize(recent_scores),
        "last": row_standardize(last_scores),
        "repeat": row_standardize(repeat_scores),
        "category": row_standardize(categories),
        "user": row_standardize(users),
    }


def history_buckets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    lengths = frame["item_seq_raw"].map(lambda value: len(read_items(value))).to_numpy()
    return {
        "empty": lengths == 0,
        "short": (lengths > 0) & (lengths <= 10),
        "medium": (lengths > 10) & (lengths < 200),
        "long": lengths >= 200,
    }


def tune_rule_blend(
    components: dict[str, np.ndarray],
    frame: pd.DataFrame,
    targets: pd.Series,
    class_items: list[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, object]]:
    base_weights = {
        "global": 0.55,
        "recent": 1.10,
        "last": 0.55,
        "repeat": 0.75,
        "category": 0.55,
        "user": 0.15,
    }
    masks = history_buckets(frame)
    final = np.zeros_like(next(iter(components.values())))
    selected: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, object] = {"component_scores": {}, "bucket_scores": {}}
    for name, values in components.items():
        diagnostics["component_scores"][name] = ndcg_rows(values, class_items, targets)[0]

    for bucket, mask in masks.items():
        if not mask.any():
            continue
        weights = dict(base_weights)
        if bucket == "empty":
            weights.update({"global": 1.0, "recent": 0.0, "last": 0.0, "repeat": 0.0, "category": 0.0, "user": 0.25})
        current = sum(weights[name] * components[name][mask] for name in weights)
        current_score = ndcg_rows(current, class_items, targets[mask])[0]
        for _ in range(3):
            improved = False
            for name in weights:
                best = (current_score, weights[name])
                for candidate in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.4]:
                    trial_weights = dict(weights)
                    trial_weights[name] = candidate
                    trial = sum(trial_weights[key] * components[key][mask] for key in trial_weights)
                    score = ndcg_rows(trial, class_items, targets[mask])[0]
                    if score > best[0] + 1e-7:
                        best = (score, candidate)
                if best[1] != weights[name]:
                    weights[name] = best[1]
                    current_score = best[0]
                    improved = True
            if not improved:
                break
        final[mask] = sum(weights[name] * components[name][mask] for name in weights)
        selected[bucket] = weights
        diagnostics["bucket_scores"][bucket] = {
            "rows": int(mask.sum()),
            "ndcg_at_10": current_score,
        }
    diagnostics["total_ndcg_at_10"] = ndcg_rows(final, class_items, targets)[0]
    return final, selected, diagnostics


def topk_submission(scores: np.ndarray, class_items: list[str], uids: pd.Series) -> pd.DataFrame:
    item_array = np.asarray(class_items)
    top = np.argpartition(-scores, kth=9, axis=1)[:, :10]
    top_scores = np.take_along_axis(scores, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
    predictions = [",".join(item_array[row].tolist()) for row in top]
    return pd.DataFrame({"uid": uids, "prediction": predictions})


def run_rules(validate: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    train = pd.read_csv(REC_DIR / "train.csv")
    test = pd.read_csv(REC_DIR / "test.csv")
    user = pd.read_csv(REC_DIR / "user.csv")
    item = pd.read_csv(REC_DIR / "item.csv")
    class_items = train["target_iid"].value_counts().index.tolist()
    selected_weights: dict[str, dict[str, float]]
    log: dict[str, object] = {
        "method": "sparse_transition_repeat_category_user_bucket_blend",
        "num_classes": len(class_items),
    }
    if validate:
        fit_pos, valid_pos = make_holdout(train["target_iid"])
        inner_train = train.iloc[fit_pos].reset_index(drop=True)
        inner_valid = train.iloc[valid_pos].reset_index(drop=True)
        visible_valid = make_test_like(inner_valid, test)
        print("building validation rule components...", flush=True)
        components = rule_components(inner_train, visible_valid, user, item, class_items)
        _, selected_weights, diagnostics = tune_rule_blend(
            components, visible_valid, inner_valid["target_iid"], class_items
        )
        log["validation"] = diagnostics
        log["selected_weights"] = selected_weights
        print(json.dumps(diagnostics, ensure_ascii=False), flush=True)
    else:
        selected_weights = {
            "empty": {"global": 1.0, "recent": 0.0, "last": 0.0, "repeat": 0.0, "category": 0.0, "user": 0.25},
            "short": {"global": 0.55, "recent": 1.1, "last": 0.55, "repeat": 0.75, "category": 0.55, "user": 0.15},
            "medium": {"global": 0.55, "recent": 1.1, "last": 0.55, "repeat": 0.75, "category": 0.55, "user": 0.15},
            "long": {"global": 0.55, "recent": 1.1, "last": 0.55, "repeat": 0.75, "category": 0.55, "user": 0.15},
        }
        log["selected_weights"] = selected_weights

    print("building final rule components...", flush=True)
    final_components = rule_components(train, test, user, item, class_items)
    masks = history_buckets(test)
    final_scores = np.zeros_like(next(iter(final_components.values())))
    for bucket, mask in masks.items():
        weights = selected_weights[bucket]
        final_scores[mask] = sum(weights[name] * final_components[name][mask] for name in weights)
    return topk_submission(final_scores, class_items, test["uid"]), log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "b_final_20260722")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission, log = run_rules(validate=not args.skip_validation)
    submission.to_csv(args.output_dir / "B2_rules.csv", index=False)
    (args.output_dir / "recommendation_rules_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
