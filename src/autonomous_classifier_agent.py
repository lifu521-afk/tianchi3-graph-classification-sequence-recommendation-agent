"""Rule-compliant serial experiment agent for NPZ node-classification tasks.

The agent derives all choices from the provided NPZ.  It never uses dataset
names, node IDs, external labels, or leaderboard differences as a signal.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize


SEED = 20260723


@dataclass
class TaskData:
    adjacency: sparse.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    classes: np.ndarray


@dataclass
class CandidateResult:
    name: str
    oof_accuracy: float
    testlike_accuracy: float
    testlike_min_accuracy: float
    runtime_seconds: float
    oof_scores: np.ndarray
    accepted: bool
    reason: str


def load_task(path: Path) -> TaskData:
    raw = np.load(path)
    adjacency = sparse.csr_matrix(
        (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
        shape=tuple(raw["adj_shape"]),
        dtype=np.float32,
    )
    features = sparse.csr_matrix(
        (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
        shape=tuple(raw["attr_shape"]),
        dtype=np.float32,
    ).toarray()
    labels = raw["labels"].astype(np.int64)
    train_idx = raw["train_idx"].astype(np.int64)
    test_idx = raw["test_idx"].astype(np.int64)
    classes = np.unique(labels[train_idx])
    if len(classes) < 2:
        raise ValueError("At least two visible classes are required.")
    return TaskData(adjacency, features, labels, train_idx, test_idx, classes)


def row_normalize(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree, dtype=np.float32)
    np.divide(1.0, degree, out=inverse, where=degree > 0)
    return (sparse.diags(inverse) @ matrix).tocsr()


def encode_labels(data: TaskData, indices: np.ndarray) -> np.ndarray:
    return np.searchsorted(data.classes, data.labels[indices]).astype(np.int64)


def class_weight(encoded: np.ndarray) -> dict[int, float]:
    counts = np.bincount(encoded)
    mean_count = float(counts.mean())
    return {label: float((mean_count / count) ** 0.2) for label, count in enumerate(counts)}


def graph_operators(adjacency: sparse.csr_matrix) -> dict[str, sparse.csr_matrix]:
    undirected = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    return {"out": row_normalize(adjacency), "in": row_normalize(adjacency.T), "sym": row_normalize(undirected)}


def graph_content(data: TaskData, operators: dict[str, sparse.csr_matrix]) -> np.ndarray:
    x = normalize(data.features, norm="l2", axis=1).astype(np.float32)
    out_1, in_1, sym_1 = operators["out"] @ x, operators["in"] @ x, operators["sym"] @ x
    out_2, in_2, sym_2 = operators["out"] @ out_1, operators["in"] @ in_1, operators["sym"] @ sym_1
    degrees = np.column_stack(
        (
            np.log1p(np.asarray(data.adjacency.sum(axis=1)).ravel()),
            np.log1p(np.asarray(data.adjacency.sum(axis=0)).ravel()),
        )
    ).astype(np.float32)
    return np.hstack((x, out_1, in_1, sym_1, out_2, in_2, sym_2, degrees)).astype(np.float32)


def label_augmented_scores(
    data: TaskData,
    base_features: np.ndarray,
    operators: dict[str, sparse.csr_matrix],
    fit_idx: np.ndarray,
) -> np.ndarray:
    encoded = encode_labels(data, fit_idx)
    seeds = np.zeros((len(data.labels), len(data.classes)), dtype=np.float32)
    seeds[fit_idx, encoded] = 1.0
    out_1, in_1, sym_1 = operators["out"] @ seeds, operators["in"] @ seeds, operators["sym"] @ seeds
    out_2, in_2, sym_2 = operators["out"] @ out_1, operators["in"] @ in_1, operators["sym"] @ sym_1
    sym_3 = operators["sym"] @ sym_2
    augmented = np.hstack((base_features, out_1, in_1, sym_1, out_2, in_2, sym_2, sym_3)).astype(np.float32)
    model = RidgeClassifier(alpha=2.0, class_weight=class_weight(encoded))
    model.fit(augmented[fit_idx], encoded)
    return np.asarray(model.decision_function(augmented), dtype=np.float32)


def ridge_scores(features: np.ndarray, data: TaskData, fit_idx: np.ndarray) -> np.ndarray:
    encoded = encode_labels(data, fit_idx)
    model = RidgeClassifier(alpha=3.0, class_weight=class_weight(encoded))
    model.fit(features[fit_idx], encoded)
    return np.asarray(model.decision_function(features), dtype=np.float32)


def profile_task(data: TaskData) -> tuple[dict[str, object], np.ndarray]:
    n_nodes = len(data.labels)
    out_degree = np.asarray(data.adjacency.sum(axis=1)).ravel()
    in_degree = np.asarray(data.adjacency.sum(axis=0)).ravel()
    feature_norm = np.linalg.norm(data.features, axis=1)
    summary = np.column_stack(
        (
            np.log1p(out_degree),
            np.log1p(in_degree),
            feature_norm,
            data.features.mean(axis=1),
            data.features.std(axis=1),
        )
    )
    # Small, label-free PCA makes the shift diagnostic more expressive.
    pca_components = min(8, data.features.shape[1], n_nodes - 1)
    if pca_components >= 2:
        pcs = PCA(n_components=pca_components, random_state=SEED, svd_solver="randomized").fit_transform(
            StandardScaler().fit_transform(data.features)
        )
        summary = np.hstack((summary, pcs))
    domain_labels = np.zeros(n_nodes, dtype=np.int64)
    domain_labels[data.test_idx] = 1
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    propensity = np.zeros(n_nodes, dtype=np.float32)
    for fit_pos, valid_pos in folds.split(summary, domain_labels):
        scaler = StandardScaler().fit(summary[fit_pos])
        domain_model = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        domain_model.fit(scaler.transform(summary[fit_pos]), domain_labels[fit_pos])
        propensity[valid_pos] = domain_model.predict_proba(scaler.transform(summary[valid_pos]))[:, 1]
    train_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[data.train_idx] = True
    reciprocal = data.adjacency.multiply(data.adjacency.T).nnz
    edges = data.adjacency.tocoo()
    labelled_edge = train_mask[edges.row] & train_mask[edges.col]
    same_label = data.labels[edges.row[labelled_edge]] == data.labels[edges.col[labelled_edge]]
    class_prior = np.bincount(encode_labels(data, data.train_idx), minlength=len(data.classes)) / len(data.train_idx)
    chance_homophily = float(np.square(class_prior).sum())
    profile = {
        "nodes": n_nodes,
        "feature_dim": int(data.features.shape[1]),
        "feature_density": float(np.count_nonzero(data.features) / data.features.size),
        "train_nodes": int(len(data.train_idx)),
        "test_nodes": int(len(data.test_idx)),
        "classes": data.classes.tolist(),
        "edges": int(data.adjacency.nnz),
        "reciprocal_edge_fraction": float(reciprocal / max(data.adjacency.nnz, 1)),
        "out_degree_median_train": float(np.median(out_degree[data.train_idx])),
        "out_degree_median_test": float(np.median(out_degree[data.test_idx])),
        "feature_norm_median_train": float(np.median(feature_norm[data.train_idx])),
        "feature_norm_median_test": float(np.median(feature_norm[data.test_idx])),
        "labelled_edge_homophily": float(same_label.mean()) if len(same_label) else None,
        "class_chance_homophily": chance_homophily,
        "homophily_lift": float(same_label.mean() - chance_homophily) if len(same_label) else None,
        "train_test_propensity_auc": float(roc_auc_score(domain_labels, propensity)),
        "train_propensity_median": float(np.median(propensity[data.train_idx])),
        "test_propensity_median": float(np.median(propensity[data.test_idx])),
    }
    return profile, propensity


def testlike_holdouts(
    data: TaskData, propensity: np.ndarray, n_repeats: int = 3
) -> list[np.ndarray]:
    """Build labelled holdouts matching the public test propensity distribution."""
    validation_fraction = len(data.test_idx) / len(data.labels)
    bins = min(8, max(3, int(round(math.sqrt(len(data.test_idx) / 100.0))) + 2))
    edges = np.quantile(propensity, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.array([-np.inf, np.inf], dtype=np.float32)
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    all_bins = np.digitize(propensity, edges[1:-1], right=True)
    test_hist = np.bincount(all_bins[data.test_idx], minlength=len(edges) - 1).astype(np.float64)
    holdouts: list[np.ndarray] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(SEED + repeat)
        chosen: list[np.ndarray] = []
        for label in data.classes:
            members = data.train_idx[data.labels[data.train_idx] == label]
            count = max(1, round(len(members) * validation_fraction))
            member_bins = all_bins[members]
            member_hist = np.bincount(member_bins, minlength=len(test_hist)).astype(np.float64)
            weights = (test_hist[member_bins] + 1.0) / (member_hist[member_bins] + 1.0)
            weights = weights / weights.sum()
            chosen.append(rng.choice(members, size=count, replace=False, p=weights))
        holdouts.append(np.sort(np.concatenate(chosen)))
    return holdouts


def summarize_holdouts(
    data: TaskData, holdouts: list[np.ndarray], propensity: np.ndarray
) -> dict[str, object]:
    first_holdout = holdouts[0]
    return {
        "testlike_holdout_fraction": float(len(first_holdout) / len(data.train_idx)),
        "testlike_holdout_nodes": int(len(first_holdout)),
        "testlike_holdout_repeats": len(holdouts),
        "testlike_class_counts": np.bincount(
            encode_labels(data, first_holdout), minlength=len(data.classes)
        ).tolist(),
        "testlike_holdout_propensity_median": float(np.median(propensity[first_holdout])),
        "test_propensity_median": float(np.median(propensity[data.test_idx])),
    }


def evaluate_testlike(
    score_builder: Callable[[np.ndarray], np.ndarray],
    data: TaskData,
    holdouts: list[np.ndarray],
) -> tuple[float, float, list[float]]:
    accuracies: list[float] = []
    for holdout_idx in holdouts:
        fit_idx = np.setdiff1d(data.train_idx, holdout_idx, assume_unique=True)
        scores = score_builder(fit_idx)
        accuracies.append(
            float(
                accuracy_score(
                    encode_labels(data, holdout_idx),
                    scores[holdout_idx].argmax(axis=1),
                )
            )
        )
    return float(np.mean(accuracies)), float(np.min(accuracies)), accuracies


def evaluate_candidate(
    score_builder: Callable[[np.ndarray], np.ndarray],
    data: TaskData,
    validation_idx: np.ndarray,
) -> tuple[float, np.ndarray]:
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_scores = np.zeros((len(data.labels), len(data.classes)), dtype=np.float32)
    for fit_pos, valid_pos in folds.split(validation_idx, encode_labels(data, validation_idx)):
        scores = score_builder(validation_idx[fit_pos])
        oof_scores[validation_idx[valid_pos]] = scores[validation_idx[valid_pos]]
    prediction = oof_scores[validation_idx].argmax(axis=1)
    return float(accuracy_score(encode_labels(data, validation_idx), prediction)), oof_scores


def make_submission(data: TaskData, scores: np.ndarray, template: Path) -> pd.DataFrame:
    prediction = data.classes[scores[data.test_idx].argmax(axis=1)]
    if template.exists():
        frame = pd.read_csv(template)[["test_idx"]].copy()
    else:
        frame = pd.DataFrame({"test_idx": data.test_idx})
    mapping = dict(zip(data.test_idx.tolist(), prediction.tolist()))
    frame["label"] = frame["test_idx"].map(mapping).astype(np.int64)
    return frame


def run_agent(
    data_path: Path,
    template_path: Path,
    output_dir: Path,
    budget_minutes: float,
    task_id: str,
) -> dict[str, object]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, object]] = []
    data = load_task(data_path)
    profile, propensity = profile_task(data)
    testlike_sets = testlike_holdouts(data, propensity)
    profile.update(summarize_holdouts(data, testlike_sets, propensity))
    trajectory.append({"round": 0, "action": "profile", "feedback": profile, "next_strategy": "run_feature_ridge_baseline"})

    operators = graph_operators(data.adjacency)
    standardized = StandardScaler().fit_transform(data.features).astype(np.float32)
    candidates: list[tuple[str, Callable[[np.ndarray], np.ndarray], str]] = [
        ("feature_ridge", lambda fit_idx: ridge_scores(standardized, data, fit_idx), "mandatory_feature_baseline")
    ]
    if data.features.shape[1] >= 32:
        components = min(128, data.features.shape[1] - 1, len(data.labels) - 1)
        low_rank = PCA(n_components=components, random_state=SEED, svd_solver="randomized").fit_transform(standardized).astype(np.float32)
        candidates.append(("low_rank_ridge", lambda fit_idx: ridge_scores(low_rank, data, fit_idx), "low_rank_challenger"))
    graph_enabled = profile["homophily_lift"] is not None and float(profile["homophily_lift"]) > 0.01
    if graph_enabled:
        content = graph_content(data, operators)
        candidates.append(
            ("directed_graph_ridge", lambda fit_idx: label_augmented_scores(data, content, operators, fit_idx), "graph_signal_positive")
        )
    else:
        trajectory.append(
            {
                "round": len(trajectory),
                "action": "skip_candidate",
                "configuration": "directed_graph_ridge",
                "feedback": {"homophily_lift": profile["homophily_lift"]},
                "next_strategy": "avoid_graph_model_without_visible_label_signal",
            }
        )

    results: list[CandidateResult] = []
    best: CandidateResult | None = None
    for name, builder, selection_reason in candidates:
        if time.monotonic() - started > budget_minutes * 60:
            trajectory.append({"round": len(trajectory), "action": "stop", "reason": "budget_reserve_reached"})
            break
        candidate_started = time.monotonic()
        oof_accuracy, oof_scores = evaluate_candidate(builder, data, data.train_idx)
        testlike_accuracy, testlike_min_accuracy, testlike_accuracies = evaluate_testlike(
            builder, data, testlike_sets
        )
        if best is None:
            accepted, reason = True, "baseline"
        else:
            accepted = (
                oof_accuracy >= best.oof_accuracy + 0.001
                and testlike_accuracy >= best.testlike_accuracy + 0.001
                and testlike_min_accuracy >= best.testlike_min_accuracy - 0.002
            )
            reason = "passes_oof_and_testlike_gate" if accepted else "rejected_by_stability_gate"
        result = CandidateResult(
            name=name,
            oof_accuracy=oof_accuracy,
            testlike_accuracy=testlike_accuracy,
            testlike_min_accuracy=testlike_min_accuracy,
            runtime_seconds=time.monotonic() - candidate_started,
            oof_scores=oof_scores,
            accepted=accepted,
            reason=reason,
        )
        results.append(result)
        trajectory.append(
            {
                "round": len(trajectory),
                "action": "evaluate_candidate",
                "configuration": name,
                "selection_reason": selection_reason,
                "feedback": {
                    "oof_accuracy": oof_accuracy,
                    "testlike_accuracy": testlike_accuracy,
                    "testlike_fold_accuracies": testlike_accuracies,
                    "testlike_min_accuracy": testlike_min_accuracy,
                    "runtime_seconds": result.runtime_seconds,
                },
                "next_strategy": reason,
            }
        )
        if accepted:
            best = result

    if best is None:
        raise RuntimeError("No candidate completed before the budget reserve.")
    selected_builder = next(builder for name, builder, _ in candidates if name == best.name)
    final_scores = selected_builder(data.train_idx)
    submission = make_submission(data, final_scores, template_path)
    submission_path = output_dir / f"{task_id}.csv"
    submission.to_csv(submission_path, index=False)
    report = {
        "data_path": str(data_path),
        "profile": profile,
        "selected_model": best.name,
        "candidate_results": [
            {
                "name": row.name,
                "oof_accuracy": row.oof_accuracy,
                "testlike_accuracy": row.testlike_accuracy,
                "testlike_min_accuracy": row.testlike_min_accuracy,
                "runtime_seconds": row.runtime_seconds,
                "accepted": row.accepted,
                "reason": row.reason,
            }
            for row in results
        ],
        "runtime_seconds": time.monotonic() - started,
        "submission": str(submission_path),
    }
    (output_dir / "classification_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_agent_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"trajectory_{task_id}.json").write_text(
        json.dumps({"events": trajectory}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=110.0)
    parser.add_argument("--task-id", default="B1")
    args = parser.parse_args()
    report = run_agent(
        args.data, args.template, args.output_dir, args.budget_minutes, args.task_id
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
