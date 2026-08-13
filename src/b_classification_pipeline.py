from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir

DATA_PATH = task_dir("B_classification") / "B1.npz"
TEMPLATE_PATH = task_dir("B_classification") / "sample_submission.csv"
SEED = 20260722


def row_normalize(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    inverse = np.zeros_like(totals)
    np.divide(1.0, totals, out=inverse, where=totals > 0)
    return (sparse.diags(inverse) @ matrix).tocsr()


def row_standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def load_data() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)
    adjacency = sparse.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
        dtype=np.float32,
    )
    features = sparse.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
        dtype=np.float32,
    ).toarray()
    labels = data["labels"].astype(np.int64)
    train_idx = data["train_idx"].astype(np.int64)
    test_idx = data["test_idx"].astype(np.int64)
    return adjacency, features.astype(np.float32), labels, train_idx, test_idx


def graph_operators(adjacency: sparse.csr_matrix) -> dict[str, sparse.csr_matrix]:
    symmetric_raw = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    return {
        "out": row_normalize(adjacency),
        "in": row_normalize(adjacency.T),
        "sym": row_normalize(symmetric_raw),
    }


def content_features(
    adjacency: sparse.csr_matrix,
    features: np.ndarray,
    operators: dict[str, sparse.csr_matrix],
) -> np.ndarray:
    features = normalize(features, norm="l2", axis=1).astype(np.float32)
    out_once = operators["out"] @ features
    in_once = operators["in"] @ features
    symmetric_once = operators["sym"] @ features
    out_twice = operators["out"] @ out_once
    in_twice = operators["in"] @ in_once
    symmetric_twice = operators["sym"] @ symmetric_once
    out_degree = np.log1p(np.asarray(adjacency.sum(axis=1)).ravel())
    in_degree = np.log1p(np.asarray(adjacency.sum(axis=0)).ravel())
    degrees = np.column_stack([out_degree, in_degree]).astype(np.float32)
    return np.hstack(
        [
            features,
            out_once,
            in_once,
            symmetric_once,
            out_twice,
            in_twice,
            symmetric_twice,
            degrees,
        ]
    ).astype(np.float32)


def class_weights(
    labels: np.ndarray,
    fit_idx: np.ndarray,
    power: float = 0.2,
) -> dict[int, float]:
    counts = np.bincount(labels[fit_idx])
    mean_count = float(counts.mean())
    return {
        int(label): float((mean_count / count) ** power)
        for label, count in enumerate(counts)
    }


def fold_scores(
    base_features: np.ndarray,
    operators: dict[str, sparse.csr_matrix],
    labels: np.ndarray,
    fit_idx: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    seeds = np.zeros((len(labels), num_classes), dtype=np.float32)
    seeds[fit_idx, labels[fit_idx]] = 1.0
    out_labels = operators["out"] @ seeds
    in_labels = operators["in"] @ seeds
    sym_labels = operators["sym"] @ seeds
    out_labels_2 = operators["out"] @ out_labels
    in_labels_2 = operators["in"] @ in_labels
    sym_labels_2 = operators["sym"] @ sym_labels
    sym_labels_3 = operators["sym"] @ sym_labels_2

    augmented = np.hstack(
        [
            base_features,
            out_labels,
            in_labels,
            sym_labels,
            out_labels_2,
            in_labels_2,
            sym_labels_2,
            sym_labels_3,
        ]
    ).astype(np.float32)
    model = RidgeClassifier(
        alpha=2.0,
        class_weight=class_weights(labels, fit_idx, power=0.2),
    )
    model.fit(augmented[fit_idx], labels[fit_idx])
    return np.asarray(model.decision_function(augmented), dtype=np.float32)


def train_and_predict(validate: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    adjacency, features, labels, train_idx, test_idx = load_data()
    operators = graph_operators(adjacency)
    base_features = content_features(adjacency, features, operators)
    num_classes = int(labels[train_idx].max()) + 1
    log: dict[str, object] = {
        "method": "directed_two_hop_content_and_label_features_weighted_ridge",
        "num_nodes": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "num_classes": num_classes,
        "ridge_alpha": 2.0,
        "class_weight_power": 0.2,
    }
    if validate:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros((len(labels), num_classes), dtype=np.float32)
        fold_accuracies: list[float] = []
        for fold, (fit_pos, valid_pos) in enumerate(splitter.split(train_idx, labels[train_idx]), start=1):
            fit_idx = train_idx[fit_pos]
            valid_idx = train_idx[valid_pos]
            ridge_scores = fold_scores(
                base_features, operators, labels, fit_idx, num_classes
            )
            oof[valid_idx] = ridge_scores[valid_idx]
            accuracy = float(
                accuracy_score(
                    labels[valid_idx],
                    ridge_scores[valid_idx].argmax(axis=1),
                )
            )
            fold_accuracies.append(accuracy)
            print(f"classification fold {fold}: {accuracy:.6f}", flush=True)
        log["fold_accuracies"] = fold_accuracies
        log["oof_accuracy"] = float(accuracy_score(labels[train_idx], oof[train_idx].argmax(axis=1)))

    final_scores = fold_scores(
        base_features, operators, labels, train_idx, num_classes
    )
    predictions = final_scores[test_idx].argmax(axis=1).astype(np.int64)

    template = pd.read_csv(TEMPLATE_PATH)
    prediction_map = dict(zip(test_idx.tolist(), predictions.tolist()))
    submission = template[["test_idx"]].copy()
    submission["label"] = submission["test_idx"].map(prediction_map).astype(np.int64)
    return submission, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "b_final_20260722")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission, log = train_and_predict(validate=not args.skip_validation)
    submission.to_csv(args.output_dir / "B1.csv", index=False)
    (args.output_dir / "classification_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
