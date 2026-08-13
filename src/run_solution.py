from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir

CLS_DIR = task_dir("A_classification")
REC_DIR = task_dir("A_recommendation")
CLASSIFICATION_SMOOTH_MODEL_NAMES = {"logreg_c_0.3", "logreg_c_0.7"}


def read_items(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_counts(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, str) or not value:
        return counts
    for part in value.split(","):
        if ":" not in part:
            continue
        iid, count = part.split(":", 1)
        iid = iid.strip()
        try:
            counts[iid] += int(count)
        except ValueError:
            continue
    return counts


def format_counts(items: list[str]) -> str | float:
    if not items:
        return np.nan
    counts = Counter(items)
    return ",".join(f"{iid}:{count}" for iid, count in counts.items())


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=np.float32)
    np.divide(1.0, row_sum, out=inv, where=row_sum != 0)
    return sparse.diags(inv).dot(matrix).tocsr()


def load_classification() -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(CLS_DIR / "A1.npz")
    adj = sparse.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
        dtype=np.float32,
    )
    features = sparse.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
        dtype=np.float32,
    )
    labels = data["labels"].astype(int)
    return adj, features, labels, data["train_idx"].astype(int), data["test_idx"].astype(int)


def build_classification_features(
    adj: sparse.csr_matrix,
    features: sparse.csr_matrix,
    labels: np.ndarray,
    label_seed_idx: np.ndarray,
    num_classes: int,
) -> sparse.csr_matrix:
    features = normalize(features, norm="l2", axis=1, copy=True).tocsr().astype(np.float32)
    out_adj = row_normalize(adj)
    in_adj = row_normalize(adj.T.tocsr())
    sym_adj = row_normalize(((adj + adj.T) > 0).astype(np.float32).tocsr())

    y_seed = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    y_seed[label_seed_idx, labels[label_seed_idx]] = 1.0
    y_seed_sp = sparse.csr_matrix(y_seed)

    parts = [
        features,
        out_adj.dot(features),
        in_adj.dot(features),
        sym_adj.dot(features),
        out_adj.dot(out_adj).dot(features),
        sym_adj.dot(sym_adj).dot(features),
        out_adj.dot(y_seed_sp),
        in_adj.dot(y_seed_sp),
        sym_adj.dot(y_seed_sp),
        sym_adj.dot(sym_adj).dot(y_seed_sp),
    ]
    return sparse.hstack(parts, format="csr", dtype=np.float32)


def classification_models() -> dict[str, object]:
    return {
        "ridge_alpha_0.3": RidgeClassifier(alpha=0.3, class_weight=None),
        "ridge_alpha_1.0": RidgeClassifier(alpha=1.0, class_weight=None),
        "ridge_alpha_3.0": RidgeClassifier(alpha=3.0, class_weight=None),
        "linearsvc_c_0.3": LinearSVC(C=0.3, class_weight=None, dual="auto", max_iter=5000, random_state=20260713),
        "linearsvc_c_1.0": LinearSVC(C=1.0, class_weight=None, dual="auto", max_iter=5000, random_state=20260713),
        "logreg_c_0.3": OneVsRestClassifier(LogisticRegression(C=0.3, solver="liblinear", max_iter=1000, random_state=20260713)),
        "logreg_c_0.7": OneVsRestClassifier(LogisticRegression(C=0.7, solver="liblinear", max_iter=1000, random_state=20260713)),
        "logreg_c_1.0": OneVsRestClassifier(LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=20260713)),
        "logreg_c_2.0": OneVsRestClassifier(LogisticRegression(C=2.0, solver="liblinear", max_iter=1000, random_state=20260713)),
    }


def classification_probabilities(model: object, matrix: sparse.csr_matrix, num_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(matrix)
        return np.asarray(proba, dtype=np.float32)
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(matrix), dtype=np.float32)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        decision -= decision.max(axis=1, keepdims=True)
        exp_decision = np.exp(decision)
        return exp_decision / np.maximum(exp_decision.sum(axis=1, keepdims=True), 1e-12)
    pred = model.predict(matrix).astype(int)
    return np.eye(num_classes, dtype=np.float32)[pred]


def smooth_classification_probabilities(
    probabilities: np.ndarray,
    adj: sparse.csr_matrix,
    labels: np.ndarray,
    seed_idx: np.ndarray,
    alpha: float,
    clamp: float,
    steps: int,
) -> np.ndarray:
    smooth_adj = row_normalize((adj + adj.T + sparse.eye(adj.shape[0], dtype=np.float32)).astype(np.float32).tocsr())
    num_classes = probabilities.shape[1]
    seeds = np.zeros_like(probabilities, dtype=np.float32)
    seeds[seed_idx, labels[seed_idx]] = 1.0
    base = probabilities.astype(np.float32, copy=True)
    current = base.copy()
    for _ in range(steps):
        current = (1.0 - alpha) * base + alpha * smooth_adj.dot(current)
        if clamp > 0:
            current[seed_idx] = (1.0 - clamp) * current[seed_idx] + clamp * seeds[seed_idx]
    current = np.maximum(current, 0.0)
    current_sum = current.sum(axis=1, keepdims=True)
    return current / np.maximum(current_sum, 1e-12)


def train_classification(validate: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    adj, features, labels, train_idx, test_idx = load_classification()
    num_classes = int(labels[train_idx].max()) + 1
    log: dict[str, object] = {
        "task": "classification",
        "num_nodes": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "experiments": [],
    }

    best_name = "logreg03_07_smooth_a0.65_c0.75_s6"
    best_score = -1.0
    best_smoothing: dict[str, object] | None = {
        "name": "logreg03_07_smooth_a0.65_c0.75_s6",
        "model_names": ["logreg_c_0.3", "logreg_c_0.7"],
        "alpha": 0.65,
        "clamp": 0.75,
        "steps": 6,
    }
    if validate:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=20260713)
        inner_train_pos, inner_val_pos = next(splitter.split(train_idx, labels[train_idx]))
        inner_train_idx = train_idx[inner_train_pos]
        inner_val_idx = train_idx[inner_val_pos]
        X_inner = build_classification_features(adj, features, labels, inner_train_idx, num_classes)
        y_val = labels[inner_val_idx]

        fitted_probabilities: dict[str, np.ndarray] = {}
        for name, model in classification_models().items():
            model.fit(X_inner[inner_train_idx], labels[inner_train_idx])
            probabilities = classification_probabilities(model, X_inner, num_classes)
            pred = probabilities[inner_val_idx].argmax(axis=1)
            score = float(accuracy_score(y_val, pred))
            log["experiments"].append({"name": name, "accuracy": score})
            if score > best_score:
                best_name = name
                best_score = score
                best_smoothing = None
            fitted_probabilities[name] = probabilities

        smooth_candidate_specs = [
            {
                "name": "ridge3_smooth_a0.7_c0.2_s2",
                "model_names": ["ridge_alpha_3.0"],
                "alpha": 0.7,
                "clamp": 0.2,
                "steps": 2,
            },
            {
                "name": "logreg03_07_smooth_a0.65_c0.75_s6",
                "model_names": ["logreg_c_0.3", "logreg_c_0.7"],
                "alpha": 0.65,
                "clamp": 0.75,
                "steps": 6,
            },
        ]
        smoothing_candidates = [
            {"name": "avg_smooth_mix_a0.5_c0.0_s1", "alpha": 0.5, "clamp": 0.0, "steps": 1},
            {"name": "avg_smooth_mix_a0.7_c0.0_s1", "alpha": 0.7, "clamp": 0.0, "steps": 1},
            {"name": "avg_smooth_mix_a0.7_c0.2_s1", "alpha": 0.7, "clamp": 0.2, "steps": 1},
            {"name": "avg_smooth_mix_a0.7_c0.2_s2", "alpha": 0.7, "clamp": 0.2, "steps": 2},
            {"name": "avg_smooth_mix_a0.7_c0.5_s1", "alpha": 0.7, "clamp": 0.5, "steps": 1},
        ]
        for params in smoothing_candidates:
            params = {"model_names": ["ridge_alpha_3.0"], **params}
            smooth_candidate_specs.append(params)
        for params in smooth_candidate_specs:
            model_names = list(params["model_names"])
            averaged_probabilities = np.mean([fitted_probabilities[name] for name in model_names], axis=0)
            smoothed = smooth_classification_probabilities(
                averaged_probabilities,
                adj,
                labels,
                inner_train_idx,
                alpha=float(params["alpha"]),
                clamp=float(params["clamp"]),
                steps=int(params["steps"]),
            )
            score = float(accuracy_score(y_val, smoothed[inner_val_idx].argmax(axis=1)))
            log["experiments"].append({"name": params["name"], "accuracy": score, "smoothing": params})
            if score > best_score:
                best_name = str(params["name"])
                best_score = score
                best_smoothing = params

    final_features = build_classification_features(adj, features, labels, train_idx, num_classes)
    if best_smoothing is not None:
        final_probabilities = []
        selected_model_names = set(best_smoothing.get("model_names", CLASSIFICATION_SMOOTH_MODEL_NAMES))
        for name, model in classification_models().items():
            if name not in selected_model_names:
                continue
            model.fit(final_features[train_idx], labels[train_idx])
            final_probabilities.append(classification_probabilities(model, final_features, num_classes))
        final_average = np.mean(final_probabilities, axis=0)
        final_scores = smooth_classification_probabilities(
            final_average,
            adj,
            labels,
            train_idx,
            alpha=float(best_smoothing["alpha"]),
            clamp=float(best_smoothing["clamp"]),
            steps=int(best_smoothing["steps"]),
        )
        test_pred = final_scores[test_idx].argmax(axis=1).astype(int)
    else:
        final_model = classification_models()[best_name]
        final_model.fit(final_features[train_idx], labels[train_idx])
        test_pred = final_model.predict(final_features[test_idx]).astype(int)

    log["selected_model"] = best_name
    if best_smoothing is not None:
        log["selected_smoothing"] = best_smoothing
    if validate:
        log["selected_validation_accuracy"] = best_score
    submission = pd.DataFrame({"test_idx": test_idx, "label": test_pred})
    return submission, log


@dataclass(frozen=True)
class RecConfig:
    name: str
    global_w: float
    last_w: float
    seq_w: float
    count_w: float
    user_w: float
    repeat_w: float


class Recommender:
    def __init__(self, item_ids: list[str], user_feature_cols: list[str], config: RecConfig):
        self.item_ids = item_ids
        self.item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.user_feature_cols = user_feature_cols
        self.config = config
        size = len(item_ids)
        self.global_scores = np.zeros(size, dtype=np.float32)
        self.last_to_target = np.zeros((size, size), dtype=np.float32)
        self.seq_to_target = np.zeros((size, size), dtype=np.float32)
        self.user_group_scores: dict[tuple[str, object], np.ndarray] = {}

    def fit(self, train: pd.DataFrame, user: pd.DataFrame) -> None:
        user_lookup = user.set_index("uid")
        group_counts: defaultdict[tuple[str, object], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.item_ids), dtype=np.float32)
        )
        for row in train.itertuples(index=False):
            target_idx = self.item_to_idx.get(row.target_iid)
            if target_idx is None:
                continue
            self.global_scores[target_idx] += 1.0
            raw_items = read_items(row.item_seq_raw)
            dedup_items = read_items(row.item_seq_dedup)
            count_items = parse_counts(row.item_seq_counts)

            if raw_items:
                last_idx = self.item_to_idx.get(raw_items[-1])
                if last_idx is not None:
                    self.last_to_target[last_idx, target_idx] += 1.0

            recent = raw_items[-30:] if raw_items else dedup_items[-30:]
            for offset, iid in enumerate(reversed(recent)):
                source_idx = self.item_to_idx.get(iid)
                if source_idx is not None:
                    self.seq_to_target[source_idx, target_idx] += 1.0 / math.sqrt(offset + 1.0)
            for iid, count in count_items.items():
                source_idx = self.item_to_idx.get(iid)
                if source_idx is not None:
                    self.seq_to_target[source_idx, target_idx] += min(count, 10) * 0.08

            if row.uid in user_lookup.index:
                user_row = user_lookup.loc[row.uid]
                for col in self.user_feature_cols:
                    group_counts[(col, user_row[col])][target_idx] += 1.0

        self.global_scores = np.log1p(self.global_scores)
        self.global_scores /= self.global_scores.max() or 1.0
        self.last_to_target = np.log1p(self.last_to_target)
        self.seq_to_target = np.log1p(self.seq_to_target)
        last_max = self.last_to_target.max(axis=1, keepdims=True)
        seq_max = self.seq_to_target.max(axis=1, keepdims=True)
        self.last_to_target = np.divide(self.last_to_target, last_max, out=np.zeros_like(self.last_to_target), where=last_max != 0)
        self.seq_to_target = np.divide(self.seq_to_target, seq_max, out=np.zeros_like(self.seq_to_target), where=seq_max != 0)
        for key, values in group_counts.items():
            values = np.log1p(values)
            max_value = float(values.max())
            if max_value > 0:
                values /= max_value
            self.user_group_scores[key] = values

    def score_row(self, row: object, user_lookup: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        scores = cfg.global_w * self.global_scores.copy()
        raw_items = read_items(getattr(row, "item_seq_raw", None))
        count_items = parse_counts(getattr(row, "item_seq_counts", None))

        if raw_items:
            last_idx = self.item_to_idx.get(raw_items[-1])
            if last_idx is not None:
                scores += cfg.last_w * self.last_to_target[last_idx]
            recent = raw_items[-30:]
            seen_sources: set[int] = set()
            for offset, iid in enumerate(reversed(recent)):
                source_idx = self.item_to_idx.get(iid)
                if source_idx is None:
                    continue
                scores[source_idx] += cfg.repeat_w * (1.0 / math.sqrt(offset + 1.0))
                scores += cfg.seq_w * (1.0 / math.sqrt(offset + 1.0)) * self.seq_to_target[source_idx]
                seen_sources.add(source_idx)
            for iid, count in count_items.items():
                source_idx = self.item_to_idx.get(iid)
                if source_idx is None:
                    continue
                scores[source_idx] += cfg.repeat_w * 0.08 * math.log1p(count)
                if source_idx not in seen_sources:
                    scores += cfg.count_w * min(count, 10) * self.seq_to_target[source_idx]

        uid = getattr(row, "uid")
        if uid in user_lookup.index:
            user_row = user_lookup.loc[uid]
            for col in self.user_feature_cols:
                group = self.user_group_scores.get((col, user_row[col]))
                if group is not None:
                    scores += cfg.user_w * group
        return scores

    def predict_topk(self, test: pd.DataFrame, user: pd.DataFrame, k: int = 10) -> list[list[str]]:
        user_lookup = user.set_index("uid")
        predictions: list[list[str]] = []
        for row in test.itertuples(index=False):
            scores = self.score_row(row, user_lookup)
            top_idx = np.argpartition(-scores, kth=k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            predictions.append([self.item_ids[idx] for idx in top_idx])
        return predictions


def ndcg_at_10(predictions: list[list[str]], targets: pd.Series) -> float:
    scores = []
    for pred, target in zip(predictions, targets):
        try:
            rank = pred.index(target)
        except ValueError:
            scores.append(0.0)
        else:
            scores.append(1.0 / math.log2(rank + 2.0))
    return float(np.mean(scores))


def predictions_from_score_matrix(scores: np.ndarray, item_ids: list[str], k: int = 10) -> list[list[str]]:
    item_array = np.asarray(item_ids)
    predictions: list[list[str]] = []
    for row in scores:
        top_idx = np.argpartition(-row, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        predictions.append(item_array[top_idx].tolist())
    return predictions


def recommendation_score_matrix(recommender: Recommender, frame: pd.DataFrame, user: pd.DataFrame) -> np.ndarray:
    user_lookup = user.set_index("uid")
    scores = [recommender.score_row(row, user_lookup) for row in frame.itertuples(index=False)]
    matrix = np.vstack(scores).astype(np.float32)
    return matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-6)


def recommendation_tokens(row: object, user_row: pd.Series | None) -> list[str]:
    tokens: list[str] = []
    raw_items = read_items(getattr(row, "item_seq_raw", None))
    counts = parse_counts(getattr(row, "item_seq_counts", None))
    length = len(raw_items)
    length_bucket = "empty" if length == 0 else "one" if length == 1 else "short" if length <= 3 else "mid" if length <= 30 else "long"
    tokens.append(f"len={min(length, 30)}")
    tokens.append(f"bucket={length_bucket}")
    for offset, iid in enumerate(reversed(raw_items[-20:])):
        tokens.append(f"recent:{min(offset, 9)}:{iid}")
        tokens.append(f"seen:{iid}")
    if raw_items:
        tokens.append(f"last:{raw_items[-1]}")
        tokens.append(f"first:{raw_items[0]}")
    for iid, count in counts.items():
        tokens.append(f"cnt:{iid}:{min(count, 5)}")
    if user_row is not None:
        user_tokens = []
        for col, value in user_row.items():
            token = f"{col}={value}"
            tokens.append(token)
            user_tokens.append(token)
        for left, right in zip(user_tokens, user_tokens[1:]):
            tokens.append(f"pair:{left}|{right}")
    return tokens


def recommendation_hashed_features(frame: pd.DataFrame, user: pd.DataFrame, n_features: int = 1 << 18) -> sparse.csr_matrix:
    user_lookup = user.set_index("uid")
    documents = []
    for row in frame.itertuples(index=False):
        user_row = user_lookup.loc[row.uid] if row.uid in user_lookup.index else None
        documents.append(recommendation_tokens(row, user_row))
    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    return normalize(hasher.transform(documents).astype(np.float32), norm="l2", axis=1, copy=False).tocsr()


def recommendation_lr_score_matrix(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    user: pd.DataFrame,
    item_ids: list[str],
    c_value: float = 0.7,
) -> np.ndarray:
    encoder = LabelEncoder()
    y = encoder.fit_transform(train["target_iid"])
    X_train = recommendation_hashed_features(train.drop(columns=["target_iid"]), user)
    X_frame = recommendation_hashed_features(frame, user)
    model = LogisticRegression(C=c_value, solver="saga", max_iter=250, random_state=20260713)
    model.fit(X_train, y)
    probabilities = model.predict_proba(X_frame).astype(np.float32)
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    matrix = np.zeros((len(frame), len(item_ids)), dtype=np.float32)
    for class_idx, iid in enumerate(encoder.classes_):
        target_idx = item_to_idx.get(iid)
        if target_idx is not None:
            matrix[:, target_idx] = probabilities[:, class_idx]
    return matrix / np.maximum(matrix.max(axis=1, keepdims=True), 1e-6)


def recommendation_holdout_indices(targets: pd.Series, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260713)
    counts = targets.value_counts().to_dict()
    remaining = targets.value_counts().to_dict()
    val_mask = np.zeros(len(targets), dtype=bool)
    desired_val_size = int(round(len(targets) * test_size))
    for idx in rng.permutation(len(targets)):
        target = targets.iloc[idx]
        if counts[target] < 2 or remaining[target] <= 1:
            continue
        val_mask[idx] = True
        remaining[target] -= 1
        if int(val_mask.sum()) >= desired_val_size:
            break
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def make_test_like_recommendation_validation(validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260713)
    test_lengths = test["item_seq_raw"].map(lambda value: len(read_items(value))).to_numpy()
    sampled_lengths = rng.choice(test_lengths, size=len(validation), replace=True)
    visible = validation.drop(columns=["target_iid"]).copy()
    for row_pos, keep_len in enumerate(sampled_lengths):
        raw_items = read_items(validation.iloc[row_pos]["item_seq_raw"])
        kept = raw_items[-int(keep_len) :] if keep_len > 0 else []
        if kept:
            visible.iat[row_pos, visible.columns.get_loc("item_seq_raw")] = ",".join(kept)
            visible.iat[row_pos, visible.columns.get_loc("item_seq_dedup")] = ",".join(dict.fromkeys(kept))
            visible.iat[row_pos, visible.columns.get_loc("item_seq_counts")] = format_counts(kept)
        else:
            visible.iat[row_pos, visible.columns.get_loc("item_seq_raw")] = np.nan
            visible.iat[row_pos, visible.columns.get_loc("item_seq_dedup")] = np.nan
            visible.iat[row_pos, visible.columns.get_loc("item_seq_counts")] = np.nan
    return visible


def train_recommendation(validate: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    train = pd.read_csv(REC_DIR / "train.csv")
    test = pd.read_csv(REC_DIR / "test.csv")
    user = pd.read_csv(REC_DIR / "user.csv")
    item = pd.read_csv(REC_DIR / "item.csv")
    item_ids = item["iid"].tolist()
    user_feature_cols = [col for col in user.columns if col != "uid"]
    configs = [
        RecConfig("global_only", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        RecConfig("last_seq_global", 0.45, 1.25, 0.45, 0.10, 0.03, 0.0),
        RecConfig("seq_heavy", 0.35, 1.00, 0.85, 0.12, 0.04, 0.0),
        RecConfig("user_smooth", 0.45, 1.00, 0.55, 0.10, 0.10, 0.0),
        RecConfig("history_heavy", 0.25, 1.50, 0.90, 0.15, 0.04, 0.0),
        RecConfig("short_history_balanced", 0.55, 0.95, 0.35, 0.06, 0.06, 0.15),
        RecConfig("short_history_repeat", 0.45, 0.90, 0.30, 0.05, 0.05, 0.45),
        RecConfig("short_history_global", 0.70, 0.65, 0.18, 0.03, 0.08, 0.18),
        RecConfig("history_repeat_low", 0.25, 1.50, 0.90, 0.15, 0.04, 0.35),
        RecConfig("history_repeat_mid", 0.22, 1.35, 0.75, 0.12, 0.04, 0.75),
        RecConfig("repeat_heavy", 0.20, 1.20, 0.60, 0.10, 0.03, 1.10),
    ]
    log: dict[str, object] = {
        "task": "recommendation",
        "num_train": int(len(train)),
        "num_test": int(len(test)),
        "num_items": int(len(item_ids)),
        "experiments": [],
    }
    best_config = next(config for config in configs if config.name == "history_repeat_mid")
    best_score = -1.0
    best_blend_weight = 0.0
    if validate:
        train_pos, val_pos = recommendation_holdout_indices(train["target_iid"], test_size=0.2)
        inner_train = train.iloc[train_pos].reset_index(drop=True)
        inner_val = train.iloc[val_pos].reset_index(drop=True)
        visible_val = make_test_like_recommendation_validation(inner_val, test)
        validation_scores: dict[str, float] = {}
        for config in configs:
            recommender = Recommender(item_ids, user_feature_cols, config)
            recommender.fit(inner_train, user)
            pred = recommender.predict_topk(visible_val, user, k=10)
            score = ndcg_at_10(pred, inner_val["target_iid"])
            validation_scores[config.name] = score
            log["experiments"].append({"name": config.name, "ndcg_at_10": score, "config": config.__dict__})
            if score > best_score:
                best_config = config
                best_score = score
                best_blend_weight = 0.0

        mid_config = next(config for config in configs if config.name == "history_repeat_mid")
        mid_score = validation_scores.get(mid_config.name)
        if mid_score is not None and mid_score >= best_score - 0.001:
            best_config = mid_config
            best_score = mid_score
            best_blend_weight = 0.0

        recommender = Recommender(item_ids, user_feature_cols, best_config)
        recommender.fit(inner_train, user)
        rule_scores = recommendation_score_matrix(recommender, visible_val, user)
        lr_scores = recommendation_lr_score_matrix(inner_train, visible_val, user, item_ids, c_value=0.7)
        for blend_weight in [0.1, 0.2, 0.3]:
            blended_scores = rule_scores + blend_weight * lr_scores
            pred = predictions_from_score_matrix(blended_scores, item_ids, k=10)
            score = ndcg_at_10(pred, inner_val["target_iid"])
            log["experiments"].append({
                "name": f"{best_config.name}_lr_blend_{blend_weight}",
                "ndcg_at_10": score,
                "config": best_config.__dict__,
                "lr_blend_weight": blend_weight,
            })
            if score > best_score:
                best_score = score
                best_blend_weight = blend_weight

    final_recommender = Recommender(item_ids, user_feature_cols, best_config)
    final_recommender.fit(train, user)
    if best_blend_weight > 0:
        rule_scores = recommendation_score_matrix(final_recommender, test, user)
        lr_scores = recommendation_lr_score_matrix(train, test, user, item_ids, c_value=0.7)
        predictions = predictions_from_score_matrix(rule_scores + best_blend_weight * lr_scores, item_ids, k=10)
    else:
        predictions = final_recommender.predict_topk(test, user, k=10)
    submission = pd.DataFrame(
        {
            "uid": test["uid"],
            "prediction": [",".join(items) for items in predictions],
        }
    )
    log["selected_config"] = best_config.__dict__
    log["selected_lr_blend_weight"] = best_blend_weight
    if validate:
        log["selected_validation_ndcg_at_10"] = best_score
    return submission, log


def validate_outputs(a1: pd.DataFrame, a2: pd.DataFrame) -> None:
    a1_template = pd.read_csv(CLS_DIR / "sample_submission.csv")
    a2_test = pd.read_csv(REC_DIR / "test.csv")
    item_ids = set(pd.read_csv(REC_DIR / "item.csv")["iid"])

    if list(a1.columns) != ["test_idx", "label"]:
        raise ValueError("A1.csv columns must be test_idx,label")
    if not a1["test_idx"].equals(a1_template["test_idx"]):
        raise ValueError("A1.csv test_idx order does not match template")
    if not a1["label"].between(0, 9).all():
        raise ValueError("A1.csv labels must be integers in [0, 9]")

    if list(a2.columns) != ["uid", "prediction"]:
        raise ValueError("A2.csv columns must be uid,prediction")
    if not a2["uid"].equals(a2_test["uid"]):
        raise ValueError("A2.csv uid order does not match test.csv")
    for uid, value in zip(a2["uid"], a2["prediction"]):
        items = read_items(value)
        if len(items) != 10:
            raise ValueError(f"{uid} has {len(items)} recommendations, expected 10")
        if len(set(items)) != 10:
            raise ValueError(f"{uid} has duplicate recommendations")
        illegal = [iid for iid in items if iid not in item_ids]
        if illegal:
            raise ValueError(f"{uid} has illegal item ids: {illegal[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A榜 classification and recommendation submissions.")
    parser.add_argument("--no-validate", action="store_true", help="Skip internal validation sweeps and use default configs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Directory for A1.csv, A2.csv, logs, and prediction.zip.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_validation = not args.no_validate
    print("training classification...", flush=True)
    a1, cls_log = train_classification(validate=run_validation)
    print("training recommendation...", flush=True)
    a2, rec_log = train_recommendation(validate=run_validation)
    print("validating outputs...", flush=True)
    validate_outputs(a1, a2)

    a1_path = args.output_dir / "A1.csv"
    a2_path = args.output_dir / "A2.csv"
    log_path = args.output_dir / "experiment_log.json"
    zip_path = args.output_dir / "prediction.zip"
    a1.to_csv(a1_path, index=False)
    a2.to_csv(a2_path, index=False)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump({"classification": cls_log, "recommendation": rec_log}, handle, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(a1_path, arcname="A1.csv")
        zf.write(a2_path, arcname="A2.csv")

    print(f"wrote {a1_path}")
    print(f"wrote {a2_path}")
    print(f"wrote {log_path}")
    print(f"wrote {zip_path}")
    if run_validation:
        print(f"classification best accuracy: {cls_log.get('selected_validation_accuracy'):.6f}")
        print(f"recommendation best ndcg@10: {rec_log.get('selected_validation_ndcg_at_10'):.6f}")


if __name__ == "__main__":
    main()
