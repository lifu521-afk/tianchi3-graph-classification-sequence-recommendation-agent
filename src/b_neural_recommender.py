from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

import b_recommendation_pipeline as rules


ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir

REC_DIR = task_dir("B_recommendation")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260722
MAX_LEN = 200


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PreparedRows:
    sequences: list[np.ndarray]
    users: np.ndarray
    targets: np.ndarray


def prepare_rows(
    frame: pd.DataFrame,
    user: pd.DataFrame,
    item_to_idx: dict[str, int],
    target_to_col: dict[str, int] | None,
) -> PreparedRows:
    user_cols = [col for col in user.columns if col != "uid"]
    user_lookup = user.set_index("uid")
    sequences: list[np.ndarray] = []
    user_values = np.zeros((len(frame), len(user_cols)), dtype=np.int64)
    targets = np.full(len(frame), -1, dtype=np.int64)
    for row_pos, row in enumerate(frame.itertuples(index=False)):
        sequence = [item_to_idx[iid] for iid in rules.read_items(row.item_seq_raw) if iid in item_to_idx]
        sequences.append(np.asarray(sequence[-MAX_LEN:], dtype=np.int64))
        user_values[row_pos] = user_lookup.loc[row.uid, user_cols].to_numpy(dtype=np.int64)
        if target_to_col is not None:
            targets[row_pos] = target_to_col[row.target_iid]
    return PreparedRows(sequences, user_values, targets)


class SequenceDataset(Dataset):
    def __init__(
        self,
        prepared: PreparedRows,
        sampled_lengths: np.ndarray | None = None,
        augment: bool = False,
        seed: int = SEED,
    ):
        self.prepared = prepared
        self.sampled_lengths = sampled_lengths
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.prepared.sequences)

    def __getitem__(self, index: int):
        sequence = self.prepared.sequences[index]
        if self.augment and self.sampled_lengths is not None:
            keep = int(self.rng.choice(self.sampled_lengths))
            sequence = sequence[-keep:] if keep > 0 else sequence[:0]
        length = min(len(sequence), MAX_LEN)
        padded = np.zeros(MAX_LEN, dtype=np.int64)
        if length:
            padded[:length] = sequence[-length:]
        return (
            torch.from_numpy(padded),
            torch.tensor(length, dtype=torch.long),
            torch.from_numpy(self.prepared.users[index]),
            torch.tensor(self.prepared.targets[index], dtype=torch.long),
        )


class HybridSequenceModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_metadata: np.ndarray,
        user_cardinalities: list[int],
        class_item_indices: list[int],
        class_bias: np.ndarray,
        embedding_dim: int = 96,
        hidden_dim: int = 160,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        metadata_cardinalities = item_metadata.max(axis=0).astype(int).tolist()
        self.metadata_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, 24, padding_idx=0) for cardinality in metadata_cardinalities]
        )
        self.metadata_projection = nn.Linear(24 * item_metadata.shape[1], embedding_dim)
        padded_metadata = np.vstack([np.zeros((1, item_metadata.shape[1]), dtype=np.int64), item_metadata])
        self.register_buffer("item_metadata", torch.tensor(padded_metadata, dtype=torch.long))
        self.register_buffer("class_item_indices", torch.tensor(class_item_indices, dtype=torch.long))

        item_to_class = torch.full((num_items + 1,), -1, dtype=torch.long)
        for class_col, item_index in enumerate(class_item_indices):
            item_to_class[item_index] = class_col
        self.register_buffer("item_to_class", item_to_class)

        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, 12) for cardinality in user_cardinalities]
        )
        self.length_embedding = nn.Embedding(MAX_LEN + 1, 20)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        user_dim = 12 * len(user_cardinalities)
        feature_dim = hidden_dim + embedding_dim * 4 + 20 + user_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(384, len(class_item_indices)),
        )
        self.repeat_count_weight = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.repeat_recent_weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor(class_bias, dtype=torch.float32))

    def item_representations(self, sequence: torch.Tensor) -> torch.Tensor:
        metadata = self.item_metadata[sequence]
        metadata_parts = [
            embedding(metadata[:, :, col]) for col, embedding in enumerate(self.metadata_embeddings)
        ]
        metadata_vector = self.metadata_projection(torch.cat(metadata_parts, dim=-1))
        return self.item_embedding(sequence) + metadata_vector

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        representation = self.item_representations(sequence)
        mask = (sequence != 0).float().unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        mean = (representation * mask).sum(dim=1) / denominator
        recent_weights = torch.linspace(0.1, 1.0, MAX_LEN, device=sequence.device)[None, :, None]
        recent_weights = recent_weights * mask
        recent = (representation * recent_weights).sum(dim=1) / recent_weights.sum(dim=1).clamp_min(1.0)
        maximum = representation.masked_fill(mask == 0, -1e4).max(dim=1).values
        maximum = torch.where(lengths[:, None] > 0, maximum, torch.zeros_like(maximum))
        last_position = (lengths - 1).clamp_min(0)
        last = representation[torch.arange(len(sequence), device=sequence.device), last_position]
        last = torch.where(lengths[:, None] > 0, last, torch.zeros_like(last))

        packed = pack_padded_sequence(
            representation,
            lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        gru_state = hidden[-1]
        gru_state = torch.where(lengths[:, None] > 0, gru_state, torch.zeros_like(gru_state))
        user_parts = [
            embedding(users[:, col].clamp_min(0))
            for col, embedding in enumerate(self.user_embeddings)
        ]
        features = torch.cat(
            [
                gru_state,
                mean,
                recent,
                maximum,
                last,
                self.length_embedding(lengths.clamp(max=MAX_LEN)),
                *user_parts,
            ],
            dim=1,
        )
        logits = self.network(features)

        class_cols = self.item_to_class[sequence]
        valid = class_cols >= 0
        repeat_bonus = torch.zeros_like(logits)
        repeat_bonus.scatter_add_(1, class_cols.clamp_min(0), valid.float())
        positions = torch.linspace(0.1, 1.0, MAX_LEN, device=sequence.device)[None, :].expand_as(sequence)
        recent_bonus = torch.zeros_like(logits)
        recent_bonus.scatter_add_(1, class_cols.clamp_min(0), positions * valid.float())
        logits = (
            logits
            + self.repeat_count_weight * torch.log1p(repeat_bonus)
            + self.repeat_recent_weight * recent_bonus
        )
        return logits


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for sequence, lengths, users, targets in loader:
        sequence = sequence.to(DEVICE)
        lengths = lengths.to(DEVICE)
        users = users.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(sequence, lengths, users)
        loss = F.cross_entropy(logits, targets, label_smoothing=0.015)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(targets)
        total_rows += len(targets)
    return total_loss / max(total_rows, 1)


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, lengths, users, _ in loader:
            logits = model(sequence.to(DEVICE), lengths.to(DEVICE), users.to(DEVICE))
            outputs.append(logits.cpu().numpy().astype(np.float32))
    return np.vstack(outputs)


def build_model_inputs(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    user: pd.DataFrame,
    item: pd.DataFrame,
    class_items: list[str],
) -> tuple[SequenceDataset, DataLoader, dict[str, object]]:
    item_ids = item["iid"].tolist()
    item_to_idx = {iid: idx + 1 for idx, iid in enumerate(item_ids)}
    target_to_col = {iid: idx for idx, iid in enumerate(class_items)}
    user_cols = [col for col in user.columns if col != "uid"]
    user_cardinalities = [int(user[col].max()) for col in user_cols]
    metadata_cols = ["i_cat_01", "i_cat_02", "i_cat_03", "i_bucket_01"]
    item_metadata = item[metadata_cols].to_numpy(dtype=np.int64)
    class_item_indices = [item_to_idx[iid] for iid in class_items]
    counts = train["target_iid"].value_counts().reindex(class_items).fillna(0).to_numpy(np.float32)
    class_bias = np.log((counts + 0.5) / (counts.sum() + 0.5 * len(counts)))

    prepared_train = prepare_rows(train, user, item_to_idx, target_to_col)
    prepared_frame = prepare_rows(frame, user, item_to_idx, target_to_col)
    sampled_lengths = frame["item_seq_raw"].map(lambda value: len(rules.read_items(value))).to_numpy()
    train_dataset = SequenceDataset(
        prepared_train,
        sampled_lengths=sampled_lengths,
        augment=True,
        seed=SEED,
    )
    frame_dataset = SequenceDataset(prepared_frame, augment=False)
    frame_loader = DataLoader(frame_dataset, batch_size=512, shuffle=False, num_workers=0)
    metadata = {
        "num_items": len(item_ids),
        "item_metadata": item_metadata,
        "user_cardinalities": user_cardinalities,
        "class_item_indices": class_item_indices,
        "class_bias": class_bias,
        "item_to_idx": item_to_idx,
        "target_to_col": target_to_col,
    }
    return train_dataset, frame_loader, metadata


def fit_one_model(
    train_dataset: SequenceDataset,
    eval_loader: DataLoader,
    metadata: dict[str, object],
    class_items: list[str],
    eval_targets: pd.Series | None,
    seed: int,
    epochs: int,
    evaluate_each_epoch: bool,
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    set_seed(seed)
    train_dataset.rng = np.random.default_rng(seed)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    model = HybridSequenceModel(
        num_items=int(metadata["num_items"]),
        item_metadata=metadata["item_metadata"],
        user_cardinalities=metadata["user_cardinalities"],
        class_item_indices=metadata["class_item_indices"],
        class_bias=metadata["class_bias"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=3e-4)
    best_score = -1.0
    best_logits: np.ndarray | None = None
    best_epoch = epochs
    logs: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer)
        if evaluate_each_epoch and (epoch >= 4 or epoch == epochs):
            logits = predict(model, eval_loader)
            score = rules.ndcg_rows(logits, class_items, eval_targets)[0]
            logs.append({"epoch": float(epoch), "loss": loss, "ndcg_at_10": score})
            print(f"neural seed {seed} epoch {epoch}: loss={loss:.5f} ndcg={score:.6f}", flush=True)
            if score > best_score:
                best_score = score
                best_logits = logits
                best_epoch = epoch
        elif epoch in {1, epochs}:
            print(f"neural seed {seed} epoch {epoch}: loss={loss:.5f}", flush=True)
    if best_logits is None:
        best_logits = predict(model, eval_loader)
    return best_logits, best_epoch, logs


def rule_score_matrix(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    user: pd.DataFrame,
    item: pd.DataFrame,
    class_items: list[str],
    weights: dict[str, dict[str, float]] | None = None,
    targets: pd.Series | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, object]]:
    components = rules.rule_components(train, frame, user, item, class_items)
    if weights is None:
        scores, weights, diagnostics = rules.tune_rule_blend(
            components, frame, targets, class_items
        )
        return scores, weights, diagnostics
    masks = rules.history_buckets(frame)
    scores = np.zeros_like(next(iter(components.values())))
    for bucket, mask in masks.items():
        scores[mask] = sum(weights[bucket][name] * components[name][mask] for name in weights[bucket])
    return scores, weights, {}


def tune_neural_rule_blend(
    neural: np.ndarray,
    rule_scores: np.ndarray,
    frame: pd.DataFrame,
    targets: pd.Series,
    class_items: list[str],
) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    neural = rules.row_standardize(neural)
    rule_scores = rules.row_standardize(rule_scores)
    masks = rules.history_buckets(frame)
    final = np.zeros_like(neural)
    weights: dict[str, float] = {}
    bucket_results: dict[str, object] = {}
    grid = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.4]
    for bucket, mask in masks.items():
        neural_score = rules.ndcg_rows(neural[mask], class_items, targets[mask])[0]
        rule_score = rules.ndcg_rows(rule_scores[mask], class_items, targets[mask])[0]
        best = (neural_score, 0.0)
        for weight in grid:
            score = rules.ndcg_rows(
                neural[mask] + weight * rule_scores[mask], class_items, targets[mask]
            )[0]
            if score > best[0]:
                best = (score, weight)
        weights[bucket] = best[1]
        final[mask] = neural[mask] + best[1] * rule_scores[mask]
        bucket_results[bucket] = {
            "rows": int(mask.sum()),
            "neural": neural_score,
            "rules": rule_score,
            "blend": best[0],
            "rule_weight": best[1],
        }
    diagnostics = {
        "buckets": bucket_results,
        "neural_total": rules.ndcg_rows(neural, class_items, targets)[0],
        "rules_total": rules.ndcg_rows(rule_scores, class_items, targets)[0],
        "blend_total": rules.ndcg_rows(final, class_items, targets)[0],
    }
    return final, weights, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "b_final_20260722")
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("device", DEVICE, flush=True)

    train = pd.read_csv(REC_DIR / "train.csv")
    test = pd.read_csv(REC_DIR / "test.csv")
    user = pd.read_csv(REC_DIR / "user.csv")
    item = pd.read_csv(REC_DIR / "item.csv")
    class_items = train["target_iid"].value_counts().index.tolist()
    fit_pos, valid_pos = rules.make_holdout(train["target_iid"])
    inner_train = train.iloc[fit_pos].reset_index(drop=True)
    inner_valid = train.iloc[valid_pos].reset_index(drop=True)
    visible_valid = rules.make_test_like(inner_valid, test)
    visible_valid_with_target = visible_valid.copy()
    visible_valid_with_target["target_iid"] = inner_valid["target_iid"].values

    print("preparing validation neural data...", flush=True)
    train_dataset, valid_loader, metadata = build_model_inputs(
        inner_train, visible_valid_with_target, user, item, class_items
    )
    valid_logits, best_epoch, epoch_logs = fit_one_model(
        train_dataset,
        valid_loader,
        metadata,
        class_items,
        inner_valid["target_iid"],
        seed=SEED,
        epochs=args.epochs,
        evaluate_each_epoch=True,
    )
    print("building validation rule scores...", flush=True)
    valid_rule_scores, rule_weights, rule_diagnostics = rule_score_matrix(
        inner_train,
        visible_valid,
        user,
        item,
        class_items,
        targets=inner_valid["target_iid"],
    )
    _, blend_weights, blend_diagnostics = tune_neural_rule_blend(
        valid_logits,
        valid_rule_scores,
        visible_valid,
        inner_valid["target_iid"],
        class_items,
    )
    print(json.dumps(blend_diagnostics, ensure_ascii=False), flush=True)

    test_with_target = test.copy()
    test_with_target["target_iid"] = class_items[0]
    final_dataset, test_loader, final_metadata = build_model_inputs(
        train, test_with_target, user, item, class_items
    )
    final_logits_list: list[np.ndarray] = []
    final_logs: list[dict[str, object]] = []
    for seed in [SEED, SEED + 1]:
        logits, _, logs = fit_one_model(
            final_dataset,
            test_loader,
            final_metadata,
            class_items,
            eval_targets=None,
            seed=seed,
            epochs=best_epoch,
            evaluate_each_epoch=False,
        )
        final_logits_list.append(logits)
        final_logs.append({"seed": seed, "epochs": best_epoch, "training": logs})
    final_logits = rules.row_standardize(np.mean(final_logits_list, axis=0))

    print("building final rule scores...", flush=True)
    final_rule_scores, _, _ = rule_score_matrix(
        train, test, user, item, class_items, weights=rule_weights
    )
    final_rule_scores = rules.row_standardize(final_rule_scores)
    final_scores = np.zeros_like(final_logits)
    for bucket, mask in rules.history_buckets(test).items():
        final_scores[mask] = final_logits[mask] + blend_weights[bucket] * final_rule_scores[mask]
    submission = rules.topk_submission(final_scores, class_items, test["uid"])
    submission.to_csv(args.output_dir / "B2.csv", index=False)

    log = {
        "method": "two_seed_hybrid_gru_metadata_repeat_plus_bucket_rule_blend",
        "device": str(DEVICE),
        "best_validation_epoch": best_epoch,
        "validation_epochs": epoch_logs,
        "rule_diagnostics": rule_diagnostics,
        "rule_weights": rule_weights,
        "blend_weights": blend_weights,
        "blend_diagnostics": blend_diagnostics,
        "final_models": final_logs,
    }
    (args.output_dir / "recommendation_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
