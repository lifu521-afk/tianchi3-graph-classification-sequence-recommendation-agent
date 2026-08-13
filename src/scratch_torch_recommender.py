from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import run_solution as rs

ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50
SEED = 20260713


def ndcg_from_logits(logits: np.ndarray, class_items: list[str], targets: pd.Series) -> float:
    vals = []
    arr = np.asarray(class_items)
    for row, target in zip(logits, targets):
        idx = np.argpartition(-row, kth=9)[:10]
        idx = idx[np.argsort(-row[idx])]
        pred = arr[idx].tolist()
        vals.append(0.0 if target not in pred else 1.0 / math.log2(pred.index(target) + 2.0))
    return float(np.mean(vals))


class RecDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, user: pd.DataFrame, item_to_idx: dict[str, int], target_to_idx: dict[str, int] | None, test_lengths: np.ndarray | None = None, augment: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.user_lookup = user.set_index("uid")
        self.user_cols = [c for c in user.columns if c != "uid"]
        self.item_to_idx = item_to_idx
        self.target_to_idx = target_to_idx
        self.test_lengths = test_lengths
        self.augment = augment
        self.rng = np.random.default_rng(SEED)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        raw = rs.read_items(row.get("item_seq_raw"))
        if self.augment and self.test_lengths is not None:
            keep = int(self.rng.choice(self.test_lengths))
            raw = raw[-keep:] if keep > 0 else []
        ids = [self.item_to_idx.get(iid, 0) for iid in raw[-MAX_LEN:]]
        length = len(ids)
        pad = [0] * (MAX_LEN - length) + ids
        user_vals = self.user_lookup.loc[row["uid"], self.user_cols].to_numpy(dtype=np.int64)
        if self.target_to_idx is None:
            y = -1
        else:
            y = self.target_to_idx[row["target_iid"]]
        return torch.tensor(pad, dtype=torch.long), torch.tensor(min(length, MAX_LEN), dtype=torch.long), torch.tensor(user_vals, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SeqModel(nn.Module):
    def __init__(self, num_items: int, user_cardinalities: list[int], num_classes: int, class_bias: np.ndarray | None = None, dim: int = 96):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.len_emb = nn.Embedding(MAX_LEN + 1, 16)
        self.user_embs = nn.ModuleList([nn.Embedding(card + 1, 12) for card in user_cardinalities])
        user_dim = 12 * len(user_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + 16 + user_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )
        if class_bias is not None:
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.tensor(class_bias, dtype=torch.float32))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(seq)
        mask = (seq != 0).float().unsqueeze(-1)
        mean = (emb * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        last_pos = torch.clamp(lengths - 1, min=0)
        batch_idx = torch.arange(seq.shape[0], device=seq.device)
        last_tokens = seq[batch_idx, MAX_LEN - 1]
        last = self.item_emb(last_tokens)
        # Recent weighted mean emphasizes last interactions but still works for empty histories.
        weights = torch.linspace(0.2, 1.0, MAX_LEN, device=seq.device).view(1, MAX_LEN, 1) * mask
        recent = (emb * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        user_parts = [emb_layer(users[:, i].clamp_min(0)) for i, emb_layer in enumerate(self.user_embs)]
        features = torch.cat([mean, last, recent, self.len_emb(lengths.clamp(max=MAX_LEN)), *user_parts], dim=1)
        return self.net(features)


def class_weights_from_counts(counts: np.ndarray, power: float = 0.25, max_weight: float = 3.0) -> torch.Tensor:
    counts = counts.astype(np.float32, copy=False)
    smoothed = counts + 0.5
    weights = (float(smoothed.mean()) / smoothed) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-6)
    weights = np.clip(weights, 1.0 / float(max_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


class TiedAttentionSeqModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        user_cardinalities: list[int],
        class_item_indices: list[int],
        class_bias: np.ndarray | None = None,
        dim: int = 128,
        hidden: int = 320,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.len_emb = nn.Embedding(MAX_LEN + 1, 16)
        self.user_embs = nn.ModuleList([nn.Embedding(card + 1, 12) for card in user_cardinalities])
        user_dim = 12 * len(user_cardinalities)
        self.query = nn.Sequential(
            nn.Linear(dim + 16 + user_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.context = nn.Sequential(
            nn.Linear(dim * 4 + 16 + user_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.register_buffer("class_item_indices", torch.tensor(class_item_indices, dtype=torch.long))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        self.output_bias = nn.Parameter(torch.zeros(len(class_item_indices), dtype=torch.float32))
        if class_bias is not None:
            with torch.no_grad():
                self.output_bias.copy_(torch.tensor(class_bias, dtype=torch.float32))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(seq)
        mask = (seq != 0).float().unsqueeze(-1)
        mean = (emb * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        last = self.item_emb(seq[:, -1])
        weights = torch.linspace(0.2, 1.0, MAX_LEN, device=seq.device).view(1, MAX_LEN, 1) * mask
        recent = (emb * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        user_parts = [emb_layer(users[:, i].clamp_min(0)) for i, emb_layer in enumerate(self.user_embs)]
        len_part = self.len_emb(lengths.clamp(max=MAX_LEN))
        query = self.query(torch.cat([last, len_part, *user_parts], dim=1)).unsqueeze(-1)
        attn_logits = torch.bmm(emb, query).squeeze(-1) / math.sqrt(emb.shape[-1])
        attn_logits = attn_logits.masked_fill(seq == 0, -1e4)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1) * mask
        attn = attn / torch.clamp(attn.sum(dim=1, keepdim=True), min=1e-6)
        pooled = (emb * attn).sum(dim=1)
        features = torch.cat([pooled, mean, recent, last, len_part, *user_parts], dim=1)
        context = F.normalize(self.context(features), dim=1)
        class_emb = F.normalize(self.item_emb(self.class_item_indices), dim=1)
        scale = self.logit_scale.exp().clamp(max=50.0)
        return scale * (context @ class_emb.t()) + self.output_bias


def run_epoch(model, loader, optimizer=None, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.02, focal_gamma: float = 0.0):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for seq, lengths, users, y in loader:
        seq, lengths, users, y = seq.to(DEVICE), lengths.to(DEVICE), users.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(seq, lengths, users)
            weight = class_weights.to(DEVICE) if class_weights is not None else None
            if focal_gamma > 0:
                ce = F.cross_entropy(logits, y, weight=weight, label_smoothing=label_smoothing, reduction="none")
                pt = torch.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
                loss = ((1.0 - pt) ** float(focal_gamma) * ce).mean()
            else:
                loss = F.cross_entropy(logits, y, weight=weight, label_smoothing=label_smoothing)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)


def predict_logits(model, loader):
    model.eval()
    outs = []
    with torch.no_grad():
        for seq, lengths, users, _ in loader:
            logits = model(seq.to(DEVICE), lengths.to(DEVICE), users.to(DEVICE))
            outs.append(logits.cpu().numpy())
    return np.vstack(outs)


def main() -> None:
    print("device", DEVICE, flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data_dir = task_dir("A_recommendation")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    user = pd.read_csv(data_dir / "user.csv")
    item = pd.read_csv(data_dir / "item.csv")
    item_to_idx = {iid: idx + 1 for idx, iid in enumerate(item["iid"].tolist())}
    train_pos, val_pos = rs.recommendation_holdout_indices(train["target_iid"], test_size=0.2)
    inner_train = train.iloc[train_pos].reset_index(drop=True)
    inner_val = train.iloc[val_pos].reset_index(drop=True)
    visible_val = rs.make_test_like_recommendation_validation(inner_val, test)
    class_items = inner_train["target_iid"].value_counts().index.tolist()
    target_to_idx = {iid: idx for idx, iid in enumerate(class_items)}
    user_cols = [c for c in user.columns if c != "uid"]
    user_cardinalities = [int(user[c].max()) for c in user_cols]
    counts = inner_train["target_iid"].value_counts().reindex(class_items).to_numpy(dtype=np.float32)
    class_bias = np.log(counts / counts.sum() + 1e-8)
    test_lengths = test["item_seq_raw"].map(lambda v: len(rs.read_items(v))).to_numpy()

    train_ds = RecDataset(inner_train, user, item_to_idx, target_to_idx, test_lengths=test_lengths, augment=True)
    val_ds = RecDataset(visible_val.assign(target_iid=inner_val["target_iid"].values), user, item_to_idx, target_to_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    cfg = rs.RecConfig("mid_more_user", 0.22, 1.35, 0.75, 0.12, 0.08, 0.75)
    rec = rs.Recommender(item["iid"].tolist(), user_cols, cfg)
    rec.fit(inner_train, user)
    rule_all = rs.recommendation_score_matrix(rec, visible_val, user)
    item_to_all_idx = {iid: idx for idx, iid in enumerate(item["iid"].tolist())}
    class_item_indices = [item_to_all_idx[iid] for iid in class_items]
    rule_scores = rule_all[:, class_item_indices]
    rule_scores = rule_scores / np.maximum(rule_scores.max(axis=1, keepdims=True), 1e-6)
    print("rule", ndcg_from_logits(rule_scores, class_items, inner_val["target_iid"]), flush=True)
    model = SeqModel(len(item_to_idx), user_cardinalities, len(class_items), class_bias=class_bias, dim=96).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best = (-1.0, None, None)
    for epoch in range(1, 31):
        loss = run_epoch(model, train_loader, opt)
        logits = predict_logits(model, val_loader)
        score = ndcg_from_logits(logits, class_items, inner_val["target_iid"])
        norm_logits = logits / np.maximum(logits.max(axis=1, keepdims=True), 1e-6)
        blend_scores = []
        for blend in [0.15, 0.3, 0.5, 0.8, 1.2]:
            blend_scores.append((blend, ndcg_from_logits(norm_logits + blend * rule_scores, class_items, inner_val["target_iid"])))
        print("epoch", epoch, "loss", round(loss, 5), "ndcg", score, "blend", blend_scores, flush=True)
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, logits)
    best_logits = best[2]
    norm_logits = best_logits / np.maximum(best_logits.max(axis=1, keepdims=True), 1e-6)
    print("best", best[0], flush=True)
    for blend in [0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 1.8]:
        print("best_blend", blend, ndcg_from_logits(norm_logits + blend * rule_scores, class_items, inner_val["target_iid"]), flush=True)


if __name__ == "__main__":
    main()
