from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_solution as rs
from scratch_torch_recommender import DEVICE, MAX_LEN, SEED, RecDataset, ndcg_from_logits, predict_logits, run_epoch

ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir


class GRUSeqModel(nn.Module):
    def __init__(self, num_items: int, user_cardinalities: list[int], num_classes: int, class_bias: np.ndarray | None = None, dim: int = 96, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hidden, num_layers=1, batch_first=True)
        self.len_emb = nn.Embedding(MAX_LEN + 1, 16)
        self.user_embs = nn.ModuleList([nn.Embedding(card + 1, 12) for card in user_cardinalities])
        user_dim = 12 * len(user_cardinalities)
        self.proj = nn.Sequential(
            nn.Linear(hidden + dim * 2 + 16 + user_dim, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, num_classes),
        )
        if class_bias is not None:
            with torch.no_grad():
                self.proj[-1].bias.copy_(torch.tensor(class_bias, dtype=torch.float32))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(seq)
        _, hidden = self.gru(emb)
        gru_last = hidden[-1]
        mask = (seq != 0).float().unsqueeze(-1)
        mean = (emb * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        last_tokens = seq[:, -1]
        last = self.item_emb(last_tokens)
        user_parts = [emb_layer(users[:, i].clamp_min(0)) for i, emb_layer in enumerate(self.user_embs)]
        features = torch.cat([gru_last, mean, last, self.len_emb(lengths.clamp(max=MAX_LEN)), *user_parts], dim=1)
        return self.proj(features)


def train_gru(dim: int, hidden: int, dropout: float, lr: float, wd: float, epochs: int, seeds: list[int]):
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
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    logits_list = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))
        train_ds.rng = np.random.default_rng(seed)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
        model = GRUSeqModel(len(item_to_idx), user_cardinalities, len(class_items), class_bias=class_bias, dim=dim, hidden=hidden, dropout=dropout).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        for epoch in range(1, epochs + 1):
            loss = run_epoch(model, train_loader, opt)
        logits = predict_logits(model, val_loader)
        logits_list.append(logits)
        print("gru", dim, hidden, dropout, lr, wd, epochs, "seed", seed, "single", ndcg_from_logits(logits, class_items, inner_val["target_iid"]), "loss", loss, flush=True)
        print("gru_avg_so_far", len(logits_list), ndcg_from_logits(np.mean(logits_list, axis=0), class_items, inner_val["target_iid"]), flush=True)
    avg = np.mean(logits_list, axis=0)
    score = ndcg_from_logits(avg, class_items, inner_val["target_iid"])
    print("GRU_FINAL", dim, hidden, dropout, lr, wd, epochs, len(seeds), score, flush=True)
    return score


def main() -> None:
    print("device", DEVICE, flush=True)
    configs = [
        (96, 128, 0.25, 2e-3, 2e-4, 17),
        (96, 160, 0.25, 1.5e-3, 2e-4, 18),
        (128, 160, 0.30, 1.5e-3, 3e-4, 16),
    ]
    for cfg in configs:
        train_gru(*cfg, seeds=[20260713, 20260714, 20260715])


if __name__ == "__main__":
    main()
