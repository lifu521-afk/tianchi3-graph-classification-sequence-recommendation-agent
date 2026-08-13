from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_solution as rs
from scratch_torch_gru import GRUSeqModel
from scratch_torch_recommender import DEVICE, RecDataset, SeqModel, predict_logits, run_epoch

ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir

REC_DIR = task_dir("A_recommendation")
MLP_RUNS = [
    {"name": "seq96", "dim": 96, "lr": 2e-3, "weight_decay": 2e-4, "epochs": 17, "seeds": [20260713, 20260714, 20260715, 20260716, 20260717, 20260718]},
    {"name": "seq128", "dim": 128, "lr": 2e-3, "weight_decay": 2e-4, "epochs": 14, "seeds": [20260713, 20260714, 20260715]},
]
GRU_RUN = {"name": "gru128", "dim": 128, "hidden": 160, "dropout": 0.30, "lr": 1.5e-3, "weight_decay": 3e-4, "epochs": 16, "seeds": [20260713, 20260714, 20260715]}
GRU_BLEND_WEIGHT = 0.35


def train_mlp_logits(train_ds, test_loader, item_to_idx, user_cardinalities, class_items, class_bias):
    logits_list = []
    logs = []
    for run in MLP_RUNS:
        for seed in run["seeds"]:
            print(f"training MLP {run['name']} seed {seed}...", flush=True)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))
            train_ds.rng = np.random.default_rng(seed)
            train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
            model = SeqModel(len(item_to_idx), user_cardinalities, len(class_items), class_bias=class_bias, dim=int(run["dim"])).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(run["lr"]), weight_decay=float(run["weight_decay"]))
            loss = 0.0
            for epoch in range(1, int(run["epochs"]) + 1):
                loss = run_epoch(model, train_loader, optimizer)
                if epoch in {1, 5, 10, int(run["epochs"])}:
                    print(f"MLP {run['name']} seed {seed} epoch {epoch} loss {loss:.5f}", flush=True)
            logits_list.append(predict_logits(model, test_loader))
            logs.append({"family": "mlp", "model": run["name"], "seed": seed, "epochs": int(run["epochs"]), "final_loss": float(loss)})
    return np.mean(logits_list, axis=0), logs


def train_gru_logits(train_ds, test_loader, item_to_idx, user_cardinalities, class_items, class_bias):
    logits_list = []
    logs = []
    for seed in GRU_RUN["seeds"]:
        print(f"training GRU seed {seed}...", flush=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))
        train_ds.rng = np.random.default_rng(seed)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
        model = GRUSeqModel(
            len(item_to_idx),
            user_cardinalities,
            len(class_items),
            class_bias=class_bias,
            dim=int(GRU_RUN["dim"]),
            hidden=int(GRU_RUN["hidden"]),
            dropout=float(GRU_RUN["dropout"]),
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(GRU_RUN["lr"]), weight_decay=float(GRU_RUN["weight_decay"]))
        loss = 0.0
        for epoch in range(1, int(GRU_RUN["epochs"]) + 1):
            loss = run_epoch(model, train_loader, optimizer)
            if epoch in {1, 5, 10, int(GRU_RUN["epochs"])}:
                print(f"GRU seed {seed} epoch {epoch} loss {loss:.5f}", flush=True)
        logits_list.append(predict_logits(model, test_loader))
        logs.append({"family": "gru", "model": GRU_RUN["name"], "seed": seed, "epochs": int(GRU_RUN["epochs"]), "final_loss": float(loss)})
    return np.mean(logits_list, axis=0), logs


def recommendation_submission() -> tuple[pd.DataFrame, dict[str, object]]:
    train = pd.read_csv(REC_DIR / "train.csv")
    test = pd.read_csv(REC_DIR / "test.csv")
    user = pd.read_csv(REC_DIR / "user.csv")
    item = pd.read_csv(REC_DIR / "item.csv")
    item_to_idx = {iid: idx + 1 for idx, iid in enumerate(item["iid"].tolist())}
    class_items = train["target_iid"].value_counts().index.tolist()
    target_to_idx = {iid: idx for idx, iid in enumerate(class_items)}
    user_cols = [col for col in user.columns if col != "uid"]
    user_cardinalities = [int(user[col].max()) for col in user_cols]
    counts = train["target_iid"].value_counts().reindex(class_items).to_numpy(dtype=np.float32)
    class_bias = np.log(counts / counts.sum() + 1e-8)
    test_lengths = test["item_seq_raw"].map(lambda value: len(rs.read_items(value))).to_numpy()

    train_ds = RecDataset(train, user, item_to_idx, target_to_idx, test_lengths=test_lengths, augment=True)
    test_frame = test.copy()
    test_frame["target_iid"] = class_items[0]
    test_ds = RecDataset(test_frame, user, item_to_idx, target_to_idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

    mlp_logits, mlp_logs = train_mlp_logits(train_ds, test_loader, item_to_idx, user_cardinalities, class_items, class_bias)
    gru_logits, gru_logs = train_gru_logits(train_ds, test_loader, item_to_idx, user_cardinalities, class_items, class_bias)
    logits = mlp_logits + GRU_BLEND_WEIGHT * gru_logits

    class_array = np.asarray(class_items)
    predictions = []
    for row in logits:
        top_idx = np.argpartition(-row, kth=9)[:10]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        predictions.append(",".join(class_array[top_idx].tolist()))
    submission = pd.DataFrame({"uid": test["uid"], "prediction": predictions})
    log = {
        "task": "recommendation",
        "method": "mlp9_plus_gru3_weighted_ensemble",
        "device": str(DEVICE),
        "num_train": int(len(train)),
        "num_test": int(len(test)),
        "num_classes": int(len(class_items)),
        "mlp_runs": MLP_RUNS,
        "gru_run": GRU_RUN,
        "gru_blend_weight": GRU_BLEND_WEIGHT,
        "seeds": mlp_logs + gru_logs,
        "validation_reference": {
            "online_best_previous": 0.4990,
            "mlp9_validation_ndcg": 0.48766722425659953,
            "gru3_validation_ndcg": 0.4876874961631269,
            "blend_weight_0.35_validation_ndcg": 0.4897336818155371,
            "equal_12_model_validation_ndcg": 0.48968475113986537,
        },
    }
    return submission, log


def main() -> None:
    print("training classification...", flush=True)
    a1, cls_log = rs.train_classification(validate=False)
    print("training recommendation MLP+GRU...", flush=True)
    a2, rec_log = recommendation_submission()
    print("validating outputs...", flush=True)
    rs.validate_outputs(a1, a2)

    a1_path = ROOT / "A1.csv"
    a2_path = ROOT / "A2.csv"
    zip_path = ROOT / "prediction.zip"
    log_path = ROOT / "experiment_log.json"
    a1.to_csv(a1_path, index=False)
    a2.to_csv(a2_path, index=False)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump({"classification": cls_log, "recommendation": rec_log}, handle, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(a1_path, arcname="A1.csv")
        zf.write(a2_path, arcname="A2.csv")
    print(f"wrote {a1_path}", flush=True)
    print(f"wrote {a2_path}", flush=True)
    print(f"wrote {log_path}", flush=True)
    print(f"wrote {zip_path}", flush=True)


if __name__ == "__main__":
    main()
