from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import run_solution as rs
from generate_mlp_gru_submission import GRU_RUN, MLP_RUNS, train_gru_logits, train_mlp_logits
from scratch_torch_recommender import DEVICE, RecDataset

ROOT = Path(__file__).resolve().parent
try:
    from project_paths import task_dir
except ImportError:
    from .project_paths import task_dir

REC_DIR = task_dir("A_recommendation")
BUCKET_GRU_WEIGHTS = {
    "empty": 0.7,
    "len1": 0.1,
    "len2_3": 1.4,
    "len4p": 1.4,
}


def bucket_weights(lengths: np.ndarray) -> np.ndarray:
    weights = np.empty(len(lengths), dtype=np.float32)
    weights[lengths == 0] = BUCKET_GRU_WEIGHTS["empty"]
    weights[lengths == 1] = BUCKET_GRU_WEIGHTS["len1"]
    weights[(lengths >= 2) & (lengths <= 3)] = BUCKET_GRU_WEIGHTS["len2_3"]
    weights[lengths >= 4] = BUCKET_GRU_WEIGHTS["len4p"]
    return weights


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
    weights = bucket_weights(test_lengths)
    logits = mlp_logits + weights[:, None] * gru_logits

    class_array = np.asarray(class_items)
    predictions = []
    for row in logits:
        top_idx = np.argpartition(-row, kth=9)[:10]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        predictions.append(",".join(class_array[top_idx].tolist()))
    submission = pd.DataFrame({"uid": test["uid"], "prediction": predictions})
    bucket_counts = {
        "empty": int((test_lengths == 0).sum()),
        "len1": int((test_lengths == 1).sum()),
        "len2_3": int(((test_lengths >= 2) & (test_lengths <= 3)).sum()),
        "len4p": int((test_lengths >= 4).sum()),
    }
    log = {
        "task": "recommendation",
        "method": "mlp9_plus_gru3_bucket_weighted_by_history_length",
        "device": str(DEVICE),
        "num_train": int(len(train)),
        "num_test": int(len(test)),
        "num_classes": int(len(class_items)),
        "mlp_runs": MLP_RUNS,
        "gru_run": GRU_RUN,
        "bucket_gru_weights": BUCKET_GRU_WEIGHTS,
        "bucket_counts": bucket_counts,
        "seeds": mlp_logs + gru_logs,
        "validation_reference": {
            "mlp9_validation_ndcg": 0.48766722425659953,
            "gru3_validation_ndcg": 0.4876874961631269,
            "global_weight_0.35_validation_ndcg": 0.4897336818155371,
            "bucket_weight_validation_ndcg": 0.49032857213376563,
            "bucket_weights": BUCKET_GRU_WEIGHTS,
        },
    }
    return submission, log


def main() -> None:
    print("training classification...", flush=True)
    a1, cls_log = rs.train_classification(validate=False)
    print("training recommendation bucket MLP+GRU...", flush=True)
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
