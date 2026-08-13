from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

import run_solution as rs
from generate_bucket_mlp_gru_submission import bucket_weights
from generate_fine_bucket_mlp_gru_submission import fine_bucket_weights
from scratch_torch_gru import GRUSeqModel


class RegressionTests(unittest.TestCase):
    def test_bucket_boundaries_are_total(self) -> None:
        lengths = np.array([0, 1, 2, 3, 4, 50], dtype=np.int64)
        np.testing.assert_array_equal(
            bucket_weights(lengths),
            np.array([0.7, 0.1, 1.4, 1.4, 1.4, 1.4], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            fine_bucket_weights(lengths),
            np.array([0.7, 0.2, 0.5, 1.2, 1.2, 1.2], dtype=np.float32),
        )

    def test_gru_handles_mixed_history_lengths(self) -> None:
        torch.manual_seed(7)
        model = GRUSeqModel(
            num_items=20,
            user_cardinalities=[3, 4],
            num_classes=5,
            dim=8,
            hidden=10,
            dropout=0.0,
        ).eval()
        seq = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4],
                [0, 0, 0, 4, 6],
                [0, 0, 4, 6, 8],
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        users = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
        with torch.no_grad():
            logits = model(seq, lengths, users)
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_existing_submission_files_are_valid(self) -> None:
        if not (rs.ROOT / "A1.csv").exists() or not (rs.ROOT / "A2.csv").exists():
            self.skipTest("competition submission files are intentionally not published")
        a1 = pd.read_csv(rs.ROOT / "A1.csv")
        a2 = pd.read_csv(rs.ROOT / "A2.csv")
        rs.validate_outputs(a1, a2)


if __name__ == "__main__":
    unittest.main()
