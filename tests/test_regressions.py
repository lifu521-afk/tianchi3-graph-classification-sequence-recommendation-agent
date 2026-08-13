import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_bucket_mlp_gru_submission import bucket_weights
from generate_fine_bucket_mlp_gru_submission import fine_bucket_weights
from scratch_torch_gru import GRUSeqModel


class RegressionTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        lengths = np.array([0, 1, 2, 3, 4, 50])
        np.testing.assert_allclose(bucket_weights(lengths), [0.7, 0.1, 1.4, 1.4, 1.4, 1.4])
        np.testing.assert_allclose(fine_bucket_weights(lengths), [0.7, 0.2, 0.5, 1.2, 1.2, 1.2])

    def test_gru_mixed_lengths(self):
        model = GRUSeqModel(20, [3, 4], 5, dim=8, hidden=10, dropout=0.0).eval()
        seq = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 4], [0, 0, 0, 4, 6]])
        lengths = torch.tensor([0, 1, 2])
        users = torch.tensor([[0, 1], [1, 2], [2, 3]])
        with torch.no_grad():
            logits = model(seq, lengths, users)
        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
