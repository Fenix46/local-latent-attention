import sys
import unittest
from pathlib import Path

import torch


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from prototype.models import build_model


class LocalLatentCausalityTest(unittest.TestCase):
    def test_blockwise_latent_queries_do_not_peek_within_block(self) -> None:
        torch.manual_seed(0)
        model = build_model(
            "local_latent",
            vocab_size=101,
            max_seq_len=32,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            local_window=2,
            local_block_size=2,
            latent_tokens=8,
            latent_d_model=16,
            latent_heads=2,
            latent_query_block_size=4,
            checkpoint_blocks=False,
        )
        model.eval()

        # Same prefix through position 4, different future tokens in the same
        # latent query block. A causal model must produce the same logit there.
        sample_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        sample_b = torch.tensor([[1, 2, 3, 4, 5, 99, 98, 97]])

        with torch.no_grad():
            logits_a = model(sample_a)[0, 4]
            logits_b = model(sample_b)[0, 4]

        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
