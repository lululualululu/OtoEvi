import os
import unittest
from pathlib import Path

import torch

from otoevi import OtoEvi
from otoevi.training.alignment import slot_clause_loss


@unittest.skipUnless(
    os.environ.get("DINOV3_CHECKPOINT"),
    "Set DINOV3_CHECKPOINT to run real-backbone tests",
)
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = OtoEvi(
            48,
            0,
            12,
            checkpoint=Path(os.environ["DINOV3_CHECKPOINT"]),
            max_len=10,
            decoder_layers=2,
        ).eval()
        cls.images = torch.randn(1, 2, 3, 384, 384)
        cls.mask = torch.tensor([[True, False]])
        cls.inputs = torch.tensor([[1, 4, 5, 6]])

    def test_backbone_frozen_and_counted(self):
        self.assertFalse(
            any(p.requires_grad for p in self.model.vision.backbone.parameters())
        )
        audit = self.model.parameter_audit()
        self.assertGreater(audit["visual_tower_parameters"], 20_000_000)
        self.assertTrue(audit["under_100m"])

    def test_masked_nan_view_cannot_affect_generation(self):
        padded = self.images.clone()
        padded[~self.mask] = torch.nan
        with torch.inference_mode():
            original = self.model(self.images, self.inputs, view_mask=self.mask)[
                "logits"
            ]
            output = self.model(padded, self.inputs, view_mask=self.mask)["logits"]
        torch.testing.assert_close(original, output, atol=0, rtol=0)

    def test_decoder_does_not_read_future_tokens(self):
        modified = self.inputs.clone()
        modified[:, -1] = 9
        with torch.inference_mode():
            memory, contexts = self.model.encode(self.images, self.mask)
            original = self.model.decode(self.inputs, memory, contexts)
            output = self.model.decode(modified, memory, contexts)
        torch.testing.assert_close(original[:, :-1], output[:, :-1], atol=0, rtol=0)

    def test_view_order_does_not_change_logits(self):
        with torch.inference_mode():
            a = self.model(self.images, self.inputs, view_mask=self.mask)["logits"]
            b = self.model(
                self.images.flip(1), self.inputs, view_mask=self.mask.flip(1)
            )["logits"]
        torch.testing.assert_close(a, b, atol=4e-5, rtol=2e-5)

    def test_clause_supervision_does_not_move_token_embeddings(self):
        self.model.zero_grad(set_to_none=True)
        slots = torch.randn(2, 4, 256, requires_grad=True)
        valid = torch.ones(2, 4, dtype=torch.bool)
        tokens = torch.tensor([[[1, 2, 0], [3, 4, 0]], [[3, 4, 0], [5, 6, 0]]])
        mask = tokens.ne(0)
        loss, _ = slot_clause_loss(
            self.model,
            slots,
            valid,
            tokens,
            mask,
            mask.any(-1),
            torch.ones(2, 2, dtype=torch.bool),
        )
        loss.backward()
        self.assertIsNotNone(slots.grad)
        self.assertIsNotNone(self.model.clause_text_projection[1].weight.grad)
        self.assertIsNone(self.model.captioner.token_embed.weight.grad)


if __name__ == "__main__":
    unittest.main()
