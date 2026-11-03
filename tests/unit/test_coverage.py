"""Mechanism checks for the CCL objective and checkpoint naming."""

import unittest

import torch

from otoevi.evaluation.ten_concept_evaluator import score
from otoevi.inference.checkpoints import checkpoint_variant
from otoevi.training.alignment import (
    clause_pair_masks,
    coverage_loss,
    unique_positive_mask,
)


class CoverageTests(unittest.TestCase):
    def test_duplicates_deduplicate_only_positive_mean(self):
        tokens = torch.tensor([[[4, 5], [6, 7]], [[4, 5], [8, 9]]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        valid = mask.any(-1)
        positives, allowed, _, _ = clause_pair_masks(
            tokens, mask, valid, torch.ones(2, 2, dtype=torch.bool)
        )
        unique = unique_positive_mask(tokens, mask, valid, positives)
        self.assertEqual(unique.sum(1).tolist(), [2, 2])
        self.assertEqual(allowed.sum(1).tolist(), [4, 4])
        self.assertEqual(positives.sum(1).tolist(), [3, 3])

    def test_coverage_matches_explicit_formula_and_has_finite_gradient(self):
        scores = torch.tensor(
            [[3.0, -2.0, 3.0, 0.0], [1.0, 0.0, 1.0, 2.0]], requires_grad=True
        )
        positives = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool)
        allowed = torch.ones_like(positives)
        unique = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=torch.bool)
        forward = torch.stack(
            [torch.logsumexp(row, 0) - row[u].mean() for row, u in zip(scores, unique)]
        ).mean()
        reverse = torch.stack(
            [
                torch.logsumexp(scores[:, j], 0)
                - torch.logsumexp(scores[positives[:, j], j], 0)
                for j in range(4)
            ]
        ).mean()
        loss = coverage_loss(scores, positives, allowed, unique)
        torch.testing.assert_close(loss, (forward + reverse) / 2)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_masked_clause_does_not_create_nan(self):
        scores = torch.tensor(
            [[1.0, 2.0, -float("inf")], [2.0, 1.0, -float("inf")]], requires_grad=True
        )
        pos = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
        allow = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        loss = coverage_loss(scores, pos, allow, pos)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(scores.grad[:, -1].eq(0).all())

    def test_current_checkpoint_names_are_explicit(self):
        self.assertEqual(checkpoint_variant({"variant": "D1_clause_coverage"}), "full")
        self.assertEqual(checkpoint_variant({"variant": "H101_AD1"}), "memory_coverage")
        with self.assertRaises(ValueError):
            checkpoint_variant({"variant": "A0_full"})

    def test_unmentioned_reference_is_not_an_explicit_negative(self):
        result, _ = score(
            [dict(reference="左侧鼓膜完整。", prediction="右侧鼓膜穿孔。")]
        )
        self.assertEqual(result["explicit_negative"]["perforation"]["support"], 0)
        self.assertEqual(result["concepts"]["perforation"]["fp"], 1)
        self.assertEqual(result["laterality"]["opposite"], 1)


if __name__ == "__main__":
    unittest.main()
