import unittest

import torch

from otoevi.data.clauses import TrainClauseStatistics, candidate_mask
from otoevi.data.text import build_vocabulary
from otoevi.models.evidence import sparsemax
from otoevi.training.alignment import clause_pair_masks
from otoevi.training.gradients import protect_slot_gradient


class MechanismTests(unittest.TestCase):
    def test_repeated_clause_remains_positive_across_reports(self):
        tokens = torch.tensor([[[1, 2, 0], [3, 4, 0]], [[3, 4, 0], [5, 6, 0]]])
        mask = tokens.ne(0)
        positives, allowed, _, _ = clause_pair_masks(
            tokens, mask, mask.any(-1), torch.eye(2).bool()
        )
        self.assertTrue(positives[0, 2] and positives[1, 1])
        self.assertTrue((allowed | ~positives).all())
        self.assertFalse(allowed[0, 3])

    def test_same_patient_is_not_used_as_a_negative(self):
        stats = TrainClauseStatistics([{"caption": "甲乙。"}, {"caption": "丙丁。"}])
        features = [
            stats.features("甲乙。", "patient"),
            stats.features("丙丁。", "patient"),
        ]
        self.assertFalse(candidate_mask(features)[0, 1])

    def test_conflicting_gradient_is_removed(self):
        slots = torch.tensor([[1.0, 2.0]], requires_grad=True)
        main, auxiliary = slots.sum(), -10 * slots.sum()
        statistics = protect_slot_gradient(main, auxiliary, slots)
        (main + auxiliary).backward()
        torch.testing.assert_close(slots.grad, torch.ones_like(slots))
        self.assertEqual(statistics["conflict_fraction"], 1.0)

    def test_gradient_protection_respects_accumulation_weight(self):
        slots = torch.tensor([[1.0, 2.0]], requires_grad=True)
        main, auxiliary = (
            slots.sum() * 0.25,
            (slots * torch.tensor([[2.0, -1.0]])).sum() * 0.25,
        )
        statistics = protect_slot_gradient(main, auxiliary, slots)
        (main + auxiliary).backward()
        self.assertLessEqual(statistics["protected_aux_to_main_norm"], 0.500001)
        torch.testing.assert_close(
            slots.grad.sum(),
            torch.tensor(0.5)
            + (0.25 * torch.tensor([[2.0, -1.0]]) * (0.5 * (2**0.5) / (5**0.5))).sum(),
        )

    def test_sparse_reader_can_assign_exact_zero(self):
        weights = sparsemax(torch.tensor([[10.0, 0.0, -2.0], [1.0, 1.0, 1.0]]))
        torch.testing.assert_close(weights[0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))

    def test_vocabulary_has_no_validation_tokens(self):
        vocabulary = build_vocabulary([{"caption": "甲乙。"}], [0], 1)
        self.assertNotIn("丙", vocabulary["stoi"])
        self.assertEqual(vocabulary["mode"], "character")


if __name__ == "__main__":
    unittest.main()
