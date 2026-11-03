import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from otoevi.data.clauses import TrainClauseStatistics
from otoevi.data.dataset import ReportDataset
from otoevi.data.loading import collate_reports
from otoevi.data.manifests import audit_split
from otoevi.data.text import build_vocabulary


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name, color in [("a.png", "red"), ("b.png", "blue"), ("c.png", "white")]:
            Image.new("RGB", (32, 32), color).save(self.root / name)
        self.entries = [
            dict(
                report_id="r1",
                patient_id="p1",
                filenames=["a.png"],
                caption="甲乙，丙丁。",
            ),
            dict(
                report_id="r2",
                patient_id="p2",
                filenames=["b.png", "c.png"],
                caption="戊己。",
            ),
        ]
        self.vocab = build_vocabulary(self.entries, [0, 1], 1)
        self.stats = TrainClauseStatistics(self.entries)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, mode="matched"):
        return ReportDataset(
            self.entries,
            self.root,
            self.vocab,
            ["甲乙", "戊己"],
            self.stats,
            visual_mode=mode,
        )

    def test_missing_view_is_masked_without_duplicate_pixels(self):
        data = self.dataset()
        batch = collate_reports([data[0], data[1]])
        self.assertEqual(batch["images"].shape, (2, 2, 3, 384, 384))
        self.assertEqual(batch["view_mask"].tolist(), [[True, False], [True, True]])
        self.assertEqual(float(batch["images"][0, 1].abs().sum()), 0.0)

    def test_chinese_punctuation_creates_two_clauses(self):
        item = self.dataset()[0]
        self.assertEqual(int(item["clause_valid"].sum()), 2)
        self.assertEqual(item["clause_mask"][0].sum(), 2)

    def test_shuffling_changes_images_while_reference_stays_fixed(self):
        normal, shuffled = self.dataset()[0], self.dataset("shuffled")[0]
        self.assertEqual(normal["caption"], shuffled["caption"])
        self.assertEqual(shuffled["visual_source_report_id"], "r2")
        self.assertFalse(torch.equal(normal["images"], shuffled["images"][:1]))

    def test_blank_control_keeps_real_view_mask(self):
        item = self.dataset("blank")[1]
        self.assertEqual(float(item["images"].abs().sum()), 0.0)
        self.assertEqual(item["view_mask"].tolist(), [True, True])

    def test_overlap_in_patients_is_rejected(self):
        validation = [
            dict(
                report_id="other",
                patient_id="p1",
                filenames=["other.png"],
                caption="示例。",
            )
        ]
        with self.assertRaisesRegex(ValueError, "patient_id"):
            audit_split(self.entries, validation)


if __name__ == "__main__":
    unittest.main()
