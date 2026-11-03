"""Budget validation must fail before a training run writes any output."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from otoevi.training.config import TrainConfig


class TrainConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = TrainConfig(
            train=self.root / "train.json",
            val=self.root / "val.json",
            image_root=self.root / "images",
            vision_checkpoint=self.root / "model.safetensors",
            output=self.root / "run",
        )

    def test_rejects_invalid_accumulation_budgets(self):
        for options in (
            {"batch_size": 0},
            {"effective_batch": 0},
            {"effective_batch": 63},
            {"epochs": 1},
            {"workers": -1},
            {"initialization_bank_size": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                replace(self.config, **options).validate()
        self.assertFalse(self.config.output.exists())

    def test_populated_output_requires_explicit_resume(self):
        self.config.output.mkdir()
        sentinel = self.config.output / "history.json"
        sentinel.write_text("existing run")
        with self.assertRaisesRegex(ValueError, "Output already contains"):
            self.config.validate()
        replace(self.config, resume=True).validate()
        self.assertEqual(sentinel.read_text(), "existing run")


if __name__ == "__main__":
    unittest.main()
