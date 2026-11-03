"""Exercise training and all CLI workflows on synthetic reports and images."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


@unittest.skipUnless(
    os.environ.get("DINOV3_CHECKPOINT"),
    "Set DINOV3_CHECKPOINT and install the eval extra to run the full workflow",
)
class WorkflowTests(unittest.TestCase):
    def test_train_checkpoint_predict_evaluate(self):
        from otoevi.cli.evaluate import main as evaluate_main
        from otoevi.cli.predict import main as predict_main
        from otoevi.cli.train import main as train_main

        torch.set_num_threads(2)
        vision = os.environ["DINOV3_CHECKPOINT"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            captions = [
                "左侧鼓膜穿孔。",
                "右侧鼓膜凹陷。",
                "左侧鼓膜充血。",
                "右侧鼓室积液。",
            ]
            entries = []
            for index, caption in enumerate(captions + captions[:2]):
                name = f"{index}.png"
                Image.new(
                    "RGB", (64, 64), (30 + index * 25, 150, 200 - index * 20)
                ).save(images / name)
                entries.append(
                    dict(
                        report_id=f"r{index}",
                        patient_id=f"p{index}",
                        filenames=[name],
                        caption=caption,
                    )
                )
            for split, rows in (("train", entries[:4]), ("val", entries[4:])):
                (root / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
            common = ["--vision-checkpoint", vision, "--image-root", str(images)]
            training_args = [
                "--train",
                str(root / "train.json"),
                "--val",
                str(root / "val.json"),
                "--output",
                str(root / "run"),
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--effective-batch",
                "4",
                "--initialization-bank-size",
                "4",
                "--workers",
                "0",
                "--device",
                "cpu",
            ] + common
            with contextlib.redirect_stdout(io.StringIO()):
                train_main(training_args)
                # A completed run must resume without extra updates or state changes.
                history = (root / "run/history.json").read_bytes()
                train_main(training_args + ["--resume"])
                self.assertEqual((root / "run/history.json").read_bytes(), history)
            checkpoint = str(root / "run/last_model.pt")
            config = json.loads((root / "run/config.json").read_text())
            self.assertEqual(config["model"], "OtoEvi")
            self.assertEqual(config["variant"], "full")
            prediction = io.StringIO()
            with contextlib.redirect_stdout(prediction):
                predict_main(
                    [
                        "--checkpoint",
                        checkpoint,
                        "--vision-checkpoint",
                        vision,
                        "--images",
                        str(images / "4.png"),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertTrue(prediction.getvalue().strip())
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_main(
                    [
                        "--checkpoint",
                        checkpoint,
                        "--manifest",
                        str(root / "val.json"),
                        "--train",
                        str(root / "train.json"),
                        "--output",
                        str(root / "evaluation"),
                        "--batch-size",
                        "2",
                        "--workers",
                        "0",
                        "--device",
                        "cpu",
                    ]
                    + common
                )
            result = json.loads((root / "evaluation/metrics.json").read_text())
            for mode in ("matched", "shuffled", "blank"):
                self.assertEqual(result[mode]["reports"], 2)
                self.assertIn("corpus_bleu", result[mode])
                self.assertIn("clinical_proxy", result[mode])
                predictions = (
                    root / "evaluation" / f"{mode}_predictions.jsonl"
                ).read_text()
                self.assertEqual(len(predictions.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
