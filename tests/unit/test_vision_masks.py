import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from otoevi.models.vision import DINOv3Vision


class DummyBackbone(nn.Module):
    def forward_intermediates(self, images, **kwargs):
        n = len(images)
        grid = images.new_zeros(n, 384, 24, 24)
        return images.new_zeros(n, 1, 384), [(grid, None)] * 4


class VisionMaskTests(unittest.TestCase):
    def test_empty_real_view_does_not_unmask_other_views_or_padding(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            with patch(
                "otoevi.models.vision.timm.create_model", return_value=DummyBackbone()
            ):
                model = DINOv3Vision(Path(checkpoint.name)).eval()
        pixels = torch.zeros(2, 2, 3, 384, 384)
        pixels[0, 0, :, 96:288, 96:288] = 0.7
        images = (pixels - model.image_mean) / model.image_std
        view_mask = torch.tensor([[True, False], [True, False]])
        with torch.no_grad():
            together = model(images, view_mask)
            alone = model(images[:1], view_mask[:1])
        self.assertTrue(
            torch.equal(together["spatial_valid"][:1], alone["spatial_valid"])
        )
        self.assertTrue(together["spatial_valid"][0, 0].any())
        self.assertFalse(together["spatial_valid"][0, 0].all())
        self.assertTrue(together["spatial_valid"][1, 0].all())
        self.assertFalse(together["spatial_valid"][:, 1].any())


if __name__ == "__main__":
    unittest.main()
