"""DINO image preprocessing for training and deterministic inference."""

from torchvision import transforms as T


def dino_transform(training=False):
    spatial = (
        T.RandomResizedCrop(
            (384, 384), scale=(0.78, 1.0), interpolation=T.InterpolationMode.BICUBIC
        )
        if training
        else T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC)
    )
    operations = [spatial]
    if training:
        operations.append(T.ColorJitter(0.08, 0.08, 0.06, 0.01))
    return T.Compose(
        operations
        + [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
