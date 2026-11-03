"""Eight A/B/CCL factorial configurations, sharing canonical initialization."""

from .otoevi import OtoEvi

FACTORS = {
    "base": (False, False, False),
    "memory_only": (True, False, False),
    "reading_only": (False, True, False),
    "coverage_only": (False, False, True),
    "memory_reading": (True, True, False),
    "memory_coverage": (True, False, True),
    "reading_coverage": (False, True, True),
    "full": (True, True, True),
}
VARIANTS = dict(FACTORS, no_gradient_coordination=(True, True, True))


class AblationOtoEvi(OtoEvi):
    def __init__(self, vocab_size, pad_id, num_tags=384, variant="full", **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(vocab_size, pad_id, num_tags, **kwargs)
        self.variant = variant
        self.use_memory, self.use_reading, self.use_coverage = VARIANTS[variant]
        self.captioner.use_memory = self.use_memory
        self.coordinate_gradients = variant != "no_gradient_coordination"
