"""Explicit conversion of historical research checkpoints to the CCL package."""

RETIRED_CAPTIONER = (
    "scale_gate",
    "state_mixer",
    "tag_head",
    "dynamic_prefix",
    "prefix_attention",
    "prefix_norm",
    "prefix_gate",
    "phrase_visual",
    "phrase_text",
    "alignment_temperature",
    "anchor_tag_head",
    "anchor_embeddings",
    "anchor_gate",
    "tag_positive_weights",
)
RETIRED_MODEL = ("fov_global_projection", "fov_global_output")


def discard_retired_modules(model):
    for name in RETIRED_CAPTIONER:
        delattr(model.captioner, name)
    for name in RETIRED_MODEL:
        delattr(model, name)


def active_state_dict(state):
    """Drop only the explicitly enumerated inactive legacy parameters/buffers."""
    prefixes = tuple("captioner." + name for name in RETIRED_CAPTIONER) + RETIRED_MODEL

    def retired(key):
        return any(key == p or key.startswith(p + ".") for p in prefixes)

    return {key: value for key, value in state.items() if not retired(key)}
