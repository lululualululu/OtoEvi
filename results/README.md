# Experiment records

All scores come from completed research runs at the final epoch 15. This release reorganizes the implementation; it does not claim that all experiments were newly retrained during packaging.

- `otoinsight_summary.json`: eight final memory/reading/CCL configurations, three image conditions, mean and sample SD.
- `otoinsight_metrics_per_seed.json`: corresponding seeds 42/7/3407 values.
- `external_metrics_summary.json`, `external_metrics_per_seed.json`: final CCL on separately trained IU/PEIR models; PEIR current and original-reference scoring are distinguished.
- `provenance.json`: manuscript hash, research revision and source-label mapping.
- `implementation_equivalence.json`: exact active-state/RNG and real-backbone BF16 forward/weighted-gradient comparisons against four archived CCL configurations, two synthetic examinations each.
- `data_and_checkpoint_replay.json`: exact real training-batch32 tensors and eight archived predictions replayed per dataset; full OtoInsight validation concept metrics reproduced.

BLEU, ROUGE and F1 are stored on a 0–1 scale; CIDEr is unscaled. Standard deviations are sample SD across three fitted seeds, not patient-level confidence intervals.

The main final-CCL OtoInsight experiment uses 1,583 validation reports. No final-CCL 2026 temporal score is asserted. Current and original PEIR reference scores use the same generated outputs. Baseline implementation scores and historical pooled-positive controls remain separate research records; this public package does not claim to distribute all baseline implementations.

CCL effects are mixed across configurations. Relative to memory+reading, the full model's mean concept Macro-F1 gain is about 0.68 percentage points, concentrated in seed 42; the conditional patient-bootstrap interval reported in the research analysis crosses zero. The validation results do not establish a universal or statistically significant clinical gain. See the manuscript for the complete paired analysis and limitations.

```bash
python scripts/check_results.py
```

This checks every released aggregate against its per-seed values without needing patient data.
