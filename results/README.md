# Results

Metrics use the checkpoint after 15 complete training epochs and seeds **42, 7, 3407**.

| File | Contents |
|---|---|
| `otoinsight_summary.json` | Eight memory / reading / CCL configurations and three image conditions; mean and sample standard deviation |
| `otoinsight_metrics_per_seed.json` | OtoInsight metrics for each configuration, condition and seed |
| `external_metrics_summary.json` | IU X-Ray and PEIR Gross aggregate metrics |
| `external_metrics_per_seed.json` | IU X-Ray and PEIR Gross per-seed metrics |
| `evaluation_settings.json` | Dataset partitions, seeds, checkpoint epoch and metric units |

BLEU, ROUGE and F1 are stored on a 0–1 scale; CIDEr is unscaled. Standard deviations are sample SD across three fitted seeds, not patient-level confidence intervals.

OtoInsight uses 1,583 validation reports. IU X-Ray and PEIR Gross use 383 and 745 test reports, respectively, with models trained separately on each dataset. PEIR scores distinguish history-reduced and original references; both are evaluated against the same generated outputs. See the [data protocol](../docs/data.md#peir-gross-protocol).

CCL effects vary across configurations. Relative to memory + reading, the full model increases mean concept Macro-F1 by about 0.68 percentage points, with most of the gain in seed 42. The conditional patient-bootstrap interval crosses zero; these results do not establish a statistically significant clinical gain. See the paper for the paired analysis and limitations.

Verify aggregate means and sample standard deviations against the per-seed values:

```bash
python scripts/check_results.py
```

This check does not require patient data.
