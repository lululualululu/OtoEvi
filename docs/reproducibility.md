# Reproduction and verification

## Fixed recipe

| Setting | Value |
|---|---|
| Seeds | 42, 7, 3407 |
| Budget / checkpoint | 15 full epochs / final epoch |
| Backbone | Frozen timm DINOv3 ViT-S/16, eval mode, input 384×384 |
| Encoder / decoder | 5 layers each, width256, 8 heads, FFN1024, ReLU, dropout .1 |
| Local slot resampler / readers | 4 heads, 12 slots per real view, two independent readers |
| Optimizer | AdamW, lr2e-4, weight decay .01, clip norm1.0 |
| Schedule | One epoch linear warm-up; cosine decay |
| Microbatch / effective batch | 32 / 64; smaller final accumulation group |
| Training precision | BF16 autocast on CUDA, TF32 matmul allowed |
| Evaluation | FP32 greedy, at most 90 generated tokens, full reference text |
| Tokenization | Chinese nonwhitespace characters; English lowercase regex words/punctuation |
| Teacher forcing | At most89 report tokens plus BOS/EOS boundary |
| Augmentation | RandomResizedCrop scale[.78,1], bicubic; ColorJitter(.08,.08,.06,.01) |
| Inference transform | Bicubic resize384, ImageNet mean/std |

Clause and report contrast are computed within each **microbatch**, not across the accumulated 64 reports. No text augmentation, test-based checkpoint selection, label smoothing, or weighted generation CE is used.

## Running and resuming

Use the JSON configs in the repository root, as shown in the README. Explicit command-line arguments override config values. Each seed/variant needs a separate output directory. `--resume` requires identical data, source hashes and configuration. A completed run resumes without adding epochs.

The trainer saves model/optimizer/scheduler states, vocabulary, epoch history and configuration hashes. The fixed epoch-specific seeds reproduce batch ordering and worker-side transforms. Checkpoints in this package use the pruned parameter layout; use original training code to resume an unfinished historical research checkpoint.

## Controls and metrics

- `matched`: original image/report pairing.
- `shuffled`: deterministic one-position cyclic image donor, matching the frozen current experiment loader. It has no fixed report points, but is **not** guaranteed patient-disjoint when neighboring reports belong to one patient.
- `blank`: zero the normalized image tensor while preserving real-view masks. This is not an RGB-black image.

Metrics are corpus-level COCO BLEU, ROUGE-L and CIDEr. No PTB tokenizer, sentence-BLEU substitution or reference truncation is applied. The Chinese concept evaluator counts positive reference agreement and reports explicit-negative and laterality measures separately. An unmentioned reference finding is not a confirmed negative diagnosis. English datasets do not receive the Chinese evaluator.

For PEIR, rescoring **the same generated outputs** against `test_original.json` isolates reference choice at evaluation; it is not a second generation experiment. Use `scripts/score_predictions.py` with `--references`.

## Tests

```bash
python -m unittest discover -s tests/unit -v
DINOV3_CHECKPOINT=weights/dinov3/model.safetensors \
  python -m unittest discover -s tests/integration -v
python scripts/check_results.py
```

Synthetic integration tests exercise training, save/resume, image-only prediction, controls and metric calculation. They do not train a meaningful clinical model.

The public migration was additionally checked against the archived research implementation: seeded active state/RNG, BF16 loss and weighted parameter gradients with real DINOv3 features, actual training-batch tensors, and a small archived-prediction replay on each dataset. See the machine-readable verification records in `results/`. These are **migration checks**, not a fresh 15-epoch rerun. Math SDP was used for gradient equivalence only; production training retains the original attention backend. Native fused CUDA attention may not be bitwise repeatable across repeated runs, hardware or software versions.

## Scope

Full numerical reproduction of the clinical tables requires the unavailable OtoInsight images/reports and exact split. The release supplies the runnable method and metrics, not a substitute dataset. Public datasets require your source assets to match the frozen specification; the preparation script rejects mismatched captions instead of silently changing the benchmark.
