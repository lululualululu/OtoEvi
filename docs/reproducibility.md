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

The trainer saves model/optimizer/scheduler states, vocabulary, epoch history and configuration hashes. The fixed epoch-specific seeds reproduce batch ordering and worker-side transforms. Resume from a checkpoint produced by `otoevi-train` with matching data and configuration.

## Controls and metrics

- `matched`: original image/report pairing.
- `shuffled`: deterministic one-position cyclic image donor. It has no fixed report points, but is **not** guaranteed patient-disjoint when neighboring reports belong to one patient.
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

CI runs unit tests, aggregate-result checks, lint and package builds on Python 3.8 and 3.10 with Ubuntu 22.04. The fixed runner version supports the pinned Python and PyTorch dependencies.

Native fused CUDA attention may not be bitwise repeatable across runs, hardware or software versions. Record dependency versions, hardware, configuration and random seed for each experiment.

## Scope

Reproducing the OtoInsight scores requires the clinical images, reports and fixed patient-level partitions. These data are currently not publicly available to protect patient privacy and confidential clinical information. The fictional manifests demonstrate the input schema only. For IU X-Ray and PEIR Gross, use authorized source assets matching the provided split specifications; the preparation script verifies caption hashes and split membership.
