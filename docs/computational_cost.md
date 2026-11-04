# Computational cost

The complete model was measured with trained seed-42 checkpoints on an NVIDIA RTX 4090, Python 3.8.19, PyTorch 2.3.1 + CUDA 12.1, and four CPU threads. Inference uses FP32 with TF32 matmul enabled and 384×384 inputs. Five warm-up runs precede 20 synchronized wall-clock repetitions.

The fixed-workload benchmark generates all 90 tokens without early EOS. It includes visual encoding and autoregressive decoding, using the implementation's full-prefix decoder without a KV cache. Images are already on the GPU; checkpoint loading, file reads, CPU preprocessing and host-to-device transfer are excluded.

## Complete model

Parameter counts include the frozen DINOv3 encoder. Active inference counts exclude training-only projections. Dataset-specific vocabularies account for the differences in parameter counts.

| Dataset | Total parameters (M) | Trainable (M) | Active inference (M) | Counted GFLOPs / report |
|---|---:|---:|---:|---:|
| OtoInsight | 35.176 | 13.589 | 34.912 | 114.80 |
| IU X-Ray | 35.834 | 14.247 | 35.570 | 117.49 |
| PEIR Gross | 36.575 | 14.988 | 36.311 | 88.54 |

GFLOPs count matrix multiplication, convolution and scaled dot-product attention; one multiply-add is two operations. Normalization, activations, pooling, sparsemax, elementwise operations and reductions are excluded. These are **scoped operator counts**, rather than exhaustive FLOPs. Native fused MHA is disabled only during operator counting to expose attention operations; timing uses the normal inference path.

| Dataset | Real views | Batch | Mean batch latency (ms) | P95 (ms) | Reports/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| OtoInsight | 2 | 1 | 375.5 | 381.9 | 2.66 | 0.186 |
| OtoInsight | 2 | 8 | 409.5 | 418.0 | 19.54 | 0.437 |
| OtoInsight | 2 | 32 | 480.7 | 485.1 | 66.57 | 1.279 |
| IU X-Ray | 2 | 1 | 385.4 | 387.5 | 2.59 | 0.189 |
| IU X-Ray | 2 | 8 | 513.6 | 614.1 | 15.58 | 0.441 |
| IU X-Ray | 2 | 32 | 555.0 | 675.4 | 57.65 | 1.282 |
| PEIR Gross | 1 | 1 | 428.0 | 584.5 | 2.34 | 0.172 |
| PEIR Gross | 1 | 8 | 435.2 | 595.9 | 18.38 | 0.300 |
| PEIR Gross | 1 | 32 | 445.2 | 462.5 | 71.87 | 0.730 |

Memory is the peak reported by `torch.cuda.max_memory_allocated`, including the resident model and input tensors. It excludes the CUDA context and non-PyTorch allocations. Raw times and both allocated/reserved memory are in [the measurement records](../results/computational_cost/). Runtime depends on GPU state and workload; the P95 values show variability in these repetitions.

With normal EOS stopping on the first 32 distinct held-out examinations at batch size 1, mean latency was 124.6 ms for OtoInsight, 198.8 ms for IU X-Ray and 129.9 ms for PEIR Gross. Mean generated lengths were 27.5, 44.8 and 28.1 tokens, respectively. This small sample uses actual view counts and does not estimate the whole evaluation corpus.

## Training and CCL

Each optimizer update accumulates two microbatches of 32 reports, using BF16 autocast, AdamW, clipping and the full training objective. Three warm-up updates precede eight measured updates. The same input batches are used with and without CCL. Images are prepared before timing; loading, augmentation, validation and checkpoint writing are excluded. Measurement updates remain in memory and are not saved.

| Dataset | With CCL (ms/update) | Without CCL (ms/update) | With CCL peak GiB |
|---|---:|---:|---:|
| OtoInsight | 194.0 | 178.7 | 1.331 |
| IU X-Ray | 263.4 | 228.6 | 1.467 |
| PEIR Gross | 177.6 | 176.7 | 1.177 |

These are compute-only update measurements. The corresponding completed 15-epoch seed-42 training logs record 99.53, 15.64 and 19.45 minutes of training-loop time, including loading and augmentation but excluding validation and checkpoint writing. Those runs shared GPUs with concurrent workloads, so their durations are not a controlled dedicated-GPU comparison.

CCL adds a training objective and gradient coordination. The memory + reading and full configurations have the same inference graph, active parameter count and counted GFLOPs. Disabling CCL on the same checkpoint produced identical generated tokens in the replay check. Timing differences between separately trained checkpoints are not evidence that CCL speeds up inference.

## Same-backbone controls

The controls use trained seed-42 checkpoints, the same DINOv3 encoder, the same double-view inputs and fixed 90-token workload.

| Variant | Active parameters (M) | Counted GFLOPs | B1 latency (ms) | B8 latency (ms) |
|---|---:|---:|---:|---:|
| `base` | 27.519 | 102.88 | 285.9 | 366.4 |
| `memory_reading` | 34.912 | 114.80 | 410.2 | 423.5 |
| `full` | 34.912 | 114.80 | 375.5 | 409.5 |

## Run the benchmark

Install the package, supply an authorized trained checkpoint and prepare manifests as described in [data preparation](data.md). At least 32 held-out reports with the requested real-view count are required. For optional training timing, at least 64 training reports are required. Manifests must supply `patient_id`, with `null` only where patient identity is unavailable.

```bash
python scripts/measure_cost.py \
  --dataset otoinsight \
  --checkpoint runs/otoinsight/full/seed42/last_model.pt \
  --vision-checkpoint weights/dinov3/model.safetensors \
  --train-manifest data/otoinsight/train.json \
  --eval-manifest data/otoinsight/val.json \
  --image-root data/otoinsight/images \
  --real-views 2 --device cuda:0 \
  --output output/cost/otoinsight.json \
  --measure-training
```

Use `--real-views 1` for PEIR Gross. Pass `--variant base` or `--variant memory_reading` when benchmarking those configurations. Measure one job at a time on an otherwise idle GPU. JSON output contains the checkpoint hash, runtime versions, raw repetition times and token lengths. The published JSON records contain aggregates and counts, without clinical images, report text or examination identifiers.
