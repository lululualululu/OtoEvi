"""Measure OtoEvi inference and optimizer-update cost without modifying checkpoints."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from otoevi.data.clauses import TrainClauseStatistics
from otoevi.data.dataset import ReportDataset
from otoevi.data.loading import collate_reports
from otoevi.data.manifests import load_manifest
from otoevi.inference.checkpoints import load_model
from otoevi.training.objective import objective


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset label in the output")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--real-views", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--variant", default=None, help="Optional checkpoint variant override"
    )
    parser.add_argument(
        "--measure-training",
        action="store_true",
        help="Also time in-memory optimizer updates",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("This synchronized benchmark requires a CUDA device")
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    checkpoint, image_root = args.checkpoint, args.image_root
    device = torch.device(args.device)
    model, vocab, _ = load_model(
        checkpoint, args.vision_checkpoint, args.device, variant=args.variant
    )
    training = load_manifest(args.train_manifest)
    entries = load_manifest(args.eval_manifest)
    stats = TrainClauseStatistics(training)
    dataset = ReportDataset(entries, image_root, vocab, [], stats)
    bos, eos = vocab["stoi"]["<bos>"], vocab["stoi"]["<eos>"]
    result = {
        "dataset": args.dataset,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "script_sha256": sha(Path(__file__)),
        "gpu": torch.cuda.get_device_name(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": 4,
            "tf32": True,
            "image_size": 384,
            "inference_dtype": "FP32",
            "training_autocast": "BF16",
        },
        "vocabulary_size": len(vocab["tokens"]),
        "parameter_audit": model.parameter_audit(),
        "protocol": {
            "warmup": 5,
            "repeats": args.repeats,
            "sync": "torch.cuda.synchronize before/after each sample",
            "timed_scope": "GPU-resident tensors; includes visual encoder and autoregressive decoder; excludes file IO, CPU transforms, H2D, checkpoint loading",
            "fixed_length": 90,
            "decoder_cache": "none; recomputes prefix at each token",
            "memory": "torch CUDA maximum allocated/reserved includes resident model and input, excludes CUDA context and non-PyTorch allocation",
            "flops": "FlopCounterMode matmul/convolution/SDPA, multiply-add=2; normalization, activation, pooling, reductions, sparsemax and elementwise ops excluded; no-grad with native fused MHA disabled for operator visibility; timing uses normal inference",
        },
        "inference": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    def to_device(batch):
        return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

    def time_runs(fn, warmup=5, repeats=None):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats or args.repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
        return {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)),
            "sd_ms": statistics.stdev(samples),
            "samples_ms": samples,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }

    maxviews = args.real_views
    selected = [
        i for i, e in enumerate(entries) if min(len(e["filenames"]), 2) == maxviews
    ][:32]
    if len(selected) != 32:
        raise ValueError(
            "Need at least 32 held-out reports with the requested real-view count"
        )
    items = [dataset[i] for i in selected]
    # One representative real examination, two real views where available.
    one = to_device(collate_reports(items[:1]))
    with torch.enable_grad():
        out = model(one["images"], one["inputs"], view_mask=one["view_mask"])["logits"]
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            out.sum(), [p for _, p in params], allow_unused=True
        )
        active = [n for (n, p), g in zip(params, grads) if g is not None]
        result["inference_connected_trainable_parameters"] = sum(
            p.numel() for (n, p), g in zip(params, grads) if g is not None
        )
        result["inference_connected_parameters"] = (
            result["inference_connected_trainable_parameters"]
            + result["parameter_audit"]["visual_tower_parameters"]
        )
        result["inference_connected_parameter_names"] = active
        del out, grads, params
    model.zero_grad(set_to_none=True)
    for batch_size in (1, 8, 32):
        batch = to_device(collate_reports(items[:batch_size]))

        def fixed(batch=batch):
            return model.generate(batch["images"], bos, -1, 90, batch["view_mask"])

        with torch.inference_mode():
            fixed_result = time_runs(fixed)
            fixed_result["reports_per_second"] = (
                1000 * batch_size / fixed_result["mean_ms"]
            )
            fixed_result["amortized_ms_per_report"] = (
                fixed_result["mean_ms"] / batch_size
            )
            actual_result = time_runs(
                lambda batch=batch: model.generate(
                    batch["images"], bos, eos, 90, batch["view_mask"]
                )
            )
            tokens = model.generate(batch["images"], bos, eos, 90, batch["view_mask"])
            lengths = []
            for row in tokens.tolist():
                lengths.append(row.index(eos) + 1 if eos in row else len(row))
            actual_result["generated_lengths_including_eos"] = lengths
            actual_result["batch_decode_steps"] = tokens.shape[1]
            actual_result["reports_per_second"] = (
                1000 * batch_size / actual_result["mean_ms"]
            )
            actual_result["amortized_ms_per_report"] = (
                actual_result["mean_ms"] / batch_size
            )
        row = {
            "batch_size": batch_size,
            "real_views_per_report": maxviews,
            "fixed90": fixed_result,
            "natural_eos": actual_result,
        }
        if batch_size == 1:
            fast = torch.backends.mha.get_fastpath_enabled()
            try:
                torch.backends.mha.set_fastpath_enabled(False)
                with torch.no_grad(), FlopCounterMode(display=False) as counter:
                    model.generate.__wrapped__(
                        model, batch["images"], bos, -1, 90, batch["view_mask"]
                    )
                row["fixed90_counted_gflops_per_report"] = (
                    counter.get_total_flops() / 1e9
                )
                assert row["fixed90_counted_gflops_per_report"] > 0
                row["counted_ops"] = {
                    str(k): v for k, v in counter.get_flop_counts()["Global"].items()
                }
                with torch.no_grad(), FlopCounterMode(display=False) as counter:
                    model(
                        batch["images"], batch["inputs"], view_mask=batch["view_mask"]
                    )
                row["teacher_forced90_counted_gflops_per_report"] = (
                    counter.get_total_flops() / 1e9
                )
            finally:
                torch.backends.mha.set_fastpath_enabled(fast)
        result["inference"].append(row)
        save()
        print(
            args.dataset,
            "inference",
            batch_size,
            round(fixed_result["mean_ms"], 2),
            flush=True,
        )
        del batch, fixed
    # Natural stop latency distribution over 32 distinct held-out examinations.
    samples = []
    lengths = []
    views = []
    with torch.inference_mode():
        for item in [dataset[i] for i in range(32)]:
            b = to_device(collate_reports([item]))
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            tokens = model.generate(b["images"], bos, eos, 90, b["view_mask"])
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
            lengths.append(tokens.shape[1])
            views.append(int(b["view_mask"].sum()))
    result["natural_eos_32_distinct_examinations"] = {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
        "decode_steps": lengths,
        "real_views": views,
    }
    del one, items, b, tokens, item
    if not args.measure_training:
        result["complete"] = True
        save()
        return
    if len(training) < 64:
        raise ValueError("Training timing requires at least 64 training reports")
    # Same actual two training microbatches for with/without CCL; optimizer steps are in-memory only.
    training_dataset = ReportDataset(
        training, image_root, vocab, [], stats, training=True
    )
    indices = torch.randperm(
        len(training), generator=torch.Generator().manual_seed(43)
    ).tolist()[:64]
    torch.manual_seed(1042)
    microbatches = [
        to_device(collate_reports([training_dataset[j] for j in indices[i : i + 32]]))
        for i in (0, 32)
    ]
    result["training"] = {}
    for use_coverage in (True, False) if model.use_coverage else (False,):
        torch.manual_seed(1042)
        model.train()
        model.use_coverage = use_coverage
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)

        def step(optimizer=optimizer):
            optimizer.zero_grad(set_to_none=True)
            for b in microbatches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = objective(model, b, weight=0.5)
                if not torch.isfinite(output["loss"]):
                    raise FloatingPointError("Non-finite measured training loss")
                output["scaled_loss"].backward()
                del output
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        timed = time_runs(step, warmup=3, repeats=8)
        timed.update(
            microbatch=32,
            effective_batch=64,
            real_views_per_microbatch=[int(b["view_mask"].sum()) for b in microbatches],
            reports_per_second=64000 / timed["mean_ms"],
            projected_compute_minutes_per_epoch=timed["mean_ms"]
            / 1000
            * ((len(training) + 63) // 64)
            / 60,
            scope="steady-state in-memory AdamW optimizer update, two microbatches, includes full objective/gradient coordination when enabled; excludes IO, augmentation, validation and checkpoint writing",
        )
        result["training"]["with_ccl" if use_coverage else "without_ccl"] = timed
        del optimizer, step
        torch.cuda.empty_cache()
        save()
        print(
            args.dataset,
            "training",
            use_coverage,
            round(timed["mean_ms"], 2),
            flush=True,
        )
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
