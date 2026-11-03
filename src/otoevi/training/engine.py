"""Fixed-budget training, validation diagnostics and resumable checkpoints."""

import json
import math
import time
from pathlib import Path

import torch

from otoevi.data.clauses import TrainClauseStatistics
from otoevi.data.dataset import ReportDataset
from otoevi.data.loading import build_loader
from otoevi.data.manifests import audit_split, load_manifest
from otoevi.data.text import build_vocabulary
from otoevi.models.ablations import AblationOtoEvi
from otoevi.utils.files import file_hash, fingerprint, save_checkpoint, save_json
from otoevi.utils.random import set_seed

from .config import TrainConfig
from .generation import generation_losses
from .objective import objective


def train(args: TrainConfig) -> None:
    """Run the configured budget, keeping the final epoch checkpoint."""
    args.validate()
    torch.set_num_threads(4)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = True
    source_root = Path(__file__).resolve().parents[1]
    train, val = load_manifest(args.train), load_manifest(args.val)
    audit_split(train, val)
    vocabulary = build_vocabulary(train, list(range(len(train))), 1)
    tags = []
    statistics = TrainClauseStatistics(train)
    config = dict(
        model="OtoEvi",
        initialization_bank_size=args.initialization_bank_size,
        objective="clause_coverage",
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        effective_batch=args.effective_batch,
        workers=args.workers,
        max_len=90,
        decoder_layers=5,
        lr=2e-4,
        weight_decay=0.01,
        train_sha=file_hash(args.train),
        val_sha=file_hash(args.val),
        vision_sha=file_hash(args.vision_checkpoint),
        vocabulary_sha=fingerprint(vocabulary),
        tags_sha=fingerprint(tags),
        image_root=str(args.image_root.resolve()),
        source_sha={
            str(p.relative_to(source_root)): file_hash(p)
            for p in sorted(source_root.rglob("*.py"))
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = AblationOtoEvi(
        len(vocabulary["tokens"]),
        vocabulary["stoi"]["<pad>"],
        args.initialization_bank_size,
        args.variant,
        checkpoint=args.vision_checkpoint,
    ).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    steps = math.ceil(len(train) / args.effective_batch)

    def schedule(step):
        if step < steps:
            return (step + 1) / steps
        return 0.5 * (
            1 + math.cos(math.pi * (step - steps) / ((args.epochs - 1) * steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    first, history = 1, []
    if args.resume:
        saved = torch.load(
            args.output / "last_model.pt", map_location="cpu", weights_only=False
        )
        if saved["config"] != config:
            raise ValueError("Checkpoint data, sources or configuration changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first, history = saved["epoch"] + 1, saved["history"]
        del saved
    save_json(args.output / "config.json", config)
    save_json(args.output / "vocabulary.json", vocabulary)
    save_json(args.output / "tags.json", tags)
    dataset = ReportDataset(
        train, args.image_root, vocabulary, tags, statistics, training=True
    )
    validation = ReportDataset(val, args.image_root, vocabulary, tags, statistics)
    accumulation = args.effective_batch // args.batch_size
    device = torch.device(args.device)
    for epoch in range(first, args.epochs + 1):
        started = time.monotonic()
        model.train()
        model.vision.backbone.eval()
        set_seed(args.seed + 1000 * epoch)
        batches = build_loader(
            dataset, args.batch_size, args.workers, args.seed, epoch, True
        )
        optimizer.zero_grad(set_to_none=True)
        total, seen, updates, order = 0.0, 0, 0, []
        for index, raw in enumerate(batches):
            batch = {k: v.to(device) for k, v in raw.items() if torch.is_tensor(v)}
            count = len(raw["report_id"])
            group = min(
                args.effective_batch,
                len(dataset) - (index // accumulation) * args.effective_batch,
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                result = objective(model, batch, weight=count / group)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError("Non-finite training loss")
            result["scaled_loss"].backward()
            total += float(result["loss"].detach()) * count
            seen += count
            order.extend(raw["report_id"])
            if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                updates += 1
        expected = [
            train[i]["report_id"]
            for i in torch.randperm(
                len(train), generator=torch.Generator().manual_seed(args.seed + epoch)
            ).tolist()
        ]
        if seen != len(train) or order != expected or updates != steps:
            raise ValueError("Training report order/coverage/update check failed")
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.inference_mode():
            for raw in build_loader(validation, args.batch_size, args.workers, 42):
                batch = {k: v.to(device) for k, v in raw.items() if torch.is_tensor(v)}
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    memory, contexts = model.encode(batch["images"], batch["view_mask"])
                    logits = model.decode(batch["inputs"], memory, contexts)
                    loss, _ = generation_losses(
                        logits,
                        batch["targets"],
                        vocabulary["stoi"]["<pad>"],
                        batch["ce_weights"],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation loss")
                count = len(raw["report_id"])
                val_loss += float(loss) * count
                val_count += count
        row = dict(
            epoch=epoch,
            train_loss=total / seen,
            validation_ce=val_loss / val_count,
            optimizer_updates=updates,
            report_order_sha=fingerprint(order),
            seconds=time.monotonic() - started,
        )
        history.append(row)
        save_checkpoint(
            args.output / "last_model.pt",
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                vocabulary=vocabulary,
                tags=tags,
                epoch=epoch,
                config=config,
                history=history,
            ),
        )
        save_json(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
