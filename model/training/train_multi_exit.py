"""Train and independently evaluate all exits of the Imagenette model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.training.imagenette_data import create_datasets, download_imagenette
from model.training.multi_exit_resnet import (
    DEFAULT_EXIT_WEIGHTS,
    MultiExitResNet,
    weighted_multi_exit_loss,
)


@dataclass
class ExitAccumulator:
    loss_sum: float = 0.0
    correct: int = 0
    samples: int = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        self.loss_sum += float(loss.detach()) * batch_size
        self.correct += int((logits.argmax(dim=1) == targets).sum())
        self.samples += batch_size

    def result(self) -> dict[str, float | int]:
        return {
            "loss": self.loss_sum / self.samples,
            "accuracy": self.correct / self.samples,
            "correct": self.correct,
            "samples": self.samples,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "model" / "artifacts" / "phase2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test limit only")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Smoke-test limit only")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(
    model: MultiExitResNet,
    loader: DataLoader,
    device: torch.device,
    weights: tuple[float, ...],
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    accumulators = [ExitAccumulator() for _ in weights]
    weighted_loss_sum = 0.0
    sample_count = 0
    start = time.perf_counter()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            weighted_loss, exit_losses = weighted_multi_exit_loss(logits, targets, weights)
            if training:
                weighted_loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            weighted_loss_sum += float(weighted_loss.detach()) * batch_size
            sample_count += batch_size
            for accumulator, exit_loss, exit_logits in zip(
                accumulators, exit_losses, logits, strict=True
            ):
                accumulator.update(exit_loss, exit_logits, targets)

    if sample_count == 0:
        raise RuntimeError("No samples were processed")
    return {
        "weighted_loss": weighted_loss_sum / sample_count,
        "samples": sample_count,
        "duration_seconds": time.perf_counter() - start,
        "exits": [accumulator.result() for accumulator in accumulators],
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("Epochs, batch size, and image size must be positive")
    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root, archive_sha256 = download_imagenette(args.data_root.resolve())
    train_dataset, validation_dataset = create_datasets(dataset_root, args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = MultiExitResNet(num_classes=len(train_dataset.classes)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_final_accuracy = -1.0
    exit_weights = tuple(DEFAULT_EXIT_WEIGHTS)
    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "dataset_root": str(dataset_root),
        "archive_sha256": archive_sha256,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "class_to_idx": train_dataset.class_to_idx,
        "channels": list(model.channels),
        "blocks_per_stage": list(model.blocks_per_stage),
        "exit_weights": list(exit_weights),
        "arguments": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, exit_weights, optimizer, args.max_train_batches
            )
            validation_metrics = run_epoch(
                model, validation_loader, device, exit_weights, None, args.max_val_batches
            )
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record, indent=2), flush=True)

            final_accuracy = float(validation_metrics["exits"][-1]["accuracy"])
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "run_config": run_config,
                "validation": validation_metrics,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if final_accuracy > best_final_accuracy:
                best_final_accuracy = final_accuracy
                torch.save(checkpoint, output_dir / "best_final_exit.pt")
            scheduler.step()


if __name__ == "__main__":
    main()
