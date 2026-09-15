"""Reload a Phase-2 checkpoint and independently evaluate every exit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.training.imagenette_data import create_datasets, download_imagenette
from model.training.multi_exit_resnet import MultiExitResNet
from model.training.train_multi_exit import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["run_config"]
    image_size = int(config["arguments"]["image_size"])
    dataset_root, archive_sha256 = download_imagenette(args.data_root.resolve())
    _, validation_dataset = create_datasets(dataset_root, image_size)
    if validation_dataset.class_to_idx != config["class_to_idx"]:
        raise RuntimeError("Checkpoint and validation class mappings differ")

    model = MultiExitResNet(
        num_classes=len(validation_dataset.classes),
        channels=tuple(config["channels"]),
        blocks_per_stage=tuple(config["blocks_per_stage"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    metrics = run_epoch(
        model,
        validation_loader,
        device,
        tuple(config["exit_weights"]),
        optimizer=None,
        max_batches=None,
    )
    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "archive_sha256": archive_sha256,
        "class_to_idx": validation_dataset.class_to_idx,
        "exit_weights": config["exit_weights"],
        "validation": metrics,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
