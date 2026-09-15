"""Download and load the official 160-pixel Imagenette dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
from urllib.request import urlopen

from torchvision import datasets, transforms


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def download_imagenette(data_root: Path) -> tuple[Path, str]:
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_root = data_root / "imagenette2-160"
    archive_path = data_root / "imagenette2-160.tgz"
    extracted = (dataset_root / "train").is_dir() and (dataset_root / "val").is_dir()
    if extracted and not archive_path.exists():
        return dataset_root, "archive-not-retained"
    if not archive_path.exists():
        temporary_path = archive_path.with_suffix(".tgz.part")
        digest = hashlib.sha256()
        with urlopen(IMAGENETTE_URL) as response, temporary_path.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
        temporary_path.replace(archive_path)
        sha256 = digest.hexdigest()
    else:
        digest = hashlib.sha256()
        with archive_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        sha256 = digest.hexdigest()

    if not extracted:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(data_root, filter="data")
    if not (dataset_root / "train").is_dir() or not (dataset_root / "val").is_dir():
        raise RuntimeError("Imagenette archive did not contain expected train/val directories")
    return dataset_root, sha256


def create_datasets(dataset_root: Path, image_size: int = 128):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 16),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dataset = datasets.ImageFolder(dataset_root / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(dataset_root / "val", transform=validation_transform)
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise RuntimeError("Imagenette train/validation class mappings differ")
    if len(train_dataset.classes) != 10:
        raise RuntimeError(f"Expected 10 Imagenette classes, found {len(train_dataset.classes)}")
    return train_dataset, validation_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    dataset_root, sha256 = download_imagenette(args.data_root.resolve())
    train_dataset, validation_dataset = create_datasets(dataset_root)
    print(f"dataset_root={dataset_root}")
    print(f"archive_sha256={sha256}")
    print(f"train_samples={len(train_dataset)}")
    print(f"validation_samples={len(validation_dataset)}")
    print(f"class_to_idx={train_dataset.class_to_idx}")


if __name__ == "__main__":
    main()
