"""Memory-efficient TensorFlow loader for the fixed TouchFree Phase 1 split."""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "touchfree_split_manifest.csv"
PREPROCESSING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_manifest.csv"
PREPROCESSED_IMAGES_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_images.npy"
PHASE3_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"

CLASS_NAMES = ("dislike", "fist", "like", "no_gesture", "one", "palm", "peace")
CLASS_TO_ID = {name: class_id for class_id, name in enumerate(CLASS_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
IMAGE_SHAPE = (224, 224, 3)
VALID_STATUS = "success"


@dataclass
class DatasetMetadata:
    """Validated row alignment and valid image indices for all fixed splits."""

    manifest_rows: list[dict[str, str]]
    preprocessing_rows: list[dict[str, str]]
    valid_rows: list[dict[str, str]]
    indices_by_split: dict[str, list[int]]
    users_by_split: dict[str, set[str]]
    manifest_class_counts: Counter[str]
    valid_class_counts: Counter[str]
    excluded_failure_counts: Counter[str]
    split_counts: Counter[str]
    valid_split_counts: Counter[str]
    unique_users: int
    array_shape: tuple[int, ...]
    array_dtype: str
    pixel_min: float
    pixel_max: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def inspect_phase2_data(chunk_size: int = 64) -> DatasetMetadata:
    """Validate array/manifest alignment and scan image values in small mmap chunks."""
    manifest_rows = _read_csv(SPLIT_MANIFEST_PATH)
    preprocessing_rows = _read_csv(PREPROCESSING_MANIFEST_PATH)
    if not manifest_rows:
        raise ValueError(f"Split manifest is empty: {SPLIT_MANIFEST_PATH}")
    if len(manifest_rows) != len(preprocessing_rows):
        raise ValueError(
            f"Manifest/preprocessing row count differs: {len(manifest_rows)} vs {len(preprocessing_rows)}"
        )

    required_manifest_columns = {"image_id", "label", "user_id", "split"}
    required_preprocessing_columns = required_manifest_columns | {"processed_row", "detection_status"}
    if not required_manifest_columns.issubset(manifest_rows[0]):
        raise ValueError(f"Unexpected split manifest columns: {list(manifest_rows[0])}")
    if not required_preprocessing_columns.issubset(preprocessing_rows[0]):
        raise ValueError(f"Unexpected preprocessing manifest columns: {list(preprocessing_rows[0])}")

    manifest_ids = [row["image_id"] for row in manifest_rows]
    preprocessing_ids = [row["image_id"] for row in preprocessing_rows]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("Split manifest contains duplicate image IDs")
    if manifest_ids != preprocessing_ids:
        raise ValueError("Preprocessing manifest image IDs/order differ from the fixed split manifest")

    users_by_split = {split: set() for split in SPLIT_NAMES}
    indices_by_split = {split: [] for split in SPLIT_NAMES}
    valid_rows: list[dict[str, str]] = []
    manifest_class_counts: Counter[str] = Counter()
    valid_class_counts: Counter[str] = Counter()
    excluded_failure_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    valid_split_counts: Counter[str] = Counter()

    for index, (manifest_row, preprocessing_row) in enumerate(zip(manifest_rows, preprocessing_rows)):
        for field in ("label", "user_id", "split"):
            if manifest_row[field] != preprocessing_row[field]:
                raise ValueError(f"{field} mismatch for image {manifest_row['image_id']}")
        if int(preprocessing_row["processed_row"]) != index:
            raise ValueError(f"processed_row order mismatch at {manifest_row['image_id']}")
        label = manifest_row["label"]
        split = manifest_row["split"]
        if label not in CLASS_TO_ID:
            raise ValueError(f"Unexpected class in manifest: {label}")
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split in manifest: {split}")
        manifest_class_counts[label] += 1
        split_counts[split] += 1
        users_by_split[split].add(manifest_row["user_id"])

    overlaps = {
        "train/val": users_by_split["train"] & users_by_split["val"],
        "train/test": users_by_split["train"] & users_by_split["test"],
        "val/test": users_by_split["val"] & users_by_split["test"],
    }
    overlapping_users = {name: sorted(users) for name, users in overlaps.items() if users}
    if overlapping_users:
        raise ValueError(f"User leakage in fixed split: {overlapping_users}")

    images = np.load(PREPROCESSED_IMAGES_PATH, mmap_mode="r", allow_pickle=False)
    if images.shape != (len(manifest_rows), *IMAGE_SHAPE):
        raise ValueError(f"Unexpected image array shape {images.shape}; expected {(len(manifest_rows), *IMAGE_SHAPE)}")
    if images.dtype != np.float32:
        raise ValueError(f"Unexpected image dtype {images.dtype}; expected float32")

    pixel_min = float("inf")
    pixel_max = float("-inf")
    for start in range(0, len(manifest_rows), chunk_size):
        end = min(start + chunk_size, len(manifest_rows))
        chunk = images[start:end]
        row_valid = np.isfinite(chunk).all(axis=(1, 2, 3))
        for local_index, is_valid in enumerate(row_valid):
            index = start + local_index
            manifest_row = manifest_rows[index]
            preprocessing_row = preprocessing_rows[index]
            has_valid_crop = preprocessing_row["detection_status"] == VALID_STATUS
            if bool(is_valid) != has_valid_crop:
                raise ValueError(
                    f"Preprocessing status/value mismatch for {manifest_row['image_id']}: "
                    f"status={preprocessing_row['detection_status']}, finite={bool(is_valid)}"
                )
            if not is_valid:
                excluded_failure_counts[manifest_row["label"]] += 1
                continue

            image = chunk[local_index]
            minimum = float(image.min())
            maximum = float(image.max())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(f"Image values outside [0, 1] for {manifest_row['image_id']}")
            pixel_min = min(pixel_min, minimum)
            pixel_max = max(pixel_max, maximum)
            valid_rows.append(preprocessing_row)
            indices_by_split[manifest_row["split"]].append(index)
            valid_class_counts[manifest_row["label"]] += 1
            valid_split_counts[manifest_row["split"]] += 1

    if not valid_rows:
        raise ValueError("No valid preprocessed crops are available for training")

    return DatasetMetadata(
        manifest_rows=manifest_rows,
        preprocessing_rows=preprocessing_rows,
        valid_rows=valid_rows,
        indices_by_split=indices_by_split,
        users_by_split=users_by_split,
        manifest_class_counts=manifest_class_counts,
        valid_class_counts=valid_class_counts,
        excluded_failure_counts=excluded_failure_counts,
        split_counts=split_counts,
        valid_split_counts=valid_split_counts,
        unique_users=len(set().union(*users_by_split.values())),
        array_shape=tuple(images.shape),
        array_dtype=str(images.dtype),
        pixel_min=pixel_min,
        pixel_max=pixel_max,
    )


def make_tf_datasets(
    metadata: DatasetMetadata,
    batch_size: int = 32,
    seed: int = 2026,
) -> dict[str, tf.data.Dataset]:
    """Build batched datasets; only the training order is shuffled."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    epoch_number = 0
    output_signature = (
        tf.TensorSpec(shape=(None, *IMAGE_SHAPE), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    def dataset_generator(split: str):
        nonlocal epoch_number
        indices = list(metadata.indices_by_split[split])
        if split == "train":
            random.Random(seed + epoch_number).shuffle(indices)
            epoch_number += 1

        # Reopen as a memory map for each epoch; only one batch is materialized at a time.
        images = np.load(PREPROCESSED_IMAGES_PATH, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_images = np.asarray(images[batch_indices], dtype=np.float32)
            batch_labels = np.fromiter(
                (CLASS_TO_ID[metadata.manifest_rows[index]["label"]] for index in batch_indices),
                dtype=np.int32,
                count=len(batch_indices),
            )
            yield batch_images, batch_labels
        del images

    datasets: dict[str, tf.data.Dataset] = {}
    for split in SPLIT_NAMES:
        dataset = tf.data.Dataset.from_generator(
            lambda split_name=split: dataset_generator(split_name),
            output_signature=output_signature,
        )
        options = tf.data.Options()
        options.experimental_deterministic = True
        datasets[split] = dataset.with_options(options).prefetch(1)
    return datasets


def class_weights_from_train(metadata: DatasetMetadata) -> dict[int, float]:
    """Compute balanced class weights from valid training examples only."""
    counts = Counter(
        metadata.manifest_rows[index]["label"] for index in metadata.indices_by_split["train"]
    )
    if set(counts) != set(CLASS_NAMES):
        raise ValueError(f"Training split is missing a class: {dict(counts)}")
    total = sum(counts.values())
    return {CLASS_TO_ID[name]: total / (len(CLASS_NAMES) * counts[name]) for name in CLASS_NAMES}


def print_dataset_summary(metadata: DatasetMetadata, batch_size: int = 32, seed: int = 2026) -> None:
    """Print pre-training counts and inspect train/validation batches only."""
    print(f"Dataset shape: {metadata.array_shape}; dtype={metadata.array_dtype}; range=[{metadata.pixel_min:.4f}, {metadata.pixel_max:.4f}]")
    print(f"Total manifest rows: {len(metadata.manifest_rows)}")
    print(f"Valid CNN images: {len(metadata.valid_rows)}")
    print(f"Excluded failed/no-hand rows: {sum(metadata.excluded_failure_counts.values())}")
    print(f"Class IDs: {CLASS_TO_ID}")
    for class_name in CLASS_NAMES:
        print(
            f"  {class_name}: {metadata.valid_class_counts[class_name]} valid; "
            f"{metadata.excluded_failure_counts[class_name]} excluded"
        )
    for split in SPLIT_NAMES:
        print(f"{split}: {metadata.valid_split_counts[split]} valid / {metadata.split_counts[split]} manifest rows")
    print(f"Unique users: {metadata.unique_users}; split-user intersections: all empty")

    datasets = make_tf_datasets(metadata, batch_size=batch_size, seed=seed)
    # Keep the held-out test split untouched until final post-selection evaluation.
    for split in ("train", "val"):
        images, labels = next(iter(datasets[split]))
        tf.debugging.assert_all_finite(images, f"Non-finite image batch in {split}")
        tf.debugging.assert_greater_equal(labels, 0)
        tf.debugging.assert_less(labels, len(CLASS_NAMES))
        print(
            f"{split} example batch: shape={tuple(images.shape)}, labels="
            f"{labels.numpy().tolist()[:8]}, range=[{float(tf.reduce_min(images)):.4f}, {float(tf.reduce_max(images)):.4f}]"
        )


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(2026)
    metadata = inspect_phase2_data()
    print_dataset_summary(metadata)