"""Crop and normalize TouchFree images with MediaPipe Hand Landmarker Tasks."""

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from phase2_utils import (
    DEFAULT_PADDING,
    IMAGE_SIZE,
    PROCESSED_DIR,
    VISUAL_CHECKS_DIR,
    build_image_index,
    create_landmarker,
    crop_hand,
    detect_hands,
    hand_bounding_box,
    load_image,
    load_manifest,
    normalize_image,
    resize_image,
    resolve_image_path,
    sample_visual_rows,
    select_hand_index,
    validate_subject_splits,
)


PREPROCESSED_IMAGES_PATH = PROCESSED_DIR / "preprocessed_images.npy"
PREPROCESSING_MANIFEST_PATH = PROCESSED_DIR / "preprocessing_manifest.csv"
FAILURES_PATH = PROCESSED_DIR / "preprocessing_failures.csv"
MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "processed_row",
    "label",
    "user_id",
    "split",
    "detection_status",
    "reason",
)
FAILURE_COLUMNS = ("image_id", "image_path", "label", "split", "reason")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_visual_check(
    image_bgr: np.ndarray,
    image_id: str,
    label: str,
    split: str,
    hand_landmarks: list[Any] | None,
    box: tuple[int, int, int, int] | None,
) -> None:
    """Save an original-frame overlay for manual detection review."""
    VISUAL_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    preview = image_bgr.copy()
    if box is not None and hand_landmarks is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 255), 2)
        for landmark in hand_landmarks:
            point = (round(landmark.x * image_bgr.shape[1]), round(landmark.y * image_bgr.shape[0]))
            cv2.circle(preview, point, 2, (0, 255, 0), -1)
        status = "HAND DETECTED"
    else:
        status = "NO HAND DETECTED"

    cv2.putText(preview, f"{label} | {split} | {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output_path = VISUAL_CHECKS_DIR / f"{split}_{label}_{image_id}.jpg"
    if not cv2.imwrite(str(output_path), preview):
        raise OSError(f"Could not save visual check: {output_path}")


def process_dataset(manifest_path: Path, padding: float) -> tuple[int, int]:
    rows = load_manifest(manifest_path)
    validate_subject_splits(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Indexing JPGs under the extracted dataset for {len(rows):,} manifest rows...")
    image_index = build_image_index()
    visual_ids = sample_visual_rows(rows)
    processed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    successes = 0
    started_at = time.perf_counter()

    normalized_images = np.lib.format.open_memmap(
        PREPROCESSED_IMAGES_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    landmarker = create_landmarker(num_hands=2)
    try:
        for row_number, row in enumerate(rows):
            image_path: Path | None = None
            status = "error"
            reason = ""
            try:
                image_path = resolve_image_path(row, image_index)
                image_bgr = load_image(image_path)
                result = detect_hands(landmarker, image_bgr)
                hand_index = select_hand_index(result)
                if hand_index is None:
                    status = "no_hand"
                    reason = "no_hand_detected"
                    normalized_images[row_number] = np.nan
                else:
                    hand = result.hand_landmarks[hand_index]
                    box = hand_bounding_box(hand, image_bgr.shape[1], image_bgr.shape[0], padding)
                    crop = resize_image(crop_hand(image_bgr, box), IMAGE_SIZE)
                    normalized_images[row_number] = normalize_image(crop)
                    status = "success"
                    successes += 1

                if row["image_id"] in visual_ids:
                    visual_hand = result.hand_landmarks[hand_index] if hand_index is not None else None
                    visual_box = box if hand_index is not None else None
                    save_visual_check(image_bgr, row["image_id"], row["label"], row["split"], visual_hand, visual_box)
            except Exception as error:
                status = "error"
                reason = f"{type(error).__name__}: {error}"
                normalized_images[row_number] = np.nan

            path_value = str(image_path) if image_path is not None else f"image_id:{row['image_id']}"
            processed_row = {
                "image_id": row["image_id"],
                "image_path": path_value,
                "processed_row": row_number,
                "label": row["label"],
                "user_id": row["user_id"],
                "split": row["split"],
                "detection_status": status,
                "reason": reason,
            }
            processed_rows.append(processed_row)
            if status != "success":
                failures.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": path_value,
                        "label": row["label"],
                        "split": row["split"],
                        "reason": reason,
                    }
                )

            completed = row_number + 1
            if completed % 100 == 0 or completed == len(rows):
                elapsed = time.perf_counter() - started_at
                print(f"Processed {completed:,}/{len(rows):,} images ({elapsed / 60:.1f} min)")
    finally:
        landmarker.close()
        normalized_images.flush()
        del normalized_images

    write_csv(PREPROCESSING_MANIFEST_PATH, MANIFEST_COLUMNS, processed_rows)
    write_csv(FAILURES_PATH, FAILURE_COLUMNS, failures)
    print(f"Successful crops: {successes:,}; failures/no-hand: {len(failures):,}")
    print(f"Normalized images: {PREPROCESSED_IMAGES_PATH}")
    print(f"Visual checks: {VISUAL_CHECKS_DIR}")
    return successes, len(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="Existing split manifest CSV")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING, help="Crop padding as a fraction of hand box size")
    arguments = parser.parse_args()
    process_dataset(arguments.manifest or Path(__file__).resolve().parent / "data/manifests/touchfree_split_manifest.csv", arguments.padding)


if __name__ == "__main__":
    main()