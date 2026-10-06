"""Extract normalized 21-point XYZ hand landmarks for the existing split."""

import argparse
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from phase2_utils import (
    FEATURE_COLUMNS,
    MANIFEST_PATH,
    PROCESSED_DIR,
    build_image_index,
    create_landmarker,
    detect_hands,
    load_image,
    load_manifest,
    normalize_landmarks,
    resolve_image_path,
    select_hand_index,
    validate_subject_splits,
)


DEFAULT_LANDMARKS_PATH = PROCESSED_DIR / "landmarks.parquet"
# Preserve an existing user artifact; put this run in a separate file when necessary.
LANDMARKS_PATH = (
    PROCESSED_DIR / "landmarks_phase2.parquet"
    if DEFAULT_LANDMARKS_PATH.exists()
    else DEFAULT_LANDMARKS_PATH
)
OUTPUT_COLUMNS = (
    "image_id",
    "image_path",
    "label",
    "user_id",
    "split",
    "detection_status",
    "reason",
    *FEATURE_COLUMNS,
)


def extract_landmarks(manifest_path: Path = MANIFEST_PATH) -> tuple[int, int]:
    rows = load_manifest(manifest_path)
    validate_subject_splits(rows)
    image_index = build_image_index()
    output_rows: list[dict[str, Any]] = []
    successful_detections = 0
    started_at = time.perf_counter()
    landmarker = create_landmarker(num_hands=2)

    try:
        for row_number, row in enumerate(rows, start=1):
            image_path: Path | None = None
            detection_status = "error"
            reason = ""
            features: list[float | None] = [None] * len(FEATURE_COLUMNS)
            try:
                image_path = resolve_image_path(row, image_index)
                image_bgr = load_image(image_path)
                result = detect_hands(landmarker, image_bgr)
                hand_index = select_hand_index(result)
                if hand_index is None:
                    detection_status = "no_hand"
                    reason = "no_hand_detected"
                else:
                    features = normalize_landmarks(result.hand_landmarks[hand_index])
                    detection_status = "success"
                    successful_detections += 1
            except Exception as error:
                detection_status = "error"
                reason = f"{type(error).__name__}: {error}"

            output_row: dict[str, Any] = {
                "image_id": row["image_id"],
                "image_path": str(image_path) if image_path is not None else f"image_id:{row['image_id']}",
                "label": row["label"],
                "user_id": row["user_id"],
                "split": row["split"],
                "detection_status": detection_status,
                "reason": reason,
            }
            output_row.update(dict(zip(FEATURE_COLUMNS, features)))
            output_rows.append(output_row)

            if row_number % 100 == 0 or row_number == len(rows):
                elapsed = time.perf_counter() - started_at
                print(f"Extracted {row_number:,}/{len(rows):,} images ({elapsed / 60:.1f} min)")
    finally:
        landmarker.close()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(output_rows).select(OUTPUT_COLUMNS)
    pq.write_table(table, LANDMARKS_PATH, compression="snappy")
    failed = len(output_rows) - successful_detections
    print(f"Landmark rows: {len(output_rows):,}; detections: {successful_detections:,}; failures/no-hand: {failed:,}")
    print(f"Landmark features per row: {len(FEATURE_COLUMNS)}")
    print(f"Parquet output: {LANDMARKS_PATH}")
    return successful_detections, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Existing split manifest CSV")
    arguments = parser.parse_args()
    extract_landmarks(arguments.manifest)


if __name__ == "__main__":
    main()