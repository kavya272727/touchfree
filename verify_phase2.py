"""Validate Phase 2 image, landmark, manifest, and split outputs."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from landmarks import LANDMARKS_PATH
from preprocessing import FAILURES_PATH, PREPROCESSED_IMAGES_PATH, PREPROCESSING_MANIFEST_PATH
from phase2_utils import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    MANIFEST_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    VISUAL_CHECKS_DIR,
    build_image_index,
    load_image,
    load_manifest,
    resolve_image_path,
    sample_visual_rows,
    validate_subject_splits,
)


SUMMARY_PATH = PROCESSED_DIR / "phase2_summary.txt"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected Phase 2 output is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def verify_phase2(manifest_path: Path = MANIFEST_PATH) -> tuple[str, bool]:
    manifest_rows = load_manifest(manifest_path)
    users_by_split = validate_subject_splits(manifest_rows)
    image_index = build_image_index()
    expected_by_id = {row["image_id"]: row for row in manifest_rows}

    source_paths: dict[str, str] = {}
    source_missing: list[str] = []
    source_corrupt: list[str] = []
    for row in manifest_rows:
        try:
            path = resolve_image_path(row, image_index)
            source_paths[row["image_id"]] = str(path)
            load_image(path)
        except FileNotFoundError as error:
            source_missing.append(f"{row['image_id']}: {error}")
        except Exception as error:
            source_corrupt.append(f"{row['image_id']}: {type(error).__name__}: {error}")

    path_occurrences: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        path = source_paths.get(row["image_id"])
        if path:
            path_occurrences[path.casefold()].append(row["split"])
    duplicate_paths = {path: splits for path, splits in path_occurrences.items() if len(splits) > 1}
    duplicate_paths_across_splits = {
        path: splits for path, splits in duplicate_paths.items() if len(set(splits)) > 1
    }

    preprocessing_rows = read_csv(PREPROCESSING_MANIFEST_PATH)
    failure_rows = read_csv(FAILURES_PATH)
    landmark_table = pq.read_table(LANDMARKS_PATH)
    landmark_rows = landmark_table.to_pylist()
    landmark_ids = [row.get("image_id", "") for row in landmark_rows]
    duplicate_landmark_ids = [image_id for image_id, count in Counter(landmark_ids).items() if count > 1]
    missing_landmark_ids = sorted(set(expected_by_id) - set(landmark_ids))
    unexpected_landmark_ids = sorted(set(landmark_ids) - set(expected_by_id))
    metadata_mismatches: list[str] = []

    preprocessing_by_id = {row["image_id"]: row for row in preprocessing_rows}
    preprocessing_mismatches = sorted(set(expected_by_id) ^ set(preprocessing_by_id))
    for image_id, output_row in preprocessing_by_id.items():
        source_row = expected_by_id.get(image_id)
        if source_row and any(output_row.get(field) != source_row[field] for field in ("label", "user_id", "split")):
            metadata_mismatches.append(f"preprocessing metadata mismatch: {image_id}")

    landmark_by_id = {row.get("image_id", ""): row for row in landmark_rows}
    for image_id, output_row in landmark_by_id.items():
        source_row = expected_by_id.get(image_id)
        if source_row and any(output_row.get(field) != source_row[field] for field in ("label", "user_id", "split")):
            metadata_mismatches.append(f"landmark metadata mismatch: {image_id}")
        if source_row and output_row.get("image_path") != source_paths.get(image_id):
            metadata_mismatches.append(f"landmark image_path mismatch: {image_id}")

    feature_columns = [column for column in FEATURE_COLUMNS if column in landmark_table.column_names]
    processed_success = [row for row in preprocessing_rows if row.get("detection_status") == "success"]
    preprocessing_failures = [row for row in preprocessing_rows if row.get("detection_status") != "success"]
    failure_log_ids = {row.get("image_id", "") for row in failure_rows}
    preprocessing_failure_ids = {row.get("image_id", "") for row in preprocessing_failures}

    array_shape_error = ""
    normalized_image_sample_error = ""
    if not PREPROCESSED_IMAGES_PATH.is_file():
        array_shape_error = f"Normalized image array missing: {PREPROCESSED_IMAGES_PATH}"
    else:
        normalized_images = np.load(PREPROCESSED_IMAGES_PATH, mmap_mode="r")
        if normalized_images.shape != (len(manifest_rows), 224, 224, 3):
            array_shape_error = f"Unexpected normalized image array shape: {normalized_images.shape}"
        elif normalized_images.dtype != np.float32:
            array_shape_error = f"Unexpected normalized image dtype: {normalized_images.dtype}"
        else:
            for output_row in processed_success[:20]:
                sample = normalized_images[int(output_row["processed_row"])]
                if not np.isfinite(sample).all() or float(sample.min()) < 0 or float(sample.max()) > 1:
                    normalized_image_sample_error = f"Invalid normalized sample: {output_row['image_id']}"
                    break
        del normalized_images

    class_counts = Counter(row["label"] for row in manifest_rows)
    preprocessing_success_by_class = Counter(row["label"] for row in processed_success)
    landmark_success_by_class = Counter(
        row["label"] for row in landmark_rows if row.get("detection_status") == "success"
    )
    split_counts = Counter(row["split"] for row in manifest_rows)
    preprocessing_by_split = Counter(row["split"] for row in processed_success)
    landmark_by_split = Counter(row["split"] for row in landmark_rows if row.get("detection_status") == "success")
    landmark_detection_failures = sum(row.get("detection_status") != "success" for row in landmark_rows)

    user_overlaps = {
        "train ∩ val": users_by_split["train"] & users_by_split["val"],
        "train ∩ test": users_by_split["train"] & users_by_split["test"],
        "val ∩ test": users_by_split["val"] & users_by_split["test"],
    }
    selected_visual_ids = sample_visual_rows(manifest_rows)
    expected_visual_paths = [
        VISUAL_CHECKS_DIR / f"{row['split']}_{row['label']}_{row['image_id']}.jpg"
        for row in manifest_rows
        if row["image_id"] in selected_visual_ids
    ]
    visual_check_count = sum(path.is_file() for path in expected_visual_paths)
    all_visual_file_count = len(list(VISUAL_CHECKS_DIR.glob("*.jpg"))) if VISUAL_CHECKS_DIR.is_dir() else 0
    preexisting_visual_file_count = max(0, all_visual_file_count - visual_check_count)

    lines = [
        "TouchFree Phase 2 Verification",
        f"MediaPipe version: {__import__('mediapipe').__version__}",
        f"Model asset present: {MODEL_PATH.is_file()} ({MODEL_PATH})",
        f"Total manifest images: {len(manifest_rows)}",
        f"Preprocessing successful images: {len(processed_success)}",
        f"Preprocessing failures/no-hand: {len(preprocessing_failures)}",
        f"Preprocessing detection rate: {100 * len(processed_success) / len(manifest_rows):.2f}%",
        f"Landmark rows: {len(landmark_rows)}",
        f"Landmark successful detections: {len(landmark_rows) - landmark_detection_failures}",
        f"Landmark failed/no-hand rows: {landmark_detection_failures}",
        f"Landmark features per row: {len(feature_columns)}",
        f"Missing source images: {len(source_missing)}",
        f"Corrupt/unreadable source images: {len(source_corrupt)}",
        f"Preprocessing failure log rows: {len(failure_rows)}",
        f"Duplicate source paths: {len(duplicate_paths)}",
        f"Duplicate source paths crossing splits: {len(duplicate_paths_across_splits)}",
        f"Duplicate landmark image IDs: {len(duplicate_landmark_ids)}",
        f"Missing landmark image IDs: {len(missing_landmark_ids)}",
        f"Unexpected landmark image IDs: {len(unexpected_landmark_ids)}",
        f"Phase 2 visual check images: {visual_check_count}",
        f"Other pre-existing visual check images preserved: {preexisting_visual_file_count}",
        "",
        "Class counts (manifest / preprocessed / landmark detections):",
    ]
    for class_name in CLASS_NAMES:
        lines.append(
            f"  {class_name}: {class_counts[class_name]} / "
            f"{preprocessing_success_by_class[class_name]} / {landmark_success_by_class[class_name]}"
        )
    lines.extend(("", "Split counts (manifest / preprocessed / landmark detections):"))
    for split in ("train", "val", "test"):
        lines.append(
            f"  {split}: {split_counts[split]} / {preprocessing_by_split[split]} / {landmark_by_split[split]} "
            f"({len(users_by_split[split])} users)"
        )
    lines.append("Subject overlap:")
    for name, overlapping_users in user_overlaps.items():
        lines.append(f"  {name}: {len(overlapping_users)}")
        if overlapping_users:
            lines.append(f"    IDs: {', '.join(sorted(overlapping_users))}")

    issues: list[str] = []
    if any(user_overlaps.values()):
        issues.append("subject overlap exists")
    if source_missing or source_corrupt:
        issues.append("source image missing or corrupt")
    if duplicate_paths or duplicate_landmark_ids:
        issues.append("duplicate image paths or landmark IDs exist")
    if preprocessing_mismatches or missing_landmark_ids or unexpected_landmark_ids:
        issues.append("output image IDs do not match the existing manifest")
    if metadata_mismatches:
        issues.append("output metadata does not match the manifest")
    if len(feature_columns) != 63 or len(landmark_table) != len(manifest_rows):
        issues.append("landmark row or feature count is incorrect")
    if failure_log_ids != preprocessing_failure_ids:
        issues.append("preprocessing failure log does not match failure statuses")
    if array_shape_error or normalized_image_sample_error:
        issues.append("normalized image output is invalid")
    if visual_check_count != len(expected_visual_paths):
        issues.append("one or more expected Phase 2 visual-check images are missing")

    lines.extend(("", f"Verification: {'FAIL' if issues else 'PASS'}"))
    if issues:
        lines.extend(f"  - {issue}" for issue in issues)
    for detail in (array_shape_error, normalized_image_sample_error):
        if detail:
            lines.append(f"  - {detail}")
    if source_missing:
        lines.extend(("Missing images (first 10):", *[f"  {item}" for item in source_missing[:10]]))
    if source_corrupt:
        lines.extend(("Corrupt images (first 10):", *[f"  {item}" for item in source_corrupt[:10]]))

    report = "\n".join(lines) + "\n"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Summary saved to: {SUMMARY_PATH}")
    return report, not issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Existing split manifest CSV")
    arguments = parser.parse_args()
    _, passed = verify_phase2(arguments.manifest)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()