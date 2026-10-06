"""Shared paths and helpers for TouchFree Phase 2 preprocessing."""

import csv
import random
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "touchfree_split_manifest.csv"
IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hagrid_extracted"
    / "hagrid-sample-30k-384p"
    / "hagrid_30k"
)
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VISUAL_CHECKS_DIR = PROCESSED_DIR / "visual_checks"

CLASS_NAMES = ("dislike", "fist", "like", "no_gesture", "one", "palm", "peace")
SPLIT_NAMES = ("train", "val", "test")
IMAGE_SIZE = 224
DEFAULT_PADDING = 0.20
FEATURE_COLUMNS = tuple(
    f"landmark_{landmark_index:02d}_{axis}"
    for landmark_index in range(21)
    for axis in ("x", "y", "z")
)
MANIFEST_FIELDS = ("image_id", "label", "user_id", "split")


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    """Load and validate the existing split manifest without changing it."""
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        columns = set(reader.fieldnames or ())
        missing_columns = set(MANIFEST_FIELDS) - columns
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]

    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")

    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        if not row["image_id"] or not row["user_id"]:
            raise ValueError(f"Manifest row {row_number} has a missing image_id or user_id")
        if row["image_id"] in seen_ids:
            raise ValueError(f"Duplicate image_id in manifest: {row['image_id']}")
        seen_ids.add(row["image_id"])
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unexpected class in manifest row {row_number}: {row['label']!r}")
        if row["split"] not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split in manifest row {row_number}: {row['split']!r}")

    return rows


def validate_subject_splits(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """Return user sets and fail if a subject occurs in more than one split."""
    users_by_split = {
        split: {row["user_id"] for row in rows if row["split"] == split}
        for split in SPLIT_NAMES
    }
    overlaps = {
        "train_val": users_by_split["train"] & users_by_split["val"],
        "train_test": users_by_split["train"] & users_by_split["test"],
        "val_test": users_by_split["val"] & users_by_split["test"],
    }
    nonempty = {name: sorted(users) for name, users in overlaps.items() if users}
    if nonempty:
        raise ValueError(f"Subject overlap found in the existing split manifest: {nonempty}")
    return users_by_split


def build_image_index(image_root: Path = IMAGE_ROOT) -> dict[str, list[Path]]:
    """Index JPGs by UUID; no_gesture images can live under another class folder."""
    if not image_root.is_dir():
        raise FileNotFoundError(f"Extracted image directory not found: {image_root}")

    image_index: dict[str, list[Path]] = {}
    for image_path in image_root.rglob("*.jpg"):
        image_index.setdefault(image_path.stem.lower(), []).append(image_path)
    return image_index


def resolve_image_path(
    row: dict[str, str],
    image_index: dict[str, list[Path]],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a manifest path when present, otherwise resolve its UUID to a JPG."""
    supplied_path = row.get("image_path", "").strip()
    if supplied_path:
        path = Path(supplied_path)
        candidates = [path] if path.is_absolute() else [project_root / path, IMAGE_ROOT / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Manifest image_path does not exist: {supplied_path}")

    matches = image_index.get(row["image_id"].lower(), [])
    if not matches:
        raise FileNotFoundError(f"No JPG found for image_id {row['image_id']}")
    if len(matches) != 1:
        raise ValueError(f"Expected one JPG for {row['image_id']}, found {len(matches)}")
    return matches[0].resolve()


def load_image(image_path: Path) -> np.ndarray:
    """Load a color image with OpenCV and reject unreadable files."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Image is missing or cannot be decoded: {image_path}")
    return image


def create_landmarker(num_hands: int = 2) -> Any:
    """Create one reusable MediaPipe Tasks Hand Landmarker in IMAGE mode."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Official Hand Landmarker task asset not found: {MODEL_PATH}")

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker: Any, image_bgr: np.ndarray) -> Any:
    """Convert an OpenCV BGR frame to SRGB and run IMAGE-mode detection."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return landmarker.detect(mp_image)


def select_hand_index(result: Any) -> int | None:
    """Select the detected hand with the highest handedness confidence."""
    if not result.hand_landmarks:
        return None

    def confidence(index: int) -> float:
        categories = result.handedness[index] if index < len(result.handedness) else []
        return float(categories[0].score) if categories else 0.0

    return max(range(len(result.hand_landmarks)), key=confidence)


def hand_bounding_box(
    hand_landmarks: list[Any],
    image_width: int,
    image_height: int,
    padding: float = DEFAULT_PADDING,
) -> tuple[int, int, int, int]:
    """Build a padded pixel box from normalized hand landmark extents."""
    if padding < 0:
        raise ValueError("Padding must be non-negative")

    x_values = [float(landmark.x) for landmark in hand_landmarks]
    y_values = [float(landmark.y) for landmark in hand_landmarks]
    x1 = int(np.floor(min(x_values) * image_width))
    y1 = int(np.floor(min(y_values) * image_height))
    x2 = int(np.ceil(max(x_values) * image_width))
    y2 = int(np.ceil(max(y_values) * image_height))
    padding_x = int(round((x2 - x1) * padding))
    padding_y = int(round((y2 - y1) * padding))
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(image_width, x2 + padding_x)
    y2 = min(image_height, y2 + padding_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detected hand produced an empty bounding box")
    return x1, y1, x2, y2


def crop_hand(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a hand box, using half-open pixel coordinates."""
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Hand crop is empty")
    return crop


def resize_image(image_bgr: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize a hand crop to a square image suitable for later CNN use."""
    interpolation = cv2.INTER_AREA if image_bgr.shape[0] > image_size else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (image_size, image_size), interpolation=interpolation)


def normalize_image(image_bgr: np.ndarray) -> np.ndarray:
    """Return RGB float32 image data scaled to [0, 1] for CNN input."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def normalize_landmarks(hand_landmarks: list[Any]) -> list[float]:
    """Make 21 XYZ points wrist-relative and scale by wrist-to-middle-MCP length."""
    if len(hand_landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(hand_landmarks)}")

    coordinates = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype=np.float32,
    )
    # Wrist origin removes translation; wrist-to-middle-MCP palm length removes most scale variation.
    relative = coordinates - coordinates[0]
    scale = float(np.linalg.norm(relative[9]))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.max(np.linalg.norm(relative[1:], axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Landmark hand-size scale is zero or invalid")
    return (relative / scale).reshape(-1).astype(np.float32).tolist()


def sample_visual_rows(rows: list[dict[str, str]], per_class: int = 2, seed: int = 2026) -> set[str]:
    """Choose deterministic image IDs for manual visual review."""
    generator = random.Random(seed)
    selected: set[str] = set()
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["label"] == class_name]
        selected.update(
            row["image_id"] for row in generator.sample(class_rows, min(per_class, len(class_rows)))
        )
    return selected