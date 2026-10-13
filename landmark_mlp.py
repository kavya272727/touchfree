"""Train and evaluate a lightweight MLP on existing Phase 2 landmarks."""

import argparse
import csv
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "touchfree_split_manifest.csv"
LANDMARKS_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_phase2.parquet"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
PHASE5_DIR = PROJECT_ROOT / "data" / "processed" / "phase5"
BEST_MODEL_PATH = MODELS_DIR / "landmark_mlp_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "landmark_mlp_final.keras"
HISTORY_PATH = PHASE5_DIR / "training_history.csv"
REPORT_PATH = PHASE5_DIR / "classification_report.txt"
CONFUSION_CSV_PATH = PHASE5_DIR / "confusion_matrix.csv"
CONFUSION_PNG_PATH = PHASE5_DIR / "confusion_matrix.png"
ACCURACY_PNG_PATH = PHASE5_DIR / "accuracy_curve.png"
LOSS_PNG_PATH = PHASE5_DIR / "loss_curve.png"
SUMMARY_PATH = PHASE5_DIR / "phase5_summary.txt"

CLASS_NAMES = ("dislike", "fist", "like", "no_gesture", "one", "palm", "peace")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "val", "test")
FEATURE_COLUMNS = tuple(
    f"landmark_{landmark_index:02d}_{axis}"
    for landmark_index in range(21)
    for axis in ("x", "y", "z")
)
METADATA_COLUMNS = ("image_id", "label", "user_id", "split", "detection_status")
SEED = 2026
BATCH_SIZE = 64
MAX_EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8


def _read_manifest() -> list[dict[str, str]]:
    with SPLIT_MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _read_phase2_metadata() -> list[dict[str, Any]]:
    parquet_file = pq.ParquetFile(LANDMARKS_PATH)
    columns = parquet_file.schema.names
    if not set(METADATA_COLUMNS).issubset(columns):
        raise ValueError(f"Phase 2 landmark metadata columns are missing: {columns}")
    if not set(FEATURE_COLUMNS).issubset(columns):
        raise ValueError("Phase 2 Parquet does not contain all expected 63 normalized landmark features")
    return parquet_file.read(columns=list(METADATA_COLUMNS)).to_pylist()


def _validate_fixed_split(
    manifest_rows: list[dict[str, str]],
    parquet_metadata: list[dict[str, Any]],
) -> dict[str, set[str]]:
    if len(manifest_rows) != len(parquet_metadata):
        raise ValueError(f"Manifest/Parquet row count differs: {len(manifest_rows)} vs {len(parquet_metadata)}")
    manifest_ids = [row["image_id"] for row in manifest_rows]
    parquet_ids = [row["image_id"] for row in parquet_metadata]
    if len(set(manifest_ids)) != len(manifest_ids) or manifest_ids != parquet_ids:
        raise ValueError("Phase 2 landmark rows do not exactly match the fixed manifest ID order")

    users_by_split = {split: set() for split in SPLITS}
    for index, (manifest_row, landmark_row) in enumerate(zip(manifest_rows, parquet_metadata)):
        for field in ("label", "user_id", "split"):
            if manifest_row[field] != landmark_row[field]:
                raise ValueError(f"{field} mismatch at manifest row {index}: {manifest_row['image_id']}")
        if manifest_row["label"] not in CLASS_TO_ID:
            raise ValueError(f"Unexpected class: {manifest_row['label']}")
        if manifest_row["split"] not in SPLITS:
            raise ValueError(f"Unexpected split: {manifest_row['split']}")
        users_by_split[manifest_row["split"]].add(manifest_row["user_id"])

    overlaps = {
        "train/val": users_by_split["train"] & users_by_split["val"],
        "train/test": users_by_split["train"] & users_by_split["test"],
        "val/test": users_by_split["val"] & users_by_split["test"],
    }
    overlapping_users = {name: sorted(users) for name, users in overlaps.items() if users}
    if overlapping_users:
        raise ValueError(f"Subject leakage detected: {overlapping_users}")
    return users_by_split


def _load_split_features(
    metadata_rows: list[dict[str, Any]],
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load only one split's successful feature rows from Parquet."""
    selected_metadata = [
        row for row in metadata_rows if row["split"] == split and row["detection_status"] == "success"
    ]
    if not selected_metadata:
        raise ValueError(f"No successful landmark rows for split {split}")

    # The filter prevents train/validation feature loading from touching test feature columns.
    table = pq.read_table(
        LANDMARKS_PATH,
        columns=["image_id", *FEATURE_COLUMNS],
        filters=[("split", "=", split), ("detection_status", "=", "success")],
    )
    loaded_ids = table["image_id"].to_pylist()
    expected_ids = [row["image_id"] for row in selected_metadata]
    if loaded_ids != expected_ids:
        raise ValueError(f"Filtered {split} Parquet IDs do not match metadata order")

    features = np.column_stack(
        [table[column].to_numpy(zero_copy_only=False) for column in FEATURE_COLUMNS]
    ).astype(np.float32, copy=False)
    if features.shape != (len(selected_metadata), 63):
        raise ValueError(f"Unexpected {split} feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite landmark feature in {split} split")

    labels = np.fromiter(
        (CLASS_TO_ID[row["label"]] for row in selected_metadata),
        dtype=np.int32,
        count=len(selected_metadata),
    )
    return features, labels, expected_ids


def _class_weights(train_labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing one or more classes: {counts.tolist()}")
    total = int(counts.sum())
    return {index: total / (len(CLASS_NAMES) * int(count)) for index, count in enumerate(counts)}


def _make_dataset(features: np.ndarray, labels: np.ndarray, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(1)


def build_model() -> tf.keras.Model:
    """Create a small 63-feature MLP with a seven-class softmax output."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,), name="normalized_hand_landmarks"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
            tf.keras.layers.Dropout(0.30, name="dropout_030"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_64"),
            tf.keras.layers.Dropout(0.20, name="dropout_020"),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities"),
        ],
        name="touchfree_landmark_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(true_positive, confusion.sum(axis=0), out=np.zeros_like(true_positive), where=confusion.sum(axis=0) != 0)
    recall = np.divide(true_positive, confusion.sum(axis=1), out=np.zeros_like(true_positive), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall != 0)
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(confusion[i].sum())}
        for i, name in enumerate(CLASS_NAMES)
    }
    return confusion, per_class, float(precision.mean()), float(recall.mean()), float(f1.mean())


class BestValidationCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_loss = float("inf")
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        validation_loss = float((logs or {}).get("val_loss", float("inf")))
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_epoch = epoch + 1
            self.model.save(BEST_MODEL_PATH)


def _evaluate_test_once(features: np.ndarray, labels: np.ndarray):
    """This is the sole loader of test landmark values, called after checkpoint selection."""
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    probabilities = model(features, training=False).numpy()
    losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities).numpy()
    predicted = np.argmax(probabilities, axis=1).astype(np.int32)
    confusion, per_class, macro_precision, macro_recall, macro_f1 = _metrics(labels, predicted)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    return model, float(losses.mean()), accuracy, macro_precision, macro_recall, macro_f1, per_class, confusion


def _save_history(history: tf.keras.callbacks.History) -> None:
    keys = ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate")
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("epoch", *keys))
        for epoch in range(len(history.history["loss"])):
            writer.writerow((epoch + 1, *(history.history.get(key, [""] * (epoch + 1))[epoch] for key in keys)))


def _save_evaluation_outputs(
    confusion: np.ndarray,
    per_class: dict[str, dict[str, float]],
    accuracy: float,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
) -> None:
    with CONFUSION_CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("true\\predicted", *CLASS_NAMES))
        for class_name, row in zip(CLASS_NAMES, confusion):
            writer.writerow((class_name, *(int(value) for value in row)))

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class", title="Landmark MLP test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for r in range(7):
        for c in range(7):
            axis.text(c, r, str(int(confusion[r, c])), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(CONFUSION_PNG_PATH, dpi=150)
    plt.close(figure)

    report = ["Landmark MLP test classification report", "Class                 Precision   Recall   F1-score   Support"]
    for name in CLASS_NAMES:
        values = per_class[name]
        report.append(f"{name:<20} {values['precision']:.4f}      {values['recall']:.4f}   {values['f1']:.4f}      {values['support']}")
    report.extend((f"Accuracy: {accuracy:.4f}", f"Macro precision: {macro_precision:.4f}", f"Macro recall: {macro_recall:.4f}", f"Macro F1: {macro_f1:.4f}"))
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _save_curves(history: tf.keras.callbacks.History) -> None:
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    for output_path, train_key, val_key, title, ylabel in (
        (ACCURACY_PNG_PATH, "accuracy", "val_accuracy", "Landmark MLP accuracy", "Accuracy"),
        (LOSS_PNG_PATH, "loss", "val_loss", "Landmark MLP loss", "Sparse categorical cross-entropy"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, history.history[train_key], label="Training")
        axis.plot(epochs, history.history[val_key], label="Validation")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def run_sanity_only() -> None:
    manifest_rows = _read_manifest()
    parquet_metadata = _read_phase2_metadata()
    users_by_split = _validate_fixed_split(manifest_rows, parquet_metadata)
    x_train, y_train, _ = _load_split_features(parquet_metadata, "train")
    x_val, y_val, _ = _load_split_features(parquet_metadata, "val")
    model = build_model()
    train_output = model(x_train[: min(BATCH_SIZE, len(x_train))], training=False).numpy()
    assert train_output.shape == (min(BATCH_SIZE, len(x_train)), len(CLASS_NAMES))
    assert np.isfinite(train_output).all()
    print(f"Train features/labels: {x_train.shape} / {y_train.shape}; classes={sorted(set(y_train.tolist()))}")
    print(f"Validation features/labels: {x_val.shape} / {y_val.shape}; classes={sorted(set(y_val.tolist()))}")
    print(f"Forward-pass output: {train_output.shape}; finite=True; class mapping={CLASS_TO_ID}")
    print(f"User split counts: { {split: len(users) for split, users in users_by_split.items()} }; overlap=none")
    print("Test feature values were not loaded.")


def train_and_evaluate() -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest()
    parquet_metadata = _read_phase2_metadata()
    users_by_split = _validate_fixed_split(manifest_rows, parquet_metadata)

    # Load only train and validation numerical features before best-epoch selection.
    x_train, y_train, train_ids = _load_split_features(parquet_metadata, "train")
    x_val, y_val, val_ids = _load_split_features(parquet_metadata, "val")
    assert set(train_ids).isdisjoint(val_ids)
    if set(users_by_split["train"]) & set(users_by_split["val"]):
        raise ValueError("Train/validation subject leakage")

    for split_name, features, labels in (("train", x_train, y_train), ("validation", x_val, y_val)):
        if features.shape[1] != 63 or not np.isfinite(features).all():
            raise ValueError(f"Invalid {split_name} landmark feature matrix: {features.shape}")
        if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
            raise ValueError(f"Out-of-range {split_name} labels")
        print(f"{split_name}: X={features.shape}, y={labels.shape}, classes={sorted(set(labels.tolist()))}")

    class_counts = np.bincount(y_train, minlength=7)
    class_weights = {index: len(y_train) / (7 * int(count)) for index, count in enumerate(class_counts)}
    train_dataset = _make_dataset(x_train, y_train, training=True)
    validation_dataset = _make_dataset(x_val, y_val, training=False)
    model = build_model()
    print(f"MLP parameters: {model.count_params():,}; output shape: {model.output_shape}")

    checkpoint = BestValidationCheckpoint()
    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            checkpoint,
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        ],
        verbose=1,
    )
    training_seconds = time.perf_counter() - started
    _save_history(history)
    _save_curves(history)
    model.save(FINAL_MODEL_PATH)

    # Test feature columns are loaded only after training and validation-based selection finish.
    x_test, y_test, test_ids = _load_split_features(parquet_metadata, "test")
    if set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("Test image IDs overlap train/validation")
    if users_by_split["train"] & users_by_split["test"] or users_by_split["val"] & users_by_split["test"]:
        raise ValueError("Subject overlap involving test split")

    best_model, test_loss, test_accuracy, macro_precision, macro_recall, macro_f1, per_class, confusion = _evaluate_test_once(x_test, y_test)
    _save_evaluation_outputs(confusion, per_class, test_accuracy, macro_precision, macro_recall, macro_f1)
    inference_batch = x_test[: min(BATCH_SIZE, len(x_test))]
    best_model(inference_batch, training=False).numpy()
    inference_started = time.perf_counter()
    for _ in range(5):
        best_model(inference_batch, training=False).numpy()
    inference_seconds = (time.perf_counter() - inference_started) / 5

    summary = [
        "TouchFree Phase 5 Landmark MLP Summary",
        f"Python: {platform.python_version()}; TensorFlow: {tf.__version__}; NumPy: {np.__version__}; PyArrow: {pq.__version__ if hasattr(pq, '__version__') else 'installed'}",
        f"Feature source: {LANDMARKS_PATH}",
        f"Input features: {len(FEATURE_COLUMNS)} normalized XYZ landmark values",
        f"Valid detections used: {len(parquet_metadata) - sum(row['detection_status'] != 'success' for row in parquet_metadata)}; no-hand rows excluded: {sum(row['detection_status'] != 'success' for row in parquet_metadata)}",
        f"Class order/IDs: {CLASS_TO_ID}",
        f"Train/validation/test: {len(y_train)}/{len(y_val)}/{len(y_test)}",
        f"Users train/validation/test: {len(users_by_split['train'])}/{len(users_by_split['val'])}/{len(users_by_split['test'])}; overlap: none",
        f"Model: Dense(128, ReLU) -> Dropout(0.30) -> Dense(64, ReLU) -> Dropout(0.20) -> Dense(7, softmax); parameters={best_model.count_params():,}",
        f"Training: max_epochs={MAX_EPOCHS}; completed={len(history.history['loss'])}; best_epoch={checkpoint.best_epoch}; batch={BATCH_SIZE}; lr={LEARNING_RATE}; seed={SEED}; early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        f"Training-only class weights: {class_weights}",
        f"Training time: {training_seconds:.2f} seconds",
        f"Best validation loss: {checkpoint.best_loss:.4f}",
        f"Test loss={test_loss:.4f}; accuracy={test_accuracy:.4f}; macro_precision={macro_precision:.4f}; macro_recall={macro_recall:.4f}; macro_F1={macro_f1:.4f}",
        f"Inference time: {inference_seconds:.6f} seconds per batch of {len(inference_batch)} landmark rows",
        f"Best model: {BEST_MODEL_PATH.name} ({BEST_MODEL_PATH.stat().st_size / (1024**2):.2f} MiB)",
        f"Final model: {FINAL_MODEL_PATH.name} ({FINAL_MODEL_PATH.stat().st_size / (1024**2):.2f} MiB)",
        "Phase 3 CNN: accuracy=0.8215, macro F1=0.8310",
        "Phase 4 MobileNetV2: accuracy=0.8263, macro F1=0.8409",
        "Test data was loaded only after best-epoch selection and evaluated once.",
        "",
        "Per-class test metrics:",
    ]
    for name in CLASS_NAMES:
        values = per_class[name]
        summary.append(f"  {name}: precision={values['precision']:.4f}, recall={values['recall']:.4f}, f1={values['f1']:.4f}, support={values['support']}")
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-only", action="store_true", help="Validate train/validation landmarks and model output without training")
    arguments = parser.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    if arguments.sanity_only:
        run_sanity_only()
    else:
        train_and_evaluate()


if __name__ == "__main__":
    main()