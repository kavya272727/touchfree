"""Train and evaluate the Phase 3 baseline CNN on the fixed TouchFree split."""

import argparse
import csv
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import (
    CLASS_NAMES,
    CLASS_TO_ID,
    PHASE3_DIR,
    DatasetMetadata,
    class_weights_from_train,
    inspect_phase2_data,
    make_tf_datasets,
    print_dataset_summary,
)


MODELS_DIR = Path(__file__).resolve().parent / "data" / "models"
BEST_MODEL_PATH = MODELS_DIR / "baseline_cnn_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "baseline_cnn.keras"
HISTORY_PATH = PHASE3_DIR / "training_history.csv"
CLASSIFICATION_REPORT_PATH = PHASE3_DIR / "classification_report.txt"
CONFUSION_MATRIX_CSV_PATH = PHASE3_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_PATH = PHASE3_DIR / "confusion_matrix.png"
ACCURACY_CURVE_PATH = PHASE3_DIR / "accuracy_curve.png"
LOSS_CURVE_PATH = PHASE3_DIR / "loss_curve.png"
SUMMARY_PATH = PHASE3_DIR / "phase3_summary.txt"

SEED = 2026
BATCH_SIZE = 32
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3


def build_model(learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Build the requested simple three-block baseline CNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3), name="image"),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv_32"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_1"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv_64"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_2"),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv_128"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_3"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
            tf.keras.layers.Dropout(0.5, name="dropout_0_5"),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities"),
        ],
        name="touchfree_baseline_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History) -> None:
    keys = list(history.history)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(("epoch", *keys))
        for epoch_index in range(len(history.history.get("loss", []))):
            writer.writerow((epoch_index + 1, *(history.history[key][epoch_index] for key in keys)))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]], float, float]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = _safe_ratio(true_positive, confusion.sum(axis=0))
    recall = _safe_ratio(true_positive, confusion.sum(axis=1))
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    per_class = {
        class_name: {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(confusion[class_id].sum()),
        }
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    accuracy = float(true_positive.sum() / max(1, confusion.sum()))
    macro_f1 = float(f1.mean())
    return confusion, per_class, accuracy, macro_f1


def save_confusion_matrix(confusion: np.ndarray) -> None:
    with CONFUSION_MATRIX_CSV_PATH.open("w", newline="", encoding="utf-8") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(("true\\predicted", *CLASS_NAMES))
        for class_name, row in zip(CLASS_NAMES, confusion):
            writer.writerow((class_name, *(int(value) for value in row)))

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Baseline CNN test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = confusion.max() / 2 if confusion.size else 0
    for row_index in range(len(CLASS_NAMES)):
        for column_index in range(len(CLASS_NAMES)):
            value = int(confusion[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=150)
    plt.close(figure)


def save_training_curves(history: tf.keras.callbacks.History) -> None:
    epochs = np.arange(1, len(history.history.get("loss", [])) + 1)
    accuracy_key = "accuracy"
    validation_accuracy_key = "val_accuracy"

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history.history.get(accuracy_key, []), label="Training accuracy")
    axis.plot(epochs, history.history.get(validation_accuracy_key, []), label="Validation accuracy")
    axis.set(xlabel="Epoch", ylabel="Accuracy", title="Baseline CNN accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ACCURACY_CURVE_PATH, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history.history.get("loss", []), label="Training loss")
    axis.plot(epochs, history.history.get("val_loss", []), label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Sparse categorical cross-entropy", title="Baseline CNN loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(LOSS_CURVE_PATH, dpi=150)
    plt.close(figure)


def write_classification_report(per_class: dict[str, dict[str, float]], accuracy: float, macro_f1: float) -> None:
    lines = [
        "Baseline CNN test classification report",
        "Class                 Precision   Recall   F1-score   Support",
    ]
    for class_name in CLASS_NAMES:
        values = per_class[class_name]
        lines.append(
            f"{class_name:<20} {values['precision']:.4f}      {values['recall']:.4f}   "
            f"{values['f1']:.4f}      {values['support']}"
        )
    lines.extend((f"Accuracy: {accuracy:.4f}", f"Macro F1: {macro_f1:.4f}"))
    CLASSIFICATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    metadata: DatasetMetadata,
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    training_seconds: float,
    best_epoch: int,
    class_weights: dict[int, float],
    test_loss: float,
    test_accuracy: float,
    macro_f1: float,
    per_class: dict[str, dict[str, float]],
) -> None:
    best_epoch_index = best_epoch - 1
    train_accuracy = history.history["accuracy"][best_epoch_index]
    validation_accuracy = history.history["val_accuracy"][best_epoch_index]
    train_loss = history.history["loss"][best_epoch_index]
    validation_loss = history.history["val_loss"][best_epoch_index]
    physical_devices = ", ".join(device.device_type for device in tf.config.list_physical_devices())
    lines = [
        "TouchFree Phase 3 Baseline CNN Summary",
        f"Python: {platform.python_version()}",
        f"TensorFlow: {tf.__version__}",
        f"NumPy: {np.__version__}",
        f"Compute devices: {physical_devices or 'CPU'}",
        f"Total fixed manifest rows: {len(metadata.manifest_rows)}",
        f"Valid CNN images: {len(metadata.valid_rows)}",
        f"Excluded no-hand rows: {sum(metadata.excluded_failure_counts.values())}",
        f"Unique users: {metadata.unique_users}",
        f"Image shape/dtype/range: {metadata.array_shape} / {metadata.array_dtype} / [{metadata.pixel_min}, {metadata.pixel_max}]",
        f"Class order and IDs: {CLASS_TO_ID}",
        f"Manifest class counts: {dict(metadata.manifest_class_counts)}",
        f"Valid class counts: {dict(metadata.valid_class_counts)}",
        f"Valid split counts: {dict(metadata.valid_split_counts)}",
        "Split: fixed user-based train/val/test assignments from Phase 1; no re-splitting",
        f"User overlap: none ({ {split: len(users) for split, users in metadata.users_by_split.items()} })",
        "Architecture: Conv2D(32,3)-MaxPool; Conv2D(64,3)-MaxPool; Conv2D(128,3)-MaxPool; Flatten; Dense(128); Dropout(0.5); Dense(7, softmax)",
        f"Input size: {model.input_shape}",
        f"Trainable parameters: {model.count_params():,}",
        f"Batch size: {BATCH_SIZE}",
        f"Maximum epochs: {MAX_EPOCHS}",
        f"Completed epochs: {len(history.history['loss'])}",
        f"Best validation-loss epoch: {best_epoch}",
        f"Learning rate (initial): {LEARNING_RATE}",
        f"Random seed: {SEED}",
        "Augmentation: none",
        "Class weights: training-only balanced weights; " + ", ".join(
            f"{CLASS_NAMES[class_id]}={weight:.4f}" for class_id, weight in class_weights.items()
        ),
        f"Training time seconds: {training_seconds:.2f}",
        f"Best epoch train loss/accuracy: {train_loss:.4f} / {train_accuracy:.4f}",
        f"Best epoch validation loss/accuracy: {validation_loss:.4f} / {validation_accuracy:.4f}",
        f"Held-out test loss: {test_loss:.4f}",
        f"Held-out test accuracy: {test_accuracy:.4f}",
        f"Held-out test macro F1: {macro_f1:.4f}",
        "The test split is untouched until the best validation checkpoint is selected; it is then evaluated once and is not used for training or callbacks.",
        "",
        "Per-class test metrics:",
    ]
    for class_name in CLASS_NAMES:
        values = per_class[class_name]
        lines.append(
            f"  {class_name}: precision={values['precision']:.4f}, recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, support={values['support']}"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_baseline(batch_size: int = BATCH_SIZE, max_epochs: int = MAX_EPOCHS) -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    cpu_count = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(4, cpu_count))
    tf.config.threading.set_inter_op_parallelism_threads(2)

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = inspect_phase2_data()
    print_dataset_summary(metadata, batch_size=batch_size, seed=SEED)

    datasets = make_tf_datasets(metadata, batch_size=batch_size, seed=SEED)
    train_dataset = datasets["train"]
    validation_dataset = datasets["val"]
    test_dataset = datasets["test"]
    class_weights = class_weights_from_train(metadata)
    model = build_model()
    print(f"Model parameters: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    training_started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=math.ceil(metadata.valid_split_counts["train"] / batch_size),
        validation_steps=math.ceil(metadata.valid_split_counts["val"] / batch_size),
        epochs=max_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=False,
        verbose=1,
    )
    training_seconds = time.perf_counter() - training_started
    save_history(history)
    save_training_curves(history)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    model.save(FINAL_MODEL_PATH)

    # Make one held-out test pass for loss, accuracy, and class predictions together.
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    total_test_loss = 0.0
    for image_batch, label_batch in test_dataset:
        label_tensor = tf.cast(label_batch, tf.int32)
        probability_tensor = best_model(image_batch, training=False)
        batch_loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, probability_tensor)
        total_test_loss += float(tf.reduce_sum(batch_loss))
        label_values = label_tensor.numpy().astype(np.int64)
        true_labels.extend(label_values.tolist())
        probabilities = probability_tensor.numpy()
        predicted_labels.extend(np.argmax(probabilities, axis=1).astype(np.int64).tolist())

    confusion, per_class, _, macro_f1 = classification_metrics(
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(predicted_labels, dtype=np.int64),
    )
    test_loss = total_test_loss / max(1, len(true_labels))
    test_accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    save_confusion_matrix(confusion)
    write_classification_report(per_class, float(test_accuracy), macro_f1)
    _write_summary(
        metadata,
        best_model,
        history,
        training_seconds,
        best_epoch,
        class_weights,
        float(test_loss),
        float(test_accuracy),
        macro_f1,
        per_class,
    )
    print(f"Best validation-loss epoch: {best_epoch}")
    print(f"Test accuracy: {test_accuracy:.4f}; macro F1: {macro_f1:.4f}")
    print(f"Phase 3 report: {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    arguments = parser.parse_args()
    train_baseline(batch_size=arguments.batch_size, max_epochs=arguments.epochs)


if __name__ == "__main__":
    main()