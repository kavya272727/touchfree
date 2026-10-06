"""Phase 4 MobileNetV2 transfer learning for TouchFree's fixed seven-class split."""

import argparse
import csv
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

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
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
PHASE4_DIR = PROJECT_ROOT / "data" / "processed" / "phase4"
BEST_MODEL_PATH = MODELS_DIR / "mobilenetv2_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "mobilenetv2_final.keras"
HISTORY_PATH = PHASE4_DIR / "mobilenetv2_training_history.csv"
REPORT_PATH = PHASE4_DIR / "mobilenetv2_classification_report.txt"
CONFUSION_CSV_PATH = PHASE4_DIR / "mobilenetv2_confusion_matrix.csv"
CONFUSION_PNG_PATH = PHASE4_DIR / "mobilenetv2_confusion_matrix.png"
ACCURACY_PNG_PATH = PHASE4_DIR / "mobilenetv2_accuracy_curve.png"
LOSS_PNG_PATH = PHASE4_DIR / "mobilenetv2_loss_curve.png"
SUMMARY_PATH = PHASE4_DIR / "phase4_summary.txt"

SEED = 2026
BATCH_SIZE = 32
STAGE_A_EPOCHS = 8
STAGE_B_EPOCHS = 5
STAGE_A_LEARNING_RATE = 1e-3
STAGE_B_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 2
FINE_TUNE_LAST_LAYERS = 30
HEAD_DROPOUT = 0.2
RESUMED_STAGE_A_EPOCH_SECONDS = 117.0
RESUMED_STAGE_A_EPOCH_METRICS = {
    "loss": 1.0275,
    "accuracy": 0.6176,
    "val_loss": 0.86347,
    "val_accuracy": 0.7233,
    "learning_rate": STAGE_A_LEARNING_RATE,
}


def make_transfer_datasets(
    metadata: DatasetMetadata,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> dict[str, tf.keras.utils.PyDataset]:
    """Make finite-cardinality batches from the existing read-only image mmap."""
    return {
        split: MobileNetSplitDataset(metadata, split, batch_size, seed)
        for split in ("train", "val", "test")
    }


class MobileNetSplitDataset(tf.keras.utils.PyDataset):
    """Keras PyDataset with one-batch materialization and train-only reshuffling."""

    def __init__(self, metadata: DatasetMetadata, split: str, batch_size: int, seed: int):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.metadata = metadata
        self.split = split
        self.batch_size = batch_size
        self.image_indices = np.asarray(metadata.indices_by_split[split], dtype=np.int64)
        self.order = np.arange(len(self.image_indices), dtype=np.int64)
        self.rng = random.Random(seed)
        self.images = np.load(
            PROJECT_ROOT / "data" / "processed" / "preprocessed_images.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if split == "train":
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        start = batch_index * self.batch_size
        manifest_indices = self.image_indices[self.order[start : start + self.batch_size]]
        # Advanced indexing reads only this batch from the read-only memory map.
        images = np.asarray(self.images[manifest_indices], dtype=np.float32)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images * 255.0)
        labels = np.fromiter(
            (CLASS_TO_ID[self.metadata.manifest_rows[index]["label"]] for index in manifest_indices),
            dtype=np.int32,
            count=len(manifest_indices),
        )
        return images, labels

    def __iter__(self):
        for batch_index in range(len(self)):
            yield self[batch_index]

    def on_epoch_end(self) -> None:
        if self.split == "train":
            self.rng.shuffle(self.order)


def build_model(
    learning_rate: float = STAGE_A_LEARNING_RATE,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a MobileNetV2 ImageNet backbone and seven-class classification head."""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name="mobilenetv2_preprocessed_image")
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(HEAD_DROPOUT, name="head_dropout")(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="touchfree_mobilenetv2")
    compile_model(model, learning_rate)
    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def sanity_check(
    model: tf.keras.Model,
    metadata: DatasetMetadata,
    datasets: dict[str, tf.data.Dataset],
) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Load one train and validation batch and run one forward pass on train only."""
    train_batch = next(iter(datasets["train"]))
    validation_batch = next(iter(datasets["val"]))
    for split, (images, labels) in (("train", train_batch), ("validation", validation_batch)):
        tf.debugging.assert_equal(tf.shape(images)[1:], (224, 224, 3))
        tf.debugging.assert_all_finite(images, f"Non-finite {split} images")
        tf.debugging.assert_greater_equal(labels, 0)
        tf.debugging.assert_less(labels, len(CLASS_NAMES))
        print(
            f"{split} batch: image_shape={tuple(images.shape)}, label_shape={tuple(labels.shape)}, "
            f"class_ids={sorted(set(np.asarray(labels).tolist()))}, "
            f"preprocessed_range=[{float(tf.reduce_min(images)):.4f}, {float(tf.reduce_max(images)):.4f}]"
        )

    train_outputs = model(train_batch[0], training=False)
    tf.debugging.assert_equal(tf.shape(train_outputs), (tf.shape(train_batch[0])[0], len(CLASS_NAMES)))
    tf.debugging.assert_all_finite(train_outputs, "Non-finite MobileNetV2 output")
    tf.debugging.assert_near(tf.reduce_sum(train_outputs, axis=1), tf.ones(tf.shape(train_outputs)[0]), atol=1e-4)
    trainable_parameters = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
    total_parameters = int(model.count_params())
    print(f"MobileNetV2 output shape: {tuple(train_outputs.shape)}")
    print(f"Output classes: {len(CLASS_NAMES)} ({CLASS_TO_ID})")
    print(f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable (frozen-backbone stage)")
    print("Sanity check: PASS; test dataset was not iterated.")
    return train_batch, validation_batch


class BestValidationCheckpoint(tf.keras.callbacks.Callback):
    """Keep one best checkpoint across both transfer-learning stages."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_stage = ""
        self.global_epoch = 0
        self.current_stage = ""

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.global_epoch += 1

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        validation_loss = float((logs or {}).get("val_loss", float("inf")))
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_epoch = self.global_epoch
            self.best_stage = self.current_stage
            self.model.save(self.path)
            print(f"Saved global best {self.current_stage} checkpoint (val_loss={validation_loss:.5f})")


def callbacks_for_stage(
    checkpoint: BestValidationCheckpoint,
    stage: str,
) -> list[tf.keras.callbacks.Callback]:
    checkpoint.current_stage = stage
    return [
        checkpoint,
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
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def unfreeze_upper_layers(backbone: tf.keras.Model) -> int:
    """Unfreeze only the last portion of MobileNetV2, keeping BatchNorm frozen."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-FINE_TUNE_LAST_LAYERS:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return int(sum(np.prod(variable.shape) for variable in backbone.trainable_variables))


def _write_history(
    path: Path,
    histories: list[tuple[str, tf.keras.callbacks.History]],
    resumed_stage_a: bool = False,
) -> None:
    history_columns = ("stage", "stage_epoch", "global_epoch", "loss", "accuracy", "val_loss", "val_accuracy", "learning_rate")
    with path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history_columns)
        writer.writeheader()
        global_epoch = 0
        if resumed_stage_a:
            global_epoch = 1
            writer.writerow(
                {
                    "stage": "stage_a_frozen",
                    "stage_epoch": 1,
                    "global_epoch": 1,
                    **RESUMED_STAGE_A_EPOCH_METRICS,
                }
            )
        for stage, history in histories:
            count = len(history.history.get("loss", []))
            for stage_epoch in range(count):
                global_epoch += 1
                stage_epoch_offset = 1 if resumed_stage_a and stage == "stage_a_frozen" else 0
                row: dict[str, Any] = {
                    "stage": stage,
                    "stage_epoch": stage_epoch + 1 + stage_epoch_offset,
                    "global_epoch": global_epoch,
                }
                for key in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"):
                    values = history.history.get(key)
                    row[key] = values[stage_epoch] if values and stage_epoch < len(values) else ""
                writer.writerow(row)


def _classification_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(true_positive, confusion.sum(axis=0), out=np.zeros_like(true_positive), where=confusion.sum(axis=0) != 0)
    recall = np.divide(true_positive, confusion.sum(axis=1), out=np.zeros_like(true_positive), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(confusion[index].sum()),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return confusion, per_class, float(precision.mean()), float(recall.mean()), float(f1.mean())


def _save_evaluation(
    test_dataset: tf.data.Dataset,
    best_model: tf.keras.Model,
) -> tuple[float, float, float, float, float, dict[str, dict[str, float]], np.ndarray]:
    """Traverse the held-out test dataset once for predictions and loss."""
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    loss_total = 0.0
    for images, labels in test_dataset:
        labels = tf.cast(labels, tf.int32)
        probabilities = best_model(images, training=False)
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)
        loss_total += float(tf.reduce_sum(losses))
        true_labels.extend(labels.numpy().astype(np.int64).tolist())
        predicted_labels.extend(np.argmax(probabilities.numpy(), axis=1).astype(np.int64).tolist())
    true = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    confusion, per_class, macro_precision, macro_recall, macro_f1 = _classification_metrics(true, predicted)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    mean_loss = loss_total / max(1, len(true))
    return mean_loss, accuracy, macro_precision, macro_recall, macro_f1, per_class, confusion


def _save_outputs(
    histories: list[tuple[str, tf.keras.callbacks.History]],
    confusion: np.ndarray,
    per_class: dict[str, dict[str, float]],
    test_accuracy: float,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    resumed_stage_a: bool = False,
) -> None:
    _write_history(HISTORY_PATH, histories, resumed_stage_a=resumed_stage_a)
    with CONFUSION_CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("true\\predicted", *CLASS_NAMES))
        for name, row in zip(CLASS_NAMES, confusion):
            writer.writerow((name, *(int(value) for value in row)))

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class", title="MobileNetV2 test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row_index in range(len(CLASS_NAMES)):
        for column_index in range(len(CLASS_NAMES)):
            axis.text(column_index, row_index, str(int(confusion[row_index, column_index])), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(CONFUSION_PNG_PATH, dpi=150)
    plt.close(figure)

    combined: dict[str, list[float]] = {key: [] for key in ("accuracy", "val_accuracy", "loss", "val_loss")}
    for _, history in histories:
        for key in combined:
            combined[key].extend(history.history.get(key, []))
    if resumed_stage_a:
        for key in combined:
            combined[key].insert(0, RESUMED_STAGE_A_EPOCH_METRICS[key])
    epochs = np.arange(1, len(combined["loss"]) + 1)
    for path, train_key, validation_key, title, ylabel in (
        (ACCURACY_PNG_PATH, "accuracy", "val_accuracy", "MobileNetV2 accuracy", "Accuracy"),
        (LOSS_PNG_PATH, "loss", "val_loss", "MobileNetV2 loss", "Sparse categorical cross-entropy"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, combined[train_key], label="Training")
        axis.plot(epochs, combined[validation_key], label="Validation")
        axis.set(xlabel="Global epoch", ylabel=ylabel, title=title)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)

    report = [
        "MobileNetV2 test classification report",
        "Class                 Precision   Recall   F1-score   Support",
    ]
    for name in CLASS_NAMES:
        values = per_class[name]
        report.append(f"{name:<20} {values['precision']:.4f}      {values['recall']:.4f}   {values['f1']:.4f}      {values['support']}")
    report.extend((f"Accuracy: {test_accuracy:.4f}", f"Macro precision: {macro_precision:.4f}", f"Macro recall: {macro_recall:.4f}", f"Macro F1: {macro_f1:.4f}"))
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_summary(
    metadata: DatasetMetadata,
    model: tf.keras.Model,
    histories: list[tuple[str, tf.keras.callbacks.History]],
    checkpoint: BestValidationCheckpoint,
    stage_a_trainable_parameters: int,
    stage_b_trainable_parameters: int,
    fine_tune_backbone_parameters: int,
    training_seconds: float,
    test_loss: float,
    test_accuracy: float,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    per_class: dict[str, dict[str, float]],
    inference_seconds_per_batch: float,
    resumed_stage_a: bool = False,
) -> None:
    summary = [
        "TouchFree Phase 4 MobileNetV2 Summary",
        f"Python: {platform.python_version()}",
        f"TensorFlow: {tf.__version__}",
        f"NumPy: {np.__version__}",
        "Backbone: MobileNetV2 ImageNet weights, include_top=False, alpha=1.0",
        "Input transform: Phase 2 [0,1] RGB values scaled to [0,255], then Keras MobileNetV2 preprocess_input to [-1,1]",
        f"Class order/IDs: {CLASS_TO_ID}",
        f"Total fixed manifest images: {len(metadata.manifest_rows)}",
        f"Valid CNN images: {len(metadata.valid_rows)}; excluded no-hand: {sum(metadata.excluded_failure_counts.values())}",
        f"Valid class counts: {dict(metadata.valid_class_counts)}",
        f"Valid split counts: {dict(metadata.valid_split_counts)}",
        f"Unique users: {metadata.unique_users}; subject overlap: none",
        f"Stage A: frozen backbone, max epochs={STAGE_A_EPOCHS}, lr={STAGE_A_LEARNING_RATE}",
        f"Stage B: last {FINE_TUNE_LAST_LAYERS} backbone layers selectively unfrozen, BatchNorm frozen, max epochs={STAGE_B_EPOCHS}, lr={STAGE_B_LEARNING_RATE}",
        f"Head dropout: {HEAD_DROPOUT}; data augmentation: none",
        f"Batch size: {BATCH_SIZE}; early stopping patience: {EARLY_STOPPING_PATIENCE}; seed: {SEED}",
        f"Total parameters: {model.count_params():,}",
        f"Stage A trainable parameters: {stage_a_trainable_parameters:,}",
        f"Stage B total trainable parameters: {stage_b_trainable_parameters:,}",
        f"Stage B backbone trainable parameters: {fine_tune_backbone_parameters:,}",
        f"Stage A resumed from the saved pilot checkpoint: {resumed_stage_a}",
        f"Training time: {training_seconds:.2f} seconds",
        f"Best checkpoint: {BEST_MODEL_PATH.name}; selected at global epoch {checkpoint.best_epoch} in {checkpoint.best_stage}, best val_loss={checkpoint.best_loss:.5f}",
        f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}; macro precision: {macro_precision:.4f}; macro recall: {macro_recall:.4f}; macro F1: {macro_f1:.4f}",
        f"Approximate inference time: {inference_seconds_per_batch:.4f} seconds per batch of up to 8 train images (warmup + 5 repeats)",
        "Test was not used in pre-training sanity checks, training, checkpoint selection, or callbacks; it was traversed once after training for final metrics.",
        "Phase 3 baseline comparison: accuracy 0.8215; macro F1 0.8310",
        "",
        "Per-class test metrics:",
    ]
    for name in CLASS_NAMES:
        values = per_class[name]
        summary.append(f"  {name}: precision={values['precision']:.4f}, recall={values['recall']:.4f}, f1={values['f1']:.4f}, support={values['support']}")
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")


def _inference_timing(model: tf.keras.Model, train_dataset: tf.data.Dataset) -> float:
    images, _ = next(iter(train_dataset))
    sample = images[:8]
    model(sample, training=False).numpy()
    started = time.perf_counter()
    for _ in range(5):
        model(sample, training=False).numpy()
    return (time.perf_counter() - started) / 5


def train_phase4(
    stage_a_epochs: int,
    stage_b_epochs: int,
    batch_size: int,
    resume_stage_a: bool = False,
) -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    cpu_count = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(4, cpu_count))
    tf.config.threading.set_inter_op_parallelism_threads(2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    metadata = inspect_phase2_data()
    datasets = make_transfer_datasets(metadata, batch_size=batch_size, seed=SEED)
    class_weights = class_weights_from_train(metadata)
    steps_per_epoch = math.ceil(metadata.valid_split_counts["train"] / batch_size)
    validation_steps = math.ceil(metadata.valid_split_counts["val"] / batch_size)
    checkpoint = BestValidationCheckpoint(BEST_MODEL_PATH)
    resumed_epoch = 0
    if resume_stage_a:
        if not BEST_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Cannot resume Stage A; checkpoint is missing: {BEST_MODEL_PATH}")
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
        backbone = model.get_layer("mobilenetv2_1.00_224")
        optimizer_iterations = int(model.optimizer.iterations.numpy())
        resumed_epoch, remainder = divmod(optimizer_iterations, steps_per_epoch)
        if remainder != 0 or resumed_epoch != 1:
            raise ValueError(
                "Resume checkpoint is not the expected completed pilot epoch: "
                f"optimizer_iterations={optimizer_iterations}, steps_per_epoch={steps_per_epoch}"
            )
        if backbone.trainable or backbone.trainable_variables:
            raise ValueError("Resume checkpoint is not a frozen-backbone Stage A checkpoint")
        if stage_a_epochs <= resumed_epoch:
            raise ValueError(f"Stage A total epochs ({stage_a_epochs}) must exceed resumed epoch {resumed_epoch}")
        resume_validation = model.evaluate(datasets["val"], verbose=0)
        checkpoint.best_loss = float(resume_validation[0])
        checkpoint.best_epoch = resumed_epoch
        checkpoint.best_stage = "stage_a_frozen"
        checkpoint.global_epoch = resumed_epoch
        print(
            f"Resuming frozen Stage A from checkpoint epoch {resumed_epoch}; "
            f"optimizer iterations={optimizer_iterations}, val_loss={checkpoint.best_loss:.5f}"
        )
    else:
        model, backbone = build_model()

    # Sanity check completes before either training stage begins.
    train_batch, _ = sanity_check(model, metadata, datasets)
    stage_a_trainable_parameters = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))

    histories: list[tuple[str, tf.keras.callbacks.History]] = []
    training_started = time.perf_counter()

    checkpoint.current_stage = "stage_a_frozen"
    stage_a_history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        initial_epoch=resumed_epoch,
        epochs=stage_a_epochs,
        callbacks=callbacks_for_stage(checkpoint, "stage_a_frozen"),
        class_weight=class_weights,
        shuffle=False,
        verbose=1,
    )
    histories.append(("stage_a_frozen", stage_a_history))

    # Continue Stage B from the globally best validation checkpoint, not a worse late epoch.
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    backbone = model.get_layer("mobilenetv2_1.00_224")
    fine_tune_backbone_parameters = unfreeze_upper_layers(backbone)
    stage_b_trainable_parameters = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
    compile_model(model, STAGE_B_LEARNING_RATE)
    checkpoint.current_stage = "stage_b_finetune"
    stage_b_history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=stage_b_epochs,
        callbacks=callbacks_for_stage(checkpoint, "stage_b_finetune"),
        class_weight=class_weights,
        shuffle=False,
        verbose=1,
    )
    histories.append(("stage_b_finetune", stage_b_history))
    training_seconds = time.perf_counter() - training_started
    if resume_stage_a:
        training_seconds += RESUMED_STAGE_A_EPOCH_SECONDS
    model.save(FINAL_MODEL_PATH)

    # The best checkpoint is selected only by validation loss; test is traversed once below.
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    test_loss, test_accuracy, macro_precision, macro_recall, macro_f1, per_class, confusion = _save_evaluation(datasets["test"], best_model)
    _save_outputs(histories, confusion, per_class, test_accuracy, macro_precision, macro_recall, macro_f1, resumed_stage_a=resume_stage_a)
    inference_seconds = _inference_timing(best_model, datasets["train"])
    _write_summary(
        metadata,
        best_model,
        histories,
        checkpoint,
        stage_a_trainable_parameters,
        stage_b_trainable_parameters,
        fine_tune_backbone_parameters,
        training_seconds,
        test_loss,
        test_accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        per_class,
        inference_seconds,
        resumed_stage_a=resume_stage_a,
    )

    print(f"Best model size: {BEST_MODEL_PATH.stat().st_size / (1024**2):.2f} MiB")
    print(f"Final model size: {FINAL_MODEL_PATH.stat().st_size / (1024**2):.2f} MiB")
    print(f"Phase 4 summary: {SUMMARY_PATH}")


def run_sanity_only(batch_size: int) -> None:
    tf.keras.utils.set_random_seed(SEED)
    metadata = inspect_phase2_data()
    datasets = make_transfer_datasets(metadata, batch_size=batch_size, seed=SEED)
    model, _ = build_model()
    sanity_check(model, metadata, datasets)


def run_stage_a_pilot(batch_size: int) -> None:
    """Run exactly one frozen-backbone epoch; do not save models or touch test batches."""
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    cpu_count = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(4, cpu_count))
    tf.config.threading.set_inter_op_parallelism_threads(2)

    metadata = inspect_phase2_data()
    datasets = make_transfer_datasets(metadata, batch_size=batch_size, seed=SEED)
    class_weights = class_weights_from_train(metadata)
    model, _ = build_model()
    sanity_check(model, metadata, datasets)

    started = time.perf_counter()
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        steps_per_epoch=math.ceil(metadata.valid_split_counts["train"] / batch_size),
        validation_steps=math.ceil(metadata.valid_split_counts["val"] / batch_size),
        epochs=1,
        class_weight=class_weights,
        shuffle=False,
        verbose=1,
    )
    elapsed = time.perf_counter() - started

    train_loss = float(history.history["loss"][0])
    train_accuracy = float(history.history["accuracy"][0])
    validation_loss = float(history.history["val_loss"][0])
    validation_accuracy = float(history.history["val_accuracy"][0])
    measured_epoch_seconds = elapsed
    # Estimate the 8+5 epoch plan at observed Stage A speed; fine-tuning is budgeted 1.5x slower.
    estimated_full_seconds = measured_epoch_seconds * (STAGE_A_EPOCHS + (STAGE_B_EPOCHS * 1.5))
    print("\nStage A frozen-backbone pilot results (exactly one epoch)")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print("Loss trend: one epoch is insufficient to determine whether loss is decreasing.")
    print(
        f"Estimated full {STAGE_A_EPOCHS}+{STAGE_B_EPOCHS} epoch run: "
        f"{estimated_full_seconds / 60:.1f} minutes using measured Stage A speed and a 1.5x Stage B allowance."
    )
    print("Pilot completed; Stage B and test evaluation were not started. No checkpoint was saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-only", action="store_true", help="Load ImageNet weights and validate train/val batches without training")
    parser.add_argument("--pilot-stage-a", action="store_true", help="Train exactly one Stage A epoch; do not run Stage B or test evaluation")
    parser.add_argument("--resume-stage-a", action="store_true", help="Resume the existing one-epoch Stage A checkpoint before Stage B")
    parser.add_argument("--train", action="store_true", help="Run Stage A and Stage B training after the sanity check")
    parser.add_argument("--stage-a-epochs", type=int, default=STAGE_A_EPOCHS)
    parser.add_argument("--stage-b-epochs", type=int, default=STAGE_B_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()
    if arguments.pilot_stage_a:
        if arguments.train or arguments.sanity_only or arguments.resume_stage_a:
            parser.error("--pilot-stage-a cannot be combined with --train, --resume-stage-a, or --sanity-only")
        run_stage_a_pilot(arguments.batch_size)
        return
    if arguments.resume_stage_a and not arguments.train:
        parser.error("--resume-stage-a requires --train")
    if arguments.sanity_only or not arguments.train:
        run_sanity_only(arguments.batch_size)
        if not arguments.train:
            print("Sanity-only mode complete; full training not started.")
        return
    train_phase4(arguments.stage_a_epochs, arguments.stage_b_epochs, arguments.batch_size, resume_stage_a=arguments.resume_stage_a)


if __name__ == "__main__":
    main()