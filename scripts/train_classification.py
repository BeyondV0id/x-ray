import sys
import os
import json
import argparse
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed, check_gpu_status

class PercentageProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to log batch step and total epoch completion percentages clearly."""
    def __init__(self, total_epochs: int, logger=None, update_pct_step: int = 10):
        super().__init__()
        self.total_epochs = total_epochs
        self.logger = logger
        self.update_pct_step = update_pct_step
        self.current_epoch = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        steps = self.params.get("steps") if hasattr(self, "params") and self.params else None
        if steps:
            curr_step = batch + 1
            freq = max(1, steps * self.update_pct_step // 100)
            if curr_step == 1 or curr_step % freq == 0 or curr_step == steps:
                epoch_pct = (curr_step / steps) * 100
                overall_pct = ((self.current_epoch - 1 + (curr_step / steps)) / self.total_epochs) * 100
                acc = logs.get("accuracy", 0.0)
                tot_loss = logs.get("loss", 0.0)
                msg = f"  [Epoch {self.current_epoch}/{self.total_epochs} | Batch {curr_step}/{steps} ({epoch_pct:.0f}%)] -> Total Progress: {overall_pct:.1f}% | Loss: {tot_loss:.4f} | Acc: {acc:.4f}"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg, flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        overall_pct = (epoch_num / self.total_epochs) * 100
        val_loss = logs.get("val_loss", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        tot_loss = logs.get("loss", 0.0)
        msg = f"=== EPOCH {epoch_num}/{self.total_epochs} COMPLETE ({overall_pct:.1f}% Overall) | Train Loss: {tot_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ==="
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

def build_classifier_model(image_size: int = 512, num_classes: int = 2, freeze_backbone: bool = False):
    """
    Build EfficientNetB0 based Transfer Learning Classifier.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )
    
    if freeze_backbone:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Fine-tune starting from layer 100 onwards
        for layer in base_model.layers[:100]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    
    # Data Augmentation pipeline
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.05)(x)
    
    # Preprocessing
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=not freeze_backbone)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="predictions")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="XRayClassifier_EfficientNetB0")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Keras Binary Classifier on Chest X-Ray Archive Dataset.")
    parser.add_argument("--data-dir", type=str, default="data/raw/dataset1", help="Path to pre-split dataset directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=384, help="Input image resolution (e.g., 384 or 512).")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable Keras mixed float16 precision.")
    parser.add_argument("--two-stage", action="store_true", help="Use 2-stage training (frozen head warmup -> fine-tuning).")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("train_classifier")
    logger.info("=== STEP 05 (CLASSIFIER): TRAINING TENSORFLOW CLASSIFICATION MODEL ===")

    if args.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Enabled Keras Mixed Precision Policy: mixed_float16")
        except Exception as mp_err:
            logger.warning(f"Could not enable mixed precision: {mp_err}")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))
    check_gpu_status(logger=logger)

    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        logger.error(f"Train/Val directories not found under: {data_dir}")
        sys.exit(1)

    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate

    logger.info(f"Loading datasets from: {data_dir}")
    logger.info(f"Hyperparameters -> Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Image Size: {image_size}x{image_size}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        label_mode="binary"
    )

    # Prefetch optimization with AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    model = build_classifier_model(image_size=image_size, num_classes=2, freeze_backbone=args.two_stage)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    paths = config["paths"]
    ckpt_dir = Path(paths["models_checkpoints"])
    prod_dir = Path(paths["models_production"])
    logs_dir = Path(paths["logs_tensorboard"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prod_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    progress_cb = PercentageProgressCallback(total_epochs=epochs, logger=logger, update_pct_step=10)

    callbacks = [
        progress_cb,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "cls_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(prod_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1
        )
    ]

    logger.info("Starting training loop...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    hist_path = prod_dir / "training_history_classification.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    logger.info(f"Training completed successfully! Saved history to: {hist_path}")
    logger.info(f"Best model saved to: {prod_dir / 'best_model.keras'}")
    logger.info("=== STEP 05 (CLASSIFIER) COMPLETE ===")

if __name__ == "__main__":
    main()
