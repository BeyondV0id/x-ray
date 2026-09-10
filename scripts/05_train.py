import sys
import os
import json
import argparse
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed, check_gpu_status
from src.dataset import build_tf_dataset
from src.model import build_retinanet_model, FocalLoss, SmoothL1Loss

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
                cls_loss = logs.get("cls_predictions_loss", logs.get("loss", 0.0))
                box_loss = logs.get("box_predictions_loss", 0.0)
                tot_loss = logs.get("loss", 0.0)
                msg = f"  [Epoch {self.current_epoch}/{self.total_epochs} | Batch {curr_step}/{steps} ({epoch_pct:.0f}%)] -> Total Progress: {overall_pct:.1f}% | Loss: {tot_loss:.4f} (Cls: {cls_loss:.4f}, Box: {box_loss:.4f})"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg, flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        overall_pct = (epoch_num / self.total_epochs) * 100
        val_loss = logs.get("val_loss", 0.0)
        tot_loss = logs.get("loss", 0.0)
        msg = f"=== EPOCH {epoch_num}/{self.total_epochs} COMPLETE ({overall_pct:.1f}% Overall) | Train Loss: {tot_loss:.4f} | Val Loss: {val_loss:.4f} ==="
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow RetinaNet Chest X-Ray Object Detector.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("train")
    logger.info("=== STEP 05: TRAINING TENSORFLOW OBJECT DETECTION MODEL ===")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    gpu_status = check_gpu_status(logger=logger)

    # Resolve CLI parameters with fallback to config
    epochs = args.epochs if args.epochs is not None else config["training"].get("epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else config["training"].get("batch_size", 8)
    learning_rate = args.learning_rate if args.learning_rate is not None else config["training"].get("learning_rate", 0.0001)

    logger.info(f"Training Configuration:\n  - Epochs: {epochs}\n  - Batch Size: {batch_size}\n  - Learning Rate: {learning_rate}")

    paths = config["paths"]
    proc_dir = Path(paths["data_processed"])
    train_manifest = proc_dir / "train.json"
    val_manifest = proc_dir / "val.json"

    if not train_manifest.exists() or not val_manifest.exists():
        logger.error(f"Dataset manifests missing in {proc_dir}. Please run 03_prepare_dataset.py first.")
        sys.exit(1)

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    logger.info(f"Loaded {len(train_records)} training records, {len(val_records)} validation records.")

    image_size = config["model"].get("image_size", 512)
    num_classes = config.get("num_classes", 2)

    # Build tf.data pipelines
    train_ds = build_tf_dataset(train_records, image_size=image_size, batch_size=batch_size, is_training=True, num_classes=num_classes)
    val_ds = build_tf_dataset(val_records, image_size=image_size, batch_size=batch_size, is_training=False, num_classes=num_classes)

    # Build or load model
    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        model = tf.keras.models.load_model(
            args.resume,
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )
    else:
        logger.info(f"Initializing RetinaNet model with {config['model'].get('backbone', 'EfficientNetB0')} backbone...")
        model = build_retinanet_model(
            input_shape=(image_size, image_size, 3),
            num_classes=num_classes,
            backbone_name=config["model"].get("backbone", "EfficientNetB0")
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss={
            "cls_predictions": FocalLoss(
                alpha=config["training"].get("focal_loss_alpha", 0.25),
                gamma=config["training"].get("focal_loss_gamma", 2.0)
            ),
            "box_predictions": SmoothL1Loss()
        },
        loss_weights={
            "cls_predictions": 1.0,
            "box_predictions": config["training"].get("box_loss_weight", 1.0)
        }
    )

    # Callbacks
    ckpt_dir = Path(paths["models_checkpoints"])
    prod_dir = Path(paths["models_production"])
    logs_dir = Path(paths["logs_tensorboard"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prod_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_dir / "epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    best_model_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(prod_dir / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=str(logs_dir),
        histogram_freq=1
    )

    progress_cb = PercentageProgressCallback(total_epochs=epochs, logger=logger, update_pct_step=10)

    callbacks = [progress_cb, checkpoint_cb, best_model_cb, early_stop_cb, reduce_lr_cb, tensorboard_cb]

    logger.info("Starting model training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        # Save training history
        hist_path = prod_dir / "training_history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)

        logger.info(f"Training completed successfully! Saved history to {hist_path}")
        logger.info(f"Best model saved under: {prod_dir / 'best_model.keras'}")

    except Exception as e:
        logger.error(f"Training interrupted or encountered error: {e}")
        try:
            fallback_path = prod_dir / "interrupted_model.keras"
            model.save(str(fallback_path))
            logger.info(f"Fallback model saved to: {fallback_path}")
        except Exception as save_err:
            logger.warning(f"Could not save fallback model: {save_err}")

    logger.info("=== STEP 05 COMPLETE ===")

if __name__ == "__main__":
    main()
