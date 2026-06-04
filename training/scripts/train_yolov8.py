#!/usr/bin/env python3
"""
Training Script: Fine-tune YOLOv8n on fall detection datasets
University of Ruhuna - EE7204/EC7205

Free & Open-Source Datasets (prepared by prepare_datasets.py):
  - Roboflow Fall Detection : 4,497 pre-annotated YOLO images (CC BY 4.0)
    https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8
  - URFD (UR Fall Detection): 30 fall + 40 ADL sequences, ~2,600 frames
    https://fenix.ur.edu.pl/~mkepski/ds/uf.html  (direct download, no login)
  - Le2i FDD               : 221 annotated videos (home + coffee room)
    https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia  (free Kaggle)
  - MCFD                   : 8 fall scenarios × 3 cameras
    http://www.iro.umontreal.ca/~labimage/Dataset/

Usage:
  python training/scripts/train_yolov8.py --config training/configs/yolov8_fall.yaml
  python training/scripts/train_yolov8.py --resume runs/detect/elder_watch/weights/last.pt
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("TrainYOLO")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 for fall detection")
    p.add_argument("--config",  type=str, default="training/configs/yolov8_fall.yaml")
    p.add_argument("--resume",  type=str, default=None,  help="Resume from checkpoint .pt")
    p.add_argument("--data",    type=str, default=None,  help="Override dataset YAML path")
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--batch",   type=int, default=None)
    p.add_argument("--device",  type=str, default="0",
                   help="CUDA device(s): 0, 0,1, or 'cpu'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--export",  action="store_true",
                   help="Export best model to TFLite after training")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Install ultralytics: pip install ultralytics")

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})

    # Override with CLI args
    epochs      = args.epochs  or train_cfg.get("epochs", 50)
    batch_size  = args.batch   or train_cfg.get("batch_size", 16)
    imgsz       = train_cfg.get("image_size", 640)
    lr          = train_cfg.get("learning_rate", 0.001)
    weights     = args.resume  or train_cfg.get("pretrained_weights", "yolov8n.pt")
    data_yaml   = args.data    or "training/configs/dataset.yaml"
    project_dir = "runs/detect"
    run_name    = "elder_watch"

    logger.info("=" * 60)
    logger.info("Elder Watch - YOLOv8 Fine-tuning")
    logger.info(f"  Weights    : {weights}")
    logger.info(f"  Dataset    : {data_yaml}")
    logger.info(f"  Epochs     : {epochs}")
    logger.info(f"  Batch size : {batch_size}")
    logger.info(f"  Image size : {imgsz}")
    logger.info(f"  Device     : {args.device}")
    logger.info("=" * 60)

    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr,
        lrf=0.01,                   # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation (built-in Ultralytics augment)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,               # rotation
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,                 # Don't flip vertically (ceiling cam)
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,                # Cosine LR schedule
        close_mosaic=10,            # Disable mosaic last 10 epochs
        resume=bool(args.resume),
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,
        
        # Callbacks
        patience=15,                # Early stopping patience
    )

    best_model = Path(project_dir) / run_name / "weights" / "best.pt"
    logger.info(f"Training complete. Best model: {best_model}")

    # Copy to models/finetuned/
    os.makedirs("models/finetuned", exist_ok=True)
    dest = "models/finetuned/yolov8n_fall.pt"
    shutil.copy(best_model, dest)
    logger.info(f"Saved to: {dest}")

    # Optionally export to ONNX + TFLite
    if args.export:
        logger.info("Exporting model to TFLite (INT8)...")
        from training.scripts.quantize import quantize_model
        quantize_model(dest, data_yaml=data_yaml)

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
