#!/usr/bin/env python3
"""
Quantization Script: Post-Training Quantization (PTQ) to INT8 TFLite
University of Ruhuna - EE7204/EC7205

Pipeline:
  YOLOv8 .pt  →  ONNX  →  TensorFlow SavedModel  →  INT8 TFLite

Expected results:
  - Size:      ~25 MB (FP32) → ~6-7 MB (INT8)  [~75% reduction]
  - Speed:     ~50 ms/frame  → ~20 ms/frame on RPi 5
  - Accuracy:  <2% mAP drop

Usage:
  python training/scripts/quantize.py \
      --model models/finetuned/yolov8n_fall.pt \
      --data  training/configs/dataset.yaml \
      --out   models/quantized/yolov8n_fall_int8.tflite
"""

import argparse
import logging
import os
import glob
import random
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("Quantize")


def parse_args():
    p = argparse.ArgumentParser(description="INT8 PTQ for Raspberry Pi deployment")
    p.add_argument("--model",    required=True,  help="Path to fine-tuned .pt model")
    p.add_argument("--data",     required=True,  help="Dataset YAML for calibration images")
    p.add_argument("--out",      default="models/quantized/yolov8n_fall_int8.tflite")
    p.add_argument("--cal-size", type=int, default=1000, help="Calibration sample count")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--method",   choices=["PTQ", "QAT"], default="PTQ")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Step 1: Export .pt → ONNX → TensorFlow SavedModel
# ──────────────────────────────────────────────────────────────────────

def export_to_tf_savedmodel(pt_path: str, imgsz: int) -> str:
    """Export YOLOv8 .pt to TensorFlow SavedModel format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("pip install ultralytics")

    logger.info(f"Exporting {pt_path} → TF SavedModel...")
    model = YOLO(pt_path)
    
    # Export to TF SavedModel (via ONNX internally)
    # This creates <model_dir>/<name>_saved_model/
    saved_model_path = model.export(
        format="saved_model",
        imgsz=imgsz,
        half=False,           # FP32 for PTQ calibration
        int8=False,           # We do INT8 manually for more control
        dynamic=False,
        simplify=True,
        nms=False,            # We apply NMS post-inference
    )
    logger.info(f"TF SavedModel exported: {saved_model_path}")
    return str(saved_model_path)


# ──────────────────────────────────────────────────────────────────────
# Step 2: Calibration dataset generator
# ──────────────────────────────────────────────────────────────────────

def load_calibration_images(data_yaml: str, n_samples: int, imgsz: int) -> list:
    """Load calibration images from training set."""
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    base_dir = Path(data_yaml).parent
    train_dir = base_dir / data_cfg.get("train", "train/images")

    if not train_dir.exists():
        logger.warning(f"Train dir not found: {train_dir}. Using random noise calibration.")
        return [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                for _ in range(n_samples)]

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_imgs = []
    for pat in patterns:
        all_imgs.extend(glob.glob(str(train_dir / "**" / pat), recursive=True))

    if not all_imgs:
        logger.warning("No calibration images found. Using random noise.")
        return [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                for _ in range(n_samples)]

    random.shuffle(all_imgs)
    selected = all_imgs[:n_samples]

    calibration_data = []
    for img_path in selected:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        calibration_data.append(img)

    logger.info(f"Loaded {len(calibration_data)} calibration images from {train_dir}")
    return calibration_data


def make_representative_dataset(images: list) -> Generator:
    """Generator for TFLite converter calibration."""
    def generator():
        for img in images:
            yield [np.expand_dims(img, axis=0)]  # Add batch dim
    return generator


# ──────────────────────────────────────────────────────────────────────
# Step 3: TFLite INT8 conversion
# ──────────────────────────────────────────────────────────────────────

def convert_to_int8_tflite(saved_model_path: str,
                             representative_dataset,
                             output_path: str,
                             imgsz: int):
    """Convert TF SavedModel to INT8 quantized TFLite."""
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("pip install tensorflow>=2.12.0")

    logger.info("Starting INT8 PTQ conversion...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    # Full integer quantization (weights + activations)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    # Allow custom ops (needed for some YOLO layers)
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True

    logger.info("Running calibration + conversion (this may take a few minutes)...")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✓ INT8 TFLite model saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Step 4: Validate the quantized model
# ──────────────────────────────────────────────────────────────────────

def validate_tflite(tflite_path: str, test_images: list, imgsz: int):
    """Quick sanity-check: run inference on a few test images."""
    try:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        interp = tflite.Interpreter(model_path=tflite_path, num_threads=4)
        interp.allocate_tensors()
        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        in_scale, in_zp = in_det["quantization"]
        logger.info(f"Input: dtype={in_det['dtype'].__name__}, "
                    f"scale={in_scale:.6f}, zp={in_zp}")
        logger.info(f"Output: {out_det['shape']}")

        import time
        latencies = []
        for img in test_images[:10]:
            # Quantize
            q_img = (img / in_scale + in_zp).clip(-128, 127).astype(np.int8)
            interp.set_tensor(in_det["index"], q_img[np.newaxis, ...])
            t0 = time.perf_counter()
            interp.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(latencies) / len(latencies)
        logger.info(f"Validation: avg latency = {avg_ms:.1f} ms/frame "
                    f"(on {os.uname().nodename})")
        logger.info(f"Projected RPi 5 FPS: ~{1000/avg_ms:.1f} "
                    f"(actual may differ)")

    except Exception as e:
        logger.warning(f"Validation failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Ultralytics shortcut (alternative to manual pipeline)
# ──────────────────────────────────────────────────────────────────────

def quantize_via_ultralytics(pt_path: str, data_yaml: str, output_path: str, imgsz: int):
    """
    Alternative: Use Ultralytics built-in INT8 TFLite export.
    Simpler but less control over calibration.
    """
    from ultralytics import YOLO
    model = YOLO(pt_path)
    exported = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=True,
        data=data_yaml,
    )
    # Move to output path
    import shutil
    shutil.move(str(exported), output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Ultralytics INT8 TFLite: {output_path} ({size_mb:.2f} MB)")
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def quantize_model(pt_path: str,
                   data_yaml: str = "training/configs/dataset.yaml",
                   output_path: str = "models/quantized/yolov8n_fall_int8.tflite",
                   imgsz: int = 640,
                   n_cal: int = 1000):
    """
    Full PTQ pipeline. Called from train.py --export or directly.
    Tries manual pipeline first; falls back to Ultralytics export.
    """
    logger.info("=" * 60)
    logger.info("INT8 Post-Training Quantization")
    logger.info(f"  Input model : {pt_path}")
    logger.info(f"  Data YAML   : {data_yaml}")
    logger.info(f"  Output      : {output_path}")
    logger.info(f"  Cal samples : {n_cal}")
    logger.info("=" * 60)

    cal_images = load_calibration_images(data_yaml, n_cal, imgsz)

    try:
        import tensorflow as tf
        logger.info("Using manual TF PTQ pipeline...")
        saved_model = export_to_tf_savedmodel(pt_path, imgsz)
        rep_dataset = make_representative_dataset(cal_images)
        convert_to_int8_tflite(saved_model, rep_dataset, output_path, imgsz)
    except Exception as e:
        logger.warning(f"Manual pipeline failed ({e}). Trying Ultralytics export...")
        quantize_via_ultralytics(pt_path, data_yaml, output_path, imgsz)

    validate_tflite(output_path, cal_images, imgsz)
    logger.info("Quantization complete.")
    return output_path


if __name__ == "__main__":
    args = parse_args()
    quantize_model(
        pt_path=args.model,
        data_yaml=args.data,
        output_path=args.out,
        imgsz=args.imgsz,
        n_cal=args.cal_size,
    )
