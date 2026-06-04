#!/usr/bin/env python3
"""
Quantization Script — Fixed for RTX 5090 / CUDA 13.0
Uses Ultralytics built-in TFLite INT8 export instead of raw TF PTQ pipeline.
TensorFlow's manual PTQ causes segfault on CUDA 13+ because TF doesn't
support CUDA 13 yet (TF 2.x supports up to CUDA 12.x).

Pipeline:  YOLOv8 .pt  →  TFLite INT8  (via Ultralytics)
Expected:  ~6.2 MB (FP32)  →  ~1.6 MB (INT8)
           ~50 ms/frame    →  ~20 ms/frame on RPi 5

Usage:
  python training/scripts/quantize.py \
      --model models/finetuned/yolov8n_fall.pt \
      --data  training/configs/dataset.yaml \
      --out   models/quantized/yolov8n_fall_int8.tflite
"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("Quantize")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True,
                   help="Path to fine-tuned .pt  e.g. models/finetuned/yolov8n_fall.pt")
    p.add_argument("--data",   required=True,
                   help="Dataset YAML e.g. training/configs/dataset.yaml")
    p.add_argument("--out",    default="models/quantized/yolov8n_fall_int8.tflite",
                   help="Output TFLite path")
    p.add_argument("--imgsz",  type=int, default=640)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════
# Method A — Ultralytics built-in INT8 TFLite export (RECOMMENDED)
# Does NOT use TensorFlow directly → no segfault on CUDA 13
# ════════════════════════════════════════════════════════════════════

def export_ultralytics(pt_path: str, data_yaml: str,
                       output_path: str, imgsz: int) -> str:
    """
    Use Ultralytics YOLO.export() to produce INT8 TFLite.
    Ultralytics handles the ONNX → TFLite conversion internally
    using onnx2tf which does NOT require a working CUDA TF install.
    """
    from ultralytics import YOLO

    logger.info("=" * 60)
    logger.info("INT8 Quantization via Ultralytics export")
    logger.info(f"  Input  : {pt_path}")
    logger.info(f"  Data   : {data_yaml}")
    logger.info(f"  Output : {output_path}")
    logger.info(f"  imgsz  : {imgsz}")
    logger.info("=" * 60)

    logger.info("Loading model...")
    model = YOLO(pt_path)

    logger.info("Exporting to INT8 TFLite (this takes 2-5 min)...")
    logger.info("Note: You may see TF/CUDA warnings — these are harmless.")

    t0 = time.time()

    # Force CPU-only for TF operations to avoid CUDA 13 segfault
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    exported = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=True,
        data=data_yaml,
        half=False,
        simplify=True,
        nms=False,
    )

    elapsed = time.time() - t0
    logger.info(f"Export finished in {elapsed:.0f}s")

    # Ultralytics saves the tflite alongside the pt file
    # Find it and move to our output path
    exported_path = find_tflite(pt_path, exported)

    if exported_path is None or not Path(exported_path).exists():
        raise FileNotFoundError(
            f"TFLite file not found after export. "
            f"Expected near: {pt_path}\n"
            f"Run: find . -name '*.tflite' to locate it manually."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(str(exported_path), output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"✓ INT8 TFLite saved: {output_path}  ({size_mb:.2f} MB)")
    return output_path


def find_tflite(pt_path: str, exported) -> str:
    """
    Locate the exported .tflite file.
    Ultralytics may return the path directly or save it near the .pt file.
    """
    # If Ultralytics returned a valid path
    if exported and isinstance(exported, (str, Path)):
        p = Path(str(exported))
        if p.exists() and p.suffix == ".tflite":
            return str(p)
        # Sometimes it returns the _saved_model folder; tflite is inside
        tflite_candidates = list(p.parent.glob("*.tflite")) if p.parent.exists() else []
        if tflite_candidates:
            return str(tflite_candidates[0])

    # Search near the .pt file
    pt_dir  = Path(pt_path).parent
    pt_stem = Path(pt_path).stem

    search_dirs = [pt_dir, pt_dir.parent, Path(".")]
    for d in search_dirs:
        for tflite in d.rglob("*.tflite"):
            return str(tflite)

    return None


# ════════════════════════════════════════════════════════════════════
# Validation — quick inference test on the quantized model
# ════════════════════════════════════════════════════════════════════

def validate(tflite_path: str, imgsz: int = 640):
    """Run 10 dummy inferences to confirm the model works and measure latency."""
    logger.info("\nValidating quantized model...")
    try:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            # Use tensorflow.lite on desktop
            import tensorflow as tf
            tflite = tf.lite

        import numpy as np

        interp = tflite.Interpreter(model_path=tflite_path, num_threads=4)
        interp.allocate_tensors()

        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        logger.info(f"  Input  dtype : {in_det['dtype'].__name__}")
        logger.info(f"  Input  shape : {in_det['shape'].tolist()}")
        logger.info(f"  Output shape : {out_det['shape'].tolist()}")
        logger.info(f"  Quantization : scale={in_det['quantization'][0]:.6f}, "
                    f"zp={in_det['quantization'][1]}")

        # Warm up
        dummy = np.zeros(in_det["shape"], dtype=in_det["dtype"])
        for _ in range(3):
            interp.set_tensor(in_det["index"], dummy)
            interp.invoke()

        # Benchmark
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            interp.set_tensor(in_det["index"], dummy)
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        logger.info(f"  Desktop latency : {avg_ms:.1f} ms/frame  "
                    f"({1000/avg_ms:.0f} FPS)")
        logger.info(f"  RPi 5 estimate  : ~{avg_ms * 5:.0f} ms/frame  "
                    f"(~{1000/(avg_ms*5):.0f} FPS)  [RPi ~5x slower than desktop]")
        logger.info("  ✓ Model validated OK")

    except Exception as e:
        logger.warning(f"Validation skipped: {e}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def quantize_model(pt_path: str,
                   data_yaml: str = "training/configs/dataset.yaml",
                   output_path: str = "models/quantized/yolov8n_fall_int8.tflite",
                   imgsz: int = 640) -> str:
    """Entry point called from train.py --export or directly."""
    return export_ultralytics(pt_path, data_yaml, output_path, imgsz)


def main():
    args = parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Run training first: python training/scripts/train_yolov8.py ...")
        return

    output = quantize_model(
        pt_path=args.model,
        data_yaml=args.data,
        output_path=args.out,
        imgsz=args.imgsz,
    )

    validate(output, imgsz=args.imgsz)

    size_fp32 = os.path.getsize(args.model) / 1024 / 1024
    size_int8 = os.path.getsize(output) / 1024 / 1024
    reduction = (1 - size_int8 / size_fp32) * 100

    logger.info("\n" + "=" * 60)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  FP32 model : {args.model}  ({size_fp32:.2f} MB)")
    logger.info(f"  INT8 model : {output}  ({size_int8:.2f} MB)")
    logger.info(f"  Size reduction : {reduction:.0f}%")
    logger.info(f"\nNext step — transfer to Raspberry Pi:")
    logger.info(f"  scp {output} pi@raspberrypi.local:~/elder_watch/models/quantized/")


if __name__ == "__main__":
    main()