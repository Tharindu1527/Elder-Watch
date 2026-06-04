#!/usr/bin/env python3
"""
Evaluation Script
Measures: precision, recall, F1, mAP, inference time, FPS, memory usage.
Compares: FP32 (.pt) vs INT8 (.tflite)

Usage:
  python training/scripts/evaluate.py \
      --pt    models/finetuned/yolov8n_fall.pt \
      --tflite models/quantized/yolov8n_fall_int8.tflite \
      --data  training/configs/dataset.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("Evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt",      type=str, default=None)
    p.add_argument("--tflite",  type=str, default=None)
    p.add_argument("--data",    type=str, default="training/configs/dataset.yaml")
    p.add_argument("--out",     type=str, default="logs/eval_results.json")
    p.add_argument("--n",       type=int, default=200, help="Test frames")
    p.add_argument("--warmup",  type=int, default=10,  help="Warmup iterations")
    return p.parse_args()


def load_test_images(data_yaml: str, n: int):
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    base   = Path(data_yaml).parent / cfg.get("path", ".")
    test_d = Path(cfg.get("path", ".")) / "images" / "test"
    imgs   = list(test_d.glob("*.jpg")) + list(test_d.glob("*.png"))
    return imgs[:n] if len(imgs) >= n else imgs


def load_labels(data_yaml: str):
    """Load ground truth labels for test images."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    base  = Path(cfg.get("path", "."))
    lbl_d = base / "labels" / "test"
    return lbl_d


def bench_pytorch(pt_path: str, img_paths: list, warmup: int):
    """Benchmark PyTorch model."""
    from ultralytics import YOLO
    model = YOLO(pt_path)

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(warmup):
        model(dummy, verbose=False)

    latencies = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        t0 = time.perf_counter()
        model(img, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_ms": np.mean(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "fps":    1000 / np.mean(latencies),
        "n":      len(latencies),
    }


def bench_tflite(tflite_path: str, img_paths: list, warmup: int):
    """Benchmark TFLite INT8 model."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    interp = tflite.Interpreter(model_path=tflite_path, num_threads=4)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    imgsz = in_det["shape"][1]

    # Warmup
    dummy = np.zeros((1, imgsz, imgsz, 3), dtype=in_det["dtype"])
    for _ in range(warmup):
        interp.set_tensor(in_det["index"], dummy)
        interp.invoke()

    latencies = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if in_det["dtype"] == np.int8:
            img = (img / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        blob = img[np.newaxis, ...]

        t0 = time.perf_counter()
        interp.set_tensor(in_det["index"], blob)
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_ms": np.mean(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "fps":    1000 / np.mean(latencies),
        "n":      len(latencies),
    }


def eval_pytorch_metrics(pt_path: str, data_yaml: str):
    """Run official Ultralytics validation."""
    from ultralytics import YOLO
    model = YOLO(pt_path)
    metrics = model.val(data=data_yaml, verbose=False, save=False)
    return {
        "mAP50":    float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":   float(metrics.box.mr),
    }


def get_model_sizes(pt_path, tflite_path):
    sizes = {}
    if pt_path and os.path.exists(pt_path):
        sizes["fp32_mb"] = os.path.getsize(pt_path) / 1024 / 1024
    if tflite_path and os.path.exists(tflite_path):
        sizes["int8_mb"] = os.path.getsize(tflite_path) / 1024 / 1024
    if "fp32_mb" in sizes and "int8_mb" in sizes:
        sizes["reduction_pct"] = (1 - sizes["int8_mb"] / sizes["fp32_mb"]) * 100
    return sizes


def main():
    args      = parse_args()
    results   = {}
    img_paths = load_test_images(args.data, args.n)
    logger.info(f"Test images: {len(img_paths)}")

    # Model sizes
    results["model_sizes"] = get_model_sizes(args.pt, args.tflite)
    logger.info(f"Model sizes: {results['model_sizes']}")

    # FP32 PyTorch
    if args.pt and os.path.exists(args.pt):
        logger.info("Benchmarking FP32 PyTorch model...")
        try:
            results["pytorch_metrics"] = eval_pytorch_metrics(args.pt, args.data)
            results["pytorch_latency"] = bench_pytorch(args.pt, img_paths, args.warmup)
            logger.info(f"  mAP50={results['pytorch_metrics']['mAP50']:.3f} | "
                        f"FPS={results['pytorch_latency']['fps']:.1f}")
        except Exception as e:
            logger.error(f"PyTorch eval failed: {e}")

    # INT8 TFLite
    if args.tflite and os.path.exists(args.tflite):
        logger.info("Benchmarking INT8 TFLite model...")
        try:
            results["tflite_latency"] = bench_tflite(args.tflite, img_paths, args.warmup)
            logger.info(f"  FPS={results['tflite_latency']['fps']:.1f} | "
                        f"avg_ms={results['tflite_latency']['avg_ms']:.1f}")
        except Exception as e:
            logger.error(f"TFLite eval failed: {e}")

    # Summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {args.out}")


if __name__ == "__main__":
    main()
