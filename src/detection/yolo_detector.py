"""
YOLOv8 Detector
Supports:
  - .pt  (PyTorch, Ultralytics) — for desktop training/evaluation
  - .tflite (INT8 quantized)    — for Raspberry Pi deployment
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Wraps YOLOv8 inference for both PyTorch (.pt) and TFLite (.tflite) backends.
    Automatically selects backend based on model file extension.
    """

    CLASS_NAMES = {0: "person", 1: "fall"}

    def __init__(self, config: dict):
        self.config = config
        self.input_size   = config.get("input_size", 640)
        self.conf_thresh  = config.get("confidence_threshold", 0.50)
        self.fall_thresh  = config.get("fall_confidence_threshold", 0.70)
        self.nms_thresh   = config.get("nms_threshold", 0.45)
        self.use_tflite   = config.get("use_tflite", True)

        self.model     = None
        self.backend   = None
        self._load_model()

    # ──────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────

    def _load_model(self):
        model_path = Path(self.config.get("model_path", ""))
        fallback   = Path(self.config.get("fallback_model", ""))

        # Try primary path first
        if model_path.exists():
            self._init_from_path(model_path)
        elif fallback.exists():
            logger.warning(f"Primary model not found: {model_path}. Using fallback: {fallback}")
            self._init_from_path(fallback)
        else:
            logger.warning(
                "No model file found. Attempting to download yolov8n from Ultralytics..."
            )
            self._load_pytorch("yolov8n.pt")

    def _init_from_path(self, path: Path):
        ext = path.suffix.lower()
        if ext == ".tflite":
            self._load_tflite(str(path))
        elif ext in (".pt", ".pth"):
            self._load_pytorch(str(path))
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def _load_tflite(self, path: str):
        """Load INT8 TFLite model — preferred on Raspberry Pi 5."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        logger.info(f"Loading TFLite model: {path}")
        self.interpreter = tflite.Interpreter(
            model_path=path,
            num_threads=4  # Use all 4 cores on RPi 5
        )
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.backend = "tflite"

        # Determine if INT8 or FLOAT32
        self.input_dtype = self.input_details[0]["dtype"]
        self.input_scale, self.input_zero_point = (
            self.input_details[0]["quantization"]
            if self.input_dtype == np.int8
            else (1.0, 0)
        )
        logger.info(f"TFLite model loaded. dtype={self.input_dtype.__name__}, "
                    f"scale={self.input_scale:.6f}, zp={self.input_zero_point}")

    def _load_pytorch(self, path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("ultralytics package not installed.")
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ADD THIS LINE
        
        logger.info(f"Loading PyTorch YOLO model: {path}")
        self.pt_model = YOLO(path)
        self.backend  = "pytorch"
        logger.info("PyTorch model loaded.")

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR frame.
        Returns list of: {bbox: [x1,y1,x2,y2], class: str, class_id: int, confidence: float}
        """
        if self.backend == "tflite":
            return self._detect_tflite(frame)
        elif self.backend == "pytorch":
            return self._detect_pytorch(frame)
        return []

    def _detect_tflite(self, frame: np.ndarray) -> List[Dict]:
        """TFLite inference path (INT8 or FLOAT32)."""
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        blob = self._preprocess_tflite(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], blob)

        t0 = time.perf_counter()
        self.interpreter.invoke()
        inference_ms = (time.perf_counter() - t0) * 1000

        # Get output — YOLOv8 TFLite output: [1, 6, num_boxes] or [1, num_boxes, 6]
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Dequantize if INT8
        if self.output_details[0]["dtype"] == np.int8:
            out_scale, out_zp = self.output_details[0]["quantization"]
            output = (output.astype(np.float32) - out_zp) * out_scale

        detections = self._postprocess(output, orig_w, orig_h)
        logger.debug(f"TFLite inference: {inference_ms:.1f}ms | {len(detections)} detections")
        return detections

    def _detect_pytorch(self, frame: np.ndarray) -> List[Dict]:
        """PyTorch / Ultralytics inference path."""
        results = self.pt_model(frame, conf=self.conf_thresh,
                                iou=self.nms_thresh, verbose=False,
                                device="cpu")
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id    = int(box.cls[0])
                conf      = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name  = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                if conf < self.conf_thresh:
                    continue
                if cls_name == "fall" and conf < self.fall_thresh:
                    continue

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": conf
                })
        return detections

    # ──────────────────────────────────────────────────────────────────
    # Pre/post-processing helpers
    # ──────────────────────────────────────────────────────────────────

    def _preprocess_tflite(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and optionally quantize to INT8."""
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.int8:
            # Quantize: float_val = (int8_val - zero_point) * scale
            # So: int8_val = float_val / scale + zero_point
            normalized = rgb.astype(np.float32) / 255.0
            quantized  = (normalized / self.input_scale + self.input_zero_point)
            quantized  = np.clip(quantized, -128, 127).astype(np.int8)
            return quantized[np.newaxis, ...]  # [1, H, W, 3]
        else:
            normalized = rgb.astype(np.float32) / 255.0
            return normalized[np.newaxis, ...]

    def _postprocess(self, raw_output: np.ndarray,
                     orig_w: int, orig_h: int) -> List[Dict]:
        """
        Parse YOLOv8 TFLite output and apply NMS.
        YOLOv8 exports in shape [1, 6, 8400] (cx,cy,w,h,cls0,cls1...)
        or transposed [1, 8400, 6].
        """
        out = raw_output[0]
        if out.shape[0] < out.shape[-1]:
            out = out.T  # Transpose to [8400, 6]

        boxes_raw, scores_raw = [], []
        for row in out:
            x_c, y_c, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf   = float(class_scores[cls_id])

            if conf < self.conf_thresh:
                continue
            if cls_id == 1 and conf < self.fall_thresh:
                continue

            # Normalize coords are in [0,1] for input_size space
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size

            x1 = int((x_c - w / 2) * self.input_size * scale_x)
            y1 = int((y_c - h / 2) * self.input_size * scale_y)
            x2 = int((x_c + w / 2) * self.input_size * scale_x)
            y2 = int((y_c + h / 2) * self.input_size * scale_y)

            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            boxes_raw.append([x1, y1, x2, y2, cls_id, conf])
            scores_raw.append(conf)

        if not boxes_raw:
            return []

        # OpenCV NMS
        boxes_for_nms = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_raw]
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, scores_raw,
            self.conf_thresh, self.nms_thresh
        )
        if isinstance(indices, tuple) and len(indices) == 0:
            return []

        detections = []
        for i in indices.flatten():
            b = boxes_raw[i]
            cls_name = self.CLASS_NAMES.get(b[4], f"class_{b[4]}")
            detections.append({
                "bbox": [b[0], b[1], b[2], b[3]],
                "class": cls_name,
                "class_id": b[4],
                "confidence": b[5]
            })
        return detections
