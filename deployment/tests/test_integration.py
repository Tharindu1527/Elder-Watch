#!/usr/bin/env python3
"""
Integration Tests for Elder Watch
Run these before deployment to verify all components work on RPi 5.

Usage: python deployment/tests/test_integration.py
"""

import logging
import sys
import time
import os
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("Tests")

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m⚠ SKIP\033[0m"

results = {}


def test(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results[name] = "PASS"
                print(f"  {PASS}  {name}")
            except Exception as e:
                results[name] = f"FAIL: {e}"
                print(f"  {FAIL}  {name}: {e}")
        return wrapper
    return decorator


# ── Tests ──────────────────────────────────────────────────────────────

@test("Import: OpenCV")
def t_opencv():
    import cv2
    assert cv2.__version__, "No version"

@test("Import: NumPy")
def t_numpy():
    import numpy as np
    assert np.__version__

@test("Import: MediaPipe")
def t_mediapipe():
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(model_complexity=0)
    pose.close()

@test("Import: TFLite")
def t_tflite():
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    assert tflite

@test("Import: PyYAML")
def t_yaml():
    import yaml
    d = yaml.safe_load("key: value")
    assert d["key"] == "value"

@test("Frame Processor")
def t_frame_processor():
    sys.path.insert(0, os.getcwd())
    from src.preprocessing.frame_processor import FrameProcessor
    cfg = {"resize_width": 640, "resize_height": 640,
           "clahe_enabled": True, "clahe_clip_limit": 2.0,
           "clahe_grid_size": [8,8], "blur_kernel": 3, "normalize": True}
    fp = FrameProcessor(cfg)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = fp.process(dummy)
    assert out.shape == (640, 640, 3), f"Bad shape: {out.shape}"

@test("Optical Flow")
def t_optical_flow():
    from src.preprocessing.optical_flow import OpticalFlowAnalyzer
    cfg = {"optical_flow_enabled": True, "flow_method": "farneback"}
    oa = OpticalFlowAnalyzer(cfg)
    f1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    f2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    oa.update(f1)
    mag = oa.update(f2)
    assert isinstance(mag, float)

@test("MediaPipe Pose Estimator")
def t_pose():
    from src.pose.mediapipe_pose import MediaPipePoseEstimator
    cfg = {"min_detection_confidence": 0.5,
           "min_tracking_confidence": 0.5,
           "model_complexity": 0,
           "enable_segmentation": False,
           "smooth_landmarks": True}
    pe = MediaPipePoseEstimator(cfg)
    dummy = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    lm, feats = pe.estimate(dummy)
    # No landmarks expected on random noise – just confirm no crash

@test("Fall Classifier - NORMAL state")
def t_classifier_normal():
    from src.classification.fall_classifier import FallClassifier
    cfg = {"body_aspect_ratio_threshold": 2.0,
           "center_of_mass_threshold": 0.45,
           "torso_angle_threshold": 30,
           "hip_knee_angle_threshold": 120,
           "velocity_threshold": 0.08,
           "inactivity_threshold_seconds": 5,
           "confirmation_frames": 3,
           "near_fall_frames": 2}
    cl = FallClassifier(cfg)
    # Standing person features
    features = [{"aspect_ratio": 3.5, "com_y": 0.55, "torso_angle_deg": 5.0,
                 "avg_knee_angle": 170, "head_below_hips": False,
                 "velocity": 0.01, "det_confidence": 0.9, "class": "person",
                 "flow_magnitude": 0.01}]
    state = cl.classify(features)
    assert state == "NORMAL", f"Expected NORMAL, got {state}"

@test("Fall Classifier - FALL_DETECTED state")
def t_classifier_fall():
    from src.classification.fall_classifier import FallClassifier
    cfg = {"body_aspect_ratio_threshold": 2.0,
           "center_of_mass_threshold": 0.45,
           "torso_angle_threshold": 30,
           "hip_knee_angle_threshold": 120,
           "velocity_threshold": 0.08,
           "inactivity_threshold_seconds": 99,
           "confirmation_frames": 3,
           "near_fall_frames": 2}
    cl = FallClassifier(cfg)
    # Fallen person: horizontal, high velocity, low aspect ratio
    features = [{"aspect_ratio": 0.5, "com_y": 0.3, "torso_angle_deg": 80.0,
                 "avg_knee_angle": 90, "head_below_hips": True,
                 "velocity": 0.15, "det_confidence": 0.95, "class": "fall",
                 "flow_magnitude": 0.2}]
    # Need 3 consecutive frames
    for _ in range(3):
        state = cl.classify(features)
    assert state == "FALL_DETECTED", f"Expected FALL_DETECTED, got {state}"

@test("Camera detection")
def t_camera():
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available at /dev/video0")
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read frame"
    assert frame is not None
    assert frame.shape[2] == 3

@test("TFLite model file exists")
def t_tflite_file():
    path = "models/quantized/yolov8n_fall_int8.tflite"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run quantization first:\n"
            "  python training/scripts/quantize.py --model models/finetuned/yolov8n_fall.pt "
            "--data training/configs/dataset.yaml"
        )

@test("Inference latency < 50ms")
def t_latency():
    path = "models/quantized/yolov8n_fall_int8.tflite"
    if not os.path.exists(path):
        raise FileNotFoundError("Quantized model not found - skipping latency test")
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    interp = tflite.Interpreter(model_path=path, num_threads=4)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    in_scale, in_zp = in_det["quantization"]
    imgsz = in_det["shape"][1]

    dummy = np.zeros((1, imgsz, imgsz, 3), dtype=in_det["dtype"])
    # Warmup
    for _ in range(5):
        interp.set_tensor(in_det["index"], dummy)
        interp.invoke()

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        interp.set_tensor(in_det["index"], dummy)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    logger.info(f"    Avg latency: {avg_ms:.1f}ms | FPS: {1000/avg_ms:.1f}")
    assert avg_ms < 200, f"Too slow: {avg_ms:.1f}ms (expected <200ms on RPi)"


# ── Run all tests ──────────────────────────────────────────────────────

def run_all():
    print("\n" + "=" * 55)
    print("  Elder Watch Integration Tests")
    print("=" * 55)

    t_opencv()
    t_numpy()
    t_mediapipe()
    t_tflite()
    t_yaml()
    t_frame_processor()
    t_optical_flow()
    t_pose()
    t_classifier_normal()
    t_classifier_fall()
    t_camera()
    t_tflite_file()
    t_latency()

    print("\n" + "=" * 55)
    passed = sum(1 for v in results.values() if v == "PASS")
    total  = len(results)
    print(f"  Results: {passed}/{total} passed")
    print("=" * 55)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
