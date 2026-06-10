#!/usr/bin/env python3
"""
Elder Watch - Computer Vision-Based Elder Monitoring System
University of Ruhuna | EE7204 / EC7205
Main entry point
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XNNPACK_FORCE_DISABLE"] = "1"
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import argparse
import logging
import sys
import signal
import time
from pathlib import Path

import cv2
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import YOLODetector
from src.pose.mediapipe_pose import MediaPipePoseEstimator
from src.preprocessing.frame_processor import FrameProcessor
from src.preprocessing.optical_flow import OpticalFlowAnalyzer
from src.classification.fall_classifier import FallClassifier
from src.alerts.alert_manager import AlertManager
from src.utils.logger import setup_logger
from src.utils.fps_counter import FPSCounter
from src.utils.visualizer import Visualizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Elder Watch - Fall Detection System")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: camera index (0), video file path, or RTSP URL")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (for RPi deployment)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: use webcam and show all visualizations")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


class ElderWatchSystem:
    """
    Main orchestrator for the Elder Watch fall detection pipeline.

    Pipeline:
        Camera → Preprocess → YOLO Detect → (MediaPipe Pose if enabled) →
        Optical Flow → Classify → Alert
    """

    def __init__(self, config: dict, headless: bool = False):
        self.config = config
        self.headless = headless
        self.running = False

        self.logger = logging.getLogger("ElderWatch")
        self.logger.info("Initializing Elder Watch System...")

        # ── Components ──────────────────────────────────────────────
        self.frame_processor = FrameProcessor(config["preprocessing"])
        self.yolo_detector   = YOLODetector(config["yolo"])

        # Warm up YOLO — forces XNNPACK/TFLite to fully init BEFORE MediaPipe
        import numpy as np
        _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo_detector.detect(_dummy)
        self.logger.info("YOLO warmup complete.")

        self.pose_estimator  = MediaPipePoseEstimator(config["mediapipe"])
        self.flow_analyzer   = OpticalFlowAnalyzer(config["preprocessing"])
        self.classifier      = FallClassifier(config["classification"])
        self.alert_manager   = AlertManager(config["alert"])
        self.fps_counter     = FPSCounter(smoothing=30)

        if not headless:
            self.visualizer = Visualizer(config["display"])
        else:
            self.visualizer = None

        self.logger.info("All components initialized successfully.")

    def open_camera(self, source) -> cv2.VideoCapture:
        """Open video source: integer index, filepath, or RTSP URL."""
        if source is None:
            source = self.config["camera"]["device_id"]

        # Try integer index
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Apply camera settings if it's a live camera
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config["camera"]["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
            cap.set(cv2.CAP_PROP_FPS,          self.config["camera"]["fps"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency on RPi

        self.logger.info(f"Camera opened: {source} @ "
                         f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                         f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
        return cap

    def process_frame(self, frame):
        """
        Full inference pipeline for a single frame.
        Returns: annotated_frame, fall_state, detections
        """
        # 1. Preprocessing
        processed = self.frame_processor.process(frame)

        # 2. YOLO detection — keep both 'person' and 'fall' class detections
        detections = self.yolo_detector.detect(processed)
        person_detections = [d for d in detections if d["class"] in ("person", "fall")]

        if not person_detections:
            # No person visible — reset classifier state
            self.classifier.reset_state()
            self.flow_analyzer.update(processed)
            return frame, "NORMAL", []

        # 3. Optical flow for motion analysis
        flow_magnitude = self.flow_analyzer.update(processed)

        # 4. Build feature dicts — use MediaPipe pose if enabled, else YOLO-only
        pose_features_list = []
        img_h, img_w = processed.shape[:2]

        for det in person_detections:
            x1, y1, x2, y2 = det["bbox"]
            landmarks    = None
            pose_features = None

            # --- Try MediaPipe if it is enabled ---
            if self.pose_estimator._pose is not None:
                margin = 20
                x1c = max(0, x1 - margin)
                y1c = max(0, y1 - margin)
                x2c = min(img_w, x2 + margin)
                y2c = min(img_h, y2 + margin)
                crop = processed[y1c:y2c, x1c:x2c]
                if crop.size > 0:
                    landmarks, pose_features = self.pose_estimator.estimate(crop)
                    if pose_features:
                        pose_features["bbox"]           = det["bbox"]
                        pose_features["class"]          = det["class"]
                        pose_features["det_confidence"] = det["confidence"]
                        pose_features["flow_magnitude"] = flow_magnitude
                        pose_features["landmarks"]      = landmarks
                        pose_features["crop_offset"]    = (x1c, y1c)

            # --- YOLO-only fallback (MediaPipe disabled or no landmarks found) ---
            if pose_features is None:
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                pose_features = {
                    "bbox":             det["bbox"],
                    "class":            det["class"],
                    "det_confidence":   det["confidence"],
                    "flow_magnitude":   flow_magnitude,
                    "aspect_ratio":     bh / bw,
                    "com_y":            ((y1 + y2) / 2) / img_h,
                    "torso_angle_deg":  0.0,
                    "avg_knee_angle":   180.0,
                    "head_below_hips":  False,
                    "velocity":         flow_magnitude,
                    "landmarks":        None,
                    "crop_offset":      (x1, y1),
                }

            pose_features_list.append(pose_features)

        # 5. Classification
        fall_state = self.classifier.classify(pose_features_list) if pose_features_list else "NORMAL"

        # 6. Trigger alert if needed
        if fall_state in ("FALL_DETECTED", "INACTIVE_ALERT"):
            snapshot = frame.copy() if self.config["alert"]["capture_snapshot"] else None
            self.alert_manager.trigger(fall_state, snapshot)

        # 7. Visualize (if not headless)
        annotated = frame
        if self.visualizer:
            annotated = self.visualizer.draw(
                frame=frame,
                detections=person_detections,
                pose_features_list=pose_features_list,
                fall_state=fall_state,
                fps=self.fps_counter.fps,
                flow_magnitude=flow_magnitude
            )

        return annotated, fall_state, person_detections

    def run(self, source=None):
        """Main loop."""
        self.running = True
        cap = self.open_camera(source)

        self.logger.info("Starting Elder Watch monitoring loop...")
        self.logger.info("Press Ctrl+C or 'q' to stop.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    # Clean exit for finite video files
                    if isinstance(source, str) and not source.startswith("rtsp"):
                        self.logger.info("Video file ended.")
                        break
                    # Live camera — attempt reconnect
                    self.logger.warning("Frame read failed – attempting reconnect...")
                    time.sleep(0.5)
                    cap.release()
                    try:
                        cap = self.open_camera(source)
                    except RuntimeError:
                        self.logger.error("Cannot reconnect to camera. Exiting.")
                        break
                    continue

                # Optional vertical flip for ceiling mount
                if self.config["camera"].get("flip_vertical"):
                    frame = cv2.flip(frame, 0)

                # Process
                annotated, fall_state, _ = self.process_frame(frame)
                self.fps_counter.tick()

                # Display
                if not self.headless and annotated is not None:
                    cv2.imshow(self.config["display"]["window_title"], annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # q or ESC
                        self.logger.info("User requested stop.")
                        break

                # Log state periodically
                if self.fps_counter.frame_count % 300 == 0:
                    self.logger.info(
                        f"[Frame {self.fps_counter.frame_count}] "
                        f"FPS={self.fps_counter.fps:.1f} | State={fall_state}"
                    )

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received.")
        finally:
            self.running = False
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.logger.info("Elder Watch stopped.")

    def stop(self, *_):
        self.logger.info("Stop signal received.")
        self.running = False


def main():
    args = parse_args()
    config = load_config(args.config)

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(log_level, config["system"].get("log_dir", "logs/"))

    system = ElderWatchSystem(
        config=config,
        headless=args.headless
    )

    # Graceful shutdown on SIGTERM (systemd)
    signal.signal(signal.SIGTERM, system.stop)

    source = args.source if args.source else None
    system.run(source=source)


if __name__ == "__main__":
    main()