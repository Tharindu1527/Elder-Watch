"""
MediaPipe Pose Estimator
Extracts 33 body landmarks and computes biomechanical features
used by the fall classifier.
"""

import logging
import math
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MediaPipePoseEstimator:
    """
    Wraps MediaPipe Pose for landmark extraction and feature computation.
    Computes:
      - Body aspect ratio (height / width)
      - Centre of mass normalized y-coordinate
      - Torso inclination angle from vertical
      - Hip-knee joint angles
      - Head-to-hip velocity (frame-to-frame delta)
    """

    # MediaPipe Pose landmark indices
    LM = {
        "nose":           0,
        "left_shoulder":  11,
        "right_shoulder": 12,
        "left_hip":       23,
        "right_hip":      24,
        "left_knee":      25,
        "right_knee":     26,
        "left_ankle":     27,
        "right_ankle":    28,
        "left_heel":      29,
        "right_heel":     30,
    }

    def __init__(self, config: dict):
        self.config = config
        self._prev_landmarks: Optional[np.ndarray] = None
        self._pose = None
        self._mp_pose = None
        self._mp_drawing = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        if not self.config.get("enabled", True):
            logger.info("MediaPipe disabled - YOLO-only mode.")
            self._pose = None
            return
        try:
            import mediapipe as mp
            self._mp_pose    = mp.solutions.pose
            self._mp_drawing = mp.solutions.drawing_utils
            self._pose = self._mp_pose.Pose(
                min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5),
                model_complexity=self.config.get("model_complexity", 0),
                enable_segmentation=self.config.get("enable_segmentation", False),
                smooth_landmarks=self.config.get("smooth_landmarks", True),
            )
            logger.info("MediaPipe Pose initialized (model_complexity="
                        f"{self.config.get('model_complexity', 0)})")
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
            raise

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def estimate(self, crop: np.ndarray
                 ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Run pose estimation on a cropped person region.
        Returns:
            landmarks: MediaPipe landmark object (for drawing)
            features:  dict of biomechanical features
        """
        if crop is None or crop.size == 0:
            return None, None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark
        h, w = crop.shape[:2]

        # Build numpy array [33, 3] of (x_norm, y_norm, visibility)
        pts = np.array([[lm[i].x, lm[i].y, lm[i].visibility]
                        for i in range(33)], dtype=np.float32)

        features = self._compute_features(pts, w, h)
        self._prev_landmarks = pts

        return results.pose_landmarks, features

    def draw_landmarks(self, image: np.ndarray, landmarks, offset=(0, 0)):
        """Draw pose skeleton on the full frame (with crop offset)."""
        if landmarks is None or self._mp_drawing is None:
            return image
        # Draw on a temp canvas then overlay (handles offset)
        temp = image.copy()
        self._mp_drawing.draw_landmarks(
            temp,
            landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=self._mp_drawing.DrawingSpec(
                color=(0, 128, 255), thickness=2),
        )
        return temp

    def reset(self):
        self._prev_landmarks = None

    # ──────────────────────────────────────────────────────────────────
    # Feature computation
    # ──────────────────────────────────────────────────────────────────

    def _compute_features(self, pts: np.ndarray, w: int, h: int) -> Dict[str, Any]:
        """
        Compute biomechanical features from normalized landmarks.
        All coordinates are in [0, 1] normalized space.
        """
        def xy(idx): return pts[idx, :2]

        # Key points
        nose          = xy(self.LM["nose"])
        l_shoulder    = xy(self.LM["left_shoulder"])
        r_shoulder    = xy(self.LM["right_shoulder"])
        l_hip         = xy(self.LM["left_hip"])
        r_hip         = xy(self.LM["right_hip"])
        l_knee        = xy(self.LM["left_knee"])
        r_knee        = xy(self.LM["right_knee"])
        l_ankle       = xy(self.LM["left_ankle"])
        r_ankle       = xy(self.LM["right_ankle"])

        # ── Bounding box of the person ──────────────────────────────
        visible = pts[pts[:, 2] > 0.3, :2]
        if len(visible) < 3:
            return {}

        x_min, y_min = visible.min(axis=0)
        x_max, y_max = visible.max(axis=0)
        bb_w = max(x_max - x_min, 1e-5)
        bb_h = max(y_max - y_min, 1e-5)
        aspect_ratio = bb_h / bb_w  # > 2 → standing; < 1 → prone

        # ── Centre of mass (mid-hip) ────────────────────────────────
        mid_hip = (l_hip + r_hip) / 2
        com_y   = float(mid_hip[1])   # 0 = top, 1 = bottom

        # ── Torso inclination angle from vertical ───────────────────
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        torso_vec    = mid_hip - mid_shoulder
        # Angle w.r.t. vertical axis (y-axis)
        torso_angle_deg = math.degrees(
            math.atan2(abs(torso_vec[0]), abs(torso_vec[1]) + 1e-6)
        )

        # ── Hip-Knee-Ankle angle (both sides) ───────────────────────
        def angle_3pts(a, b, c):
            """Angle at b formed by a-b-c."""
            ba = a - b
            bc = c - b
            cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))

        l_hip_knee_angle = angle_3pts(l_hip, l_knee, l_ankle)
        r_hip_knee_angle = angle_3pts(r_hip, r_knee, r_ankle)
        avg_knee_angle   = (l_hip_knee_angle + r_hip_knee_angle) / 2

        # ── Head vertical position ──────────────────────────────────
        # If nose y > mid_hip y (nose below hips) → likely fallen
        head_below_hips = float(nose[1]) > float(mid_hip[1])
        nose_y          = float(nose[1])

        # ── Velocity (landmark displacement from previous frame) ────
        velocity = 0.0
        if self._prev_landmarks is not None:
            delta      = pts[:, :2] - self._prev_landmarks[:, :2]
            # Weighted by visibility
            vis_mask   = pts[:, 2] > 0.3
            if vis_mask.sum() > 0:
                velocity = float(np.linalg.norm(delta[vis_mask], axis=1).mean())

        # ── Shoulder width (proxy for frontal vs. overhead view) ────
        shoulder_width = float(np.linalg.norm(l_shoulder - r_shoulder))

        return {
            "aspect_ratio":     aspect_ratio,
            "com_y":            com_y,
            "torso_angle_deg":  torso_angle_deg,
            "avg_knee_angle":   avg_knee_angle,
            "head_below_hips":  head_below_hips,
            "nose_y":           nose_y,
            "velocity":         velocity,
            "shoulder_width":   shoulder_width,
            "bb_w":             bb_w,
            "bb_h":             bb_h,
            "mid_hip":          mid_hip.tolist(),
            "mid_shoulder":     mid_shoulder.tolist(),
            "raw_landmarks":    pts.tolist(),
        }
