"""
Optical Flow Analyzer
Computes dense (Farneback) or sparse (Lucas-Kanade) optical flow
to measure inter-frame motion — used as a velocity signal for the classifier.
"""

import logging
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class OpticalFlowAnalyzer:
    """
    Maintains previous frame and computes optical flow magnitude.
    On RPi 5 with 4 GB RAM, Farneback at half resolution is real-time (~10ms).
    """

    def __init__(self, config: dict):
        flow_cfg        = config.get("optical_flow_enabled", True)
        self.enabled    = flow_cfg
        self.method     = config.get("flow_method", "farneback")
        self._prev_gray: Optional[np.ndarray] = None

        # Farneback parameters (tuned for RPi speed)
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        logger.info(f"OpticalFlowAnalyzer: enabled={self.enabled}, method={self.method}")

    def update(self, frame: np.ndarray) -> float:
        """
        Update with the latest frame and return mean flow magnitude.
        Returns 0.0 if disabled or no previous frame.
        """
        if not self.enabled:
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample to quarter size for speed on RPi
        small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

        if self._prev_gray is None:
            self._prev_gray = small
            return 0.0

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, small, None, **self.fb_params
            )
            # Magnitude of flow vectors
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion = float(np.mean(mag))
        except cv2.error as e:
            logger.warning(f"Optical flow error: {e}")
            motion = 0.0

        self._prev_gray = small
        return motion

    def reset(self):
        self._prev_gray = None

    def get_flow_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Return HSV-colorized flow image for debugging."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, **self.fb_params
        )
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
