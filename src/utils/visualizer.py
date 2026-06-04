"""
Visualizer
Draws bounding boxes, pose skeleton, fall state, and FPS overlay on frames.
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

STATE_COLORS = {
    "NORMAL":         (0, 200, 0),      # Green
    "NEAR_FALL":      (0, 165, 255),    # Orange
    "FALL_DETECTED":  (0, 0, 255),      # Red
    "INACTIVE_ALERT": (255, 0, 180),    # Magenta
}


class Visualizer:
    """Draws all debug overlays on frames for the preview window."""

    def __init__(self, config: dict):
        self.config       = config
        self.draw_skel    = config.get("draw_skeleton", True)
        self.draw_bbox    = config.get("draw_bounding_box", True)
        self.show_fps     = config.get("show_fps", True)
        self.show_conf    = config.get("show_confidence", True)
        self.draw_flow    = config.get("draw_optical_flow", False)

    def draw(self,
             frame: np.ndarray,
             detections: List[Dict[str, Any]],
             pose_features_list: List[Dict[str, Any]],
             fall_state: str,
             fps: float,
             flow_magnitude: float = 0.0) -> np.ndarray:

        out = frame.copy()
        color = STATE_COLORS.get(fall_state, (200, 200, 200))

        # ── Bounding boxes ───────────────────────────────────────────
        if self.draw_bbox:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls  = det.get("class", "person")
                conf = det.get("confidence", 0)
                label = f"{cls} {conf:.2f}" if self.show_conf else cls
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── Pose features text ───────────────────────────────────────
        for i, feats in enumerate(pose_features_list):
            y_start = 60 + i * 80
            info_lines = [
                f"Aspect: {feats.get('aspect_ratio', 0):.2f}",
                f"CoM_y:  {feats.get('com_y', 0):.2f}",
                f"Torso:  {feats.get('torso_angle_deg', 0):.1f}°",
                f"Vel:    {feats.get('velocity', 0):.3f}",
            ]
            for j, line in enumerate(info_lines):
                cv2.putText(out, line, (10, y_start + j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── State banner ─────────────────────────────────────────────
        h, w = out.shape[:2]
        banner_h = 36
        # Semi-transparent banner at bottom
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        state_text = f"STATE: {fall_state}"
        (tw, th), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(out, state_text,
                    ((w - tw) // 2, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # ── FPS ──────────────────────────────────────────────────────
        if self.show_fps:
            cv2.putText(out, f"FPS: {fps:.1f}",
                        (w - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # ── Flow magnitude bar ───────────────────────────────────────
        bar_len = int(min(flow_magnitude * 500, w - 20))
        cv2.rectangle(out, (10, 30), (10 + bar_len, 44), (100, 200, 255), -1)
        cv2.putText(out, f"Motion: {flow_magnitude:.3f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

        return out
