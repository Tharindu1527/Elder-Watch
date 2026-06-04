"""
Fall Classifier
Multi-stage decision system combining:
  - YOLO confidence
  - Pose-based biomechanical rules
  - Temporal smoothing (N-frame confirmation)
  - Inactivity detection
"""

import logging
import time
from collections import deque
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── States ──────────────────────────────────────────────────────────
NORMAL         = "NORMAL"
NEAR_FALL      = "NEAR_FALL"
FALL_DETECTED  = "FALL_DETECTED"
INACTIVE_ALERT = "INACTIVE_ALERT"

STATES = [NORMAL, NEAR_FALL, FALL_DETECTED, INACTIVE_ALERT]


class FallClassifier:
    """
    State-machine classifier for fall detection.

    Transitions:
      NORMAL → NEAR_FALL    : 1+ pose indicators (rapid velocity, low COM, small aspect ratio)
      NEAR_FALL → FALL       : N consecutive frames all showing fall indicators
      FALL_DETECTED → NORMAL : person stands up / no longer detected
      NORMAL → INACTIVE      : person motionless > threshold seconds
    """

    def __init__(self, config: dict):
        self.config = config

        # Thresholds
        self.aspect_thresh     = config.get("body_aspect_ratio_threshold", 2.0)
        self.com_thresh        = config.get("center_of_mass_threshold", 0.45)
        self.torso_thresh      = config.get("torso_angle_threshold", 30)
        self.knee_thresh       = config.get("hip_knee_angle_threshold", 120)
        self.vel_thresh        = config.get("velocity_threshold", 0.08)
        self.inactivity_secs   = config.get("inactivity_threshold_seconds", 5)
        self.confirm_frames    = config.get("confirmation_frames", 3)
        self.near_fall_frames  = config.get("near_fall_frames", 2)

        # State machine
        self._state             = NORMAL
        self._fall_vote_buffer  = deque(maxlen=self.confirm_frames)
        self._near_vote_buffer  = deque(maxlen=self.near_fall_frames)
        self._last_motion_time  = time.time()
        self._last_velocity     = 0.0
        self._inactive_flagged  = False

        logger.info("FallClassifier initialized.")
        logger.debug(f"Thresholds: aspect<{self.aspect_thresh} | "
                     f"com_y<{self.com_thresh} | torso>{self.torso_thresh}° | "
                     f"confirm={self.confirm_frames}fr")

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def classify(self, pose_features_list: List[Dict[str, Any]]) -> str:
        """
        Classify the current frame given a list of per-person pose features.
        Returns one of: NORMAL, NEAR_FALL, FALL_DETECTED, INACTIVE_ALERT
        """
        if not pose_features_list:
            self.reset_state()
            return NORMAL

        # Use the person with highest detection confidence
        primary = max(pose_features_list, key=lambda f: f.get("det_confidence", 0))

        # ── Score the primary person ─────────────────────────────────
        fall_score, indicators = self._score_features(primary)

        # ── Update velocity / inactivity tracking ────────────────────
        velocity = primary.get("velocity", 0)
        flow     = primary.get("flow_magnitude", 0)
        combined_motion = max(velocity, flow)

        if combined_motion > self.vel_thresh * 0.3:
            self._last_motion_time = time.time()
            self._inactive_flagged = False

        # ── Check inactivity ─────────────────────────────────────────
        time_still = time.time() - self._last_motion_time
        inactivity_detected = (
            time_still > self.inactivity_secs
            and self._state != INACTIVE_ALERT
        )

        # ── Vote buffers ─────────────────────────────────────────────
        is_fall_frame  = fall_score >= 3
        is_near_frame  = fall_score >= 2

        self._fall_vote_buffer.append(is_fall_frame)
        self._near_vote_buffer.append(is_near_frame)

        # ── State transitions ────────────────────────────────────────
        prev_state = self._state

        if all(self._fall_vote_buffer) and len(self._fall_vote_buffer) == self.confirm_frames:
            new_state = FALL_DETECTED
        elif inactivity_detected and self._state == FALL_DETECTED:
            new_state = INACTIVE_ALERT
        elif all(self._near_vote_buffer) and len(self._near_vote_buffer) == self.near_fall_frames:
            new_state = NEAR_FALL if self._state == NORMAL else self._state
        else:
            # Recovery: if person is upright again, reset
            if self._state in (FALL_DETECTED,) and fall_score <= 1:
                new_state = NORMAL
                self._clear_buffers()
            else:
                new_state = self._state

        self._state = new_state

        if prev_state != self._state:
            logger.info(f"[Classifier] State: {prev_state} → {self._state} "
                        f"| score={fall_score} | indicators={indicators}")

        return self._state

    def reset_state(self):
        """Call when no person is detected."""
        if self._state != NORMAL:
            logger.info(f"[Classifier] No person detected — resetting to NORMAL")
        self._state = NORMAL
        self._clear_buffers()
        self._last_motion_time = time.time()

    @property
    def state(self) -> str:
        return self._state

    # ──────────────────────────────────────────────────────────────────
    # Feature scoring
    # ──────────────────────────────────────────────────────────────────

    def _score_features(self, features: Dict[str, Any]):
        """
        Score the biomechanical features.
        Returns (score: int, active_indicators: list[str])
        Each criterion contributes 1 point; ≥3 → fall.
        """
        score = 0
        active = []

        # 1. Body aspect ratio — low = prone / horizontal
        aspect = features.get("aspect_ratio", 999)
        if aspect < self.aspect_thresh:
            score += 1
            active.append(f"aspect_ratio={aspect:.2f}")

        # 2. Centre of mass — normalized y low means person is high in frame
        #    For ceiling cam: if com_y is small, person is lying flat
        com_y = features.get("com_y", 0.5)
        if com_y < self.com_thresh:
            score += 1
            active.append(f"com_y={com_y:.2f}")

        # 3. Torso angle from vertical
        torso = features.get("torso_angle_deg", 0)
        if torso > self.torso_thresh:
            score += 1
            active.append(f"torso_angle={torso:.1f}°")

        # 4. Head below hips (strong indicator for frontal camera)
        if features.get("head_below_hips", False):
            score += 1
            active.append("head_below_hips")

        # 5. High velocity (sudden motion = falling event)
        vel = features.get("velocity", 0)
        if vel > self.vel_thresh:
            score += 1
            active.append(f"velocity={vel:.3f}")

        # 6. YOLO fall class
        det_cls = features.get("class", "person")
        det_conf = features.get("det_confidence", 0)
        if det_cls == "fall" and det_conf > 0.5:
            score += 2  # Strong signal — double weight
            active.append(f"yolo_fall_class(conf={det_conf:.2f})")

        # 7. Low knee angle — bent posture / collapsed
        knee_ang = features.get("avg_knee_angle", 180)
        if knee_ang < self.knee_thresh:
            score += 1
            active.append(f"knee_angle={knee_ang:.1f}°")

        return score, active

    def _clear_buffers(self):
        self._fall_vote_buffer.clear()
        self._near_vote_buffer.clear()
