"""
Frame Preprocessor
Applies:
  - Resize to model input size
  - CLAHE (adaptive histogram equalization) for low-light
  - Gaussian denoising
  - Pixel normalization [0, 1]
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Preprocessing pipeline optimized for edge devices.
    Keeps LAB color space CLAHE to handle:
      - Variable lighting (key challenge in home environments)
      - Night / dim lamp conditions
    """

    def __init__(self, config: dict):
        self.config      = config
        self.resize_w    = config.get("resize_width", 640)
        self.resize_h    = config.get("resize_height", 640)
        self.normalize   = config.get("normalize", True)
        self.blur_kernel = config.get("blur_kernel", 3)

        # CLAHE on the L channel of LAB
        self.clahe_enabled = config.get("clahe_enabled", True)
        if self.clahe_enabled:
            clip  = config.get("clahe_clip_limit", 2.0)
            grid  = tuple(config.get("clahe_grid_size", [8, 8]))
            self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        else:
            self.clahe = None

        logger.info(f"FrameProcessor: resize=({self.resize_w},{self.resize_h}), "
                    f"CLAHE={self.clahe_enabled}, blur={self.blur_kernel}")

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        Input:  BGR uint8 frame from camera
        Output: BGR uint8 frame ready for YOLO / MediaPipe
        """
        if frame is None:
            raise ValueError("FrameProcessor received None frame")

        # 1. Resize
        if frame.shape[1] != self.resize_w or frame.shape[0] != self.resize_h:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h),
                               interpolation=cv2.INTER_LINEAR)

        # 2. CLAHE (light normalization)
        if self.clahe_enabled and self.clahe is not None:
            frame = self._apply_clahe(frame)

        # 3. Gentle Gaussian denoising
        if self.blur_kernel > 1:
            k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            frame = cv2.GaussianBlur(frame, (k, k), 0)

        return frame

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel to equalize brightness."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def for_tflite(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert preprocessed frame to float32 normalized for TFLite FLOAT32 input.
        Not needed for INT8 path — YOLODetector handles that internally.
        """
        return frame.astype(np.float32) / 255.0
