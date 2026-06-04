"""Utility: Rolling FPS counter"""

import time
from collections import deque


class FPSCounter:
    def __init__(self, smoothing: int = 30):
        self._times      = deque(maxlen=smoothing)
        self.frame_count = 0
        self.fps         = 0.0

    def tick(self):
        now = time.perf_counter()
        self._times.append(now)
        self.frame_count += 1
        if len(self._times) >= 2:
            elapsed  = self._times[-1] - self._times[0]
            self.fps = (len(self._times) - 1) / max(elapsed, 1e-6)
