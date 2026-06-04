"""Utility: Logger setup"""

import logging
import os
from datetime import datetime


def setup_logger(level=logging.INFO, log_dir: str = "logs/"):
    os.makedirs(log_dir, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"elder_watch_{ts}.log")

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s",
        datefmt="%H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.info(f"Logging initialized → {logfile}")
