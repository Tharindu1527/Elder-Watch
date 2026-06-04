#!/usr/bin/env python3
"""
Dataset Preparation Script — FREE & OPEN-SOURCE DATASETS ONLY
Elder Watch | University of Ruhuna EE7204/EC7205

Supported datasets (all free, no registration needed):
  1. Roboflow Fall Detection (4,497 images, pre-annotated YOLO format)
     → https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8
  2. UR Fall Detection Dataset (URFD, 30 fall + 40 ADL sequences, RGB frames)
     → https://fenix.ur.edu.pl/~mkepski/ds/uf.html
  3. Le2i Fall Detection Dataset (221 videos, home + coffee room, Kaggle mirror)
     → https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
  4. Multiple Cameras Fall Dataset (MCFD, 8 scenarios × 3 cameras)
     → http://www.iro.umontreal.ca/~labimage/Dataset/
  5. Roboflow UR Fall (2,000 annotated images, pre-split YOLO format)
     → https://universe.roboflow.com/fall-detection-w7nxl/ur-fall

YOLO output format:
  datasets/processed/
    images/{train,val,test}/  *.jpg
    labels/{train,val,test}/  *.txt  (class_id cx cy w h)

Classes:
  0 = person  (standing / ADL / normal)
  1 = fall    (fallen / falling)

Usage examples:
  # Download everything automatically (needs roboflow pip package + API key)
  python training/scripts/prepare_datasets.py --download-roboflow --api-key YOUR_KEY

  # Use Roboflow dataset already downloaded as ZIP
  python training/scripts/prepare_datasets.py --roboflow datasets/raw/roboflow_fall

  # Use URFD raw frames directory
  python training/scripts/prepare_datasets.py --urfd datasets/raw/urfd

  # Use Le2i videos directory
  python training/scripts/prepare_datasets.py --le2i datasets/raw/le2i

  # Mix all available sources
  python training/scripts/prepare_datasets.py \\
      --roboflow  datasets/raw/roboflow_fall \\
      --urfd      datasets/raw/urfd \\
      --le2i      datasets/raw/le2i \\
      --mcfd      datasets/raw/mcfd \\
      --out       datasets/processed

  # Quick sanity test (limit frames)
  python training/scripts/prepare_datasets.py --urfd datasets/raw/urfd --max-frames 20
"""

import argparse
import logging
import os
import glob
import shutil
import random
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("PrepareDataset")

# ── Constants ────────────────────────────────────────────────────────
PERSON_CLASS = 0
FALL_CLASS   = 1
TARGET_FPS   = 5    # Extract 5 frames/sec from raw videos
IMG_SIZE     = 640
JPEG_QUALITY = 90
RANDOM_SEED  = 42


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare free fall-detection datasets for YOLOv8 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Dataset paths
    p.add_argument("--roboflow",  type=str, default=None,
                   help="Path to extracted Roboflow Fall Detection ZIP "
                        "(already in YOLO format — fastest option)")
    p.add_argument("--urfd",      type=str, default=None,
                   help="Path to UR Fall Detection Dataset root "
                        "(contains fall-XX/ and adl-XX/ dirs of PNG frames)")
    p.add_argument("--le2i",      type=str, default=None,
                   help="Path to Le2i FDD root (contains Home/, Coffee_room/ dirs of videos)")
    p.add_argument("--mcfd",      type=str, default=None,
                   help="Path to Multiple Cameras Fall Dataset root")

    # Download helpers
    p.add_argument("--download-roboflow", action="store_true",
                   help="Auto-download Roboflow dataset via roboflow Python package")
    p.add_argument("--api-key",  type=str, default=None,
                   help="Roboflow API key (required for --download-roboflow)")

    # Output
    p.add_argument("--out",       type=str, default="datasets/processed")
    p.add_argument("--split",     nargs=3, type=float, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max frames per video (use for quick testing)")
    p.add_argument("--no-augment", action="store_true",
                   help="Skip offline augmentation pass")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════

def resize_and_save(frame: np.ndarray, out_path: str) -> bool:
    """Resize frame to IMG_SIZE and save as JPEG. Returns True on success."""
    try:
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True
    except Exception as e:
        logger.warning(f"Save failed {out_path}: {e}")
        return False


def make_yolo_label(class_id: int,
                    cx: float = 0.5, cy: float = 0.5,
                    w: float  = 0.85, h: float = 0.90) -> str:
    """
    YOLO label line: class_id cx cy w h  (all normalized 0-1).
    Default covers ~85% of frame width and 90% height — suitable when
    a single person fills most of the frame (URFD, Le2i overhead shots).
    Roboflow datasets ship with their own precise annotations.
    """
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def extract_frames(video_path: str,
                   out_dir: str,
                   class_id: int,
                   prefix: str,
                   target_fps: int = TARGET_FPS,
                   max_frames: int = None,
                   fall_start_frame: int = None,
                   fall_end_frame: int   = None) -> List[Tuple[str, int]]:
    """
    Extract frames from a video at target_fps.
    If fall_start_frame / fall_end_frame are provided (Le2i annotations),
    each extracted frame is labelled FALL_CLASS during that window,
    PERSON_CLASS outside it.
    Returns list of (img_path, class_id) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"  Cannot open video: {video_path}")
        return []

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(src_fps / target_fps)))
    vid_id   = Path(video_path).stem
    entries  = []
    fidx     = 0
    saved    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and saved >= max_frames:
            break

        if fidx % interval == 0:
            # Determine per-frame class if annotation windows provided
            if fall_start_frame is not None and fall_end_frame is not None:
                frame_cls = (FALL_CLASS
                             if fall_start_frame <= fidx <= fall_end_frame
                             else PERSON_CLASS)
            else:
                frame_cls = class_id

            fname    = f"{prefix}_{vid_id}_f{fidx:06d}.jpg"
            out_path = os.path.join(out_dir, fname)

            if resize_and_save(frame, out_path):
                entries.append((out_path, frame_cls))
                saved += 1

        fidx += 1

    cap.release()
    return entries


def load_images_from_dir(img_dir: str,
                         class_id: int,
                         prefix: str,
                         out_dir: str) -> List[Tuple[str, int]]:
    """
    Copy/resize pre-extracted image frames from a directory.
    Used for URFD which ships as PNG frame sequences.
    """
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    imgs = []
    for pat in patterns:
        imgs.extend(sorted(Path(img_dir).glob(pat)))

    entries = []
    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        fname    = f"{prefix}_{img_path.stem}_{i:05d}.jpg"
        out_path = os.path.join(out_dir, fname)
        if resize_and_save(frame, out_path):
            entries.append((out_path, class_id))
    return entries


# ════════════════════════════════════════════════════════════════════
# Dataset 1 — Roboflow Fall Detection (pre-annotated YOLO format)
# ════════════════════════════════════════════════════════════════════

def load_roboflow(root: str, out_root: str) -> List[Tuple[str, int, str]]:
    """
    Load a Roboflow YOLO-format dataset that is already split into
    train/valid/test subdirectories.

    Roboflow ZIP structure:
      roboflow_fall/
        train/
          images/  *.jpg
          labels/  *.txt
        valid/
          images/  *.jpg
          labels/  *.txt
        test/
          images/  *.jpg
          labels/  *.txt
        data.yaml

    Returns list of (img_path, label_path, split_name) tuples.
    We return these separately because labels already exist.
    """
    if not root or not os.path.exists(root):
        logger.info("Roboflow root not provided/found — skipping.")
        return []

    root_p = Path(root)
    # Handle nested ZIP structure
    # Some Roboflow ZIPs have an extra top-level folder
    subdirs = [d for d in root_p.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not (root_p / "train").exists():
        root_p = subdirs[0]

    result = []
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for rf_split, our_split in split_map.items():
        img_dir = root_p / rf_split / "images"
        lbl_dir = root_p / rf_split / "labels"
        if not img_dir.exists():
            # Try alternate layout: images/train, labels/train
            img_dir = root_p / "images" / rf_split
            lbl_dir = root_p / "labels" / rf_split

        if not img_dir.exists():
            logger.warning(f"  Roboflow: no images dir for split '{rf_split}'")
            continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        for img_p in imgs:
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            result.append((str(img_p), str(lbl_p) if lbl_p.exists() else None, our_split))

    logger.info(f"Roboflow: found {len(result)} pre-annotated images")
    return result


def download_roboflow_dataset(api_key: str, out_dir: str) -> str:
    """
    Download Roboflow Fall Detection dataset via roboflow Python package.
    Requires: pip install roboflow
    Returns path to downloaded dataset.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise RuntimeError(
            "roboflow package not installed.\n"
            "Run: pip install roboflow\n"
            "Then get your free API key at: https://app.roboflow.com"
        )

    logger.info("Downloading Roboflow Fall Detection dataset (~4,497 images)...")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("fall-detection-ca3o8")
    dataset = project.version(4).download("yolov8", location=out_dir)
    logger.info(f"Downloaded to: {dataset.location}")
    return dataset.location


# ════════════════════════════════════════════════════════════════════
# Dataset 2 — UR Fall Detection Dataset (URFD)
# ════════════════════════════════════════════════════════════════════

def load_urfd(root: str, tmp_dir: str,
              max_frames: int = None) -> List[Tuple[str, int]]:
    """
    UR Fall Detection Dataset — free download from University of Rzeszow.
    Download: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
    Direct ZIP: https://fenix.ur.edu.pl/~mkepski/ds/data/urfall-cam0-falls.zip
                https://fenix.ur.edu.pl/~mkepski/ds/data/urfall-cam0-adls.zip

    Structure after extraction:
      urfd/
        fall-01-cam0/   (PNG frames numbered 000.png, 001.png ...)
        fall-02-cam0/
        ...
        fall-30-cam0/
        adl-01-cam0/
        ...
        adl-40-cam0/

    OR (some mirror versions):
      urfd/
        falls/
          fall-01/ ... fall-30/  (each contains PNG frames)
        adl/
          adl-01/ ... adl-40/
    """
    if not root or not os.path.exists(root):
        logger.info("URFD root not provided/found — skipping.")
        return []

    root_p = Path(root)
    entries = []

    def process_seq_dir(seq_dir: Path, class_id: int):
        """Load all PNG frames from a sequence directory."""
        frames = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
        if not frames:
            return
        prefix = f"urfd_{seq_dir.name}"
        for i, fp in enumerate(frames):
            if max_frames and i >= max_frames:
                break
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            fname = f"{prefix}_{i:05d}.jpg"
            out   = os.path.join(tmp_dir, fname)
            if resize_and_save(frame, out):
                entries.append((out, class_id))

    # Layout A: flat dirs named fall-XX-cam0 and adl-XX-cam0
    fall_dirs = sorted(root_p.glob("fall-*")) + sorted(root_p.glob("falls/fall-*"))
    adl_dirs  = sorted(root_p.glob("adl-*"))  + sorted(root_p.glob("adl/adl-*"))

    # Layout B: nested falls/ and adl/ subdirs
    if not fall_dirs:
        fall_dirs = sorted((root_p / "falls").glob("*")) if (root_p/"falls").exists() else []
    if not adl_dirs:
        adl_dirs  = sorted((root_p / "adl").glob("*"))  if (root_p/"adl").exists()  else []

    for d in fall_dirs:
        if d.is_dir():
            process_seq_dir(d, FALL_CLASS)
    for d in adl_dirs:
        if d.is_dir():
            process_seq_dir(d, PERSON_CLASS)

    # Layout C: flat PNG frames directly in root (some repacks)
    if not entries:
        all_png = sorted(root_p.rglob("*.png"))
        for i, fp in enumerate(all_png):
            if max_frames and i >= max_frames:
                break
            is_fall = "fall" in str(fp).lower()
            frame   = cv2.imread(str(fp))
            if frame is None:
                continue
            fname = f"urfd_{fp.stem}_{i:05d}.jpg"
            out   = os.path.join(tmp_dir, fname)
            if resize_and_save(frame, out):
                entries.append((out, FALL_CLASS if is_fall else PERSON_CLASS))

    logger.info(f"URFD: collected {len(entries)} frames "
                f"(fall={sum(1 for _,c in entries if c==FALL_CLASS)}, "
                f"adl={sum(1 for _,c in entries if c==PERSON_CLASS)})")
    return entries


# ════════════════════════════════════════════════════════════════════
# Dataset 3 — Le2i Fall Detection Dataset (FDD)
# ════════════════════════════════════════════════════════════════════

def parse_le2i_annotation(ann_file: str):
    """
    Parse Le2i annotation file.
    Format: two integers per file — begin_frame and end_frame of fall.
    Some files have the format:  "begin: 123\nend: 456"
    Others are just:             "123\n456"
    Returns (begin_frame, end_frame) or (None, None) if no fall.
    """
    if not ann_file or not os.path.exists(ann_file):
        return None, None
    try:
        with open(ann_file) as f:
            content = f.read().strip()
        lines = content.split("\n")
        nums = []
        for line in lines:
            # Strip labels like "begin:" or "end:"
            parts = line.replace("begin:", "").replace("end:", "").split()
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    pass
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        elif len(nums) == 1:
            return int(nums[0]), int(nums[0]) + 60  # Fallback: 2s window
    except Exception as e:
        logger.debug(f"  Annotation parse error {ann_file}: {e}")
    return None, None


def load_le2i(root: str, tmp_dir: str,
              max_frames: int = None) -> List[Tuple[str, int]]:
    """
    Le2i Fall Detection Dataset.
    Available on Kaggle: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
    (Free Kaggle account required for download)

    Structure:
      le2i/
        Home/
          video_01.avi
          Annotation_files/
            video_01.txt   ← begin/end fall frame numbers
          video_02.avi
          ...
        Coffee_room/
          video_01.avi
          Annotation_files/
            video_01.txt
          ...

    Each video is a scene; the annotation file marks which frames contain a fall.
    Frames outside the fall window = PERSON_CLASS (normal ADL).
    Frames inside the fall window  = FALL_CLASS.

    If no annotation file exists for a video, the whole video is treated as
    PERSON_CLASS (background / daily activities only).
    """
    if not root or not os.path.exists(root):
        logger.info("Le2i root not provided/found — skipping.")
        return []

    root_p  = Path(root)
    entries = []

    # Scenes: Home and Coffee_room (the annotated subsets)
    scene_dirs = []
    for name in ["Home", "Coffee_room", "home", "coffee_room", "Coffee room"]:
        d = root_p / name
        if d.exists():
            scene_dirs.append(d)

    # Fallback: search all subdirs for AVI files
    if not scene_dirs:
        scene_dirs = [root_p]

    for scene_dir in scene_dirs:
        ann_dir  = scene_dir / "Annotation_files"
        vid_files = sorted(scene_dir.glob("*.avi")) + sorted(scene_dir.glob("*.mp4"))

        for vid_path in vid_files:
            # Find annotation file
            ann_file = None
            if ann_dir.exists():
                # Try exact match first
                ann_candidate = ann_dir / (vid_path.stem + ".txt")
                if ann_candidate.exists():
                    ann_file = str(ann_candidate)
                else:
                    # Try numeric matching (video_01 → 01.txt or 1.txt)
                    for ann in ann_dir.glob("*.txt"):
                        if ann.stem in vid_path.stem or vid_path.stem in ann.stem:
                            ann_file = str(ann)
                            break

            fall_start, fall_end = parse_le2i_annotation(ann_file)
            has_fall = fall_start is not None

            prefix = f"le2i_{scene_dir.name}"
            extracted = extract_frames(
                video_path=str(vid_path),
                out_dir=tmp_dir,
                class_id=PERSON_CLASS,           # default; overridden per-frame below
                prefix=prefix,
                target_fps=TARGET_FPS,
                max_frames=max_frames,
                fall_start_frame=fall_start,
                fall_end_frame=fall_end,
            )
            entries.extend(extracted)

    logger.info(f"Le2i: collected {len(entries)} frames "
                f"(fall={sum(1 for _,c in entries if c==FALL_CLASS)}, "
                f"adl={sum(1 for _,c in entries if c==PERSON_CLASS)})")
    return entries


# ════════════════════════════════════════════════════════════════════
# Dataset 4 — Multiple Cameras Fall Dataset (MCFD)
# ════════════════════════════════════════════════════════════════════

def load_mcfd(root: str, tmp_dir: str,
              max_frames: int = None) -> List[Tuple[str, int]]:
    """
    Multiple Cameras Fall Dataset.
    Download: http://www.iro.umontreal.ca/~labimage/Dataset/
    Also available via GitHub mirrors.

    Structure:
      mcfd/
        scenario01/
          cam1/  *.png
          cam2/  *.png
          cam3/  *.png
        scenario02/ ...
        ...
        scenario08/

    Scenarios 01-08 all contain falls. The dataset does NOT include
    non-fall (ADL) sequences, so all frames = FALL_CLASS.
    Note: purely fall sequences, so expect class imbalance — combine
    with other datasets that have ADL/non-fall sequences.
    """
    if not root or not os.path.exists(root):
        logger.info("MCFD root not provided/found — skipping.")
        return []

    root_p  = Path(root)
    entries = []

    scenario_dirs = sorted(root_p.glob("scenario*")) + sorted(root_p.glob("Scenario*"))
    if not scenario_dirs:
        scenario_dirs = [d for d in root_p.iterdir() if d.is_dir()]

    for scenario in scenario_dirs:
        cam_dirs = sorted(scenario.glob("cam*")) + sorted(scenario.glob("Camera*"))
        if not cam_dirs:
            cam_dirs = [scenario]  # Flat PNG files directly in scenario dir

        # Only use cam1 to avoid 3× duplication of the same fall
        cam_dirs = cam_dirs[:1]

        for cam in cam_dirs:
            frames = sorted(cam.glob("*.png")) + sorted(cam.glob("*.jpg"))
            for i, fp in enumerate(frames):
                if max_frames and i >= max_frames:
                    break
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                fname = f"mcfd_{scenario.name}_{cam.name}_{i:05d}.jpg"
                out   = os.path.join(tmp_dir, fname)
                if resize_and_save(frame, out):
                    entries.append((out, FALL_CLASS))

    logger.info(f"MCFD: collected {len(entries)} fall frames from {len(scenario_dirs)} scenarios")
    return entries


# ════════════════════════════════════════════════════════════════════
# Offline augmentation (simple, no albumentations dependency)
# ════════════════════════════════════════════════════════════════════

def augment_fall_frames(entries: List[Tuple[str, int]],
                        tmp_dir: str) -> List[Tuple[str, int]]:
    """
    Apply light augmentation to FALL_CLASS frames to address class imbalance.
    Generates 2 additional variants per fall frame:
      - Horizontal flip
      - Brightness/contrast jitter
    """
    fall_entries  = [(p, c) for p, c in entries if c == FALL_CLASS]
    extra_entries = []

    for img_path, cls in fall_entries:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        stem = Path(img_path).stem

        # Variant 1: Horizontal flip
        flipped = cv2.flip(frame, 1)
        out1    = os.path.join(tmp_dir, f"{stem}_hflip.jpg")
        if resize_and_save(flipped, out1):
            extra_entries.append((out1, FALL_CLASS))

        # Variant 2: Brightness/contrast jitter
        alpha  = random.uniform(0.75, 1.25)   # contrast
        beta   = random.randint(-25, 25)       # brightness
        jitted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        out2   = os.path.join(tmp_dir, f"{stem}_jitter.jpg")
        if resize_and_save(jitted, out2):
            extra_entries.append((out2, FALL_CLASS))

    logger.info(f"Augmentation: added {len(extra_entries)} fall variants")
    return extra_entries


# ════════════════════════════════════════════════════════════════════
# YOLO split + label writing
# ════════════════════════════════════════════════════════════════════

def create_yolo_split_from_video_data(
        entries: List[Tuple[str, int]],
        out_root: str,
        split: Tuple[float, float, float]):
    """
    Split (img_path, class_id) entries into train/val/test.
    Writes YOLO .txt label files (whole-frame bbox).
    """
    random.shuffle(entries)
    n     = len(entries)
    n_tr  = int(n * split[0])
    n_val = int(n * split[1])

    splits_map = {
        "train": entries[:n_tr],
        "val":   entries[n_tr : n_tr + n_val],
        "test":  entries[n_tr + n_val :],
    }

    for split_name, split_entries in splits_map.items():
        img_dir = Path(out_root) / "images" / split_name
        lbl_dir = Path(out_root) / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, cls_id in split_entries:
            fname = Path(img_path).name
            stem  = Path(img_path).stem

            dst_img = img_dir / fname
            if Path(img_path).resolve() != dst_img.resolve():
                shutil.copy2(img_path, dst_img)

            dst_lbl = lbl_dir / (stem + ".txt")
            dst_lbl.write_text(make_yolo_label(cls_id))

        logger.info(f"  {split_name:5s}: {len(split_entries):6d} samples")


def merge_roboflow_into_split(roboflow_entries: List[Tuple[str, str, str]],
                               out_root: str):
    """
    Copy pre-annotated Roboflow images+labels directly into our split dirs.
    roboflow_entries: list of (img_path, label_path_or_None, split_name)
    """
    for img_path, lbl_path, split_name in roboflow_entries:
        img_dir = Path(out_root) / "images" / split_name
        lbl_dir = Path(out_root) / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy image (resize if needed)
        dst_img = img_dir / Path(img_path).name
        frame   = cv2.imread(img_path)
        if frame is not None:
            resize_and_save(frame, str(dst_img))

        # Copy or create label
        dst_lbl = lbl_dir / (Path(img_path).stem + ".txt")
        if lbl_path and os.path.exists(lbl_path):
            shutil.copy2(lbl_path, dst_lbl)
        else:
            # No label → whole-frame person annotation (background/ADL)
            dst_lbl.write_text(make_yolo_label(PERSON_CLASS))


def write_dataset_yaml(out_root: str, yaml_path: str, total_counts: dict):
    abs_root = os.path.abspath(out_root)
    content  = f"""# Elder Watch — Fall Detection Dataset
# Auto-generated by prepare_datasets.py
# Total images: {sum(total_counts.values())}

path:  {abs_root}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: person
  1: fall

# Dataset statistics
# train: {total_counts.get('train', 0)}
# val:   {total_counts.get('val', 0)}
# test:  {total_counts.get('test', 0)}
"""
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    Path(yaml_path).write_text(content)
    logger.info(f"Dataset YAML written: {yaml_path}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    args = parse_args()

    out_root = args.out
    tmp_dir  = os.path.join(out_root, "_tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Elder Watch — Free Dataset Preparation")
    logger.info("=" * 60)

    # ── Step 1: Auto-download Roboflow if requested ────────────────
    roboflow_root = args.roboflow
    if args.download_roboflow:
        if not args.api_key:
            logger.error("--api-key required for --download-roboflow")
            return
        dl_dir       = "datasets/raw/roboflow_fall"
        roboflow_root = download_roboflow_dataset(args.api_key, dl_dir)

    # ── Step 2: Load each source ───────────────────────────────────
    video_entries   = []   # (img_path, class_id) — need labels created
    roboflow_entries = []  # (img_path, lbl_path, split_name) — labels exist

    if roboflow_root:
        roboflow_entries = load_roboflow(roboflow_root, out_root)

    if args.urfd:
        video_entries.extend(load_urfd(args.urfd, tmp_dir, args.max_frames))

    if args.le2i:
        video_entries.extend(load_le2i(args.le2i, tmp_dir, args.max_frames))

    if args.mcfd:
        video_entries.extend(load_mcfd(args.mcfd, tmp_dir, args.max_frames))

    if not video_entries and not roboflow_entries:
        logger.error(
            "No dataset source provided or found!\n\n"
            "Quick start — pick ONE of these:\n\n"
            "  Option A (easiest): Download Roboflow dataset\n"
            "    1. Get free API key at https://app.roboflow.com\n"
            "    2. pip install roboflow\n"
            "    3. python prepare_datasets.py --download-roboflow --api-key YOUR_KEY\n\n"
            "  Option B: Download URFD manually (no login needed)\n"
            "    wget https://fenix.ur.edu.pl/~mkepski/ds/data/urfall-cam0-falls.zip\n"
            "    wget https://fenix.ur.edu.pl/~mkepski/ds/data/urfall-cam0-adls.zip\n"
            "    unzip urfall-cam0-falls.zip -d datasets/raw/urfd\n"
            "    unzip urfall-cam0-adls.zip  -d datasets/raw/urfd\n"
            "    python prepare_datasets.py --urfd datasets/raw/urfd\n\n"
            "  Option C: Download Le2i from Kaggle (free account needed)\n"
            "    kaggle datasets download tuyenldvn/falldataset-imvia -p datasets/raw/le2i\n"
            "    python prepare_datasets.py --le2i datasets/raw/le2i\n"
        )
        return

    # ── Step 3: Class balance summary ──────────────────────────────
    if video_entries:
        falls   = sum(1 for _, c in video_entries if c == FALL_CLASS)
        persons = sum(1 for _, c in video_entries if c == PERSON_CLASS)
        total   = len(video_entries)
        logger.info(f"Video-sourced frames: {total} total | "
                    f"fall={falls} ({falls/total*100:.1f}%) | "
                    f"person={persons} ({persons/total*100:.1f}%)")

        # ── Step 4: Augment fall frames if imbalanced ────────────────
        if not args.no_augment and falls < persons * 0.4:
            logger.info(f"Class imbalance detected (fall={falls/total*100:.0f}%). "
                        "Augmenting fall frames...")
            extra = augment_fall_frames(video_entries, tmp_dir)
            video_entries.extend(extra)
            falls2 = sum(1 for _, c in video_entries if c == FALL_CLASS)
            logger.info(f"After augment: {len(video_entries)} total | "
                        f"fall={falls2} ({falls2/len(video_entries)*100:.1f}%)")

    # ── Step 5: Write splits ────────────────────────────────────────
    logger.info(f"\nSplitting: train={args.split[0]:.0%} "
                f"val={args.split[1]:.0%} test={args.split[2]:.0%}")

    if video_entries:
        logger.info("Writing video-sourced frames...")
        create_yolo_split_from_video_data(
            video_entries, out_root, tuple(args.split)
        )

    if roboflow_entries:
        logger.info("Merging Roboflow pre-annotated images...")
        merge_roboflow_into_split(roboflow_entries, out_root)

    # ── Step 6: Count final split sizes ───────────────────────────
    counts = {}
    for split_name in ("train", "val", "test"):
        img_dir = Path(out_root) / "images" / split_name
        counts[split_name] = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0

    logger.info(f"\nFinal split sizes:")
    for k, v in counts.items():
        logger.info(f"  {k:5s}: {v:6d} images")
    logger.info(f"  Total: {sum(counts.values()):6d} images")

    # ── Step 7: Write dataset.yaml ─────────────────────────────────
    yaml_path = "training/configs/dataset.yaml"
    write_dataset_yaml(out_root, yaml_path, counts)

    # ── Cleanup tmp ────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("\nDataset preparation complete.")
    logger.info(f"Next step:\n  python training/scripts/train_yolov8.py "
                f"--data {yaml_path}")


if __name__ == "__main__":
    main()
