#!/usr/bin/env python3
"""
Dataset Preparation Script — Fixed for actual folder names:
  Le2i (falldataset-imvia) actual structure:
    le2i/
      Coffee_room_01/
        video_01.avi ... video_X.avi
        Annotation_files/
          video_01.txt  (begin_frame\nend_frame)
      Coffee_room_02/
        ...
      Home_01/
        ...
      Home_02/
        ...
      Office/            (no annotations → skip)
      Lecture_room/      (no annotations → skip)

  Roboflow v4 structure:
    roboflow_fall/
      train/images/ + train/labels/
      valid/images/ + valid/labels/
      test/images/  + test/labels/
      (class 0 = Fall-Detected → remapped to our class 1)

Our YOLO class mapping:
  0 = person  (ADL frames, outside fall annotation window)
  1 = fall    (frames inside fall window, or all Roboflow images)

Usage:
  python training/scripts/prepare_datasets.py \
      --roboflow datasets/raw/roboflow_fall \
      --le2i     datasets/raw/le2i \
      --out      datasets/processed
"""

import argparse
import logging
import os
import re
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("PrepareDataset")

# ── Constants ─────────────────────────────────────────────────────────
PERSON_CLASS = 0
FALL_CLASS   = 1
TARGET_FPS   = 5      # frames/sec extracted from videos
IMG_SIZE     = 640
JPEG_Q       = 90
SEED         = 42

# Le2i scene folders that HAVE annotation files (fall labels)
# Matches: Coffee_room_01, Coffee_room_02, Home_01, Home_02, etc.
LE2I_ANNOTATED_PREFIXES = ("coffee_room", "home")

# Le2i scene folders to SKIP (no annotation files)
LE2I_SKIP_PREFIXES = ("office", "lecture")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare Roboflow + Le2i datasets for YOLOv8 training"
    )
    p.add_argument("--roboflow",   type=str, default=None,
                   help="Path to extracted Roboflow ZIP root")
    p.add_argument("--le2i",       type=str, default=None,
                   help="Path to Le2i root (contains Coffee_room_01, Home_01, etc.)")
    p.add_argument("--out",        type=str, default="datasets/processed")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max frames per video (for quick testing, e.g. 30)")
    p.add_argument("--no-augment", action="store_true",
                   help="Skip fall-frame augmentation")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def save_jpg(frame, path: str) -> bool:
    try:
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return True
    except Exception as e:
        logger.warning(f"Save failed {path}: {e}")
        return False


def yolo_label(cls: int, cx=0.5, cy=0.5, w=0.85, h=0.90) -> str:
    return f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n"


def find_rf_root(base: str) -> Path:
    """Walk up to 2 levels to find folder containing train/ valid/ test/."""
    base_p = Path(base)
    for candidate in [base_p] + [d for d in base_p.iterdir() if d.is_dir()]:
        if (candidate / "train").exists() and (candidate / "valid").exists():
            return candidate
    return base_p


def find_le2i_root(base: str) -> Path:
    """
    Walk up to 2 levels to find the folder that contains
    Coffee_room_01, Home_01 etc.
    """
    base_p = Path(base)

    def has_le2i_scenes(d: Path) -> bool:
        if not d.is_dir():
            return False
        children = [x.name.lower() for x in d.iterdir() if x.is_dir()]
        return any(
            c.startswith(pfx)
            for c in children
            for pfx in LE2I_ANNOTATED_PREFIXES
        )

    if has_le2i_scenes(base_p):
        return base_p
    for sub in base_p.iterdir():
        if has_le2i_scenes(sub):
            return sub
    return base_p


# ════════════════════════════════════════════════════════════════════
# Dataset 1 — Roboflow Fall Detection v4
# Class 0 "Fall-Detected" → our class 1 (fall)
# ════════════════════════════════════════════════════════════════════

def remap_roboflow_label(src_lbl: Path, dst_lbl: Path):
    """
    Roboflow class 0 = Fall-Detected → remap to FALL_CLASS (1).
    Bounding box coordinates stay unchanged.
    """
    if not src_lbl.exists():
        dst_lbl.write_text(yolo_label(FALL_CLASS))
        return

    out_lines = []
    for line in src_lbl.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        # class 0 → 1 (fall), anything else → 0 (person)
        new_cls = FALL_CLASS if int(parts[0]) == 0 else PERSON_CLASS
        out_lines.append(" ".join([str(new_cls)] + parts[1:]))
    dst_lbl.write_text("\n".join(out_lines) + "\n")


def load_roboflow(root: str, out_root: str) -> int:
    """Copy Roboflow images into our split dirs with remapped labels."""
    rf_root = find_rf_root(root)
    logger.info(f"Roboflow root detected: {rf_root}")

    split_map = {"train": "train", "valid": "val", "test": "test"}
    total = 0

    for rf_split, our_split in split_map.items():
        img_dir = rf_root / rf_split / "images"
        lbl_dir = rf_root / rf_split / "labels"

        if not img_dir.exists():          # flat layout fallback
            img_dir = rf_root / rf_split
            lbl_dir = rf_root / rf_split

        if not img_dir.exists():
            logger.warning(f"  Roboflow: '{rf_split}' not found, skipping")
            continue

        out_img = Path(out_root) / "images" / our_split
        out_lbl = Path(out_root) / "labels" / our_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        imgs = sorted(list(img_dir.glob("*.jpg")) +
                      list(img_dir.glob("*.jpeg")) +
                      list(img_dir.glob("*.png")))

        for img_p in imgs:
            frame = cv2.imread(str(img_p))
            if frame is None:
                continue
            dst_img = out_img / img_p.name
            dst_lbl = out_lbl / (img_p.stem + ".txt")
            save_jpg(frame, str(dst_img))
            remap_roboflow_label(lbl_dir / (img_p.stem + ".txt"), dst_lbl)
            total += 1

        logger.info(f"  Roboflow {rf_split:5s} → {our_split}: {len(imgs)} images")

    return total


# ════════════════════════════════════════════════════════════════════
# Dataset 2 — Le2i FDD (falldataset-imvia)
# Actual folders: Coffee_room_01, Coffee_room_02, Home_01, Home_02
# ════════════════════════════════════════════════════════════════════

def parse_annotation(ann_path: Optional[Path]) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse Le2i annotation file.
    Format: two integers (begin_frame, end_frame), one per line.
    Returns (begin, end) or (None, None).
    """
    if ann_path is None or not ann_path.exists():
        return None, None
    try:
        nums = [int(x) for x in re.findall(r'\d+', ann_path.read_text())]
        if len(nums) >= 2:
            return nums[0], nums[1]
        if len(nums) == 1:
            return nums[0], nums[0] + 75   # ~3 sec at 25fps
    except Exception:
        pass
    return None, None


def find_annotation(ann_dir: Path, video_stem: str) -> Optional[Path]:
    """
    Match annotation file to video filename.
    video_stem: e.g. "video_01", "video_26", "video1"
    Tries exact match first, then numeric match.
    """
    if not ann_dir.exists():
        return None

    # Exact match
    exact = ann_dir / (video_stem + ".txt")
    if exact.exists():
        return exact

    # Extract number from video stem (e.g. "video_01" → "1", "01")
    m = re.search(r'(\d+)$', video_stem)
    if not m:
        return None
    num_str = m.group(1)
    num_int = int(num_str)

    for ann in sorted(ann_dir.glob("*.txt")):
        ann_nums = re.findall(r'\d+', ann.stem)
        if ann_nums and int(ann_nums[-1]) == num_int:
            return ann
    return None


def extract_video(video_path: Path,
                  ann_path: Optional[Path],
                  tmp_dir: str,
                  max_frames: Optional[int]) -> List[Tuple[str, int]]:
    """
    Extract frames from one Le2i video at TARGET_FPS.
    Frames inside [fall_start, fall_end] → FALL_CLASS
    Frames outside                       → PERSON_CLASS
    No annotation file → whole video = PERSON_CLASS (ADL only)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"    Cannot open: {video_path.name}")
        return []

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(src_fps / TARGET_FPS)))
    fall_start, fall_end = parse_annotation(ann_path)

    scene   = video_path.parent.name   # e.g. Coffee_room_01
    entries = []
    fidx    = 0
    saved   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and saved >= max_frames:
            break

        if fidx % interval == 0:
            if fall_start is not None and fall_start <= fidx <= fall_end:
                cls = FALL_CLASS
            else:
                cls = PERSON_CLASS

            fname = f"le2i_{scene}_{video_path.stem}_f{fidx:06d}.jpg"
            out_p = os.path.join(tmp_dir, fname)
            if save_jpg(frame, out_p):
                entries.append((out_p, cls))
                saved += 1

        fidx += 1

    cap.release()
    return entries


def load_le2i(root: str, tmp_dir: str,
              max_frames: Optional[int]) -> List[Tuple[str, int]]:
    """
    Load Le2i dataset.
    Processes all scene folders whose name starts with 'coffee_room' or 'home'
    (case-insensitive).  Skips 'office' and 'lecture_room'.
    """
    le2i_root = find_le2i_root(root)
    logger.info(f"Le2i root detected: {le2i_root}")

    # List all subdirectories
    all_dirs = sorted([d for d in le2i_root.iterdir() if d.is_dir()])
    logger.info(f"  Found folders: {[d.name for d in all_dirs]}")

    all_entries = []

    for scene_dir in all_dirs:
        name_lower = scene_dir.name.lower()

        # Skip non-annotated scenes
        if any(name_lower.startswith(skip) for skip in LE2I_SKIP_PREFIXES):
            logger.info(f"  Skipping '{scene_dir.name}' (no annotations)")
            continue

        # Only process annotated scenes
        if not any(name_lower.startswith(pfx) for pfx in LE2I_ANNOTATED_PREFIXES):
            logger.info(f"  Skipping '{scene_dir.name}' (unknown scene)")
            continue

        ann_dir = scene_dir / "Annotation_files"
        videos  = sorted(list(scene_dir.glob("*.avi")) +
                         list(scene_dir.glob("*.mp4")))

        has_ann = ann_dir.exists() and any(ann_dir.glob("*.txt"))
        logger.info(f"  Scene '{scene_dir.name}': "
                    f"{len(videos)} videos, "
                    f"annotations={'yes (' + str(len(list(ann_dir.glob('*.txt')))) + ')' if has_ann else 'NONE'}")

        for vid in videos:
            ann_p   = find_annotation(ann_dir, vid.stem) if has_ann else None
            entries = extract_video(vid, ann_p, tmp_dir, max_frames)
            all_entries.extend(entries)

    falls   = sum(1 for _, c in all_entries if c == FALL_CLASS)
    persons = sum(1 for _, c in all_entries if c == PERSON_CLASS)
    logger.info(f"  Le2i total: {len(all_entries)} frames | "
                f"fall={falls} ({falls/max(len(all_entries),1)*100:.0f}%) | "
                f"person={persons} ({persons/max(len(all_entries),1)*100:.0f}%)")
    return all_entries


# ════════════════════════════════════════════════════════════════════
# Augmentation — balance fall vs person frames
# ════════════════════════════════════════════════════════════════════

def augment_falls(entries: List[Tuple[str, int]],
                  tmp_dir: str) -> List[Tuple[str, int]]:
    falls = [e for e in entries if e[1] == FALL_CLASS]
    total = len(entries)

    if not falls or total == 0:
        return []

    ratio = len(falls) / total
    if ratio >= 0.40:
        logger.info(f"  Class balance OK (fall={ratio:.0%}), no augmentation needed")
        return []

    logger.info(f"  Imbalance: fall={ratio:.0%}. Augmenting fall frames...")
    extra = []
    for img_path, _ in falls:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        stem = Path(img_path).stem

        # Horizontal flip
        out1 = os.path.join(tmp_dir, f"{stem}_hflip.jpg")
        if save_jpg(cv2.flip(frame, 1), out1):
            extra.append((out1, FALL_CLASS))

        # Brightness jitter
        import numpy as np
        alpha  = random.uniform(0.75, 1.25)
        beta   = random.randint(-30, 30)
        jitted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        out2   = os.path.join(tmp_dir, f"{stem}_jitter.jpg")
        if save_jpg(jitted, out2):
            extra.append((out2, FALL_CLASS))

    new_r = (len(falls) + len(extra)) / (total + len(extra))
    logger.info(f"  +{len(extra)} augmented fall frames | new fall ratio = {new_r:.0%}")
    return extra


# ════════════════════════════════════════════════════════════════════
# Split + write YOLO labels (for Le2i video entries)
# ════════════════════════════════════════════════════════════════════

def write_split(entries: List[Tuple[str, int]],
                out_root: str,
                split=(0.70, 0.15, 0.15)):
    random.shuffle(entries)
    n     = len(entries)
    n_tr  = int(n * split[0])
    n_val = int(n * split[1])

    buckets = {
        "train": entries[:n_tr],
        "val":   entries[n_tr : n_tr + n_val],
        "test":  entries[n_tr + n_val:],
    }

    for sp, items in buckets.items():
        out_img = Path(out_root) / "images" / sp
        out_lbl = Path(out_root) / "labels" / sp
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path, cls in items:
            fname = Path(img_path).name
            stem  = Path(img_path).stem
            dst_i = out_img / fname
            dst_l = out_lbl / (stem + ".txt")
            if Path(img_path).resolve() != dst_i.resolve():
                shutil.copy2(img_path, dst_i)
            dst_l.write_text(yolo_label(cls))

        logger.info(f"  {sp:5s}: {len(items):6d} Le2i samples")


# ════════════════════════════════════════════════════════════════════
# dataset.yaml
# ════════════════════════════════════════════════════════════════════

def write_yaml(out_root: str, yaml_path: str):
    counts = {}
    for sp in ("train", "val", "test"):
        d = Path(out_root) / "images" / sp
        counts[sp] = len(list(d.glob("*.jpg"))) if d.exists() else 0

    content = f"""# Elder Watch — Fall Detection Dataset
# Generated by prepare_datasets.py
# Sources: Roboflow Fall Detection v4 + Le2i FDD (Coffee_room_01/02, Home_01/02)

path:  {os.path.abspath(out_root)}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: person
  1: fall

# Split sizes
# train : {counts['train']}
# val   : {counts['val']}
# test  : {counts['test']}
# total : {sum(counts.values())}
"""
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    Path(yaml_path).write_text(content)
    logger.info(f"dataset.yaml written → {yaml_path}")
    return counts


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    args = parse_args()

    if not args.roboflow and not args.le2i:
        logger.error("Provide at least --roboflow or --le2i (or both)")
        return

    out_root = args.out
    tmp_dir  = os.path.join(out_root, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Elder Watch — Dataset Preparation")
    logger.info("=" * 60)

    # ── Roboflow ───────────────────────────────────────────────────
    if args.roboflow:
        logger.info("\n[1/3] Processing Roboflow dataset...")
        n = load_roboflow(args.roboflow, out_root)
        logger.info(f"      Roboflow: {n} images processed")

    # ── Le2i ───────────────────────────────────────────────────────
    video_entries = []
    if args.le2i:
        logger.info("\n[2/3] Processing Le2i dataset...")
        video_entries = load_le2i(args.le2i, tmp_dir, args.max_frames)

        if video_entries and not args.no_augment:
            extra = augment_falls(video_entries, tmp_dir)
            video_entries.extend(extra)

        if video_entries:
            logger.info(f"      Writing Le2i split...")
            write_split(video_entries, out_root)
        else:
            logger.warning("      Le2i: no frames extracted")

    # ── YAML ───────────────────────────────────────────────────────
    logger.info("\n[3/3] Writing dataset.yaml...")
    yaml_path = "training/configs/dataset.yaml"
    counts = write_yaml(out_root, yaml_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = sum(counts.values())
    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"  train : {counts['train']:6d} images")
    logger.info(f"  val   : {counts['val']:6d} images")
    logger.info(f"  test  : {counts['test']:6d} images")
    logger.info(f"  TOTAL : {total:6d} images")
    logger.info(f"\nNext step:")
    logger.info(f"  python training/scripts/train_yolov8.py \\")
    logger.info(f"      --data {yaml_path} --device 0 --epochs 50")


if __name__ == "__main__":
    main()