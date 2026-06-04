#!/bin/bash
# ============================================================
# URFD (UR Fall Detection Dataset) - Auto Downloader
# University of Rzeszow: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
#
# Files are individual per-sequence ZIPs (RGB frames only - cam0).
# 30 fall sequences + 40 ADL sequences = 70 total.
#
# Usage:
#   chmod +x training/scripts/download_urfd.sh
#   ./training/scripts/download_urfd.sh datasets/raw/urfd
# ============================================================

set -e
OUTDIR="${1:-datasets/raw/urfd}"
BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"
mkdir -p "$OUTDIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== URFD Downloader ===${NC}"
echo "Output: $OUTDIR"
echo ""

# ── Fall sequences: fall-01 to fall-30 (RGB cam0 only) ──────────────
echo -e "${YELLOW}Downloading 30 fall sequences (RGB cam0)...${NC}"
for i in $(seq -w 1 30); do
    FILE="fall-${i}-cam0-rgb.zip"
    URL="$BASE/$FILE"
    DEST="$OUTDIR/$FILE"
    if [ -f "$DEST" ]; then
        echo "  [skip] $FILE already exists"
    else
        echo "  Downloading $FILE ..."
        wget -q --show-progress -O "$DEST" "$URL" || {
            echo "  WARNING: Failed to download $FILE (skipping)"
            rm -f "$DEST"
        }
    fi
done

# ── ADL sequences: adl-01 to adl-40 (RGB cam0 only) ─────────────────
echo -e "${YELLOW}Downloading 40 ADL sequences (RGB cam0)...${NC}"
for i in $(seq -w 1 40); do
    FILE="adl-${i}-cam0-rgb.zip"
    URL="$BASE/$FILE"
    DEST="$OUTDIR/$FILE"
    if [ -f "$DEST" ]; then
        echo "  [skip] $FILE already exists"
    else
        echo "  Downloading $FILE ..."
        wget -q --show-progress -O "$DEST" "$URL" || {
            echo "  WARNING: Failed to download $FILE (skipping)"
            rm -f "$DEST"
        }
    fi
done

# ── Unzip all downloaded ZIPs ────────────────────────────────────────
echo -e "${YELLOW}Extracting ZIPs...${NC}"
for ZIP in "$OUTDIR"/*.zip; do
    [ -f "$ZIP" ] || continue
    BASENAME=$(basename "$ZIP" .zip)
    DESTDIR="$OUTDIR/$BASENAME"
    if [ -d "$DESTDIR" ]; then
        echo "  [skip] $BASENAME already extracted"
    else
        echo "  Extracting $BASENAME ..."
        mkdir -p "$DESTDIR"
        unzip -q "$ZIP" -d "$DESTDIR" && rm -f "$ZIP"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────
FALL_DIRS=$(find "$OUTDIR" -maxdepth 1 -type d -name "fall-*" | wc -l)
ADL_DIRS=$(find  "$OUTDIR" -maxdepth 1 -type d -name "adl-*"  | wc -l)
echo ""
echo -e "${GREEN}Done!${NC}"
echo "  Fall sequences: $FALL_DIRS"
echo "  ADL  sequences: $ADL_DIRS"
echo ""
echo "Next step:"
echo "  python training/scripts/prepare_datasets.py \\"
echo "      --urfd $OUTDIR \\"
echo "      --out  datasets/processed"
