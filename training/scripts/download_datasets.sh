#!/bin/bash
# ============================================================
# Elder Watch — Master Dataset Downloader
# Downloads all free open-source fall detection datasets.
# Run this ONCE before running prepare_datasets.py
#
# Usage:
#   chmod +x training/scripts/download_datasets.sh
#   ./training/scripts/download_datasets.sh
#
# Individual dataset flags:
#   ./training/scripts/download_datasets.sh --urfd-only
#   ./training/scripts/download_datasets.sh --roboflow-only
#   ./training/scripts/download_datasets.sh --le2i-only
# ============================================================

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

DO_URFD=true
DO_ROBOFLOW=true
DO_LE2I=true

for arg in "$@"; do
    case $arg in
        --urfd-only)      DO_ROBOFLOW=false; DO_LE2I=false ;;
        --roboflow-only)  DO_URFD=false;     DO_LE2I=false ;;
        --le2i-only)      DO_URFD=false;     DO_ROBOFLOW=false ;;
        --skip-urfd)      DO_URFD=false ;;
        --skip-roboflow)  DO_ROBOFLOW=false ;;
        --skip-le2i)      DO_LE2I=false ;;
    esac
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Elder Watch — Dataset Downloader         ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# ════════════════════════════════════════════════════════════════
# DATASET 1: URFD — UR Fall Detection Dataset
# Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
# License: CC BY-NC-SA 4.0 (non-commercial research)
# Size: ~1.5 GB total (RGB frames only)
# Files: 30 fall-XX-cam0-rgb.zip + 40 adl-XX-cam0-rgb.zip
# ════════════════════════════════════════════════════════════════
if [ "$DO_URFD" = true ]; then
    echo -e "${YELLOW}[1/3] URFD — UR Fall Detection Dataset${NC}"
    echo "      Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
    echo "      Sequences: 30 falls + 40 ADL"
    echo ""
    URFD_DIR="datasets/raw/urfd"
    mkdir -p "$URFD_DIR"
    BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"

    download_urfd_seq() {
        local PREFIX=$1  # "fall" or "adl"
        local MAX=$2     # 30 or 40
        for i in $(seq -w 1 $MAX); do
            FILE="${PREFIX}-${i}-cam0-rgb.zip"
            DEST="$URFD_DIR/$FILE"
            EXTRACTED="$URFD_DIR/${PREFIX}-${i}-cam0-rgb"
            if [ -d "$EXTRACTED" ] && [ "$(ls -A $EXTRACTED)" ]; then
                echo "  [skip] ${PREFIX}-${i} already extracted"
                continue
            fi
            if [ ! -f "$DEST" ]; then
                echo "  ↓ ${FILE}"
                wget -q --show-progress --timeout=30 --tries=3 \
                     -O "$DEST" "$BASE/$FILE" 2>&1 || {
                    echo -e "  ${RED}Failed: $FILE${NC}"
                    rm -f "$DEST"
                    continue
                }
            fi
            echo "  ⊞ Extracting ${FILE}..."
            mkdir -p "$EXTRACTED"
            unzip -q "$DEST" -d "$EXTRACTED" && rm -f "$DEST"
        done
    }

    echo "  Downloading fall sequences (fall-01 to fall-30)..."
    download_urfd_seq "fall" 30
    echo "  Downloading ADL sequences (adl-01 to adl-40)..."
    download_urfd_seq "adl" 40

    FALL_N=$(find "$URFD_DIR" -maxdepth 1 -type d -name "fall-*" | wc -l)
    ADL_N=$(find  "$URFD_DIR" -maxdepth 1 -type d -name "adl-*"  | wc -l)
    echo -e "  ${GREEN}✓ URFD: $FALL_N fall + $ADL_N ADL sequences${NC}"
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# DATASET 2: Roboflow Fall Detection (4,497 pre-annotated images)
# Source: https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8
# License: CC BY 4.0
# Size: ~600 MB
# Method: roboflow Python package OR manual ZIP download
# ════════════════════════════════════════════════════════════════
if [ "$DO_ROBOFLOW" = true ]; then
    echo -e "${YELLOW}[2/3] Roboflow Fall Detection Dataset (4,497 images)${NC}"
    echo "      Source: https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8"
    echo ""

    RF_DIR="datasets/raw/roboflow_fall"

    if [ -d "$RF_DIR" ] && [ "$(ls -A $RF_DIR)" ]; then
        echo -e "  ${GREEN}[skip] Roboflow dataset already exists at $RF_DIR${NC}"
    else
        mkdir -p "$RF_DIR"

        # Check if roboflow Python package available
        if python3 -c "import roboflow" 2>/dev/null; then
            echo "  roboflow package found."
            echo "  Enter your Roboflow API key (free at https://app.roboflow.com):"
            read -r -p "  API Key: " RF_KEY
            if [ -n "$RF_KEY" ]; then
                python3 - <<PYEOF
from roboflow import Roboflow
rf = Roboflow(api_key="$RF_KEY")
project = rf.workspace("roboflow-universe-projects").project("fall-detection-ca3o8")
dataset = project.version(4).download("yolov8", location="$RF_DIR")
print(f"Downloaded to: {dataset.location}")
PYEOF
                echo -e "  ${GREEN}✓ Roboflow dataset downloaded${NC}"
            else
                echo -e "  ${YELLOW}No API key entered. Skipping Roboflow auto-download.${NC}"
                echo "  Manual download:"
                echo "    1. Go to: https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8/dataset/4"
                echo "    2. Click Download → YOLOv8 format → Download ZIP"
                echo "    3. Extract to: $RF_DIR/"
            fi
        else
            echo -e "  ${YELLOW}roboflow package not installed.${NC}"
            echo ""
            echo "  Two options:"
            echo ""
            echo "  Option A — Install roboflow and re-run:"
            echo "    pip install roboflow"
            echo "    ./training/scripts/download_datasets.sh --roboflow-only"
            echo ""
            echo "  Option B — Manual download (no Python needed):"
            echo "    1. Open: https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8/dataset/4"
            echo "    2. Sign up free → Download Dataset → YOLOv8 → Download ZIP"
            echo "    3. unzip the file into: $RF_DIR/"
            echo "    4. Continue with prepare_datasets.py --roboflow $RF_DIR"
        fi
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# DATASET 3: Le2i FDD via Kaggle
# Source: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
# License: Free for research
# Size: ~2 GB
# Requires: free Kaggle account + kaggle CLI
# ════════════════════════════════════════════════════════════════
if [ "$DO_LE2I" = true ]; then
    echo -e "${YELLOW}[3/3] Le2i Fall Detection Dataset (via Kaggle)${NC}"
    echo "      Source: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia"
    echo ""
    LE2I_DIR="datasets/raw/le2i"

    if [ -d "$LE2I_DIR" ] && [ "$(ls -A $LE2I_DIR)" ]; then
        echo -e "  ${GREEN}[skip] Le2i dataset already exists at $LE2I_DIR${NC}"
    elif command -v kaggle &>/dev/null; then
        # Check kaggle credentials
        if [ -f "$HOME/.kaggle/kaggle.json" ]; then
            echo "  kaggle CLI found with credentials. Downloading..."
            mkdir -p "$LE2I_DIR"
            kaggle datasets download tuyenldvn/falldataset-imvia \
                -p "$LE2I_DIR" --unzip
            echo -e "  ${GREEN}✓ Le2i dataset downloaded${NC}"
        else
            echo -e "  ${YELLOW}kaggle CLI found but no credentials file.${NC}"
            echo "  Setup:"
            echo "    1. Go to: https://www.kaggle.com/settings → API → Create New Token"
            echo "    2. Save kaggle.json to: ~/.kaggle/kaggle.json"
            echo "    3. chmod 600 ~/.kaggle/kaggle.json"
            echo "    4. Re-run: ./training/scripts/download_datasets.sh --le2i-only"
        fi
    else
        echo -e "  ${YELLOW}kaggle CLI not installed.${NC}"
        echo "  Setup:"
        echo "    pip install kaggle"
        echo "    # Get token from: https://www.kaggle.com/settings → API → Create New Token"
        echo "    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "    chmod 600 ~/.kaggle/kaggle.json"
        echo "    kaggle datasets download tuyenldvn/falldataset-imvia -p $LE2I_DIR --unzip"
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Summary & Next Steps
# ════════════════════════════════════════════════════════════════
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Download Summary                         ${NC}"
echo -e "${GREEN}============================================${NC}"

[ -d "datasets/raw/urfd" ]          && echo "  ✓ URFD:     datasets/raw/urfd" \
                                     || echo "  ✗ URFD:     not downloaded"
[ -d "datasets/raw/roboflow_fall" ] && echo "  ✓ Roboflow: datasets/raw/roboflow_fall" \
                                     || echo "  ✗ Roboflow: not downloaded"
[ -d "datasets/raw/le2i" ]          && echo "  ✓ Le2i:     datasets/raw/le2i" \
                                     || echo "  ✗ Le2i:     not downloaded"

echo ""
echo "Next step — prepare datasets for training:"
echo ""
echo "  python training/scripts/prepare_datasets.py \\"

[ -d "datasets/raw/urfd" ]          && echo "      --urfd      datasets/raw/urfd \\"
[ -d "datasets/raw/roboflow_fall" ] && echo "      --roboflow  datasets/raw/roboflow_fall \\"
[ -d "datasets/raw/le2i" ]          && echo "      --le2i      datasets/raw/le2i \\"

echo "      --out       datasets/processed"
