#!/bin/bash
# ============================================================
# Elder Watch - Raspberry Pi 5 Setup Script
# University of Ruhuna EE7204/EC7205
# Tested: Raspberry Pi 5 (4GB), Raspberry Pi OS Bookworm 64-bit
# ============================================================

set -e   # Exit on error

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}   Elder Watch - RPi 5 Setup Script      ${NC}"
echo -e "${GREEN}==========================================${NC}"

# ── 1. System update ──────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/8] Updating system packages...${NC}"
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev v4l-utils \
    git wget curl unzip \
    libgstreamer1.0-dev \
    i2c-tools \
    pigpio

# ── 2. Enable camera ──────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/8] Enabling camera interface...${NC}"
sudo raspi-config nonint do_camera 0 2>/dev/null || true
# For RPi 5, camera is controlled via dtoverlay
if ! grep -q "camera_auto_detect=1" /boot/config.txt 2>/dev/null; then
    if ! grep -q "camera_auto_detect=1" /boot/firmware/config.txt 2>/dev/null; then
        echo "camera_auto_detect=1" | sudo tee -a /boot/firmware/config.txt
    fi
fi
echo -e "${GREEN}Camera enabled.${NC}"

# ── 3. Python virtual environment ─────────────────────────────────────
echo -e "\n${YELLOW}[3/8] Creating Python virtual environment...${NC}"
VENV_DIR="$HOME/elder_watch_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ── 4. Install Python packages ────────────────────────────────────────
echo -e "\n${YELLOW}[4/8] Installing Python packages...${NC}"

# OpenCV (use system package as base, pip for extras)
pip install opencv-python-headless==4.8.1.78 || pip install opencv-python-headless

# TFLite Runtime (lightweight, no full TF)
pip install tflite-runtime

# MediaPipe (note: RPi ARM64 build)
pip install mediapipe

# Ultralytics (for optional PyTorch path)
# pip install ultralytics  # Uncomment if you want PyTorch inference on RPi

# Alert libraries
pip install twilio requests

# Utilities
pip install PyYAML numpy Pillow psutil

# SMBUS for I2C (sensor integration)
pip install smbus2

echo -e "${GREEN}Python packages installed.${NC}"

# ── 5. Clone / copy Elder Watch project ───────────────────────────────
echo -e "\n${YELLOW}[5/8] Setting up Elder Watch project...${NC}"
PROJECT_DIR="$HOME/elder_watch"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating project directory at $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    # If you have a git repo:
    # git clone https://github.com/yourusername/elder-watch.git "$PROJECT_DIR"
    echo -e "${YELLOW}Place your project files in $PROJECT_DIR${NC}"
else
    echo -e "${GREEN}Project directory exists: $PROJECT_DIR${NC}"
fi

# Create required subdirectories
mkdir -p "$PROJECT_DIR"/{logs/snapshots,models/{pretrained,finetuned,quantized},configs}

# ── 6. Performance tuning for RPi 5 ───────────────────────────────────
echo -e "\n${YELLOW}[6/8] Tuning RPi 5 performance...${NC}"

# Set CPU governor to performance mode
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

# Increase GPU memory split (256MB for camera processing)
CONFIG_FILE="/boot/firmware/config.txt"
if ! grep -q "gpu_mem=256" "$CONFIG_FILE" 2>/dev/null; then
    echo "gpu_mem=256" | sudo tee -a "$CONFIG_FILE"
fi

# Swap size (increase to 2GB for model loading)
sudo dphys-swapfile swapoff 2>/dev/null || true
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile 2>/dev/null || true
sudo dphys-swapfile setup 2>/dev/null || true
sudo dphys-swapfile swapon 2>/dev/null || true

echo -e "${GREEN}Performance tuning applied.${NC}"

# ── 7. Install systemd service ────────────────────────────────────────
echo -e "\n${YELLOW}[7/8] Installing systemd service...${NC}"

SERVICE_FILE="/etc/systemd/system/elder-watch.service"
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Elder Watch Fall Detection System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/python main.py --headless --config configs/config.yaml
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

# Resource limits
CPUQuota=95%
MemoryMax=3G
Nice=-5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=elder-watch

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable elder-watch
echo -e "${GREEN}Systemd service installed: elder-watch.service${NC}"
echo "  Start:   sudo systemctl start elder-watch"
echo "  Status:  sudo systemctl status elder-watch"
echo "  Logs:    sudo journalctl -u elder-watch -f"

# ── 8. Verify installation ─────────────────────────────────────────────
echo -e "\n${YELLOW}[8/8] Verifying installation...${NC}"
source "$VENV_DIR/bin/activate"

python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')" && \
    echo -e "  ${GREEN}✓ OpenCV${NC}" || echo -e "  ${RED}✗ OpenCV${NC}"

python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" && \
    echo -e "  ${GREEN}✓ NumPy${NC}" || echo -e "  ${RED}✗ NumPy${NC}"

python3 -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')" && \
    echo -e "  ${GREEN}✓ MediaPipe${NC}" || echo -e "  ${RED}✗ MediaPipe${NC}"

python3 -c "import tflite_runtime; print('TFLite Runtime OK')" && \
    echo -e "  ${GREEN}✓ TFLite Runtime${NC}" || \
    python3 -c "import tensorflow.lite; print('TF Lite OK')" && \
    echo -e "  ${GREEN}✓ TensorFlow Lite (full)${NC}" || \
    echo -e "  ${RED}✗ TFLite (install tflite-runtime)${NC}"

# Check camera
if v4l2-ctl --list-devices &>/dev/null; then
    echo -e "  ${GREEN}✓ Camera detected${NC}"
else
    echo -e "  ${YELLOW}⚠ No camera detected (connect camera and reboot)${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Setup complete!                         ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Copy model to: $PROJECT_DIR/models/quantized/yolov8n_fall_int8.tflite"
echo "  2. Edit:          $PROJECT_DIR/configs/config.yaml (set alert credentials)"
echo "  3. Test:          cd $PROJECT_DIR && python main.py --demo"
echo "  4. Deploy:        sudo systemctl start elder-watch"
echo ""
echo "Reboot recommended to apply all changes:"
echo "  sudo reboot"
