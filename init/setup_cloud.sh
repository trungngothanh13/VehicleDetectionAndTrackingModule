#!/bin/bash
# ThunderCompute / Cloud setup script for VehicleDetectionAndTrackingModule
# Run once: bash init/setup_cloud.sh
# Then start the 6-camera AI engine: python main.py

set -e

INIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$INIT_DIR/.." && pwd)"

echo "=== Installing system dependencies ==="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libgl1

echo "=== Installing Python dependencies ==="
pip install -q -U -r "$INIT_DIR/requirements.txt"

echo "=== Downloading YOLO26l model ==="
cd "$ROOT_DIR"
python -c "from ultralytics import YOLO; YOLO('yolo26l.pt')"

echo ""
echo "=== Setup complete! ==="
echo "Run: python main.py"
echo "Optional overrides: --cameras cam-01..cam-08, --tracker bytetrack|botsort|deepsort, --ws-url ws://localhost:5000/ws/inference"
