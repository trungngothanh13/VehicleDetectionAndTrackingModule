#!/bin/bash
# ThunderCompute setup script for VehicleDetectionAndTrackingModule
# Run this once after SSHing into your instance: bash setup_cloud.sh

set -e

echo "=== Installing system dependencies ==="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libgl1

echo "=== Installing Python dependencies ==="
pip install -q -r requirements.txt

echo "=== Downloading YOLO models ==="
python -c "from ultralytics import YOLO; YOLO('yolo26l.pt'); YOLO('yolov8l.pt')"

echo ""
echo "=== Setup complete! ==="
echo "Run: python main.py"
echo "Configure tracker in config.py (TRACKER_TYPE = 'botsort' | 'bytetrack' | 'deepsort')"
