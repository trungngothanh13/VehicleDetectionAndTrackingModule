#!/bin/bash
# ThunderCompute setup script for VehicleDetectionAndTrackingModule
# Run this once after SSHing into your instance: bash setup_cloud.sh

set -e

echo "=== Installing system dependencies ==="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libgl1

echo "=== Installing Python dependencies ==="
pip install -q -r requirements.txt

echo "=== Downloading YOLO26l model ==="
python -c "from ultralytics import YOLO; YOLO('yolo26l.pt')"

echo "=== Downloading BoT-SORT Re-ID weights ==="
python -c "from boxmot import BotSORT; from pathlib import Path; BotSORT(reid_weights=Path('osnet_x0_25_msmt17.pt'), device='cuda', half=False)"

echo ""
echo "=== Setup complete! ==="
echo "Run: python main.py"
echo "Output will be saved to: output_test_2.mp4"
