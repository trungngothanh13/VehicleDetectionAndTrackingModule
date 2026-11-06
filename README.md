# Vehicle Detection & Tracking Module (VDAT)

Computer Vision pipeline for real-time vehicle detection, multi-object tracking (**BoT-SORT**, **ByteTrack**, **DeepSORT**), traffic light classification, and automated red-light violation detection.

---

## Quick Start

### 1. Installation

```bash
# System dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1

# Python dependencies
pip install -r init/requirements.txt

# Pre-download YOLO models
python -c "from ultralytics import YOLO; YOLO('yolo26l.pt'); YOLO('yolov8l.pt')"
```

### 2. Ingestion & Streaming Modes

* **Web-Fed Camera Stream Mode (Ingest Video File or Laptop Webcam from `web.VDAT`)**:
  ```bash
  python main.py --web-input --ws-url ws://localhost:5000/ws/inference --no-video
  ```

* **Direct Laptop / USB Webcam Input**:
  ```bash
  python main.py --webcam 0 --ws --ws-url ws://localhost:5000/ws/inference --no-video
  ```

* **Simulated Camera from Local Video File**:
  ```bash
  python main.py -i test_1.mp4 -s --loop --ws --ws-url ws://localhost:5000/ws/inference --no-video
  ```

* **Offline Processing (Benchmark)**:
  ```bash
  python main.py -i test_1.mp4 --tracker bytetrack
  ```

* **Desktop Zone Drawer**:
  ```bash
  python main.py --draw-zones
  ```

---

## Configuration

* **[core/config.py](core/config.py)**: Select tracker (`botsort`, `bytetrack`, `deepsort`), model weights, input source, and detection thresholds.
* **[zones.json](zones.json)**: Approaching lane polygons, intersection boundaries, and traffic light coordinates (automatically synced over WebSocket when edited in `web.VDAT`).
