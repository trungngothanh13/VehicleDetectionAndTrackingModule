# Vehicle Detection & Tracking Module (VDAT)

Computer Vision pipeline for real-time vehicle detection, multi-object tracking (**BoT-SORT**, **ByteTrack**, **DeepSORT**), traffic light classification, and automated red-light violation detection.

---

## ⚡ Quick Start

### 1. Installation

```bash
# System dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1

# Python dependencies
pip install -r init/requirements.txt

# Pre-download YOLO models
python -c "from ultralytics import YOLO; YOLO('yolo26l.pt'); YOLO('yolov8l.pt')"
```

*(Or run `bash init/setup_cloud.sh` on cloud GPU instances).*

### 2. Usage

* **Offline Processing (Benchmark)**:
  ```bash
  python main.py -i test_2.mp4 --tracker botsort
  ```

* **Live Streaming (Stream to api.VDAT)**:
  ```bash
  python main.py -i test_2.mp4 -s --loop --ws --ws-url ws://localhost:5000/ws/inference --no-video
  ```

* **Interactive Zone Drawer**:
  ```bash
  python main.py --draw-zones
  ```

---

## ⚙️ Configuration

* **[config.py](config.py)**: Select tracker (`botsort`, `bytetrack`, `deepsort`), model weights, input source, and detection thresholds.
* **[zones.json](zones.json)**: Approaching lane polygons, intersection boundaries, and traffic light coordinates.