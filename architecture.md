# Architecture

## Overview

A modular, multi-threaded Python pipeline for traffic video analytics that reads video frames asynchronously (from video files or live RTSP streams), runs vehicle detection and tracking (supporting **BoT-SORT**, **ByteTrack**, and **DeepSORT**), classifies traffic light state, checks for lane-to-intersection red light violations, and dispatches annotated frames and violation events to pluggable sinks (local MP4 file, text report, live desktop preview, or real-time WebSocket streaming to `api.VDAT`).

---

## High-Level Data Flow

```
Video File / RTSP Stream
  │
  ▼
BaseFrameReader (VideoFileReader [threaded buffer] / RTSPStreamReader)
  │  decoded BGR frame
  ├──────────────────────► BaseTracker (BoT-SORT / ByteTrack / DeepSORT)
  │                              │ sv.Detections (xyxy, confidence, class_id, tracker_id)
  │                              ▼
  ├──────────────────────► TrafficLightDetector (HSV sliding kernel + majority vote)
  │                              │ light state: red / green / unknown
  │                              ▼
  └──────────────────────► ZoneChecker (point-in-polygon)
                                 │ violation events
                                 ▼
                           Visualizer (HUD overlay)
                                 │ annotated frame + FrameTelemetry
                                 ▼
                           BaseSink Router
                                 ├─► FileSink (output MP4 + txt report + evidence snapshots)
                                 ├─► WebSocketSink (real-time stream & alerts to api.VDAT)
                                 └─► PreviewSink (local OpenCV window)
```

---

## Modular Package Structure

```
VehicleDetectionAndTrackingModule/
├── core/
│   ├── config.py                 # Centralized configuration with env overrides
│   ├── events.py                 # Structured dataclasses: FrameTelemetry, ViolationEvent
│   └── pipeline.py               # Pure orchestrator: TrafficPipeline
├── ingestion/
│   ├── base_reader.py            # Abstract BaseFrameReader interface
│   ├── file_reader.py            # Thread-buffered VideoFileReader for local .mp4
│   └── rtsp_reader.py            # Resilient RTSPStreamReader with auto-reconnect
├── spatial/
│   └── zone_checker.py           # Polygon collision & lane-to-intersection violation logic
├── traffic_light/
│   └── traffic_light_detector.py # HSV color sampling & temporal majority voting
├── trackers/
│   ├── base_tracker.py           # Abstract BaseTracker interface & class mappings
│   ├── botsort/                  # BoT-SORT (camera motion compensation)
│   ├── bytetrack/                # ByteTrack (high-speed association)
│   └── deepsort/                 # DeepSORT (deep appearance ReID)
├── visualization/
│   └── visualizer.py             # Bounding boxes, zone overlays, and HUD stats
├── sinks/
│   ├── base_sink.py              # Abstract BaseSink interface
│   ├── file_sink.py              # Saves MP4, logs .txt report, saves JPEG snapshots
│   ├── websocket_sink.py         # Asynchronous WebSocket client to api.VDAT
│   └── preview_sink.py           # OpenCV GUI preview window
├── main.py                       # CLI entry point with command-line arguments
└── zone_drawer.py                # Interactive zone calibration GUI tool
```

---

## Running the Pipeline

### 1. Fast Smoke Test (< 100 frames)
```bash
python3 main.py --max-frames 30 --tracker bytetrack
```

### 2. Full Offline Processing (Benchmark Mode)
```bash
python3 main.py --tracker botsort --input test_2.mp4
```

### 3. Integrated Platform Mode (Streaming to api.VDAT)
```bash
python3 main.py --tracker bytetrack --ws --ws-url ws://localhost:5000/ws/inference
```

### 4. Interactive Zone Calibration Tool
```bash
python3 main.py --draw-zones
```
