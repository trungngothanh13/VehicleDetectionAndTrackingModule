# Architecture

## Overview

A modular, multi-threaded Python pipeline for traffic video analytics that reads video frames asynchronously, runs vehicle detection and tracking (supporting **BoT-SORT**, **ByteTrack**, and **DeepSORT**), classifies traffic light state, checks for lane-to-intersection red light violations, and exports an annotated video stream.

---

## High-Level Data Flow

```
Video File
  │
  ▼
FrameReader (daemon thread, 8-frame buffer queue)
  │  decoded BGR frame
  ├──────────────────────► BaseTracker (BoT-SORT / ByteTrack / DeepSORT)
  │                              │ sv.Detections (xyxy, confidence, class_id, tracker_id)
  │                              ▼
  └──────────────────────► TrafficLightDetector (HSV sliding kernel + majority vote)
                                 │ light state: red / green / unknown
                                 ▼
                           ZoneChecker (point-in-polygon)
                                 │ violation events
                                 ▼
                           Visualizer (HUD overlay)
                                 │ annotated frame
                                 ▼
                           VideoWriter / Live Preview
```

---

## Modular Tracker Architecture

All tracking algorithms reside in the `trackers/` package and implement a unified interface (`BaseTracker`):

```
trackers/
├── __init__.py               # Factory function `create_tracker(config)` & registry
├── base_tracker.py           # Abstract BaseTracker (interface, ID bookkeeping, class mappings)
├── botsort/
│   ├── __init__.py
│   └── botsort_tracker.py    # Ultralytics YOLO with BoT-SORT (camera motion compensation)
├── bytetrack/
│   ├── __init__.py
│   └── bytetrack_tracker.py  # Ultralytics YOLO with ByteTrack (high-speed association)
└── deepsort/
    ├── __init__.py
    └── deepsort_tracker.py   # YOLO detector + DeepSORT (deep-sort-realtime appearance ReID)
```

---

## Modules

| Module | Responsibility |
|---|---|
| **config.py** | Centralized configuration: `TRACKER_TYPE` ('botsort', 'bytetrack', 'deepsort'), model weights, hyperparams, video I/O, display options. |
| **trackers/** | Modular tracker implementations inheriting from `BaseTracker` and instantiated via `create_tracker(config)`. |
| **vehicle_model.py** | Backward-compatible wrapper delegating to `trackers/`. |
| **main.py** | Pipeline orchestrator. Owns `FrameReader` (async decode thread) and the per-frame processing loop. |
| **traffic_light_detector.py** | Samples HSV sub-patches around user-defined light coordinates using sliding kernel peak detection and majority voting. |
| **zone_checker.py** | Loads lane/intersection polygons from `zones.json`. Uses `cv2.pointPolygonTest` on bottom-center of bounding boxes. Fires violations when entering intersection on red. |
| **zone_drawer.py** | Interactive OpenCV tool to define lane polygons, intersection polygon, and traffic light coordinate points. |
| **visualizer.py** | Renders bounding boxes, tracker ID labels, zone overlays, and HUD statistics panel. |

---

## Switching Trackers

In `config.py`, change:
```python
TRACKER_TYPE = 'botsort'   # 'botsort' | 'bytetrack' | 'deepsort'
```

Each tracker has its dedicated parameter section in `config.py` (`BOTSORT_CONFIG`, `BYTETRACK_CONFIG`, `DEEPSORT_CONFIG`).
