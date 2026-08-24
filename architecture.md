# Architecture

## Overview

A single-process, multi-threaded Python pipeline that reads a traffic video, runs YOLO26l detection + BoT-SORT tracking on every frame, classifies the traffic light state, checks for red-light violations, and writes an annotated output video.

## High-Level Data Flow

```
Video File
  │
  ▼
FrameReader (daemon thread, 8-frame queue)
  │  decoded BGR frame
  ├──────────────────────► VehicleModel (YOLO26l + BoT-SORT, FP16)
  │                              │ sv.Detections (xyxy, class_id, tracker_id)
  │                              ▼
  └──────────────────────► TrafficLightDetector (HSV sampling + 3-frame vote)
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

## Modules

| Module | Responsibility |
|---|---|
| **config.py** | All tunables: model weights, thresholds, I/O paths, tracker choice, display flags. |
| **main.py** | Pipeline orchestrator. Owns `FrameReader` (async decode thread) and the per-frame processing loop. |
| **vehicle_model.py** | Wraps `ultralytics.YOLO.track()` with FP16, filters to target COCO classes, returns `supervision.Detections`. Tracks cumulative unique IDs. |
| **traffic_light_detector.py** | Samples a 7×7 HSV region at user-defined red/green pixel coordinates. Uses majority-vote smoothing over 3 frames. |
| **zone_checker.py** | Loads lane/intersection polygons from `zones.json`. Uses `cv2.pointPolygonTest` on the bottom-center of each bbox. Fires a violation when a vehicle transitions from a lane into the intersection while the light is red. |
| **zone_drawer.py** | Interactive OpenCV GUI to define lane polygons, intersection polygon, and traffic-light sample points. Saves to `zones.json`. |
| **visualizer.py** | Renders bounding boxes, tracker ID labels, zone overlays, and a statistics panel (vehicle count, violations, light state). |

## Key Design Decisions

- **Async frame decode** — `FrameReader` runs on a daemon thread with an 8-frame `Queue` so CPU video decoding never stalls GPU inference.
- **Single-pass detect+track** — `model.track(persist=True)` keeps BoT-SORT state across frames in one call, avoiding a separate tracker step.
- **FP16 inference** — `quantize='fp16'` for higher throughput on CUDA tensor cores.
- **1280 px input resolution** — trades compute for accuracy on small/distant motorcycles in Vietnam traffic.
- **Stateless violation logic** — `ZoneChecker` only stores two dicts (`_seen_in_lane`, `_violation_reported`) per vehicle ID; no complex FSM.

## External Dependencies

- `ultralytics` — YOLO26l model + BoT-SORT tracker
- `supervision` — `sv.Detections` data container
- `opencv-python` — video I/O, drawing, point-in-polygon tests
- `numpy` — array ops
