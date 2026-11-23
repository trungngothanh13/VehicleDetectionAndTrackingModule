#!/usr/bin/env python3
"""
Disposable Multi-Object Tracker Benchmark Script for Thesis Section 6.2.

Benchmarks ByteTrack, BoT-SORT, and DeepSORT across every video in `videotemp/`
using identical YOLO detector weights, frame resolution, and optional zone configs,
and exports a single thesis-ready Markdown report (`tracker_benchmark_report.md`).

Usage on Thunder Compute (zero-arg default):
    python benchmark_trackers.py

Optional examples:
    python benchmark_trackers.py --legacy
    python benchmark_trackers.py --save-videos --print-md
    python benchmark_trackers.py --video-dir videotemp --zones output/zones/cam-01.json
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
import statistics
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


TRACKER_DISPLAY = {
    "bytetrack": "ByteTrack",
    "botsort": "BoT-SORT",
    "deepsort": "DeepSORT",
}
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


@dataclass
class VideoMeta:
    filename: str
    path: str
    native_width: int
    native_height: int
    eval_width: int
    eval_height: int
    fps: float
    total_frames: int
    evaluated_frames: int
    duration_sec: float
    zones_path: Optional[str] = None


@dataclass
class ViolationRecord:
    frame: int
    timestamp: str
    track_id: int
    vehicle_class: str
    light_state: str
    confidence: Optional[float]


@dataclass
class TrackerRunResult:
    tracker: str
    video_name: str
    evaluated_frames: int
    runtime_sec: float
    avg_fps: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    peak_vram_mb: Optional[float]
    total_track_ids: int
    avg_active_vehicles: float
    peak_active_vehicles: int
    mean_lifespan_frames: float
    median_lifespan_frames: float
    max_lifespan_frames: int
    short_tracks_lt5: int
    short_track_ratio_pct: float
    per_class_track_ids: Dict[str, int] = field(default_factory=dict)
    has_zones: bool = False
    violations: List[ViolationRecord] = field(default_factory=list)
    annotated_video_path: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ByteTrack, BoT-SORT, and DeepSORT for VDAT Thesis Section 6.2"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videotemp",
        help="Directory containing benchmark videos (default: videotemp)",
    )
    parser.add_argument(
        "--zones-dir",
        type=str,
        default="videotemp",
        help="Directory searched for per-video <stem>.json zone configs (default: videotemp)",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=None,
        help="Explicit zone JSON file applied to all benchmark videos",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["bytetrack", "botsort", "deepsort"],
        choices=["bytetrack", "botsort", "deepsort"],
        help="Trackers to benchmark in order (default: bytetrack botsort deepsort)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="YOLO weights override (default: weights/best.pt, or yolo26l.pt with --legacy)",
    )
    parser.add_argument(
        "--legacy",
        "--stock",
        action="store_true",
        dest="legacy",
        help="Evaluate using legacy stock COCO YOLO26l weights (yolo26l.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO inference image size (default: Config.IMGSZ = 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold override",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="854x480",
        help="Evaluation resolution WxH or 'native' (default: 854x480)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit frames per video for quick testing (0 = full video)",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save annotated MP4s under output/benchmark/ for manual TP verification",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="tracker_benchmark_report.md",
        help="Output Markdown file path (default: tracker_benchmark_report.md)",
    )
    parser.add_argument(
        "--print-md",
        action="store_true",
        help="Echo the generated Markdown report to stdout",
    )
    return parser.parse_args()


def parse_resize_spec(spec: str) -> Optional[Tuple[int, int]]:
    cleaned = (spec or "").strip().lower()
    if not cleaned or cleaned == "native":
        return None
    if "x" not in cleaned:
        raise ValueError(f"Invalid --resize value '{spec}'. Use WxH (e.g. 854x480) or 'native'.")
    w_str, h_str = cleaned.split("x", 1)
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid --resize dimensions: {spec}")
    return w, h


def find_zones_for_video(
    video_path: str,
    explicit_zones: Optional[str],
    zones_dir: str,
) -> Optional[str]:
    if explicit_zones:
        return explicit_zones if os.path.exists(explicit_zones) else None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidates = [
        os.path.join(zones_dir, f"{stem}.json"),
        os.path.join("output", "zones", f"{stem}.json"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def format_duration_human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{seconds:.1f} s ({mins} min {secs:04.1f} s)"


def _sync_cuda(torch_mod: Any) -> None:
    if torch_mod is not None and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def _probe_video(
    video_path: str,
    resize_dims: Optional[Tuple[int, int]],
    max_frames: int,
    zones_path: Optional[str],
) -> VideoMeta:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    eval_w, eval_h = resize_dims if resize_dims else (native_w, native_h)
    est_eval = min(total_frames, max_frames) if (max_frames > 0 and total_frames > 0) else (max_frames or total_frames)
    duration_sec = (est_eval / fps) if (fps > 0 and est_eval > 0) else 0.0
    return VideoMeta(
        filename=os.path.basename(video_path),
        path=video_path,
        native_width=native_w,
        native_height=native_h,
        eval_width=eval_w,
        eval_height=eval_h,
        fps=fps,
        total_frames=total_frames,
        evaluated_frames=est_eval,
        duration_sec=duration_sec,
        zones_path=zones_path,
    )


def benchmark_single_run(
    video_meta: VideoMeta,
    tracker_name: str,
    config: Any,
    batch_runner: Any,
    model_lock: threading.Lock,
    resize_dims: Optional[Tuple[int, int]],
    max_frames: int,
    save_videos: bool,
    torch_mod: Any,
) -> TrackerRunResult:
    import cv2
    from spatial import ZoneChecker
    from trackers import create_tracker
    from traffic_light import TrafficLightDetector
    from visualization import Visualizer

    config.TRACKER_TYPE = tracker_name
    tracker = create_tracker(
        config,
        shared_model=batch_runner,
        model_lock=model_lock,
        camera_id=f"bench-{tracker_name}",
    )

    zone_checker = None
    light_detector = None
    if video_meta.zones_path:
        zone_checker = ZoneChecker(video_meta.zones_path)
        light_detector = TrafficLightDetector.from_file(video_meta.zones_path)

    cap = cv2.VideoCapture(video_meta.path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_meta.path}")

    # Warm up tracker internals (Kalman filter / ReID / GMC) before timed pass
    ret_first, first_frame = cap.read()
    if ret_first and first_frame is not None:
        if resize_dims and (first_frame.shape[1], first_frame.shape[0]) != resize_dims:
            first_frame = cv2.resize(first_frame, resize_dims, interpolation=cv2.INTER_AREA)
        for _ in range(5):
            tracker.track(first_frame)
    tracker.reset()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    annotated_path = None
    visualizer = None
    if save_videos:
        out_dir = os.path.join("output", "benchmark")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(video_meta.filename)[0]
        annotated_path = os.path.join(out_dir, f"{stem}_{tracker_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            annotated_path,
            fourcc,
            video_meta.fps or 30.0,
            (video_meta.eval_width, video_meta.eval_height),
        )
        visualizer = Visualizer(config.DETECTION_CLASSES)

    track_frames: Counter = Counter()
    track_class: Dict[int, str] = {}
    active_counts: List[int] = []
    latencies_ms: List[float] = []
    violations: List[ViolationRecord] = []
    violation_ids = set()

    has_cuda = bool(torch_mod is not None and torch_mod.cuda.is_available())
    if has_cuda:
        torch_mod.cuda.reset_peak_memory_stats()
    _sync_cuda(torch_mod)
    t_run_start = time.perf_counter()

    frame_idx = 0
    try:
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if resize_dims and (frame.shape[1], frame.shape[0]) != resize_dims:
                frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_AREA)

            frame_idx += 1
            t_frame_start = time.perf_counter()

            tracked = tracker.track(frame)
            num_active = len(tracked) if tracked.tracker_id is not None else 0
            active_counts.append(num_active)

            if tracked.tracker_id is not None:
                for i in range(len(tracked)):
                    tid = int(tracked.tracker_id[i])
                    cid = int(tracked.class_id[i])
                    track_frames[tid] += 1
                    if tid not in track_class:
                        track_class[tid] = tracker.get_class_name(cid)

            if zone_checker is not None:
                light_state = light_detector.detect(frame) if light_detector else "unknown"
                zone_checker.update_light_state(light_state, frame_idx, video_meta.fps)
                if tracked.tracker_id is not None:
                    light_is_red = light_state == "red"
                    for i in range(len(tracked)):
                        cid = int(tracked.class_id[i])
                        if cid not in config.VIOLATION_CLASS_IDS:
                            continue
                        tid = int(tracked.tracker_id[i])
                        bbox = tracked.xyxy[i]
                        if zone_checker.check_lane_to_intersection(
                            tid,
                            bbox,
                            light_is_red=light_is_red,
                            frame_number=frame_idx,
                            fps=video_meta.fps,
                        ):
                            if tid not in violation_ids:
                                violation_ids.add(tid)
                                conf = (
                                    float(tracked.confidence[i])
                                    if tracked.confidence is not None
                                    else None
                                )
                                violations.append(
                                    ViolationRecord(
                                        frame=frame_idx,
                                        timestamp=f"{frame_idx / video_meta.fps:.2f}s",
                                        track_id=tid,
                                        vehicle_class=tracker.get_class_name(cid),
                                        light_state=light_state,
                                        confidence=conf,
                                    )
                                )

            if writer is not None and visualizer is not None:
                vis_frame = frame.copy()
                if zone_checker is not None:
                    lights = light_detector.traffic_lights if light_detector else None
                    vis_frame = visualizer.draw_zones(
                        vis_frame,
                        zone_checker.lanes,
                        zone_checker.intersection,
                        lights,
                        exit_zones=zone_checker.exit_zones,
                    )
                vis_frame = visualizer.draw_detections(
                    vis_frame,
                    tracked,
                    tracker,
                    violation_ids=violation_ids if zone_checker is not None else None,
                )
                writer.write(vis_frame)

            latencies_ms.append((time.perf_counter() - t_frame_start) * 1000.0)

            if frame_idx % 250 == 0:
                elapsed_so_far = time.perf_counter() - t_run_start
                cur_fps = frame_idx / elapsed_so_far if elapsed_so_far > 0 else 0.0
                print(
                    f"  [{video_meta.filename} | {TRACKER_DISPLAY.get(tracker_name, tracker_name)}] "
                    f"Frame {frame_idx} ({cur_fps:.1f} FPS, active={num_active}, ids={len(track_frames)})"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    _sync_cuda(torch_mod)
    total_runtime = max(1e-6, time.perf_counter() - t_run_start)
    peak_vram_mb = (
        round(torch_mod.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if has_cuda
        else None
    )

    lifespans = list(track_frames.values())
    total_ids = len(lifespans)
    short_lt5 = sum(1 for n in lifespans if n < 5)
    short_ratio = (short_lt5 * 100.0 / total_ids) if total_ids > 0 else 0.0
    sorted_lat = sorted(latencies_ms)
    per_class_counts = dict(sorted(Counter(track_class.values()).items()))

    return TrackerRunResult(
        tracker=tracker_name,
        video_name=video_meta.filename,
        evaluated_frames=frame_idx,
        runtime_sec=total_runtime,
        avg_fps=(frame_idx / total_runtime) if total_runtime > 0 else 0.0,
        mean_latency_ms=statistics.mean(latencies_ms) if latencies_ms else 0.0,
        p50_latency_ms=percentile(sorted_lat, 50.0),
        p95_latency_ms=percentile(sorted_lat, 95.0),
        peak_vram_mb=peak_vram_mb,
        total_track_ids=total_ids,
        avg_active_vehicles=statistics.mean(active_counts) if active_counts else 0.0,
        peak_active_vehicles=max(active_counts) if active_counts else 0,
        mean_lifespan_frames=statistics.mean(lifespans) if lifespans else 0.0,
        median_lifespan_frames=statistics.median(lifespans) if lifespans else 0.0,
        max_lifespan_frames=max(lifespans) if lifespans else 0,
        short_tracks_lt5=short_lt5,
        short_track_ratio_pct=short_ratio,
        per_class_track_ids=per_class_counts,
        has_zones=(zone_checker is not None),
        violations=violations,
        annotated_video_path=annotated_path,
    )


def render_markdown_report(
    generated_at: str,
    hardware_desc: str,
    precision_desc: str,
    model_path: str,
    model_profile: str,
    classes: Dict[int, str],
    imgsz: int,
    conf: float,
    videos: List[VideoMeta],
    results: List[TrackerRunResult],
) -> str:
    lines: List[str] = []
    lines.append("# Section 6.2 Multi-Object Tracker Comparison Benchmark Report")
    lines.append("")
    lines.append(f"- **Generated:** {generated_at}")
    lines.append(f"- **Hardware:** `{hardware_desc}` ({precision_desc})")
    lines.append(f"- **Detector Weights:** `{model_path}` (`profile={model_profile}`)")
    lines.append(f"- **Active Vehicle Classes:** `{classes}`")
    lines.append(f"- **Inference Config:** `imgsz={imgsz}`, `conf={conf}`")
    lines.append(f"- **Videos Benchmarked:** {', '.join(v.filename for v in videos)}")
    lines.append("")

    # 1. Input Videos Summary
    lines.append("## 1. Benchmark Input Videos")
    lines.append("")
    lines.append("| Video | Native Res | Eval Res | FPS | Eval Frames | Duration | Zone Config |")
    lines.append("| :--- | :---: | :---: | ---: | ---: | ---: | :--- |")
    for v in videos:
        zone_lbl = f"`{v.zones_path}`" if v.zones_path else "None (Tracking-only)"
        eval_frames_actual = next(
            (r.evaluated_frames for r in results if r.video_name == v.filename),
            v.evaluated_frames,
        )
        dur = (eval_frames_actual / v.fps) if v.fps > 0 else v.duration_sec
        lines.append(
            f"| `{v.filename}` | {v.native_width}x{v.native_height} | "
            f"{v.eval_width}x{v.eval_height} | {v.fps:.1f} | {eval_frames_actual:,} | "
            f"{dur:.1f} s | {zone_lbl} |"
        )
    lines.append("")

    # 2. Section 6.2 Thesis Table
    lines.append("## 2. Table 6.2 Summary (Ready for Thesis Section 6.2)")
    lines.append("")
    lines.append(
        "> Fill **Verified True Positives** after inspecting the output or recorded violation events below."
    )
    lines.append("")
    lines.append(
        "| Tracker | Video | Runtime | Avg FPS | Avg Latency (ms) | Total Track IDs / Violations Recorded | Verified True Positives |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | :---: | :--- |")
    for v in videos:
        vid_results = [r for r in results if r.video_name == v.filename]
        for r in vid_results:
            disp = TRACKER_DISPLAY.get(r.tracker, r.tracker)
            viol_str = f"{len(r.violations)} violations" if r.has_zones else "N/A (no zones JSON)"
            lines.append(
                f"| **{disp}** | `{r.video_name}` | {format_duration_human(r.runtime_sec)} | "
                f"{r.avg_fps:.1f} FPS | {r.mean_latency_ms:.2f} ms | "
                f"{r.total_track_ids:,} IDs / {viol_str} | [PENDING MANUAL VERIFICATION] |"
            )
    lines.append("")

    # 3. Detailed Runtime & Tracking Stability Metrics per Video
    lines.append("## 3. Detailed Runtime, Latency & Track-Stability Metrics")
    lines.append("")
    class_names = sorted({c for r in results for c in r.per_class_track_ids.keys()})

    for v in videos:
        vid_results = [r for r in results if r.video_name == v.filename]
        if not vid_results:
            continue
        slowest_time = max(r.runtime_sec for r in vid_results)

        lines.append(f"### Video: `{v.filename}`")
        lines.append("")
        lines.append(
            "| Tracker | Runtime (s) | Speedup vs Slowest | Avg FPS | Mean / P50 / P95 (ms) | Peak VRAM | Total IDs | Avg / Peak Active | Mean / Med / Max Lifespan (frames) | Short Tracks (<5f) |"
        )
        lines.append(
            "| :--- | ---: | ---: | ---: | :---: | ---: | ---: | :---: | :---: | ---: |"
        )
        for r in vid_results:
            disp = TRACKER_DISPLAY.get(r.tracker, r.tracker)
            speedup = ((slowest_time - r.runtime_sec) / slowest_time * 100.0) if slowest_time > 0 else 0.0
            speed_str = "baseline" if abs(speedup) < 0.05 else f"+{speedup:.1f}% faster"
            vram_str = f"{r.peak_vram_mb:.1f} MB" if r.peak_vram_mb is not None else "N/A"
            lines.append(
                f"| **{disp}** | {r.runtime_sec:.2f} s | {speed_str} | {r.avg_fps:.1f} | "
                f"{r.mean_latency_ms:.2f} / {r.p50_latency_ms:.2f} / {r.p95_latency_ms:.2f} | "
                f"{vram_str} | {r.total_track_ids:,} | "
                f"{r.avg_active_vehicles:.1f} / {r.peak_active_vehicles} | "
                f"{r.mean_lifespan_frames:.1f} / {r.median_lifespan_frames:.1f} / {r.max_lifespan_frames} | "
                f"{r.short_tracks_lt5:,} ({r.short_track_ratio_pct:.1f}%) |"
            )
        lines.append("")

        if class_names:
            lines.append(f"#### Unique Track IDs by Vehicle Class (`{v.filename}`)")
            lines.append("")
            header_cols = " | ".join(f"`{c}`" for c in class_names)
            sep_cols = " | ".join("---:" for _ in class_names)
            lines.append(f"| Tracker | {header_cols} | Total IDs |")
            lines.append(f"| :--- | {sep_cols} | ---: |")
            for r in vid_results:
                disp = TRACKER_DISPLAY.get(r.tracker, r.tracker)
                vals = " | ".join(f"{r.per_class_track_ids.get(c, 0):,}" for c in class_names)
                lines.append(f"| **{disp}** | {vals} | {r.total_track_ids:,} |")
            lines.append("")

    # 4. Auto-Computed Comparison Highlights
    lines.append("## 4. Comparison Highlights (For Section 6.2 Analysis)")
    lines.append("")
    for v in videos:
        vid_results = [r for r in results if r.video_name == v.filename]
        if len(vid_results) < 2:
            continue
        fastest = min(vid_results, key=lambda r: r.runtime_sec)
        slowest = max(vid_results, key=lambda r: r.runtime_sec)
        fewest_ids = min(vid_results, key=lambda r: r.total_track_ids)
        most_ids = max(vid_results, key=lambda r: r.total_track_ids)
        lowest_frag = min(vid_results, key=lambda r: r.short_track_ratio_pct)

        speed_pct = (
            (slowest.runtime_sec - fastest.runtime_sec) / slowest.runtime_sec * 100.0
            if slowest.runtime_sec > 0
            else 0.0
        )
        id_diff_pct = (
            (most_ids.total_track_ids - fewest_ids.total_track_ids) / most_ids.total_track_ids * 100.0
            if most_ids.total_track_ids > 0
            else 0.0
        )
        lines.append(
            f"- **`{v.filename}`:** Fastest tracker was **{TRACKER_DISPLAY.get(fastest.tracker, fastest.tracker)}** "
            f"({fastest.avg_fps:.1f} FPS, {speed_pct:.1f}% faster than {TRACKER_DISPLAY.get(slowest.tracker, slowest.tracker)}). "
            f"Fewest raw track IDs came from **{TRACKER_DISPLAY.get(fewest_ids.tracker, fewest_ids.tracker)}** "
            f"({fewest_ids.total_track_ids:,} vs. {most_ids.total_track_ids:,} max, {id_diff_pct:.1f}% lower ID count). "
            f"Lowest `<5`-frame fragmentation: **{TRACKER_DISPLAY.get(lowest_frag.tracker, lowest_frag.tracker)}** "
            f"({lowest_frag.short_track_ratio_pct:.1f}%)."
        )
    lines.append("")

    # 5. Violation Event Audit Log (if zones were active)
    zoned_results = [r for r in results if r.has_zones]
    if zoned_results:
        lines.append("## 5. Recorded Violation Audit Log (For Manual TP Verification)")
        lines.append("")
        for r in zoned_results:
            disp = TRACKER_DISPLAY.get(r.tracker, r.tracker)
            lines.append(f"### `{r.video_name}` — {disp} ({len(r.violations)} recorded)")
            lines.append("")
            if not r.violations:
                lines.append("_No violations triggered._")
                lines.append("")
                continue
            lines.append("| # | Frame | Time | Track ID | Class | Light | Confidence |")
            lines.append("| ---: | ---: | ---: | ---: | :--- | :--- | ---: |")
            for idx, ev in enumerate(r.violations, start=1):
                conf_str = f"{ev.confidence * 100:.1f}%" if ev.confidence is not None else "N/A"
                lines.append(
                    f"| {idx} | {ev.frame} | {ev.timestamp} | #{ev.track_id} | "
                    f"{ev.vehicle_class} | {ev.light_state} | {conf_str} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_benchmark(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    from ultralytics import YOLO

    from core.config import Config
    from core.model_resolver import apply_model_classes, resolve_profile_model
    from trackers.base_tracker import BatchModelRunner

    resize_dims = parse_resize_spec(args.resize)

    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    video_files = sorted(
        os.path.join(args.video_dir, name)
        for name in os.listdir(args.video_dir)
        if name.lower().endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(args.video_dir, name))
    )
    if not video_files:
        raise RuntimeError(f"No video files found in '{args.video_dir}'.")

    config = Config()
    if args.imgsz is not None:
        config.IMGSZ = args.imgsz

    if args.legacy:
        config.MODEL_PROFILE = "stock"
    if args.model:
        config.MODEL_NAME = args.model
        if "yolo26l.pt" in os.path.basename(args.model).lower() and not args.legacy:
            config.MODEL_PROFILE = "stock"
    else:
        resolved_profile, resolved_weights, profile_conf = resolve_profile_model(config, config.MODEL_PROFILE)
        config.MODEL_PROFILE = resolved_profile
        config.MODEL_NAME = resolved_weights
        if args.conf is None:
            config.CONFIDENCE_THRESHOLD = profile_conf

    if args.conf is not None:
        config.CONFIDENCE_THRESHOLD = args.conf

    hardware_desc = "CPU (CUDA unavailable)"
    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        hardware_desc = f"CUDA: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)"

    print(
        f"🧠 [Benchmark] Loading YOLO model '{config.MODEL_NAME}' "
        f"(profile={config.MODEL_PROFILE}, imgsz={config.IMGSZ}, conf={config.CONFIDENCE_THRESHOLD})"
    )
    raw_model = YOLO(config.MODEL_NAME)
    resolved_classes = apply_model_classes(config, getattr(raw_model, "names", None))
    batch_runner = BatchModelRunner(
        raw_model,
        max_batch_size=1,
        max_wait_sec=0.0,
        warmup_imgsz=config.IMGSZ,
    )
    precision_desc = f"precision={batch_runner._predict_precision_kwargs or 'FP32'}"
    model_lock = threading.Lock()

    videos: List[VideoMeta] = []
    for vp in video_files:
        z_path = find_zones_for_video(vp, args.zones, args.zones_dir)
        videos.append(_probe_video(vp, resize_dims, args.max_frames, z_path))

    results: List[TrackerRunResult] = []
    try:
        for video_meta in videos:
            for tracker_name in args.trackers:
                disp = TRACKER_DISPLAY.get(tracker_name, tracker_name)
                print(
                    f"\n▶ Benchmarking {disp} on '{video_meta.filename}' "
                    f"({video_meta.eval_width}x{video_meta.eval_height}, "
                    f"zones={video_meta.zones_path or 'None'})..."
                )
                res = benchmark_single_run(
                    video_meta=video_meta,
                    tracker_name=tracker_name,
                    config=config,
                    batch_runner=batch_runner,
                    model_lock=model_lock,
                    resize_dims=resize_dims,
                    max_frames=args.max_frames,
                    save_videos=args.save_videos,
                    torch_mod=torch,
                )
                results.append(res)
                print(
                    f"✅ Finished {disp} on '{video_meta.filename}': "
                    f"{res.runtime_sec:.2f}s ({res.avg_fps:.1f} FPS), "
                    f"{res.total_track_ids} IDs, {len(res.violations)} violations"
                )
    finally:
        batch_runner.stop()

    report_md = render_markdown_report(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        hardware_desc=hardware_desc,
        precision_desc=precision_desc,
        model_path=config.MODEL_NAME,
        model_profile=config.MODEL_PROFILE,
        classes=resolved_classes,
        imgsz=config.IMGSZ,
        conf=config.CONFIDENCE_THRESHOLD,
        videos=videos,
        results=results,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n📄 Saved Markdown benchmark report -> {args.output}")
    if args.print_md:
        print("\n" + report_md)


if __name__ == "__main__":
    run_benchmark(parse_args())
