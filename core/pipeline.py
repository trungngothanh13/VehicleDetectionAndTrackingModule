import time
from collections import deque
from typing import List, Optional, Set
import numpy as np

from core.events import FrameTelemetry, ViolationEvent
from core.evidence_recorder import EvidenceRecorder
from ingestion.base_reader import BaseFrameReader
from spatial.zone_checker import ZoneChecker
from traffic_light.traffic_light_detector import TrafficLightDetector
from visualization.visualizer import Visualizer
from sinks.base_sink import BaseSink


class TrafficPipeline:
    """
    Orchestrates the modular red-light violation detection and tracking lifecycle.
    Decoupled from specific I/O sources and sinks.
    """

    def __init__(
        self,
        reader: BaseFrameReader,
        tracker,
        visualizer: Visualizer,
        sinks: List[BaseSink],
        config,
        light_detector: Optional[TrafficLightDetector] = None,
        zone_checker: Optional[ZoneChecker] = None,
        max_frames: Optional[int] = None,
        camera_id: Optional[str] = None,
    ):
        self.reader = reader
        self.tracker = tracker
        self.visualizer = visualizer
        self.sinks = sinks
        self.config = config
        self.light_detector = light_detector
        self.zone_checker = zone_checker
        self.max_frames = max_frames or getattr(config, 'MAX_FRAMES', None)
        self.camera_id = camera_id or getattr(config, 'CAMERA_ID', 'cam-01')

        self.violations: Set[int] = set()
        self.violation_events: List[ViolationEvent] = []
        self._evidence_recorder: Optional[EvidenceRecorder] = None
        self._pending_reader: Optional[BaseFrameReader] = None
        self._pending_tracker = None
        self._bypass_zones: bool = False
        self._stopped = False

    def set_bypass_zones(self, bypassed: bool) -> None:
        """Temporarily disables zone drawing and violation checks without altering saved zone files."""
        next_val = bool(bypassed)
        if self._bypass_zones != next_val:
            self._bypass_zones = next_val
            state_str = "BYPASSED (detection/tracking only)" if next_val else "ACTIVE"
            print(f"🚧 [{self.camera_id}] Zone annotations -> {state_str}")

    def request_tracker_switch(self, new_tracker) -> None:
        """Thread-safely schedules a live tracker swap on the next frame."""
        self._pending_tracker = new_tracker

    def stop(self) -> None:
        """Signals this pipeline session to exit gracefully."""
        self._stopped = True
        try:
            self.reader.release()
        except Exception:
            pass

    def reset_state(self) -> None:
        """Resets tracker IDs, zone crossing history, violation set, and evidence buffer on source switch."""
        if self.tracker is not None and hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        if self.zone_checker is not None:
            self.zone_checker.reset()
        self.violations.clear()
        if self._evidence_recorder is not None:
            self._evidence_recorder.clear_buffers()
        print(f"🧹 [{self.camera_id}] Pipeline state & rolling evidence buffer reset for new source")

    def reload_zones(self, zones_file: str) -> None:
        """Dynamically reloads ZoneChecker and TrafficLightDetector when zones are updated remotely."""
        try:
            new_zone_checker = ZoneChecker(zones_file)
            new_light_detector = TrafficLightDetector.from_file(zones_file)
            self.zone_checker = new_zone_checker
            self.light_detector = new_light_detector
            light_count = len(new_light_detector.traffic_lights) if new_light_detector else 0
            print(
                f"🔄 [{self.camera_id}] Applied live zone update from {zones_file}: "
                f"{len(new_zone_checker.lanes)} approach lane(s), "
                f"{len(new_zone_checker.intersection)} intersection pts, "
                f"{len(new_zone_checker.exit_zones)} exit zone(s), "
                f"{light_count} light point(s), "
                f"crossing_mode={new_zone_checker.crossing_mode} ({new_zone_checker.clearance_seconds}s grace)"
            )
        except Exception as e:
            print(f"[{self.camera_id}] Error reloading zones from {zones_file}: {e}")

    def request_reader_switch(self, new_reader: BaseFrameReader, source_label: str = "new source") -> None:
        """Queues a hot-swap of the active frame ingestion reader without restarting YOLO."""
        print(f"🎬 [{self.camera_id}] Queued hot-switch to camera input source: {source_label}")
        self.reset_state()
        self._pending_reader = new_reader
        try:
            self.reader.release()
        except Exception:
            pass

    def run(self) -> List[ViolationEvent]:
        """Runs the main inference loop across all frames until completion or max_frames."""
        fps = self.reader.fps
        width = self.reader.width
        height = self.reader.height

        pipeline_info = {
            'input_source': getattr(self.config, 'INPUT_VIDEO', 'Stream'),
            'tracker_type': getattr(self.config, 'TRACKER_TYPE', 'unknown'),
            'model_name': getattr(self.config, 'MODEL_NAME', 'unknown'),
            'model_profile': getattr(self.config, 'MODEL_PROFILE', 'trained'),
            'confidence': getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.45),
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': self.reader.total_frames,
            'camera_id': self.camera_id,
        }

        print("=" * 60)
        print(f"[{self.camera_id}] Pipeline Starting: {pipeline_info['input_source']}")
        print(f"[{self.camera_id}] Resolution:       {width}x{height} @ {fps} FPS")
        print(f"[{self.camera_id}] Active Tracker:   {pipeline_info['tracker_type'].upper()} (Model: {pipeline_info['model_name']})")
        print(f"[{self.camera_id}] Max Frames:       {self.max_frames if self.max_frames else 'Full stream'}")
        print(f"[{self.camera_id}] Active Sinks:     {[s.__class__.__name__ for s in self.sinks]}")
        print("=" * 60 + "\n")

        # Initialize the evidence video clip recorder
        evidence_dir = getattr(self.config, 'EVIDENCE_DIR', 'output/evidence')
        pre_seconds = getattr(self.config, 'EVIDENCE_PRE_SECONDS', 3.0)
        post_seconds = getattr(self.config, 'EVIDENCE_POST_SECONDS', 3.0)
        evidence_mode = getattr(self.config, 'EVIDENCE_MODE', 'raw')

        self._evidence_recorder = EvidenceRecorder(
            fps=fps,
            width=width,
            height=height,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            evidence_dir=evidence_dir,
            on_clip_ready=self._on_evidence_clip_ready,
        )

        # In raw mode, hook directly into the ingestion reader so evidence records
        # smooth, continuous camera footage at native camera FPS
        self._raw_consumer_registered = False
        if evidence_mode == 'raw' and hasattr(self.reader, 'register_raw_consumer'):
            self.reader.register_raw_consumer(self._evidence_recorder.buffer_frame)
            self._raw_consumer_registered = True
            print(f"[Evidence] Hooked raw recorder into ingestion stream at native {fps} FPS")

        print(f"Evidence Recorder: {pre_seconds}s pre + {post_seconds}s post = {pre_seconds + post_seconds}s clips (Mode: {evidence_mode})")

        # Notify all sinks of pipeline start
        for sink in self.sinks:
            sink.on_start(pipeline_info)

        frame_count = 0
        start_time = time.time()
        fps_window: deque = deque(maxlen=30)

        try:
            while not self._stopped:
                if self._pending_tracker is not None:
                    self.tracker = self._pending_tracker
                    self._pending_tracker = None
                    if self.zone_checker is not None:
                        self.zone_checker.reset()
                    self.violations.clear()
                    print(f"🧭 [{self.camera_id}] Swapped active tracker live -> {self.tracker.__class__.__name__}")

                if self._pending_reader is not None:
                    old_reader = self.reader
                    self.reader = self._pending_reader
                    self._pending_reader = None
                    try:
                        old_reader.release()
                    except Exception:
                        pass
                    fps = self.reader.fps or 30
                    if self._evidence_recorder is not None:
                        self._evidence_recorder.reconfigure(
                            fps=fps,
                            width=self.reader.width,
                            height=self.reader.height,
                        )
                    self._raw_consumer_registered = False
                    if evidence_mode == 'raw' and hasattr(self.reader, 'register_raw_consumer') and self._evidence_recorder:
                        self.reader.register_raw_consumer(self._evidence_recorder.buffer_frame)
                        self._raw_consumer_registered = True
                    frame_count = 0
                    start_time = time.time()
                    fps_window.clear()
                    print(f"[Pipeline] Switched active camera source ({self.reader.width}x{self.reader.height} @ {fps} FPS)")

                if self.max_frames and frame_count >= self.max_frames:
                    print(f"\n[Pipeline] Reached configured max_frames ({self.max_frames}). Stopping.")
                    break

                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self._pending_reader is not None:
                        continue
                    break

                source_frame_id = getattr(self.reader, 'last_source_frame_id', None)

                # Preserve clean, unannotated raw frame only when EvidenceRecorder isn't already hooked to reader
                raw_frame = None if self._raw_consumer_registered else frame.copy()

                frame_count += 1
                now_mono = time.monotonic()
                fps_window.append(now_mono)
                if len(fps_window) >= 2 and (fps_window[-1] - fps_window[0]) > 0.05:
                    live_fps = round((len(fps_window) - 1) / (fps_window[-1] - fps_window[0]), 1)
                else:
                    live_fps = float(fps)
                timestamp_sec = frame_count / fps
                timestamp_str = f"{timestamp_sec:.2f}s"

                # 1. Detection + Tracking
                tracked = self.tracker.track(frame)

                # 2. Traffic Light State (skipped when zone annotations are temporarily bypassed)
                light_state = 'unknown'
                if not self._bypass_zones and self.light_detector:
                    light_state = self.light_detector.detect(frame)

                # 3. Zone Violation Evaluation (skipped when zone annotations are temporarily bypassed)
                new_violations: List[ViolationEvent] = []
                if not self._bypass_zones and self.zone_checker:
                    self.zone_checker.update_light_state(light_state, frame_count, fps)
                if not self._bypass_zones and self.zone_checker and tracked.tracker_id is not None:
                    light_is_red = (light_state == 'red')
                    for i in range(len(tracked)):
                        class_id = int(tracked.class_id[i])
                        if class_id not in self.config.VIOLATION_CLASS_IDS:
                            continue

                        tracker_id = int(tracked.tracker_id[i])
                        bbox = tracked.xyxy[i]

                        if self.zone_checker.check_lane_to_intersection(
                            tracker_id,
                            bbox,
                            light_is_red=light_is_red,
                            frame_number=frame_count,
                            fps=fps,
                        ):
                            if tracker_id not in self.violations:
                                self.violations.add(tracker_id)
                                class_name = self.tracker.get_class_name(class_id)
                                conf = float(tracked.confidence[i]) if tracked.confidence is not None else None

                                event = ViolationEvent(
                                    camera_id=pipeline_info['camera_id'],
                                    track_id=tracker_id,
                                    vehicle_class=class_name,
                                    class_id=class_id,
                                    frame_number=frame_count,
                                    timestamp=timestamp_str,
                                    light_state=light_state,
                                    bbox=[float(c) for c in bbox],
                                    confidence=conf,
                                    snapshot_path=None,
                                    evidence_clip_path=None,
                                )
                                new_violations.append(event)
                                self.violation_events.append(event)
                                print(f"🚨 [{self.camera_id}] [VIOLATION] Vehicle #{tracker_id} ({class_name}) at Frame {frame_count} ({timestamp_str})")

                # 4. Drawing & HUD Visualization (skip expensive OpenCV overlays for unfocused background cameras unless a violation occurs)
                should_annotate = bool(new_violations) or any(
                    getattr(s, 'needs_annotated_frame', True) for s in self.sinks
                )
                if should_annotate:
                    if not self._bypass_zones and self.zone_checker:
                        lights = self.light_detector.traffic_lights if self.light_detector else None
                        frame = self.visualizer.draw_zones(
                            frame,
                            self.zone_checker.lanes,
                            self.zone_checker.intersection,
                            lights,
                            exit_zones=self.zone_checker.exit_zones,
                        )

                    active_violation_ids = None if self._bypass_zones else self.violations
                    frame = self.visualizer.draw_detections(
                        frame,
                        tracked,
                        self.tracker,
                        violation_ids=active_violation_ids,
                    )

                current_count = len(tracked) if tracked.tracker_id is not None else 0
                total_tracked = self.tracker.get_total_tracked()

                # 5. Buffer frame for evidence (if not already hooked into reader)
                if not self._raw_consumer_registered:
                    frame_to_buffer = raw_frame if (evidence_mode == 'raw' and raw_frame is not None) else frame
                    self._evidence_recorder.buffer_frame(frame_to_buffer, frame_count)

                # 6. Trigger evidence clip recording for new violations
                for event in new_violations:
                    self._evidence_recorder.trigger_clip(event)
                    # Also dispatch the instant snapshot (annotated for operator clarity) to sinks
                    for sink in self.sinks:
                        sink.on_violation(event, frame.copy())

                # 7. Dispatch frame and telemetry to all sinks
                telemetry = FrameTelemetry(
                    frame_number=frame_count,
                    timestamp=timestamp_sec,
                    fps=live_fps,
                    vehicle_count=current_count,
                    total_tracked=total_tracked,
                    light_state=light_state,
                    violations_count=len(self.violations),
                )

                for sink in self.sinks:
                    sink.on_frame(frame, telemetry)

                # Check if any preview sink requested exit
                stop_requested = any(getattr(s, 'should_stop', False) for s in self.sinks)
                if stop_requested:
                    break

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"[{self.camera_id}] Frame {frame_count}: Vehicles={current_count}, Tracked={total_tracked}, Violations={len(self.violations)}, Light={light_state} ({current_fps:.1f} FPS)")

        finally:
            # Flush any in-progress evidence clips before shutting down
            if self._evidence_recorder:
                self._evidence_recorder.flush()

            self.reader.release()
            for sink in self.sinks:
                sink.on_complete()

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print("\n" + "=" * 60)
        print(f"Processing Complete in {elapsed:.2f}s ({avg_fps:.1f} FPS)")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Unique Vehicles:  {self.tracker.get_total_tracked()}")
        print(f"Total Violations:       {len(self.violation_events)}")
        print("=" * 60)

        return self.violation_events

    def _on_evidence_clip_ready(self, event: ViolationEvent, clip_path: str) -> None:
        """
        Callback from EvidenceRecorder when a clip has been fully encoded.
        Dispatches the completed clip to all sinks that support it.
        """
        for sink in self.sinks:
            if hasattr(sink, 'on_evidence_clip'):
                sink.on_evidence_clip(event, clip_path)
