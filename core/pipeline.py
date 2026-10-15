import time
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
    ):
        self.reader = reader
        self.tracker = tracker
        self.visualizer = visualizer
        self.sinks = sinks
        self.config = config
        self.light_detector = light_detector
        self.zone_checker = zone_checker
        self.max_frames = max_frames or getattr(config, 'MAX_FRAMES', None)

        self.violations: Set[int] = set()
        self.violation_events: List[ViolationEvent] = []
        self._evidence_recorder: Optional[EvidenceRecorder] = None

    def run(self) -> List[ViolationEvent]:
        """Runs the main inference loop across all frames until completion or max_frames."""
        fps = self.reader.fps
        width = self.reader.width
        height = self.reader.height

        pipeline_info = {
            'input_source': getattr(self.config, 'INPUT_VIDEO', 'Stream'),
            'tracker_type': getattr(self.config, 'TRACKER_TYPE', 'unknown'),
            'model_name': getattr(self.config, 'MODEL_NAME', 'unknown'),
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': self.reader.total_frames,
            'camera_id': getattr(self.config, 'CAMERA_ID', 'cam-01'),
        }

        print("=" * 60)
        print(f"Pipeline Starting: {pipeline_info['input_source']}")
        print(f"Resolution:       {width}x{height} @ {fps} FPS")
        print(f"Active Tracker:   {pipeline_info['tracker_type'].upper()} (Model: {pipeline_info['model_name']})")
        print(f"Max Frames:       {self.max_frames if self.max_frames else 'Full stream'}")
        print(f"Active Sinks:     {[s.__class__.__name__ for s in self.sinks]}")
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

        try:
            while True:
                if self.max_frames and frame_count >= self.max_frames:
                    print(f"\n[Pipeline] Reached configured max_frames ({self.max_frames}). Stopping.")
                    break

                ret, frame = self.reader.read()
                if not ret or frame is None:
                    break

                # Preserve clean, unannotated raw frame for evidence archive
                raw_frame = frame.copy()

                frame_count += 1
                timestamp_sec = frame_count / fps
                timestamp_str = f"{timestamp_sec:.2f}s"

                # 1. Detection + Tracking
                tracked = self.tracker.track(frame)

                # 2. Traffic Light State
                light_state = 'unknown'
                if self.light_detector:
                    light_state = self.light_detector.detect(frame)

                # 3. Zone Violation Evaluation
                new_violations: List[ViolationEvent] = []
                if self.zone_checker and tracked.tracker_id is not None:
                    light_is_red = (light_state == 'red')
                    for i in range(len(tracked)):
                        class_id = int(tracked.class_id[i])
                        if class_id not in self.config.VIOLATION_CLASS_IDS:
                            continue

                        tracker_id = int(tracked.tracker_id[i])
                        bbox = tracked.xyxy[i]

                        if self.zone_checker.check_lane_to_intersection(tracker_id, bbox, light_is_red):
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
                                print(f"🚨 [VIOLATION] Vehicle #{tracker_id} ({class_name}) at Frame {frame_count} ({timestamp_str})")

                # 4. Drawing & HUD Visualization
                if self.zone_checker:
                    frame = self.visualizer.draw_zones(frame, self.zone_checker.lanes, self.zone_checker.intersection)

                frame = self.visualizer.draw_detections(frame, tracked, self.tracker, violation_ids=self.violations)

                current_count = len(tracked) if tracked.tracker_id is not None else 0
                total_tracked = self.tracker.get_total_tracked()

                self.visualizer.set_violations_count(len(self.violations))
                self.visualizer.set_light_state(light_state)
                frame = self.visualizer.draw_statistics(frame, current_count, total_tracked, frame_count)

                # 5. Buffer frame for evidence (if not already hooked into reader)
                if not self._raw_consumer_registered:
                    frame_to_buffer = raw_frame if evidence_mode == 'raw' else frame
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
                    fps=fps,
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
                    print(f"Frame {frame_count}: Vehicles={current_count}, Tracked={total_tracked}, Violations={len(self.violations)}, Light={light_state} ({current_fps:.1f} FPS)")

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
