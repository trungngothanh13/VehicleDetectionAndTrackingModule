import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from sinks.base_sink import BaseSink
from core.events import FrameTelemetry, ViolationEvent


class FileSink(BaseSink):
    """
    Sinks output frames to an MP4 video file, saves high-res JPEG violation evidence snapshots,
    and writes the final text report upon completion.
    """

    def __init__(
        self,
        output_video_path: Optional[str] = None,
        violation_log_path: Optional[str] = None,
        evidence_dir: Optional[str] = None,
        save_video: bool = True,
        save_log: bool = True,
        save_evidence: bool = True,
    ):
        self.output_video_path = output_video_path
        self.violation_log_path = violation_log_path
        self.evidence_dir = evidence_dir
        self.save_video = save_video and (output_video_path is not None)
        self.save_log = save_log and (violation_log_path is not None)
        self.save_evidence = save_evidence

        self._writer: Optional[cv2.VideoWriter] = None
        self._violations: List[ViolationEvent] = []
        self._pipeline_info: Dict[str, Any] = {}
        self._last_telemetry: Optional[FrameTelemetry] = None

    def on_start(self, pipeline_info: Dict[str, Any]) -> None:
        self._pipeline_info = pipeline_info

        # Create output folders
        if self.save_video and self.output_video_path:
            os.makedirs(os.path.dirname(self.output_video_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = pipeline_info.get('fps', 30)
            width = pipeline_info.get('width', 1920)
            height = pipeline_info.get('height', 1080)
            self._writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

        if self.save_log and self.violation_log_path:
            os.makedirs(os.path.dirname(self.violation_log_path) or '.', exist_ok=True)

        if self.save_evidence and self.evidence_dir:
            os.makedirs(self.evidence_dir, exist_ok=True)

    def on_frame(self, frame: np.ndarray, telemetry: FrameTelemetry) -> None:
        self._last_telemetry = telemetry
        if self._writer is not None:
            self._writer.write(frame)

    def on_violation(self, event: ViolationEvent, snapshot: np.ndarray) -> None:
        self._violations.append(event)

        # Save snapshot image
        if self.save_evidence and self.evidence_dir:
            filename = f"violation_{event.camera_id}_id{event.track_id}_f{event.frame_number}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            cv2.imwrite(filepath, snapshot)
            event.snapshot_path = filepath

    def on_complete(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self.save_log and self.violation_log_path:
            self._write_report()

    def _write_report(self) -> None:
        total_frames = self._last_telemetry.frame_number if self._last_telemetry else 0
        total_tracked = self._last_telemetry.total_tracked if self._last_telemetry else 0
        fps = self._pipeline_info.get('fps', 30)

        with open(self.violation_log_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VEHICLE RED-LIGHT VIOLATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Input Video:      {self._pipeline_info.get('input_source', 'N/A')}\n")
            f.write(f"Active Tracker:   {self._pipeline_info.get('tracker_type', 'N/A')}\n")
            f.write(f"Detector Model:   {self._pipeline_info.get('model_name', 'N/A')}\n")
            f.write(f"Total Frames:     {total_frames} (FPS: {fps})\n")
            f.write(f"Total Vehicles:   {total_tracked}\n")
            f.write(f"Total Violations: {len(self._violations)}\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Vehicle ID':<12} | {'Frame':<8} | {'Timestamp':<12} | {'Class':<12}\n")
            f.write("-" * 60 + "\n")
            for r in self._violations:
                f.write(f"{r.track_id:<12} | {r.frame_number:<8} | {r.timestamp:<12} | {r.vehicle_class:<12}\n")
            f.write("=" * 60 + "\n")
