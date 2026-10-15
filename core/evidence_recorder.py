import cv2
import os
import threading
from collections import deque
from typing import Dict, Optional, List, Tuple, Callable
import numpy as np

from core.events import ViolationEvent


class EvidenceRecorder:
    """
    Maintains a rolling buffer of recent annotated frames and produces
    short MP4 evidence clips when violations are triggered.

    Flow:
        1. Pipeline feeds every annotated frame via `buffer_frame()`.
        2. When a violation is detected, `trigger_clip()` is called.
        3. The recorder snapshots the buffered PRE-trigger frames and
           continues collecting POST-trigger frames.
        4. Once the post-trigger window expires, the full clip is
           encoded to MP4 on a background thread.
        5. The completed clip path is delivered via a callback.
    """

    def __init__(
        self,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
        evidence_dir: str = "output/evidence",
        on_clip_ready: Optional[Callable[[ViolationEvent, str], None]] = None,
    ):
        self.fps = fps
        self.width = width
        self.height = height
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.evidence_dir = evidence_dir
        self.on_clip_ready = on_clip_ready

        self._pre_frames = int(fps * pre_seconds)
        self._post_frames = int(fps * post_seconds)

        # Rolling circular buffer: keeps the last `_pre_frames` frames
        self._buffer: deque = deque(maxlen=self._pre_frames)

        # Active recordings: {track_id: _PendingClip}
        self._pending: Dict[int, '_PendingClip'] = {}
        self._lock = threading.Lock()

        os.makedirs(self.evidence_dir, exist_ok=True)

    def buffer_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """
        Called on EVERY captured raw frame (or processed frame). Stores the frame in
        the rolling buffer and feeds it to any active pending clip recordings.
        Thread-safe for background camera ingestion threads.
        """
        frame_copy = frame.copy()
        completed = []

        with self._lock:
            self._buffer.append((frame_number, frame_copy))

            # Feed post-trigger frames to any active recordings
            for track_id, clip in self._pending.items():
                clip.post_buffer.append(frame_copy)
                if len(clip.post_buffer) >= self._post_frames:
                    completed.append(track_id)

            for track_id in completed:
                clip = self._pending.pop(track_id)
                self._encode_clip_async(clip)

    def trigger_clip(self, event: ViolationEvent) -> None:
        """
        Called when a violation is confirmed. Snapshots the rolling pre-trigger
        buffer and begins collecting post-trigger frames.
        """
        with self._lock:
            if event.track_id in self._pending:
                return  # Already recording for this track

            clip = _PendingClip(
                event=event,
                pre_buffer=list(self._buffer),  # Snapshot the current rolling buffer
                post_buffer=[],
                post_target=self._post_frames,
            )
            self._pending[event.track_id] = clip

    def flush(self) -> None:
        """
        Finalize any in-progress clips (e.g., video ended before post-trigger
        window was fully collected). Encodes whatever frames were captured.
        """
        with self._lock:
            clips = list(self._pending.values())
            self._pending.clear()

        for clip in clips:
            self._encode_clip_async(clip)

    def _encode_clip_async(self, clip: '_PendingClip') -> None:
        """Spawns a background thread to encode the clip to MP4."""
        thread = threading.Thread(
            target=self._encode_clip,
            args=(clip,),
            daemon=True,
        )
        thread.start()

    def _encode_clip(self, clip: '_PendingClip') -> None:
        """Encodes pre + post frames into an MP4 file."""
        event = clip.event
        filename = (
            f"violation_{event.camera_id}_id{event.track_id}"
            f"_f{event.frame_number}.mp4"
        )
        filepath = os.path.join(self.evidence_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))

        if not writer.isOpened():
            print(f"[EvidenceRecorder] ERROR: Could not open writer for {filepath}")
            return

        # Write pre-trigger frames
        for _, frame in clip.pre_buffer:
            writer.write(frame)

        # Write post-trigger frames
        for frame in clip.post_buffer:
            writer.write(frame)

        writer.release()

        total_frames = len(clip.pre_buffer) + len(clip.post_buffer)
        duration = total_frames / self.fps if self.fps > 0 else 0
        print(
            f"📹 [EvidenceRecorder] Clip saved: {filepath} "
            f"({total_frames} frames, {duration:.1f}s)"
        )

        # Update event with the clip path
        event.evidence_clip_path = filepath

        # Notify via callback
        if self.on_clip_ready:
            try:
                self.on_clip_ready(event, filepath)
            except Exception as e:
                print(f"[EvidenceRecorder] Callback error: {e}")


class _PendingClip:
    """Internal state for a clip that is still collecting post-trigger frames."""

    __slots__ = ('event', 'pre_buffer', 'post_buffer', 'post_target')

    def __init__(
        self,
        event: ViolationEvent,
        pre_buffer: List[Tuple[int, np.ndarray]],
        post_buffer: List[np.ndarray],
        post_target: int,
    ):
        self.event = event
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.post_target = post_target
