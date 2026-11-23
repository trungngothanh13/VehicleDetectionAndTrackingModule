import cv2
import os
import time
import threading
from typing import Tuple, Optional, Callable
import numpy as np

from ingestion.base_reader import BaseFrameReader


class SimulatedCameraReader(BaseFrameReader):
    """
    Simulates a live IP/RTSP camera stream using a local video file.

    Frames are decoded at the video's native FPS on a strictly timed wall-clock.
    The inference pipeline receives frames via real-time sampling:
    if inference is slower than the video's FPS, intermediate frames are
    dropped just like on a live camera stream.

    Meanwhile, any registered raw frame consumer (e.g., EvidenceRecorder)
    receives every frame at the native frame rate.
    """

    def __init__(
        self,
        file_path: str,
        loop: bool = False,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        delete_on_release: bool = False,
        on_eof: Optional[Callable[[], None]] = None,
    ):
        self.file_path = file_path
        self.loop = bool(loop)
        self._target_width = int(target_width) if target_width else None
        self._target_height = int(target_height) if target_height else None
        self._delete_on_release = bool(delete_on_release)
        self._on_eof = on_eof
        self._released = False
        self._stopped = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_idx: int = 0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._raw_consumer: Optional[Callable[[np.ndarray, int], None]] = None
        self._frame_count = 0

        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.file_path}")

        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or 30
        native_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = self._target_width or native_w
        self._height = self._target_height or native_h
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._thread = threading.Thread(target=self._pacing_loop, daemon=True)
        self._thread.start()

    def set_loop(self, loop: bool) -> None:
        self.loop = bool(loop)

    def register_raw_consumer(self, consumer: Optional[Callable[[np.ndarray, int], None]]) -> None:
        """Register a callback that receives every raw frame at native camera FPS."""
        self._raw_consumer = consumer

    def _pacing_loop(self):
        """Streams frames at strictly timed native FPS intervals."""
        frame_duration = 1.0 / self._fps
        next_tick = time.perf_counter()

        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if callable(self._on_eof):
                    try:
                        self._on_eof()
                    except Exception as e:
                        print(f"[SimulatedCamera] on_eof callback error: {e}")
                break

            if (
                self._target_width
                and self._target_height
                and (frame.shape[1], frame.shape[0]) != (self._target_width, self._target_height)
            ):
                frame = cv2.resize(
                    frame,
                    (self._target_width, self._target_height),
                    interpolation=cv2.INTER_AREA,
                )

            self._frame_count += 1
            current_count = self._frame_count

            # Forward to raw frame consumer (e.g. EvidenceRecorder) at native FPS
            if self._raw_consumer is not None:
                try:
                    self._raw_consumer(frame, current_count)
                except Exception as e:
                    print(f"[SimulatedCamera] Error in raw consumer callback: {e}")

            # Update latest frame for pipeline inference
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_idx = current_count
                self._new_frame_event.set()

            # Pace to exact native FPS
            next_tick += frame_duration
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If falling behind, resync clock
                next_tick = time.perf_counter()

        self._stopped = True
        self._new_frame_event.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stopped and self._latest_frame is None:
            return False, None

        if self._new_frame_event.wait(timeout=2.0):
            self._new_frame_event.clear()
            with self._lock:
                frame = self._latest_frame
            return (frame is not None), frame

        return False, None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._stopped = True
        self._new_frame_event.set()
        if (
            hasattr(self, '_thread')
            and self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=1.5)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        if self._delete_on_release and self.file_path and os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                print(f"🗑️ [SimulatedCamera] Deleted temporary video: {os.path.basename(self.file_path)}")
            except OSError as e:
                print(f"[SimulatedCamera] Failed to delete temporary video '{self.file_path}': {e}")
