import os
import cv2
import time
import threading
from typing import Tuple, Optional, Callable
import numpy as np

from ingestion.base_reader import BaseFrameReader


class RTSPStreamReader(BaseFrameReader):
    """
    Ingests live RTSP/IP camera streams with automatic reconnection logic,
    zero-buffer FFmpeg configuration, and real-time latest-frame synchronization.
    Supports a raw consumer callback to capture all native camera frames
    (e.g., for rolling evidence video clips) without slowing down inference.
    """

    def __init__(self, rtsp_url: str, reconnect_interval: int = 5):
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval
        self._stopped = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_idx: int = 0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._raw_consumer: Optional[Callable[[np.ndarray, int], None]] = None
        self._frame_count = 0

        # Enforce zero-latency buffer in OpenCV FFmpeg backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"

        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or 25
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def register_raw_consumer(self, consumer: Optional[Callable[[np.ndarray, int], None]]) -> None:
        """Register a callback that receives every captured raw frame at native camera FPS."""
        self._raw_consumer = consumer

    def _capture_loop(self):
        while not self._stopped:
            if not self._cap.isOpened():
                print(f"[RTSP] Stream disconnected. Reconnecting in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
                self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                print(f"[RTSP] Failed to read frame. Reconnecting...")
                self._cap.release()
                time.sleep(1)
                continue

            self._frame_count += 1
            current_count = self._frame_count

            # Forward to raw frame consumer (e.g. EvidenceRecorder rolling buffer)
            if self._raw_consumer is not None:
                try:
                    self._raw_consumer(frame, current_count)
                except Exception as e:
                    print(f"[RTSP] Error in raw consumer callback: {e}")

            # Update latest frame for real-time inference (pipeline)
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_idx = current_count
                self._new_frame_event.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stopped:
            return False, None

        # Wait up to 2 seconds for a fresh frame
        if self._new_frame_event.wait(timeout=2.0):
            self._new_frame_event.clear()
            with self._lock:
                frame = self._latest_frame
            return True, frame
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

    def release(self) -> None:
        self._stopped = True
        self._new_frame_event.set()
        if self._cap and self._cap.isOpened():
            self._cap.release()
