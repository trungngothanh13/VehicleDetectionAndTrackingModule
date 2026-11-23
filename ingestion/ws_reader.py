import cv2
import threading
from collections import deque
from typing import Tuple, Optional, Callable
import numpy as np

from ingestion.base_reader import BaseFrameReader


class WebSocketFrameReader(BaseFrameReader):
    """
    Ingests live camera frames streamed from the web frontend (either a browser-fed
    video file acting as a standalone camera stream or the user's laptop webcam via WebRTC/Canvas)
    relayed through api.VDAT over WebSocket.
    Uses a small 3-frame ring buffer so WAN TCP jitter bursts are processed smoothly
    without overwriting back-to-back frames, while strictly preventing latency buildup.
    """

    def __init__(
        self,
        width: int = 854,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 3,
        on_drop_frame: Optional[Callable[[int], None]] = None,
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._stopped = False
        self._frames: deque = deque(maxlen=max(1, buffer_size))
        self._frame_count = 0
        self._last_frame_id: Optional[int] = None
        self._on_drop_frame = on_drop_frame
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._raw_consumer: Optional[Callable[[np.ndarray, int], None]] = None

    @property
    def last_source_frame_id(self) -> Optional[int]:
        return self._last_frame_id

    def register_raw_consumer(self, consumer: Optional[Callable[[np.ndarray, int], None]]) -> None:
        self._raw_consumer = consumer

    def push_frame(self, frame: np.ndarray, source_frame_id: Optional[int] = None) -> None:
        """Called by WebSocketSink when an 'ingest_frame' packet arrives from api.VDAT."""
        if self._stopped or frame is None:
            return

        h, w = frame.shape[:2]
        if (w, h) != (self._width, self._height):
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)

        self._frame_count += 1
        current_count = self._frame_count

        if self._raw_consumer is not None:
            try:
                self._raw_consumer(frame, current_count)
            except Exception as e:
                print(f"[WebSocketFrameReader] Raw consumer error: {e}")

        with self._lock:
            dropped = len(self._frames) == self._frames.maxlen
            self._frames.append((frame, source_frame_id))
            self._new_frame_event.set()

        if dropped and callable(self._on_drop_frame):
            try:
                self._on_drop_frame(1)
            except Exception:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        while not self._stopped:
            with self._lock:
                if self._frames:
                    frame, source_frame_id = self._frames.popleft()
                    self._last_frame_id = source_frame_id
                    if not self._frames:
                        self._new_frame_event.clear()
                    return True, frame
            self._new_frame_event.wait(timeout=1.0)
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

