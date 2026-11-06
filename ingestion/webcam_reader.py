import cv2
import time
import threading
from typing import Tuple, Optional, Callable, Union
import numpy as np

from ingestion.base_reader import BaseFrameReader


class WebcamReader(BaseFrameReader):
    """
    Ingests live video directly from a local hardware webcam (laptop integrated camera
    or USB camera, e.g. index 0 or /dev/video0).
    Uses a non-blocking background capture thread so inference always processes the
    freshest camera frame while EvidenceRecorder receives continuous frames.
    """

    def __init__(
        self,
        device: Union[int, str] = 0,
        target_width: int = 854,
        target_height: int = 480,
        target_fps: int = 30,
    ):
        if isinstance(device, str) and device.isdigit():
            self.device: Union[int, str] = int(device)
        else:
            self.device = device

        self._stopped = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_idx: int = 0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._raw_consumer: Optional[Callable[[np.ndarray, int], None]] = None
        self._frame_count = 0

        # Prefer V4L2 on Linux when opening by integer index to avoid GStreamer lag
        if isinstance(self.device, int) and hasattr(cv2, "CAP_V4L2"):
            self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.device)
        else:
            self._cap = cv2.VideoCapture(self.device)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open laptop/hardware camera: {self.device}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or target_fps
        self._width = target_width
        self._height = target_height

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def register_raw_consumer(self, consumer: Optional[Callable[[np.ndarray, int], None]]) -> None:
        self._raw_consumer = consumer

    def _capture_loop(self) -> None:
        while not self._stopped:
            if not self._cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            # Normalize resolution to match configured zone coordinates (854x480)
            h, w = frame.shape[:2]
            if (w, h) != (self._width, self._height):
                frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)

            self._frame_count += 1
            current_count = self._frame_count

            if self._raw_consumer is not None:
                try:
                    self._raw_consumer(frame, current_count)
                except Exception as e:
                    print(f"[WebcamReader] Error in raw consumer callback: {e}")

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_idx = current_count
                self._new_frame_event.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stopped:
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

    def release(self) -> None:
        self._stopped = True
        self._new_frame_event.set()
        if self._cap and self._cap.isOpened():
            self._cap.release()

