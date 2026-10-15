import cv2
import threading
from queue import Queue
from typing import Tuple, Optional, Callable
import numpy as np

from ingestion.base_reader import BaseFrameReader


class VideoFileReader(BaseFrameReader):
    """
    Reads video frames from a local video file on a background thread
    so disk/CPU decode does not block GPU inference.
    """

    def __init__(self, file_path: str, buffer_size: int = 8):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {file_path}")

        self._fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.queue = Queue(maxsize=buffer_size)
        self._stopped = False
        self._raw_consumer: Optional[Callable[[np.ndarray, int], None]] = None
        self._frame_count = 0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def register_raw_consumer(self, consumer: Optional[Callable[[np.ndarray, int], None]]) -> None:
        """Register a callback that receives every read raw frame."""
        self._raw_consumer = consumer

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self.queue.put((ret, frame))
            if not ret:
                break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stopped and self.queue.empty():
            return False, None
        ret, frame = self.queue.get()
        if ret and frame is not None:
            self._frame_count += 1
            if self._raw_consumer is not None:
                try:
                    self._raw_consumer(frame, self._frame_count)
                except Exception as e:
                    print(f"[FileReader] Error in raw consumer callback: {e}")
        return ret, frame

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
        self._stopped = True
        if self.cap.isOpened():
            self.cap.release()
