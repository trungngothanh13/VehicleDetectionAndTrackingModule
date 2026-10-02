import cv2
import threading
from queue import Queue
from typing import Tuple, Optional
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
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self.queue.put((ret, frame))
            if not ret:
                break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stopped and self.queue.empty():
            return False, None
        return self.queue.get()

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
