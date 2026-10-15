from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseFrameReader(ABC):
    """Abstract base class for all video and camera ingestion sources."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads and returns the next frame.
        Returns:
            (ret, frame): ret is True if a valid frame was read, False on EOF or fatal error.
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        """Frame rate of the stream."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""
        pass

    @property
    def total_frames(self) -> int:
        """Total frame count if known (e.g. video files), 0 for live streams."""
        return 0

    @abstractmethod
    def release(self) -> None:
        """Release any underlying hardware, threads, or OpenCV captures."""
        pass

    def register_raw_consumer(self, consumer) -> None:
        """
        Optional hook: Register a callback to receive every raw frame captured
        at native camera frame rate (e.g., for rolling evidence video clips).
        """
        pass

