from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from core.events import FrameTelemetry, ViolationEvent


class BaseSink(ABC):
    """Abstract base class for all output consumers (file saving, WebSocket, preview, etc.)."""

    def on_start(self, pipeline_info: Dict[str, Any]) -> None:
        """Called once when pipeline begins processing."""
        pass

    @abstractmethod
    def on_frame(self, frame: np.ndarray, telemetry: FrameTelemetry) -> None:
        """Called on every processed frame with rendered video frame and telemetry."""
        pass

    def on_violation(self, event: ViolationEvent, snapshot: np.ndarray) -> None:
        """Called whenever a new red-light violation is confirmed."""
        pass

    def on_complete(self) -> None:
        """Called when video processing has finished for cleanup and resource release."""
        pass
