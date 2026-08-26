from abc import ABC, abstractmethod
import numpy as np
import supervision as sv


class BaseTracker(ABC):
    """
    Abstract Base Tracker class for vehicle detection and tracking.
    All tracker implementations must inherit from this class and implement the `track` method.
    """

    def __init__(
        self,
        model_name: str = 'yolo26l.pt',
        target_classes: dict = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 1280,
        **kwargs,
    ):
        self.model_name = model_name
        self.target_classes = target_classes or {}
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._seen_ids = set()

    @abstractmethod
    def track(self, frame: np.ndarray) -> sv.Detections:
        """
        Process a single video frame. Detect and track vehicles.
        
        Args:
            frame: np.ndarray (BGR image)
            
        Returns:
            sv.Detections: Object with xyxy, confidence, class_id, and tracker_id.
        """
        pass

    def get_class_name(self, class_id: int) -> str:
        """Return the human-readable class name for a given class ID."""
        return self.target_classes.get(int(class_id), 'unknown')

    def get_total_tracked(self) -> int:
        """Total unique vehicles tracked across all frames."""
        return len(self._seen_ids)

    def reset(self) -> None:
        """Reset internal tracker state and seen tracking IDs."""
        self._seen_ids.clear()
