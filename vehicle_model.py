"""
Vehicle Model Adapter / Wrapper.
Provides backward compatibility with previous VehicleModel interface while delegating
to the modular tracker implementations in `trackers/`.
"""

import numpy as np
import supervision as sv
from trackers import create_tracker, normalize_tracker_type, BotSortTracker, ByteTrackTracker, DeepSortTracker
from trackers.base_tracker import BaseTracker


class VehicleModel(BaseTracker):
    """
    Unified detection + tracking wrapper.
    Delegates to BotSortTracker, ByteTrackTracker, or DeepSortTracker based on configuration.
    """

    def __init__(
        self,
        model_name: str = 'yolo26l.pt',
        target_classes: dict = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 1280,
        tracker_type: str = 'botsort',
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            **kwargs,
        )
        self.tracker_type = tracker_type
        norm_type = normalize_tracker_type(tracker_type)

        if norm_type in ('botsort', 'botsort.yaml'):
            tracker_yaml = kwargs.get('tracker_yaml', 'botsort.yaml')
            self._delegate = BotSortTracker(
                model_name=model_name,
                target_classes=target_classes,
                confidence_threshold=confidence_threshold,
                imgsz=imgsz,
                tracker_yaml=tracker_yaml,
                **kwargs,
            )
        elif norm_type in ('bytetrack', 'bytetrack.yaml'):
            tracker_yaml = kwargs.get('tracker_yaml', 'bytetrack.yaml')
            self._delegate = ByteTrackTracker(
                model_name=model_name,
                target_classes=target_classes,
                confidence_threshold=confidence_threshold,
                imgsz=imgsz,
                tracker_yaml=tracker_yaml,
                **kwargs,
            )
        elif norm_type == 'deepsort':
            self._delegate = DeepSortTracker(
                model_name=model_name,
                target_classes=target_classes,
                confidence_threshold=confidence_threshold,
                imgsz=imgsz,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tracker_type '{tracker_type}'")

    def track(self, frame: np.ndarray) -> sv.Detections:
        """Run detection and tracking on a single video frame."""
        detections = self._delegate.track(frame)
        self._seen_ids = self._delegate._seen_ids
        return detections

    def get_class_name(self, class_id: int) -> str:
        return self._delegate.get_class_name(class_id)

    def get_total_tracked(self) -> int:
        return self._delegate.get_total_tracked()

    def reset(self) -> None:
        self._delegate.reset()
        self._seen_ids = self._delegate._seen_ids


__all__ = ['VehicleModel', 'create_tracker']
