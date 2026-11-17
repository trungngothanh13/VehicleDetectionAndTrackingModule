import os
from trackers.base_tracker import UltralyticsMotTracker


class ByteTrackTracker(UltralyticsMotTracker):
    """
    Vehicle Detection and Tracking using YOLO + isolated Ultralytics ByteTrack state.
    Reads tracker parameters from `bytetrack.yaml`.
    """

    def __init__(
        self,
        model_name: str = 'yolo26l.pt',
        target_classes: dict = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 1280,
        tracker_yaml: str = None,
        shared_model=None,
        model_lock=None,
        **kwargs,
    ):
        local_yaml = os.path.join(os.path.dirname(__file__), 'bytetrack.yaml')
        default_yaml = local_yaml if os.path.exists(local_yaml) else 'bytetrack.yaml'
        super().__init__(
            tracker_type='bytetrack',
            default_yaml_path=default_yaml,
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            tracker_yaml=tracker_yaml,
            shared_model=shared_model,
            model_lock=model_lock,
            **kwargs,
        )
