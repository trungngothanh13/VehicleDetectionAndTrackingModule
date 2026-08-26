import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from trackers.base_tracker import BaseTracker


class ByteTrackTracker(BaseTracker):
    """
    Vehicle Detection and Tracking using Ultralytics YOLO with built-in ByteTrack.
    Reads tracker parameters from `bytetrack.yaml`.
    """

    def __init__(
        self,
        model_name: str = 'yolo26l.pt',
        target_classes: dict = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 1280,
        tracker_yaml: str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            **kwargs,
        )
        if tracker_yaml is None:
            local_yaml = os.path.join(os.path.dirname(__file__), 'bytetrack.yaml')
            tracker_yaml = local_yaml if os.path.exists(local_yaml) else 'bytetrack.yaml'
        self.tracker_yaml = tracker_yaml
        self.model = YOLO(self.model_name)

    def track(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect + track in one pass using Ultralytics ByteTrack tracker.
        persist=True maintains tracker state between frames.
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker_yaml,
            quantize='fp16',
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return sv.Detections.empty()

        boxes = result.boxes
        cls_np = boxes.cls.cpu().numpy().astype(int)

        # Filter to configured vehicle classes only
        mask = np.isin(cls_np, list(self.target_classes.keys()))
        if not mask.any():
            return sv.Detections.empty()

        xyxy = boxes.xyxy.cpu().numpy()[mask].astype(np.float32)
        confidences = boxes.conf.cpu().numpy()[mask].astype(np.float32)
        class_ids = cls_np[mask]
        tracker_ids = boxes.id.cpu().numpy()[mask].astype(int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
            tracker_id=tracker_ids,
        )

        for tid in tracker_ids:
            self._seen_ids.add(tid)

        return sv_detections
