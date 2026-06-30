from ultralytics import YOLO
import supervision as sv
import numpy as np


class VehicleModel:
    """
    Unified detection + tracking model.
    Uses Ultralytics YOLO26 with built-in BoT-SORT tracker (no extra dependencies).
    """

    def __init__(self, model_name, target_classes, confidence_threshold, imgsz=640):
        self.model = YOLO(model_name)
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._seen_ids = set()

    def track(self, frame):
        """
        Detect + track in one GPU pass using Ultralytics built-in BoT-SORT.
        persist=True keeps tracker state alive between frames.
        Returns: sv.Detections with tracker_id populated, filtered to target_classes.
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            persist=True,
            tracker='botsort.yaml',
            quantize='fp16',  # FP16 inference — faster on L40/A100 tensor cores
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

        xyxy        = boxes.xyxy.cpu().numpy()[mask].astype(np.float32)
        confidences = boxes.conf.cpu().numpy()[mask].astype(np.float32)
        class_ids   = cls_np[mask]
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

    def get_class_name(self, class_id):
        """Return the human-readable class name for a given class ID."""
        return self.target_classes.get(class_id, 'unknown')

    def get_total_tracked(self):
        """Total unique vehicles tracked across all frames."""
        return len(self._seen_ids)
