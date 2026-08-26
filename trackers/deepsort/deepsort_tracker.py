import os
import yaml
import numpy as np
import supervision as sv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from trackers.base_tracker import BaseTracker


class DeepSortTracker(BaseTracker):
    """
    Vehicle Detection and Tracking using Ultralytics YOLO for object detection
    and DeepSORT (deep-sort-realtime) for tracking.
    Reads configuration from `deepsort.yaml`.
    """

    def __init__(
        self,
        model_name: str = 'yolov8l.pt',
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
        # Load configuration from YAML file
        yaml_config = {}
        if tracker_yaml is None:
            tracker_yaml = os.path.join(os.path.dirname(__file__), 'deepsort.yaml')
        if os.path.exists(tracker_yaml):
            try:
                with open(tracker_yaml, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[DeepSORT] Warning: Could not read {tracker_yaml}: {e}")

        # Merge YAML defaults with explicitly passed kwargs
        use_reid = kwargs.get('use_reid', yaml_config.get('use_reid', False))
        max_age = kwargs.get('max_age', yaml_config.get('max_age', 30))
        n_init = kwargs.get('n_init', yaml_config.get('n_init', 3))
        max_cosine_distance = kwargs.get(
            'max_cosine_distance',
            yaml_config.get('max_cosine_distance', 0.2 if use_reid else 0.4)
        )
        nn_budget = kwargs.get(
            'nn_budget',
            yaml_config.get('nn_budget', 100 if use_reid else None)
        )
        embedder = kwargs.get('embedder', yaml_config.get('embedder', 'mobilenet')) if use_reid else None
        half = kwargs.get('half', yaml_config.get('half', True))
        embedder_gpu = kwargs.get('embedder_gpu', yaml_config.get('embedder_gpu', True))

        self.model = YOLO(self.model_name)
        self.use_reid = use_reid
        self.half = half

        # Initialize DeepSORT tracker
        tracker_kwargs = {
            'max_age': max_age,
            'n_init': n_init,
            'max_cosine_distance': max_cosine_distance,
            'nn_budget': nn_budget,
        }

        if embedder is not None:
            tracker_kwargs.update({
                'embedder': embedder,
                'half': half,
                'bgr': True,
                'embedder_gpu': embedder_gpu,
            })
        else:
            tracker_kwargs['embedder'] = None

        self.tracker = DeepSort(**tracker_kwargs)

        # Log mode
        if embedder is not None:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    embedder_device_str = f"CUDA (GPU 0: {gpu_name})" if embedder_gpu else "CPU (embedder_gpu=False)"
                else:
                    embedder_device_str = "CPU (CUDA not available)"
                print(f"[DeepSORT] Mode: ReID Appearance Matching | Embedder: {embedder} on {embedder_device_str}")
            except Exception as e:
                print(f"[DeepSORT] Mode: ReID Appearance Matching | Embedder: {embedder} ({e})")
        else:
            print("[DeepSORT] Mode: Fast IoU-only Tracking (use_reid=False in deepsort.yaml)")

    def track(self, frame: np.ndarray) -> sv.Detections:
        """
        Run YOLO detection, associate tracks via DeepSORT (IoU or ReID),
        and update Kalman filter state.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]

        raw_detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            cls_np = boxes.cls.cpu().numpy().astype(int)
            conf_np = boxes.conf.cpu().numpy().astype(np.float32)
            xyxy_np = boxes.xyxy.cpu().numpy().astype(np.float32)

            for i in range(len(cls_np)):
                c_id = cls_np[i]
                if c_id not in self.target_classes:
                    continue
                x1, y1, x2, y2 = xyxy_np[i]
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                if w < 5 or h < 5:
                    continue
                conf = float(conf_np[i])
                raw_detections.append(([float(x1), float(y1), w, h], conf, int(c_id)))

        # Update DeepSORT tracker
        if self.tracker.embedder is None:
            embeds = [np.ones(128, dtype=np.float32) for _ in raw_detections]
            tracks = self.tracker.update_tracks(raw_detections, embeds=embeds)
        else:
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        xyxy_list = []
        conf_list = []
        class_id_list = []
        tracker_id_list = []

        for track in tracks:
            # Only output tracks that are confirmed AND were matched/updated on the current frame
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            det_cls = track.get_det_class()
            if det_cls is None:
                continue

            # orig=True returns the actual YOLO measurement bounding box for this frame
            ltrb = track.to_ltrb(orig=True)
            if ltrb is None:
                continue

            track_id = int(track.track_id)
            det_conf = track.get_det_conf()
            conf = float(det_conf) if det_conf is not None else 1.0

            xyxy_list.append(ltrb)
            conf_list.append(conf)
            class_id_list.append(int(det_cls))
            tracker_id_list.append(track_id)
            self._seen_ids.add(track_id)

        if len(xyxy_list) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array(conf_list, dtype=np.float32),
            class_id=np.array(class_id_list, dtype=int),
            tracker_id=np.array(tracker_id_list, dtype=int),
        )

    def reset(self) -> None:
        super().reset()
        self.tracker.delete_all_tracks()
