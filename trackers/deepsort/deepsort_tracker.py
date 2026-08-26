import numpy as np
import supervision as sv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from trackers.base_tracker import BaseTracker


class DeepSortTracker(BaseTracker):
    """
    Vehicle Detection and Tracking using Ultralytics YOLO for object detection
    and DeepSORT (deep-sort-realtime) for appearance-based or IoU tracking.
    """

    def __init__(
        self,
        model_name: str = 'yolov8l.pt',
        target_classes: dict = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 1280,
        max_age: int = 60,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        embedder: str = 'mobilenet',
        half: bool = True,
        embedder_gpu: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            **kwargs,
        )
        self.model = YOLO(self.model_name)
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

        # Log device / tracking mode
        if embedder is not None:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    embedder_device_str = f"CUDA (GPU 0: {gpu_name})" if embedder_gpu else "CPU (embedder_gpu=False)"
                else:
                    embedder_device_str = "CPU (CUDA not available)"
                print(f"[DeepSORT] PyTorch CUDA: {cuda_available} | ReID Embedder: {embedder_device_str}")
            except Exception as e:
                print(f"[DeepSORT] Embedder initialized: {e}")
        else:
            print("[DeepSORT] Running in IoU-only mode (ReID embedder disabled)")

    def track(self, frame: np.ndarray) -> sv.Detections:
        """
        Run YOLO detection, extract appearance embeddings via DeepSORT ReID,
        and update Kalman filter tracks.
        """
        # Run YOLO detection with FP16 inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            quantize='fp16',
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
                # deep-sort-realtime expects: ([left, top, w, h], confidence, detection_class)
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
            if not track.is_confirmed():
                continue
            det_cls = track.get_det_class()
            if det_cls is None:
                continue

            ltrb = track.to_ltrb()
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
        # Reset internal DeepSort tracker if needed
        self.tracker.delete_all_tracks()
