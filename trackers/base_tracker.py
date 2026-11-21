from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future
import queue
import threading
import time
import numpy as np
import supervision as sv


class BatchModelRunner:
    """
    Coordinates micro-batched GPU inference across multiple concurrent CameraSession
    threads sharing a single loaded Ultralytics YOLO model instance.
    Collects up to `max_batch_size` same-shape frames arriving within `max_wait_sec`
    and executes one batched `model.predict(..., half=...)` pass on GPU.
    """

    def __init__(
        self,
        model,
        max_batch_size: int = 8,
        max_wait_sec: float = 0.004,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_sec
        try:
            import torch
            self.use_half = bool(torch.cuda.is_available())
        except Exception:
            self.use_half = False

        self._queue: queue.Queue = queue.Queue()
        self._leftovers: deque = deque()
        self._stopped = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="YOLO-MicroBatcher",
            daemon=True,
        )
        self._worker.start()

    def predict_one(self, frame: np.ndarray, conf: float, imgsz: int):
        """Enqueues a frame for micro-batched inference and returns its Result."""
        fut: Future = Future()
        self._queue.put((frame, conf, imgsz, fut))
        return fut.result()

    def stop(self) -> None:
        self._stopped = True
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _next_item(self):
        if self._leftovers:
            return self._leftovers.popleft()
        return self._queue.get()

    def _worker_loop(self) -> None:
        while not self._stopped:
            first = self._next_item()
            if first is None:
                break

            batch = [first]
            first_frame, first_conf, first_imgsz, _ = first
            target_shape = first_frame.shape[:2]

            deadline = time.monotonic() + self.max_wait_sec
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break

                if item is None:
                    self._stopped = True
                    break

                f, c, sz, _ = item
                if f.shape[:2] == target_shape and c == first_conf and sz == first_imgsz:
                    batch.append(item)
                else:
                    self._leftovers.append(item)
                    break

            frames = [item[0] for item in batch]
            futures = [item[3] for item in batch]
            try:
                results = self.model.predict(
                    frames,
                    conf=first_conf,
                    imgsz=first_imgsz,
                    half=self.use_half,
                    verbose=False,
                )
                for fut, res in zip(futures, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)


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
        shared_model=None,
        model_lock: threading.Lock = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.target_classes = target_classes or {}
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._shared_model = shared_model
        self._model_lock = model_lock
        self._seen_ids = set()

    def _run_model_predict(self, frame: np.ndarray):
        """Executes YOLO detection via BatchModelRunner or thread-locked YOLO model."""
        if isinstance(self._shared_model, BatchModelRunner):
            return [
                self._shared_model.predict_one(
                    frame,
                    conf=self.confidence_threshold,
                    imgsz=self.imgsz,
                )
            ]
        if self._model_lock is not None:
            with self._model_lock:
                return self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    imgsz=self.imgsz,
                    verbose=False,
                )
        return self.model.predict(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

    @abstractmethod
    def track(self, frame: np.ndarray) -> sv.Detections:
        """
        Process a single video frame. Detect and track vehicles.
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


class UltralyticsMotTracker(BaseTracker):
    """
    Shared base for Ultralytics MOT trackers (ByteTrack, BoT-SORT) that decouples
    micro-batched YOLO object detection from per-camera stateful object tracking.
    """

    def __init__(
        self,
        tracker_type: str,
        default_yaml_path: str,
        model_name: str = 'yolo26l.pt',
        target_classes: dict = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 1280,
        tracker_yaml: str = None,
        shared_model=None,
        model_lock: threading.Lock = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            shared_model=shared_model,
            model_lock=model_lock,
            **kwargs,
        )
        from ultralytics import YOLO
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.trackers.track import TRACKER_MAP

        self.tracker_type = tracker_type
        self.tracker_yaml = tracker_yaml if tracker_yaml else default_yaml_path
        if isinstance(shared_model, BatchModelRunner):
            self.model = shared_model.model
        else:
            self.model = shared_model if shared_model is not None else YOLO(self.model_name)

        cfg = IterableSimpleNamespace(**YAML.load(self.tracker_yaml))
        self._tracker_cfg = cfg
        self._tracker_cls = TRACKER_MAP[self.tracker_type]
        self._tracker = self._tracker_cls(args=self._tracker_cfg)

    def track(self, frame: np.ndarray) -> sv.Detections:
        """Runs shared/batched YOLO detection then updates this camera session's isolated tracker."""
        results = self._run_model_predict(frame)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()

        det = result.boxes.cpu().numpy()
        tracks = self._tracker.update(det, frame)
        if len(tracks) == 0:
            return sv.Detections.empty()

        # Ultralytics tracker.update returns columns: [x1, y1, x2, y2, track_id, conf, cls, orig_idx]
        cls_np = tracks[:, 6].astype(int)
        mask = np.isin(cls_np, list(self.target_classes.keys()))
        if not mask.any():
            return sv.Detections.empty()

        filtered = tracks[mask]
        xyxy = filtered[:, 0:4].astype(np.float32)
        tracker_ids = filtered[:, 4].astype(int)
        confidences = filtered[:, 5].astype(np.float32)
        class_ids = filtered[:, 6].astype(int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
            tracker_id=tracker_ids,
        )

        for tid in tracker_ids:
            self._seen_ids.add(int(tid))

        return sv_detections

    def reset(self) -> None:
        super().reset()
        self._tracker = self._tracker_cls(args=self._tracker_cfg)
