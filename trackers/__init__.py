import os
from typing import Any
from trackers.base_tracker import BaseTracker
from trackers.botsort.botsort_tracker import BotSortTracker
from trackers.bytetrack.bytetrack_tracker import ByteTrackTracker
from trackers.deepsort.deepsort_tracker import DeepSortTracker

TRACKER_REGISTRY = {
    'botsort': BotSortTracker,
    'botsort.yaml': BotSortTracker,
    'bytetrack': ByteTrackTracker,
    'bytetrack.yaml': ByteTrackTracker,
    'deepsort': DeepSortTracker,
}


def normalize_tracker_type(tracker_type: str) -> str:
    """Normalize tracker type string to lower-case without whitespace."""
    if not tracker_type:
        return 'botsort'
    normalized = tracker_type.strip().lower()
    return normalized


def create_tracker(config: Any) -> BaseTracker:
    """
    Factory function to instantiate a tracker based on configuration.
    
    Supports:
    - config.TRACKER_TYPE: 'botsort' (or 'botsort.yaml'), 'bytetrack' (or 'bytetrack.yaml'), 'deepsort'
    """
    tracker_type = normalize_tracker_type(getattr(config, 'TRACKER_TYPE', 'botsort'))
    
    # Common detector params
    model_name = getattr(config, 'MODEL_NAME', 'yolo26l.pt')
    target_classes = getattr(config, 'DETECTION_CLASSES', {})
    confidence_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', 0.30)
    imgsz = getattr(config, 'IMGSZ', 1280)

    if tracker_type in ('botsort', 'botsort.yaml'):
        botsort_cfg = getattr(config, 'BOTSORT_CONFIG', {})
        tracker_yaml = botsort_cfg.get('tracker_yaml', 'botsort.yaml')
        return BotSortTracker(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            tracker_yaml=tracker_yaml,
        )

    elif tracker_type in ('bytetrack', 'bytetrack.yaml'):
        bytetrack_cfg = getattr(config, 'BYTETRACK_CONFIG', {})
        tracker_yaml = bytetrack_cfg.get('tracker_yaml', 'bytetrack.yaml')
        return ByteTrackTracker(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            tracker_yaml=tracker_yaml,
        )

    elif tracker_type == 'deepsort':
        deepsort_cfg = getattr(config, 'DEEPSORT_CONFIG', {})
        tracker_device = getattr(config, 'TRACKER_DEVICE', 'cuda')
        embedder_gpu = deepsort_cfg.get('embedder_gpu', tracker_device == 'cuda')
        return DeepSortTracker(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            max_age=deepsort_cfg.get('max_age', getattr(config, 'MAX_AGE', 60)),
            n_init=deepsort_cfg.get('n_init', getattr(config, 'N_INIT', 3)),
            max_cosine_distance=deepsort_cfg.get('max_cosine_distance', getattr(config, 'MAX_COSINE_DISTANCE', 0.2)),
            nn_budget=deepsort_cfg.get('nn_budget', getattr(config, 'NN_BUDGET', 100)),
            embedder=deepsort_cfg.get('embedder', getattr(config, 'EMBEDDER', 'mobilenet')),
            half=deepsort_cfg.get('half', getattr(config, 'HALF', True)),
            embedder_gpu=embedder_gpu,
        )

    else:
        valid_options = list(TRACKER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported TRACKER_TYPE: '{tracker_type}'. Supported options are: {valid_options}"
        )


__all__ = [
    'BaseTracker',
    'BotSortTracker',
    'ByteTrackTracker',
    'DeepSortTracker',
    'create_tracker',
    'TRACKER_REGISTRY',
]
