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
    - config.TRACKER_TYPE: 'botsort', 'bytetrack', 'deepsort'
    Each tracker loads its detailed parameters from its respective YAML file:
    - trackers/botsort/botsort.yaml
    - trackers/bytetrack/bytetrack.yaml
    - trackers/deepsort/deepsort.yaml
    """
    tracker_type = normalize_tracker_type(getattr(config, 'TRACKER_TYPE', 'botsort'))
    
    # Common detector params
    model_name = getattr(config, 'MODEL_NAME', 'yolo26l.pt')
    target_classes = getattr(config, 'DETECTION_CLASSES', {})
    confidence_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', 0.30)
    imgsz = getattr(config, 'IMGSZ', 1280)

    if tracker_type in ('botsort', 'botsort.yaml'):
        botsort_cfg = getattr(config, 'BOTSORT_CONFIG', {})
        tracker_yaml = botsort_cfg.get('tracker_yaml', None)
        return BotSortTracker(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            tracker_yaml=tracker_yaml,
        )

    elif tracker_type in ('bytetrack', 'bytetrack.yaml'):
        bytetrack_cfg = getattr(config, 'BYTETRACK_CONFIG', {})
        tracker_yaml = bytetrack_cfg.get('tracker_yaml', None)
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
        return DeepSortTracker(
            model_name=model_name,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            tracker_yaml=deepsort_cfg.get('tracker_yaml', None),
            embedder_gpu=(tracker_device == 'cuda'),
            **{k: v for k, v in deepsort_cfg.items() if k != 'tracker_yaml'},
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
