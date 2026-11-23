import os
from typing import Any, Dict, Tuple


def resolve_profile_model(config: Any, profile: str = None) -> Tuple[str, str, float]:
    """
    Resolves (profile_key, weights_path, default_conf) for 'trained' or 'stock' (legacy).
    Automatically falls back to stock yolo26l.pt if trained weights are not present locally.
    """
    target_profile = (profile or config.MODEL_PROFILE or 'trained').strip().lower()
    if target_profile in ('stock', 'legacy', 'coco', 'default'):
        return ('stock', config.LEGACY_MODEL_NAME, config.LEGACY_CONFIDENCE_THRESHOLD)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        config.TRAINED_MODEL_NAME,
        os.path.join(project_root, config.TRAINED_MODEL_NAME),
        os.path.join(project_root, '..', 'training.VDAT', 'weights', 'best.pt'),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return ('trained', os.path.normpath(cand), config.TRAINED_CONFIDENCE_THRESHOLD)

    print(
        f"⚠️  [AI Core] Trained weights '{config.TRAINED_MODEL_NAME}' not found; "
        f"falling back to legacy stock '{config.LEGACY_MODEL_NAME}'."
    )
    return ('stock', config.LEGACY_MODEL_NAME, config.LEGACY_CONFIDENCE_THRESHOLD)


def apply_model_classes(config: Any, model_names: Any) -> Dict[int, str]:
    """
    Maps the loaded YOLO model's class IDs to canonical vehicle names so both
    80-class COCO weights (yolo26l.pt) and custom-trained weights (e.g. 5-class
    Roboflow best.pt: {0: bicycle, 1: bus, 2: car, 3: motorcycle, 4: truck})
    work seamlessly without hardcoded COCO IDs.
    """
    if not model_names:
        return config.DETECTION_CLASSES

    if isinstance(model_names, (list, tuple)):
        items = [(i, str(n)) for i, n in enumerate(model_names)]
    elif isinstance(model_names, dict):
        items = [(int(k), str(v)) for k, v in model_names.items()]
    else:
        return config.DETECTION_CLASSES

    aliases = getattr(config, 'VEHICLE_CLASS_ALIASES', {})
    resolved: Dict[int, str] = {}
    for cid, raw_name in items:
        key = raw_name.strip().lower().replace('-', ' ').replace('_', ' ')
        canonical = aliases.get(key)
        if canonical:
            resolved[cid] = canonical

    if resolved:
        config.DETECTION_CLASSES = resolved
        config.VIOLATION_CLASS_IDS = set(resolved.keys())
    return config.DETECTION_CLASSES
