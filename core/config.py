import os


class Config:
    # =========================================================================
    # SYSTEM IDENTIFIERS
    # =========================================================================
    CAMERA_ID = os.getenv('CAMERA_ID', 'cam-01')

    # =========================================================================
    # TRACKER SELECTION
    # Available options: 'botsort', 'bytetrack', 'deepsort'
    # =========================================================================
    TRACKER_TYPE = os.getenv('TRACKER_TYPE', 'bytetrack')

    # =========================================================================
    # GLOBAL MODEL & DETECTION SETTINGS
    # =========================================================================
    TRAINED_MODEL_NAME = os.getenv('TRAINED_MODEL_NAME', 'weights/best.pt')
    LEGACY_MODEL_NAME = os.getenv('LEGACY_MODEL_NAME', 'yolo26l.pt')
    MODEL_PROFILE = os.getenv('MODEL_PROFILE', 'trained').strip().lower()  # 'trained' | 'stock'
    TRAINED_CONFIDENCE_THRESHOLD = float(os.getenv('TRAINED_CONFIDENCE_THRESHOLD', '0.45'))
    LEGACY_CONFIDENCE_THRESHOLD = float(os.getenv('LEGACY_CONFIDENCE_THRESHOLD', '0.30'))
    MODEL_NAME = os.getenv('MODEL_NAME', '')  # Resolved automatically from MODEL_PROFILE if unset
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.45'))
    IMGSZ = int(os.getenv('IMGSZ', '864'))

    # Canonical vehicle class aliases (used by core.model_resolver to match model.names dynamically)
    VEHICLE_CLASS_ALIASES = {
        'bicycle': 'bicycle',
        'bike': 'bicycle',
        'car': 'car',
        'auto': 'car',
        'sedan': 'car',
        'suv': 'car',
        'van': 'car',
        'motorcycle': 'motorcycle',
        'motorbike': 'motorcycle',
        'moto': 'motorcycle',
        'scooter': 'motorcycle',
        'bus': 'bus',
        'coach': 'bus',
        'truck': 'truck',
        'lorry': 'truck',
        'pickup': 'truck',
    }

    # Default COCO vehicle classes (overridden at runtime when loading custom models like 5-class best.pt)
    DETECTION_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # Only configured classes will be checked for lane/intersection violations
    VIOLATION_CLASS_IDS = set(DETECTION_CLASSES)

    # Compute device ('cuda' for GPU, 'cpu' for local CPU)
    TRACKER_DEVICE = os.getenv('TRACKER_DEVICE', 'cuda')

    # =========================================================================
    # VIDEO & I/O SETTINGS
    # =========================================================================
    INPUT_VIDEO = os.getenv('INPUT_VIDEO', 'ws')
    MAX_FRAMES = int(os.getenv('MAX_FRAMES', '0')) or None  # None = full video
    SIMULATE_CAMERA = os.getenv('SIMULATE_CAMERA', 'False').lower() in ('true', '1', 't')
    LOOP_VIDEO = os.getenv('LOOP_VIDEO', 'False').lower() in ('true', '1', 't')

    
    # Output directory & file paths
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
    OUTPUT_VIDEO = os.getenv('OUTPUT_VIDEO', 'output/bytetrack_test_1.mp4')
    VIOLATION_LOG = os.getenv('VIOLATION_LOG', 'output/bytetrack_test_1.txt')
    EVIDENCE_DIR = os.path.join(OUTPUT_DIR, 'evidence')

    # Output saving flags
    SAVE_OUTPUT_VIDEO = os.getenv('SAVE_OUTPUT_VIDEO', 'False').lower() in ('true', '1', 't')
    SAVE_VIOLATION_LOG = os.getenv('SAVE_VIOLATION_LOG', 'False').lower() in ('true', '1', 't')
    SAVE_EVIDENCE_SNAPSHOTS = True

    # Evidence video clip settings
    EVIDENCE_PRE_SECONDS = float(os.getenv('EVIDENCE_PRE_SECONDS', '3.0'))
    EVIDENCE_POST_SECONDS = float(os.getenv('EVIDENCE_POST_SECONDS', '3.0'))
    EVIDENCE_MODE = os.getenv('EVIDENCE_MODE', 'raw').lower()  # 'raw' (legal proof) or 'annotated'

    # Display settings
    SHOW_LIVE_PREVIEW = os.getenv('SHOW_LIVE_PREVIEW', 'False').lower() in ('true', '1', 't')

    # =========================================================================
    # ZONE & TRAFFIC LIGHT SETTINGS
    # =========================================================================
    ZONES_FILE = os.getenv('ZONES_FILE', 'zones.json')
    ENABLE_ZONE_DRAWER = os.getenv('ENABLE_ZONE_DRAWER', 'False').lower() in ('true', '1', 't')

    # =========================================================================
    # WEBSOCKET STREAMING SETTINGS (api.VDAT Integration)
    # =========================================================================
    ENABLE_WEBSOCKET_STREAM = os.getenv('ENABLE_WEBSOCKET_STREAM', 'True').lower() in ('true', '1', 't')
    API_WS_URL = os.getenv('API_WS_URL', 'ws://localhost:5000/ws/inference')
    JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '60'))
    STREAM_TARGET_FPS = float(os.getenv('STREAM_TARGET_FPS', '15.0'))
    STREAM_MAX_WIDTH = int(os.getenv('STREAM_MAX_WIDTH', '854'))
