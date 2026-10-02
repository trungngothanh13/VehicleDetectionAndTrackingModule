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
    MODEL_NAME = os.getenv('MODEL_NAME', 'yolo26l.pt')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.30'))
    IMGSZ = int(os.getenv('IMGSZ', '1280'))

    # COCO classes to detect and track
    # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
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
    INPUT_VIDEO = os.getenv('INPUT_VIDEO', 'test_1.mp4')
    MAX_FRAMES = int(os.getenv('MAX_FRAMES', '0')) or None  # None = full video
    
    # Output directory & file paths
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
    OUTPUT_VIDEO = os.getenv('OUTPUT_VIDEO', 'output/bytetrack_test_1.mp4')
    VIOLATION_LOG = os.getenv('VIOLATION_LOG', 'output/bytetrack_test_1.txt')
    EVIDENCE_DIR = os.path.join(OUTPUT_DIR, 'evidence')

    # Output saving flags
    SAVE_OUTPUT_VIDEO = os.getenv('SAVE_OUTPUT_VIDEO', 'True').lower() in ('true', '1', 't')
    SAVE_VIOLATION_LOG = os.getenv('SAVE_VIOLATION_LOG', 'True').lower() in ('true', '1', 't')
    SAVE_EVIDENCE_SNAPSHOTS = True

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
    ENABLE_WEBSOCKET_STREAM = os.getenv('ENABLE_WEBSOCKET_STREAM', 'False').lower() in ('true', '1', 't')
    API_WS_URL = os.getenv('API_WS_URL', 'ws://localhost:5000/ws/inference')
    JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '75'))
