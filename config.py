class Config:
    # =========================================================================
    # TRACKER SELECTION
    # Available options: 'botsort', 'bytetrack', 'deepsort'
    # =========================================================================
    TRACKER_TYPE = 'botsort'

    # =========================================================================
    # GLOBAL MODEL & DETECTION SETTINGS
    # =========================================================================
    MODEL_NAME = 'yolo26l.pt'  # Detector weights (e.g. 'yolo26l.pt' or 'yolov8l.pt')
    CONFIDENCE_THRESHOLD = 0.30  # Minimum detection confidence
    IMGSZ = 1280                 # Inference resolution (1280 for small/distant vehicles)

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
    TRACKER_DEVICE = 'cuda'

    # =========================================================================
    # TRACKER-SPECIFIC CONFIGURATIONS
    # =========================================================================
    
    # BoT-SORT (Ultralytics built-in with camera motion compensation)
    BOTSORT_CONFIG = {
        'tracker_yaml': 'botsort.yaml',
        'track_thresh': 0.35,
        'match_thresh': 0.6,
        'track_buffer': 60,
    }

    # ByteTrack (Ultralytics built-in high-speed association)
    BYTETRACK_CONFIG = {
        'tracker_yaml': 'bytetrack.yaml',
        'track_thresh': 0.35,
        'match_thresh': 0.6,
        'track_buffer': 60,
    }

    # DeepSORT (deep-sort-realtime appearance-based tracking)
    DEEPSORT_CONFIG = {
        'max_age': 60,              # Max frames to keep track alive without detection
        'n_init': 3,                # Number of consecutive detections to confirm track
        'max_cosine_distance': 0.2, # Maximum cosine distance for ReID matching (0.4 if IoU-only)
        'nn_budget': 100,           # Max feature vectors per track
        'embedder': 'mobilenet',    # ReID embedder ('mobilenet' or None for IoU-only)
        'half': True,               # FP16 ReID inference for speed
        'embedder_gpu': True,       # Run ReID embedder on GPU if available
    }

    # =========================================================================
    # VIDEO & I/O SETTINGS
    # =========================================================================
    INPUT_VIDEO = 'test_1.mp4'
    OUTPUT_VIDEO = 'output_test_1.mp4'

    # Display settings
    SHOW_LIVE_PREVIEW = False  # Must be False for headless cloud servers
    SAVE_OUTPUT_VIDEO = True

    # =========================================================================
    # ZONE & TRAFFIC LIGHT SETTINGS
    # =========================================================================
    ZONES_FILE = 'zones.json'
    ENABLE_ZONE_DRAWER = True  # Set True to run interactive zone drawer, then exit