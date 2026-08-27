class Config:
    # =========================================================================
    # TRACKER SELECTION
    # Available options: 'botsort', 'bytetrack', 'deepsort'
    #
    # Each tracker's fine-grained parameters are managed in its own YAML file:
    # - BoT-SORT:  trackers/botsort/botsort.yaml
    # - ByteTrack: trackers/bytetrack/bytetrack.yaml
    # - DeepSORT:  trackers/deepsort/deepsort.yaml (toggle use_reid: false/true)
    # =========================================================================
    TRACKER_TYPE = 'botsort'

    # =========================================================================
    # GLOBAL MODEL & DETECTION SETTINGS
    # =========================================================================
    MODEL_NAME = 'yolo26l.pt'    # Detector weights (e.g. 'yolo26l.pt' or 'yolov8l.pt')
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
    # VIDEO & I/O SETTINGS
    # =========================================================================
    INPUT_VIDEO = 'test_1.mp4'
    
    # Output directory & file paths
    OUTPUT_DIR = 'output'
    OUTPUT_VIDEO = 'output/output_test_1.mp4'
    VIOLATION_LOG = 'output/violations_test_1.txt'

    # Output saving flags
    SAVE_OUTPUT_VIDEO = True
    SAVE_VIOLATION_LOG = True

    # Display settings
    SHOW_LIVE_PREVIEW = False    # Must be False for headless cloud servers

    # =========================================================================
    # ZONE & TRAFFIC LIGHT SETTINGS
    # =========================================================================
    ZONES_FILE = 'zones.json'
    ENABLE_ZONE_DRAWER = False   # Set True to run interactive zone drawer, then exit