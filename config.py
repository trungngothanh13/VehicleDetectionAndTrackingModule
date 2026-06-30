class Config:
    # Model settings
    MODEL_NAME = 'yolo26l.pt'  # Upgraded from yolov8l — +2.1 mAP, same T4 latency, NMS-free
    CONFIDENCE_THRESHOLD = 0.30  # Lower threshold feeds more detections into ByteTrack low-conf recovery
    
    # COCO classes supported by the default YOLOv8 weights.
    # Requested classes: bicycle, car, motorcycle, bus, truck.
    DETECTION_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Only configured detection classes should be considered for lane/intersection violations.
    VIOLATION_CLASS_IDS = set(DETECTION_CLASSES)
    
    # Video settings
    INPUT_VIDEO = 'test_2.mp4'
    OUTPUT_VIDEO = 'output_test_2.mp4'
    
    # Display settings
    SHOW_LIVE_PREVIEW = False  # Must be False for headless server (no display)
    SAVE_OUTPUT_VIDEO = True
    
    # Tracking settings
    TRACK_THRESH = 0.35  # Detection confidence threshold for tracking
    MATCH_THRESH = 0.6  # IOU threshold for matching detections to tracks
    TRACK_BUFFER = 60   # Number of frames to keep lost tracks

    # Zone settings
    ZONES_FILE = 'zones.json'
    ENABLE_ZONE_DRAWER = False  # Set True to draw zones, then exit