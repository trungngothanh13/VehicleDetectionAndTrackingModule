class Config:
    # Model settings
    MODEL_NAME = 'yolov8l.pt'  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    CONFIDENCE_THRESHOLD = 0.5
    
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
    SHOW_LIVE_PREVIEW = True
    SAVE_OUTPUT_VIDEO = True
    
    # Tracking settings
    TRACK_THRESH = 0.5  # Detection confidence threshold for tracking
    MATCH_THRESH = 0.8  # IOU threshold for matching detections to tracks
    TRACK_BUFFER = 30   # Number of frames to keep lost tracks

    # Zone settings
    ZONES_FILE = 'zones.json'
    ENABLE_ZONE_DRAWER = False  # Set True to draw zones, then exit