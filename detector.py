from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_name, target_classes, confidence_threshold):
        self.model = YOLO(model_name)
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        
    def detect(self, frame):
        """
        Detect selected COCO classes in a frame
        Returns: list of detections [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            cls = int(box.cls[0])
            
            # Only process the configured class IDs
            if cls in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                detections.append([x1, y1, x2, y2, conf, cls])
        
        return detections
    
    def get_class_name(self, class_id):
        """Get configured class name from ID"""
        return self.target_classes.get(class_id, 'unknown')