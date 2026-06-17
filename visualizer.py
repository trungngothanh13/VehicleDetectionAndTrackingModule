# Handles drawing boxes, labels, and stats

import cv2
import numpy as np

class Visualizer:
    def __init__(self, tracked_classes):
        self.tracked_classes = tracked_classes
        self.colors = {
            1: (255, 0, 255),  # bicycle - magenta
            2: (0, 255, 0),    # car - green
            3: (255, 0, 0),    # motorcycle - blue
            5: (0, 165, 255),  # bus - orange
            7: (0, 255, 255)   # truck - yellow
        }
        
    def draw_detections(self, frame, tracked_detections, detector):
        """Draw bounding boxes and labels for tracked vehicles"""
        if tracked_detections.tracker_id is None:
            return frame
        
        for i in range(len(tracked_detections)):
            x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
            conf = tracked_detections.confidence[i]
            class_id = tracked_detections.class_id[i]
            tracker_id = tracked_detections.tracker_id[i]
            
            # Get color for this detected class
            color = self.colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            class_name = detector.get_class_name(class_id)
            label = f"ID:{tracker_id} {class_name} {conf:.2f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def draw_statistics(self, frame, current_count, total_tracked, frame_count):
        """Draw statistics on frame"""
        # Background for stats
        cv2.rectangle(frame, (10, 10), (260, 125), (0, 0, 0), -1)
        
        # Current vehicles on screen
        cv2.putText(frame, f"Current Vehicles: {current_count}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # Total unique vehicles tracked
        cv2.putText(frame, f"Total Tracked: {total_tracked}", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_count}", (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if hasattr(self, 'violations_count'):
            cv2.putText(frame, f"Red Light Violations: {self.violations_count}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        if hasattr(self, 'light_state'):
            light_color = (0, 0, 255) if self.light_state == 'red' else (0, 255, 0) if self.light_state == 'green' else (128, 128, 128)
            cv2.putText(frame, f"Light: {self.light_state.upper()}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, light_color, 1)
        
        return frame

    def set_violations_count(self, count):
        self.violations_count = count
    
    def set_light_state(self, state):
        self.light_state = state

    def draw_zones(self, frame, lanes, intersection):
        """Draw lanes and intersection on frame"""
        # Draw lanes
        for i, lane in enumerate(lanes):
            if len(lane) >= 3:
                pts = np.array(lane, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)  # Yellow
        
        # Draw intersection
        if len(intersection) >= 3:
            pts = np.array(intersection, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)  # Blue
        
        return frame