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
        # Abbreviated class names for compact labels
        self.short_names = {
            1: 'bike', 2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'
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
            
            # Draw bounding box — thin 1px border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Compact label: "#187 moto"
            short = self.short_names.get(class_id, 'obj')
            label = f"#{tracker_id} {short}"
            
            # Draw label background
            font_scale, thickness = 0.32, 1
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0] + 2, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return frame
    
    def draw_statistics(self, frame, current_count, total_tracked, frame_count):
        """Draw compact statistics panel on frame"""
        fs = 0.38   # font scale
        pad = 8
        line_h = 18
        lines = [
            (f"Vehicles: {current_count}",  (0, 255, 255)),
            (f"Tracked:  {total_tracked}",  (0, 255, 255)),
            (f"Frame:    {frame_count}",    (200, 200, 200)),
        ]
        if hasattr(self, 'violations_count'):
            lines.append((f"Violations: {self.violations_count}", (0, 0, 255)))
        if hasattr(self, 'light_state'):
            lc = (0,0,255) if self.light_state=='red' else (0,255,0) if self.light_state=='green' else (128,128,128)
            lines.append((f"Light: {self.light_state.upper()}", lc))

        panel_h = pad * 2 + line_h * len(lines)
        cv2.rectangle(frame, (8, 8), (175, 8 + panel_h), (0, 0, 0), -1)
        for idx, (text, color) in enumerate(lines):
            y = 8 + pad + line_h * idx + 12
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)
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