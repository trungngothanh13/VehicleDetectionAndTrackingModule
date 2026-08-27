# Handles drawing boxes, labels, zones, and statistics

import cv2
import numpy as np


class Visualizer:
    def __init__(self, tracked_classes):
        self.tracked_classes = tracked_classes
        # Vibrant, high-contrast BGR colors
        self.colors = {
            1: (255, 50, 255),   # bicycle - Magenta / Pink
            2: (50, 230, 50),    # car - Bright Green
            3: (255, 215, 0),    # motorcycle - Bright Cyan/Sky Blue
            5: (0, 140, 255),    # bus - Bright Orange
            7: (0, 235, 255)     # truck - Bright Yellow
        }
        # Abbreviated class names for compact labels
        self.short_names = {
            1: 'bike',
            2: 'car',
            3: 'moto',
            5: 'bus',
            7: 'truck'
        }
        self.violations_count = 0
        self.light_state = 'unknown'

    def _get_text_color(self, bg_color):
        """Compute black or white text color for maximum luminance contrast."""
        b, g, r = bg_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return (0, 0, 0) if luminance > 135 else (255, 255, 255)

    def draw_detections(self, frame, tracked_detections, detector, violation_ids=None):
        """Draw bounding boxes and high-contrast labels for tracked vehicles"""
        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return frame

        violation_set = violation_ids if violation_ids is not None else set()

        for i in range(len(tracked_detections)):
            x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
            class_id = int(tracked_detections.class_id[i])
            tracker_id = int(tracked_detections.tracker_id[i])

            is_violation = tracker_id in violation_set

            # Use bright RED for vehicles committing a violation
            if is_violation:
                box_color = (0, 0, 255)  # Red in BGR
                short = self.short_names.get(class_id, 'obj')
                label = f"#{tracker_id} {short} [VIOLATION]"
            else:
                box_color = self.colors.get(class_id, (0, 255, 255))
                short = self.short_names.get(class_id, 'obj')
                label = f"#{tracker_id} {short}"

            # Draw bounding box (2px thickness for high visibility)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Measure label text size
            font_scale, thickness = 0.40, 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw solid background badge for label
            badge_y1 = max(0, y1 - label_h - 6)
            badge_y2 = y1
            badge_x2 = min(frame.shape[1], x1 + label_w + 6)
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

            # Draw text with optimal high-contrast color
            text_color = self._get_text_color(box_color)
            cv2.putText(
                frame,
                label,
                (x1 + 3, badge_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_statistics(self, frame, current_count, total_tracked, frame_count):
        """Draw compact statistics panel on top-left of frame"""
        fs = 0.40   # font scale
        pad = 8
        line_h = 18
        lines = [
            (f"Vehicles:   {current_count}",  (0, 255, 255)),
            (f"Tracked:    {total_tracked}",  (0, 255, 255)),
            (f"Frame:      {frame_count}",    (200, 200, 200)),
        ]
        if hasattr(self, 'violations_count'):
            lines.append((f"Violations: {self.violations_count}", (0, 0, 255)))
        if hasattr(self, 'light_state'):
            lc = (0, 0, 255) if self.light_state == 'red' else (0, 255, 0) if self.light_state == 'green' else (128, 128, 128)
            lines.append((f"Light:      {self.light_state.upper()}", lc))

        panel_w = 190
        panel_h = pad * 2 + line_h * len(lines)
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (100, 100, 100), 1)

        for idx, (text, color) in enumerate(lines):
            y = 8 + pad + line_h * idx + 12
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

        return frame

    def set_violations_count(self, count):
        self.violations_count = count

    def set_light_state(self, state):
        self.light_state = state

    def draw_zones(self, frame, lanes, intersection):
        """Draw lanes and intersection on frame"""
        # Draw lanes (Yellow outline)
        for lane in lanes:
            if len(lane) >= 3:
                pts = np.array(lane, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        # Draw intersection (Cyan outline)
        if len(intersection) >= 3:
            pts = np.array(intersection, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

        return frame