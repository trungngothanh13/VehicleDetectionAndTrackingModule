import cv2
import numpy as np
from typing import Dict, Set, Optional


class Visualizer:
    NAME_COLORS = {
        'bicycle': (255, 50, 255),     # Magenta / Pink
        'car': (50, 230, 50),          # Bright Green
        'motorcycle': (255, 215, 0),   # Bright Cyan / Sky Blue
        'bus': (0, 140, 255),          # Bright Orange
        'truck': (0, 235, 255),        # Bright Yellow
    }
    NAME_SHORT = {
        'bicycle': 'bike',
        'car': 'car',
        'motorcycle': 'moto',
        'bus': 'bus',
        'truck': 'truck',
    }

    def __init__(self, tracked_classes: Dict[int, str]):
        self.tracked_classes = {}
        self.colors: Dict[int, tuple] = {}
        self.short_names: Dict[int, str] = {}
        self.update_classes(tracked_classes)

    def update_classes(self, tracked_classes: Dict[int, str]) -> None:
        """Dynamically refreshes class-ID colors and short labels (e.g. on live model toggle)."""
        self.tracked_classes = dict(tracked_classes or {})
        self.colors = {}
        self.short_names = {}
        for cid, cname in self.tracked_classes.items():
            key = str(cname).strip().lower()
            self.colors[int(cid)] = self.NAME_COLORS.get(key, (0, 255, 255))
            self.short_names[int(cid)] = self.NAME_SHORT.get(key, key[:5] or 'obj')

    def _get_text_color(self, bg_color):
        """Compute black or white text color for maximum luminance contrast."""
        b, g, r = bg_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return (0, 0, 0) if luminance > 135 else (255, 255, 255)

    def draw_detections(self, frame: np.ndarray, tracked_detections, detector, violation_ids: Optional[Set[int]] = None) -> np.ndarray:
        """Draw bounding boxes and high-contrast labels for tracked vehicles"""
        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return frame

        violation_set = violation_ids if violation_ids is not None else set()

        # Draw normal detections first so red violation highlights always render on top
        ordered_indices = sorted(
            range(len(tracked_detections)),
            key=lambda idx: int(tracked_detections.tracker_id[idx]) in violation_set,
        )

        for i in ordered_indices:
            x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
            class_id = int(tracked_detections.class_id[i])
            tracker_id = int(tracked_detections.tracker_id[i])

            is_violation = tracker_id in violation_set
            short = self.short_names.get(class_id, 'obj')

            if is_violation:
                box_color = (0, 0, 255)  # Red in BGR
                label = f"{short} #{tracker_id}"
                box_thickness = 2
                font_scale = 0.40
            else:
                box_color = self.colors.get(class_id, (0, 255, 255))
                label = short
                box_thickness = 1
                font_scale = 0.36

            # Draw slim bounding box (1px normal, 2px red for violations)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Measure compact label text size
            thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw tight solid background badge for label
            pad_x, pad_y = 2, 2
            badge_y1 = max(0, y1 - label_h - pad_y * 2)
            badge_y2 = max(label_h + pad_y * 2, y1)
            badge_x2 = min(frame.shape[1], x1 + label_w + pad_x * 2)
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

            # Draw text with optimal high-contrast color
            text_color = self._get_text_color(box_color)
            cv2.putText(
                frame,
                label,
                (x1 + pad_x, badge_y2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_zones(
        self,
        frame: np.ndarray,
        lanes: list,
        intersection: list,
        traffic_lights: Optional[list] = None,
        exit_zones: Optional[list] = None,
    ) -> np.ndarray:
        """Draw Approach Lanes, Intersection, Exit Zones, and Traffic Light boxes on frame."""
        for lane in lanes:
            if len(lane) >= 3:
                pts = np.array(lane, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        if len(intersection) >= 3:
            pts = np.array(intersection, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

        if exit_zones:
            for idx, ez in enumerate(exit_zones):
                if len(ez) >= 3:
                    pts = np.array(ez, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0, 140, 255), 2)
                    x0, y0 = int(pts[0][0]), int(pts[0][1])
                    cv2.putText(
                        frame,
                        f"EXIT {idx + 1}",
                        (x0 + 4, max(14, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 140, 255),
                        1,
                        cv2.LINE_AA,
                    )

        if traffic_lights:
            for light in traffic_lights:
                lx = int(light.get('x', 0))
                ly = int(light.get('y', 0))
                lcolor = str(light.get('color', 'red')).lower()
                bgr = (0, 255, 0) if lcolor == 'green' else (0, 0, 255)
                cv2.circle(frame, (lx, ly), 3, bgr, -1)

        return frame
