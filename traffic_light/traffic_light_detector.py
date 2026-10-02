import cv2
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union, Any


class TrafficLightDetector:
    def __init__(self, traffic_lights: List[Dict[str, Any]], search_radius: int = 12):
        """
        Initialize detector with traffic light points.

        traffic_lights: list of [{x, y, color}, ...] where color is 'red' or 'green'
        search_radius: half-width of the search window around each configured point.
                       A 25x25 window (radius=12) absorbs ~+-12 px of wind sway.
        """
        self.traffic_lights = traffic_lights
        self.search_radius = search_radius
        self.state_history = []  # Keep last 5 frames for smoothing
        self.state = 'unknown'   # Current state: 'red', 'green', or 'unknown'

        # Separate red and green light positions
        self.red_light: Optional[Tuple[int, int]] = None
        self.green_light: Optional[Tuple[int, int]] = None
        for tl in traffic_lights:
            if tl.get('color') == 'red':
                self.red_light = (int(tl['x']), int(tl['y']))
            elif tl.get('color') == 'green':
                self.green_light = (int(tl['x']), int(tl['y']))

    @classmethod
    def from_file(cls, zones_file: str, search_radius: int = 12) -> Optional['TrafficLightDetector']:
        """Factory method to load traffic light coordinates from zones.json."""
        try:
            with open(zones_file, 'r') as f:
                data = json.load(f)
            lights = data.get('traffic_lights', [])
            if lights and len(lights) >= 2:
                return cls(lights, search_radius=search_radius)
        except Exception as e:
            print(f"[TrafficLightDetector] Could not load traffic lights: {e}")
        return None

    def detect(self, frame: np.ndarray) -> str:
        """
        Detect which traffic light is currently lit based on pixel brightness.
        Uses a wide search window to tolerate wind-induced sway.
        Returns: 'red', 'green', or 'unknown'
        """
        if not self.red_light or not self.green_light:
            return 'unknown'

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Sample pixels around the light points using the search window
        red_score = self._get_peak_score(hsv, self.red_light, 'red')
        green_score = self._get_peak_score(hsv, self.green_light, 'green')

        min_threshold = 8
        if red_score > green_score and red_score > min_threshold:
            detected_state = 'red'
        elif green_score > red_score and green_score > min_threshold:
            detected_state = 'green'
        else:
            detected_state = 'unknown'

        # Smooth with history (majority voting over last 5 frames)
        self.state_history.append(detected_state)
        if len(self.state_history) > 5:
            self.state_history.pop(0)

        if len(self.state_history) > 0:
            self.state = Counter(self.state_history).most_common(1)[0][0]

        return self.state

    def _get_peak_score(self, hsv: np.ndarray, point: Tuple[int, int], color: str) -> float:
        """
        Search a window around the configured point and find the peak color response.
        """
        x, y = point
        r = self.search_radius
        h_img, w_img = hsv.shape[:2]

        y1, y2 = max(0, y - r), min(h_img, y + r + 1)
        x1, x2 = max(0, x - r), min(w_img, x + r + 1)
        region = hsv[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        hue = region[:, :, 0]
        sat = region[:, :, 1]
        val = region[:, :, 2]

        if color == 'red':
            color_mask = np.logical_or(hue < 11, hue > 169)
        else:  # green
            color_mask = (hue >= 60) & (hue <= 90)

        bright_mask = (sat > 80) & (val > 80)
        match_mask = (color_mask & bright_mask).astype(np.float32)

        kernel_size = 7
        if match_mask.shape[0] < kernel_size or match_mask.shape[1] < kernel_size:
            return float(np.sum(match_mask))

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        response = cv2.filter2D(match_mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return float(np.max(response))

    def is_red(self) -> bool:
        return self.state == 'red'

    def is_green(self) -> bool:
        return self.state == 'green'
