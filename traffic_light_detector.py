# traffic_light_detector.py
import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self, traffic_lights, search_radius=12):
        """
        Initialize detector with traffic light points.

        traffic_lights: list of [{x, y, color}, ...] where color is 'red' or 'green'
        search_radius: half-width of the search window around each configured point.
                       A 25×25 window (radius=12) absorbs ~±12 px of wind sway.
        """
        self.traffic_lights = traffic_lights
        self.search_radius = search_radius
        self.state_history = []  # Keep last 5 frames for smoothing
        self.state = None  # Current state: 'red', 'green', or 'unknown'
        
        # Separate red and green light positions
        self.red_light = None
        self.green_light = None
        for tl in traffic_lights:
            if tl['color'] == 'red':
                self.red_light = (tl['x'], tl['y'])
            elif tl['color'] == 'green':
                self.green_light = (tl['x'], tl['y'])
    
    def detect(self, frame):
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
        
        # Determine which is lit (higher score wins)
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
        
        # Get most common state
        if len(self.state_history) > 0:
            from collections import Counter
            self.state = Counter(self.state_history).most_common(1)[0][0]
        
        return self.state
    
    def _get_peak_score(self, hsv, point, color):
        """
        Search a large window around the configured point and find the
        peak color response.  This handles the traffic light swaying
        a few pixels in any direction due to wind.

        Instead of averaging the whole window (which dilutes the signal),
        we slide a 7×7 kernel across the search region and return the
        maximum response — i.e. the score at the best-aligned sub-patch.
        """
        x, y = point
        r = self.search_radius
        h_img, w_img = hsv.shape[:2]

        # Extract the search region (clipped to frame bounds)
        y1, y2 = max(0, y - r), min(h_img, y + r + 1)
        x1, x2 = max(0, x - r), min(w_img, x + r + 1)
        region = hsv[y1:y2, x1:x2]

        if region.size == 0:
            return 0

        # Build a binary mask of pixels matching the target color
        hue = region[:, :, 0]
        sat = region[:, :, 1]
        val = region[:, :, 2]

        if color == 'red':
            # Red hue: 0-10 or 170-180
            color_mask = np.logical_or(hue < 11, hue > 169)
        else:  # green
            # Green hue: 60-90
            color_mask = (hue >= 60) & (hue <= 90)

        # Need reasonable saturation and brightness
        bright_mask = (sat > 80) & (val > 80)
        match_mask = (color_mask & bright_mask).astype(np.float32)

        # Slide a 7×7 box-sum kernel to find the densest cluster of
        # matching pixels — this is the likely position of the light.
        kernel_size = 7
        if match_mask.shape[0] < kernel_size or match_mask.shape[1] < kernel_size:
            # Region too small for sliding; fall back to total count
            return float(np.sum(match_mask))

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        response = cv2.filter2D(match_mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return float(np.max(response))
    
    def is_red(self):
        """Check if current state is red"""
        return self.state == 'red'
    
    def is_green(self):
        """Check if current state is green"""
        return self.state == 'green'
