# traffic_light_detector.py
import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self, traffic_lights):
        """
        Initialize detector with traffic light points
        traffic_lights: list of [{x, y, color}, ...] where color is 'red' or 'green'
        """
        self.traffic_lights = traffic_lights
        self.state_history = []  # Keep last 3 frames for smoothing
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
        Detect which traffic light is currently lit based on pixel brightness
        Returns: 'red', 'green', or 'unknown'
        """
        if not self.red_light or not self.green_light:
            return 'unknown'
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Sample pixels around the light points
        red_brightness = self._get_brightness(hsv, self.red_light, 'red')
        green_brightness = self._get_brightness(hsv, self.green_light, 'green')
        
        # Determine which is lit (higher score wins)
        # Require at least some brightness to detect, but be lenient with threshold
        min_threshold = 5
        if red_brightness > green_brightness and red_brightness > min_threshold:
            detected_state = 'red'
        elif green_brightness > red_brightness and green_brightness > min_threshold:
            detected_state = 'green'
        else:
            detected_state = 'unknown'
        
        # Smooth with history (majority voting over last 3 frames)
        self.state_history.append(detected_state)
        if len(self.state_history) > 3:
            self.state_history.pop(0)
        
        # Get most common state
        if len(self.state_history) > 0:
            from collections import Counter
            self.state = Counter(self.state_history).most_common(1)[0][0]
        
        return self.state
    
    def _get_brightness(self, hsv, point, color):
        """
        Get brightness/saturation score for a light point
        Sample larger region around the point for stability
        """
        x, y = point
        # Sample 7x7 region instead of 3x3 for better detection
        region = hsv[max(0, y-3):min(hsv.shape[0], y+4), 
                      max(0, x-3):min(hsv.shape[1], x+4)]
        
        if region.size == 0:
            return 0
        
        # For red light: check red hue (0-10 or 170-180) with decent saturation
        # For green light: check green hue (60-90) with decent saturation
        
        hue = region[:, :, 0]
        sat = region[:, :, 1]
        val = region[:, :, 2]
        
        if color == 'red':
            # Red hue: 0-10 or 170-180
            red_mask = np.logical_or(hue < 11, hue > 169)
            # Need reasonable saturation and brightness (lowered thresholds)
            bright_mask = (sat > 80) & (val > 80)
            score = np.sum(red_mask & bright_mask)
        else:  # green
            # Green hue: 60-90
            green_mask = (hue >= 60) & (hue <= 90)
            # Need reasonable saturation and brightness (lowered thresholds)
            bright_mask = (sat > 80) & (val > 80)
            score = np.sum(green_mask & bright_mask)
        
        return score
    
    def is_red(self):
        """Check if current state is red"""
        return self.state == 'red'
    
    def is_green(self):
        """Check if current state is green"""
        return self.state == 'green'
