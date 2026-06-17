# zone_checker.py
import cv2
import json
import numpy as np

class ZoneChecker:
    def __init__(self, zones_file):
        with open(zones_file, 'r') as f:
            zones = json.load(f)
        self.lanes = zones.get('lanes', [])
        self.intersection = zones.get('intersection', [])
        # Track if we've ever seen a vehicle in a lane
        self._seen_in_lane = {}  # {vehicle_id: True}
        # Track if violation already reported for this vehicle
        self._violation_reported = {}  # {vehicle_id: True}
    
    def get_vehicle_position(self, vehicle_bbox):
        """
        Get position of vehicle: returns lane_index (0+), 'intersection', or None
        vehicle_bbox: [x1, y1, x2, y2]
        """
        # Use bottom center of bbox as vehicle position
        x_center = (vehicle_bbox[0] + vehicle_bbox[2]) / 2
        y_bottom = vehicle_bbox[3]
        point = (x_center, y_bottom)
        
        # Check if in intersection
        if len(self.intersection) >= 3:
            intersection_polygon = np.array(self.intersection, dtype=np.int32)
            result = cv2.pointPolygonTest(intersection_polygon, point, False)
            if result >= 0:
                return 'intersection'
        
        # Check which lane
        for i, lane in enumerate(self.lanes):
            if len(lane) >= 3:
                lane_polygon = np.array(lane, dtype=np.int32)
                result = cv2.pointPolygonTest(lane_polygon, point, False)
                if result >= 0:
                    return i
        
        return None
    
    def check_lane_to_intersection(self, vehicle_id, vehicle_bbox, light_is_red=False):
        """
        Check if vehicle transitions from lane to intersection.
        Counts as violation if: vehicle was ever in a lane, and is now in intersection.
        Handles tracking gaps - only requires the vehicle be in a lane at some point.
        Returns True on first detection of vehicle in intersection after being in lane.
        """
        current_pos = self.get_vehicle_position(vehicle_bbox)
        
        # Track if vehicle was ever in a lane
        if isinstance(current_pos, int):
            self._seen_in_lane[vehicle_id] = True
        
        # Count violation if in intersection AND was previously in a lane AND light is red AND not reported yet
        if (current_pos == 'intersection' and 
            self._seen_in_lane.get(vehicle_id, False) and 
            light_is_red and
            vehicle_id not in self._violation_reported):
            self._violation_reported[vehicle_id] = True
            return True
        
        return False