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

        # Track state history per vehicle ID
        self._seen_in_lane = {}             # {vehicle_id: True}
        self._seen_in_lane_during_red = {}  # {vehicle_id: True}
        self._entered_intersection = {}     # {vehicle_id: True}
        self._entry_light_state = {}        # {vehicle_id: 'red' | 'green'}
        self._violation_reported = {}       # {vehicle_id: True}

    def get_vehicle_position(self, vehicle_bbox):
        """
        Get position of vehicle: returns lane_index (0+), 'intersection', or None.
        Uses bottom center of bbox as vehicle ground contact point.
        """
        x_center = (vehicle_bbox[0] + vehicle_bbox[2]) / 2.0
        y_bottom = float(vehicle_bbox[3])
        point = (x_center, y_bottom)

        # Check if in intersection polygon(s)
        if self.intersection and len(self.intersection) >= 3:
            # Handles both single polygon [[x, y], ...] and list of polygons [[[x, y], ...]]
            if isinstance(self.intersection[0][0], (int, float, np.number)):
                poly = np.array(self.intersection, dtype=np.int32)
                if cv2.pointPolygonTest(poly, point, False) >= 0:
                    return 'intersection'
            else:
                for inter_poly in self.intersection:
                    if len(inter_poly) >= 3:
                        poly = np.array(inter_poly, dtype=np.int32)
                        if cv2.pointPolygonTest(poly, point, False) >= 0:
                            return 'intersection'

        # Check which approach lane polygon
        for i, lane in enumerate(self.lanes):
            if len(lane) >= 3:
                lane_polygon = np.array(lane, dtype=np.int32)
                if cv2.pointPolygonTest(lane_polygon, point, False) >= 0:
                    return i

        return None

    def check_lane_to_intersection(self, vehicle_id, vehicle_bbox, light_is_red=False):
        """
        Check if vehicle commits a red-light violation.

        A violation is triggered IF AND ONLY IF:
        1. The vehicle was observed in an approach lane.
        2. The vehicle crosses from the lane into the intersection for the first time.
        3. The traffic light is RED at the exact moment of entering the intersection.

        Vehicles that entered the intersection legally while the light was GREEN
        will NOT be flagged if the light subsequently turns red while they clear the intersection.
        """
        current_pos = self.get_vehicle_position(vehicle_bbox)

        # 1. Vehicle is currently in an approach lane
        if isinstance(current_pos, int):
            self._seen_in_lane[vehicle_id] = True
            if light_is_red:
                self._seen_in_lane_during_red[vehicle_id] = True
            return False

        # 2. Vehicle is currently in the intersection
        if current_pos == 'intersection':
            # If this vehicle already entered the intersection in a previous frame,
            # its entry event was already evaluated; do not re-evaluate.
            if self._entered_intersection.get(vehicle_id, False):
                return False

            # Check if this vehicle originated from an approach lane
            was_in_lane = self._seen_in_lane.get(vehicle_id, False)

            if was_in_lane:
                # Register first entry into intersection
                self._entered_intersection[vehicle_id] = True
                self._entry_light_state[vehicle_id] = 'red' if light_is_red else 'green'

                # VIOLATION: Entered intersection while light was RED
                if light_is_red and vehicle_id not in self._violation_reported:
                    self._violation_reported[vehicle_id] = True
                    return True

        return False

    def reset(self):
        """Reset all internal tracking states."""
        self._seen_in_lane.clear()
        self._seen_in_lane_during_red.clear()
        self._entered_intersection.clear()
        self._entry_light_state.clear()
        self._violation_reported.clear()