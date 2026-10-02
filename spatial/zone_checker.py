import cv2
import json
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from core.events import ViolationEvent


class ZoneChecker:
    """
    Manages approach-lane and intersection spatial polygons.
    Evaluates whether tracked vehicles commit red-light violations
    when transitioning from an approach lane into the intersection during a red phase.
    """

    def __init__(self, zones_file_or_data: Union[str, Dict]):
        if isinstance(zones_file_or_data, str):
            with open(zones_file_or_data, 'r') as f:
                zones = json.load(f)
        else:
            zones = zones_file_or_data

        self.lanes = zones.get('lanes', [])
        self.intersection = zones.get('intersection', [])

        # Track state history per vehicle ID
        self._seen_in_lane: Dict[int, bool] = {}
        self._seen_in_lane_during_red: Dict[int, bool] = {}
        self._entered_intersection: Dict[int, bool] = {}
        self._entry_light_state: Dict[int, str] = {}
        self._violation_reported: Dict[int, bool] = {}

    def get_vehicle_position(self, vehicle_bbox: Union[List[float], np.ndarray]) -> Optional[Union[int, str]]:
        """
        Get position of vehicle: returns lane_index (0+), 'intersection', or None.
        Uses bottom center of bbox as vehicle ground contact point.
        """
        x_center = (float(vehicle_bbox[0]) + float(vehicle_bbox[2])) / 2.0
        y_bottom = float(vehicle_bbox[3])
        point = (x_center, y_bottom)

        # 1. Check if in intersection polygon(s)
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

        # 2. Check which approach lane polygon
        for i, lane in enumerate(self.lanes):
            if len(lane) >= 3:
                lane_polygon = np.array(lane, dtype=np.int32)
                if cv2.pointPolygonTest(lane_polygon, point, False) >= 0:
                    return i

        return None

    def check_lane_to_intersection(self, vehicle_id: int, vehicle_bbox: Union[List[float], np.ndarray], light_is_red: bool = False) -> bool:
        """
        Check if vehicle commits a red-light violation.

        A violation is triggered IF AND ONLY IF:
        1. The vehicle was observed in an approach lane.
        2. The vehicle crosses from the lane into the intersection for the first time.
        3. The traffic light is RED at the exact moment of entering the intersection.
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
            if self._entered_intersection.get(vehicle_id, False):
                return False

            was_in_lane = self._seen_in_lane.get(vehicle_id, False)

            if was_in_lane:
                self._entered_intersection[vehicle_id] = True
                self._entry_light_state[vehicle_id] = 'red' if light_is_red else 'green'

                if light_is_red and vehicle_id not in self._violation_reported:
                    self._violation_reported[vehicle_id] = True
                    return True

        return False

    def evaluate_detections(
        self,
        camera_id: str,
        frame_number: int,
        timestamp_str: str,
        tracked_detections,
        tracker,
        light_state: str,
        allowed_class_ids: set,
    ) -> List[ViolationEvent]:
        """
        Evaluates all tracked vehicles in the current frame and returns newly triggered ViolationEvents.
        """
        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return []

        light_is_red = (light_state == 'red')
        violations: List[ViolationEvent] = []

        for i in range(len(tracked_detections)):
            class_id = int(tracked_detections.class_id[i])
            if allowed_class_ids and class_id not in allowed_class_ids:
                continue

            tracker_id = int(tracked_detections.tracker_id[i])
            bbox = [float(c) for c in tracked_detections.xyxy[i]]
            conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else None

            if self.check_lane_to_intersection(tracker_id, bbox, light_is_red=light_is_red):
                class_name = tracker.get_class_name(class_id)
                event = ViolationEvent(
                    camera_id=camera_id,
                    track_id=tracker_id,
                    vehicle_class=class_name,
                    class_id=class_id,
                    frame_number=frame_number,
                    timestamp=timestamp_str,
                    light_state=light_state,
                    bbox=bbox,
                    confidence=conf,
                )
                violations.append(event)

        return violations

    def reset(self) -> None:
        """Reset all internal tracking states."""
        self._seen_in_lane.clear()
        self._seen_in_lane_during_red.clear()
        self._entered_intersection.clear()
        self._entry_light_state.clear()
        self._violation_reported.clear()
