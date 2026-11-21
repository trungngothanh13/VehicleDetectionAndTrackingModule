import cv2
import json
import math
from collections import deque
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Deque
from core.events import ViolationEvent


class ZoneChecker:
    """
    Manages Approach Lanes, Intersection boundaries, and Exit Zones.

    Supports two evaluation modes:
    1. Standard mode (`crossing_mode=False`):
       Flags `Approach Lane -> Intersection/Exit Zone` when `light_state == 'red'`.
    2. 3/4-Way Multi-Lane Crossing mode (`crossing_mode=True`):
       - Applies a `clearance_seconds` (default 3.0s) grace timer on `green -> red` transitions.
       - Requires vehicles to still be in an Approach Lane after the clearance timer expires
         (`seen_in_lane_during_red`) so queued left-turners or green-phase crossers are never
         falsely flagged.
       - If Exit Zones are configured (Straight Red + Turn Green): only flags vehicles that
         traverse to the far-side Exit Zone OR stand idle inside the Intersection box (>= 22 frames
         with < 14 px displacement) when cross-traffic blocks them.
    """

    def __init__(self, zones_file_or_data: Union[str, Dict]):
        if isinstance(zones_file_or_data, str):
            with open(zones_file_or_data, 'r', encoding='utf-8') as f:
                zones = json.load(f)
        else:
            zones = zones_file_or_data

        self.lanes = zones.get('lanes', [])
        self.intersection = zones.get('intersection', [])
        self.exit_zones = zones.get('exit_zones') or zones.get('exitZones') or []
        self.crossing_mode = bool(zones.get('crossing_mode') or zones.get('crossingMode') or False)

        raw_clearance = zones.get('clearance_seconds')
        if raw_clearance is None:
            raw_clearance = zones.get('clearanceSeconds')
        self.clearance_seconds = float(raw_clearance if raw_clearance is not None else 3.0)

        # Signal phase & green->red clearance timer state
        self._confirmed_light_state: Optional[str] = None
        self._red_clearance_until_frame: int = 0
        self._last_light_update_frame: int = -1

        # Track state history per vehicle ID
        self._seen_in_lane: Dict[int, bool] = {}
        self._seen_in_lane_during_red: Dict[int, bool] = {}
        self._last_lane_frame: Dict[int, int] = {}
        self._entered_intersection: Dict[int, bool] = {}
        self._legal_crosser: Dict[int, bool] = {}
        self._intersection_frames: Dict[int, int] = {}
        self._recent_intersection_pts: Dict[int, Deque[Tuple[float, float]]] = {}
        self._entry_light_state: Dict[int, str] = {}
        self._violation_reported: Dict[int, bool] = {}

    def update_light_state(self, light_state: str, frame_number: int, fps: float = 15.0) -> None:
        """
        Updates confirmed signal phase once per frame.
        Starts the clearance grace timer strictly on real `green -> red` transitions,
        ignoring transient `'unknown'` flickers while already red.
        """
        if frame_number == self._last_light_update_frame:
            return
        self._last_light_update_frame = frame_number

        normalized = str(light_state or 'unknown').lower()
        if normalized not in ('red', 'green'):
            return

        if self._confirmed_light_state == 'green' and normalized == 'red' and self.clearance_seconds > 0:
            grace_frames = int(round(self.clearance_seconds * max(float(fps or 15.0), 1.0)))
            self._red_clearance_until_frame = frame_number + grace_frames
        elif normalized == 'green':
            self._red_clearance_until_frame = 0

        self._confirmed_light_state = normalized

    def is_in_clearance(self, frame_number: int) -> bool:
        """Returns True if frame falls inside the active green->red clearance window."""
        return frame_number < self._red_clearance_until_frame

    @staticmethod
    def _point_in_polygons(point: Tuple[float, float], polys) -> bool:
        if not polys or len(polys) == 0:
            return False
        # Single polygon: [[x, y], ...]
        first = polys[0]
        if first and isinstance(first[0], (int, float, np.number)):
            if len(polys) >= 3:
                arr = np.array(polys, dtype=np.int32)
                return cv2.pointPolygonTest(arr, point, False) >= 0
            return False
        # List of polygons: [[[x, y], ...], ...]
        for poly in polys:
            if len(poly) >= 3:
                arr = np.array(poly, dtype=np.int32)
                if cv2.pointPolygonTest(arr, point, False) >= 0:
                    return True
        return False

    def get_vehicle_zone(self, vehicle_bbox: Union[List[float], np.ndarray]) -> Optional[str]:
        """
        Determines which spatial zone the vehicle's bottom-center contact point occupies:
        'exit_zone', 'intersection', 'approach_lane', or None.
        """
        x_center = (float(vehicle_bbox[0]) + float(vehicle_bbox[2])) / 2.0
        y_bottom = float(vehicle_bbox[3])
        point = (x_center, y_bottom)

        if self._point_in_polygons(point, self.exit_zones):
            return 'exit_zone'
        if self._point_in_polygons(point, self.intersection):
            return 'intersection'
        if self._point_in_polygons(point, self.lanes):
            return 'approach_lane'
        return None

    def get_vehicle_position(self, vehicle_bbox: Union[List[float], np.ndarray]) -> Optional[Union[int, str]]:
        """Backward-compatible wrapper returning zone label or lane index."""
        x_center = (float(vehicle_bbox[0]) + float(vehicle_bbox[2])) / 2.0
        y_bottom = float(vehicle_bbox[3])
        point = (x_center, y_bottom)

        if self._point_in_polygons(point, self.exit_zones):
            return 'exit_zone'
        if self._point_in_polygons(point, self.intersection):
            return 'intersection'
        for i, lane in enumerate(self.lanes):
            if len(lane) >= 3:
                if cv2.pointPolygonTest(np.array(lane, dtype=np.int32), point, False) >= 0:
                    return i
        return None

    def check_lane_to_intersection(
        self,
        vehicle_id: int,
        vehicle_bbox: Union[List[float], np.ndarray],
        light_is_red: bool = False,
        frame_number: int = 0,
        fps: float = 15.0,
    ) -> bool:
        """Evaluates whether `vehicle_id` committed a red-light violation on this frame."""
        if frame_number != self._last_light_update_frame and frame_number > 0:
            self.update_light_state('red' if light_is_red else 'green', frame_number, fps)

        zone = self.get_vehicle_zone(vehicle_bbox)
        bottom_center = (
            (float(vehicle_bbox[0]) + float(vehicle_bbox[2])) / 2.0,
            float(vehicle_bbox[3]),
        )

        # 1. Standard mode (immediate red enforcement)
        if not self.crossing_mode:
            if zone == 'approach_lane':
                self._seen_in_lane[vehicle_id] = True
                if light_is_red:
                    self._seen_in_lane_during_red[vehicle_id] = True
                return False

            if zone in ('intersection', 'exit_zone'):
                if self._entered_intersection.get(vehicle_id, False):
                    return False
                if self._seen_in_lane.get(vehicle_id, False):
                    self._entered_intersection[vehicle_id] = True
                    self._entry_light_state[vehicle_id] = 'red' if light_is_red else 'green'
                    if light_is_red and vehicle_id not in self._violation_reported:
                        self._violation_reported[vehicle_id] = True
                        return True
            return False

        # 2. 3/4-Way Multi-Lane Crossing Mode (3s Clearance + Exit Zone / Idle Box Trigger)
        in_clearance = self.is_in_clearance(frame_number)
        effective_red = bool(light_is_red and not in_clearance)

        if zone == 'approach_lane':
            self._seen_in_lane[vehicle_id] = True
            self._last_lane_frame[vehicle_id] = frame_number
            self._seen_in_lane_during_red[vehicle_id] = effective_red
            return False

        if self._legal_crosser.get(vehicle_id, False) or self._violation_reported.get(vehicle_id, False):
            return False

        if zone == 'intersection':
            if not self._seen_in_lane_during_red.get(vehicle_id, False):
                # Left the Approach Lane while GREEN or during the 3.0s clearance window
                self._legal_crosser[vehicle_id] = True
                return False

            self._entered_intersection[vehicle_id] = True
            self._intersection_frames[vehicle_id] = self._intersection_frames.get(vehicle_id, 0) + 1
            pts = self._recent_intersection_pts.setdefault(vehicle_id, deque(maxlen=25))
            pts.append(bottom_center)

            # Without Exit Zones, require >= 2 frames inside intersection during effective_red (jitter guard)
            if len(self.exit_zones) == 0:
                if effective_red and self._intersection_frames[vehicle_id] >= 2:
                    self._violation_reported[vehicle_id] = True
                    return True
                return False

            # With Exit Zones (Turn Green / All-Red Blocking): check if vehicle is standing idle inside box
            if effective_red and self._intersection_frames[vehicle_id] >= 22 and len(pts) >= 16:
                prev_pt = pts[-16]
                disp = math.hypot(bottom_center[0] - prev_pt[0], bottom_center[1] - prev_pt[1])
                if disp < 14.0:
                    self._violation_reported[vehicle_id] = True
                    return True
            return False

        if zone == 'exit_zone':
            if self._seen_in_lane_during_red.get(vehicle_id, False) and effective_red:
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
        fps: float = 15.0,
    ) -> List[ViolationEvent]:
        """Evaluates all tracked vehicles in the current frame and returns newly triggered ViolationEvents."""
        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return []

        self.update_light_state(light_state, frame_number, fps)
        light_is_red = (light_state == 'red')
        violations: List[ViolationEvent] = []

        for i in range(len(tracked_detections)):
            class_id = int(tracked_detections.class_id[i])
            if allowed_class_ids and class_id not in allowed_class_ids:
                continue

            tracker_id = int(tracked_detections.tracker_id[i])
            bbox = [float(c) for c in tracked_detections.xyxy[i]]
            conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else None

            if self.check_lane_to_intersection(
                tracker_id,
                bbox,
                light_is_red=light_is_red,
                frame_number=frame_number,
                fps=fps,
            ):
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
        """Reset all internal tracking states and clearance timers."""
        self._confirmed_light_state = None
        self._red_clearance_until_frame = 0
        self._last_light_update_frame = -1
        self._seen_in_lane.clear()
        self._seen_in_lane_during_red.clear()
        self._last_lane_frame.clear()
        self._entered_intersection.clear()
        self._legal_crosser.clear()
        self._intersection_frames.clear()
        self._recent_intersection_pts.clear()
        self._entry_light_state.clear()
        self._violation_reported.clear()
