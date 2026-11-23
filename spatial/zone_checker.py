import cv2
import json
import numpy as np
from typing import List, Dict, Optional, Union, Tuple


class ZoneChecker:
    """
    Manages Approach Lanes, Intersection boundaries, and Exit Zones.

    Supports two independent enforcement dimensions:
    1. Violation Trigger Mode (`crossing_mode`):
       - Standard Stop-Line mode (`crossing_mode=False`, or no Exit Zones defined):
         Flags `Approach Lane -> Intersection/Exit Zone` when light is effectively red.
       - Split-Phase Signal (Require Exit Zone) mode (`crossing_mode=True` with Exit Zones):
         Requires the full `Approach Lane -> Intersection -> Exit Zone` trajectory during
         effective red, eliminating false flags both on protected turns and on perpendicular
         cross-traffic turning from `Intersection -> Exit Zone`.
    2. Green->Red Clearance Grace (`clearance_seconds`):
       - `0.0`: disabled (strict immediate red).
       - `> 0.0` (e.g. `5.0`s, adjustable 1-15s): grants a post-green grace window in EITHER
         mode so vehicles already committed into the intersection in slow traffic can finish
         clearing safely without being flagged.
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
        self.clearance_seconds = max(0.0, float(raw_clearance if raw_clearance is not None else 0.0))

        # Signal phase & green->red clearance timer state
        self._confirmed_light_state: Optional[str] = None
        self._red_clearance_until_frame: int = 0
        self._last_light_update_frame: int = -1

        # Track state history per vehicle ID
        self._seen_in_lane: Dict[int, bool] = {}
        self._seen_in_lane_during_red: Dict[int, bool] = {}
        self._entered_intersection: Dict[int, bool] = {}
        self._intersection_red_frames: Dict[int, int] = {}
        self._legal_crosser: Dict[int, bool] = {}
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
        return bool(self.clearance_seconds > 0 and frame_number < self._red_clearance_until_frame)

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

    @staticmethod
    def _get_footprint_points(
        vehicle_bbox: Union[List[float], np.ndarray]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Returns (core_pts, lane_pts) for relaxed lower-footprint zone matching."""
        x1, y1, x2, y2 = (
            float(vehicle_bbox[0]),
            float(vehicle_bbox[1]),
            float(vehicle_bbox[2]),
            float(vehicle_bbox[3]),
        )
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        xc = (x1 + x2) / 2.0

        core_pts = [
            (xc, y2),
            (xc, y2 - 0.10 * h),
        ]
        lane_pts = [
            (xc, y2),
            (xc, y2 - 0.12 * h),
            (xc - 0.18 * w, y2 - 0.08 * h),
            (xc + 0.18 * w, y2 - 0.08 * h),
        ]
        return core_pts, lane_pts

    def get_vehicle_zone(self, vehicle_bbox: Union[List[float], np.ndarray]) -> Optional[str]:
        """
        Determines which spatial zone the vehicle's lower footprint occupies:
        'exit_zone', 'intersection', 'approach_lane', or None.
        """
        core_pts, lane_pts = self._get_footprint_points(vehicle_bbox)

        if any(self._point_in_polygons(p, self.exit_zones) for p in core_pts):
            return 'exit_zone'
        if any(self._point_in_polygons(p, self.intersection) for p in core_pts):
            return 'intersection'
        if any(self._point_in_polygons(p, self.lanes) for p in lane_pts):
            return 'approach_lane'
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

        # Early-exit so a legal crosser or flagged violator never flips state due to bbox jitter
        if self._violation_reported.get(vehicle_id, False) or self._legal_crosser.get(vehicle_id, False):
            return False

        zone = self.get_vehicle_zone(vehicle_bbox)
        in_clearance = self.is_in_clearance(frame_number)
        effective_red = bool(light_is_red and not in_clearance)

        if zone == 'approach_lane':
            self._seen_in_lane[vehicle_id] = True
            if effective_red:
                self._seen_in_lane_during_red[vehicle_id] = True
            return False

        # 1. Split-Phase Signal (Require Exit Zone) mode
        if self.crossing_mode and len(self.exit_zones) > 0:
            if zone == 'intersection':
                if not effective_red:
                    self._legal_crosser[vehicle_id] = True
                    return False
                if not self._seen_in_lane.get(vehicle_id, False):
                    return False
                if self.clearance_seconds > 0 and not self._seen_in_lane_during_red.get(vehicle_id, False):
                    self._legal_crosser[vehicle_id] = True
                    return False
                self._entered_intersection[vehicle_id] = True
                self._intersection_red_frames[vehicle_id] = (
                    self._intersection_red_frames.get(vehicle_id, 0) + 1
                )
                return False

            if zone == 'exit_zone':
                if not effective_red:
                    self._legal_crosser[vehicle_id] = True
                    return False
                came_from_lane = (
                    self._seen_in_lane_during_red.get(vehicle_id, False)
                    if self.clearance_seconds > 0
                    else self._seen_in_lane.get(vehicle_id, False)
                )
                if came_from_lane and self._entered_intersection.get(vehicle_id, False):
                    self._violation_reported[vehicle_id] = True
                    return True
                return False

            return False

        # 2. Standard Stop-Line mode
        if zone in ('intersection', 'exit_zone'):
            if not effective_red:
                self._legal_crosser[vehicle_id] = True
                return False

            if self._seen_in_lane.get(vehicle_id, False):
                if (
                    self.clearance_seconds > 0
                    and not self._seen_in_lane_during_red.get(vehicle_id, False)
                ):
                    self._legal_crosser[vehicle_id] = True
                    return False

                self._entered_intersection[vehicle_id] = True
                self._intersection_red_frames[vehicle_id] = (
                    self._intersection_red_frames.get(vehicle_id, 0) + 1
                )
                if zone == 'exit_zone' or self._intersection_red_frames[vehicle_id] >= 2:
                    self._violation_reported[vehicle_id] = True
                    return True

        return False

    def reset(self) -> None:
        """Reset all internal tracking states and clearance timers."""
        self._confirmed_light_state = None
        self._red_clearance_until_frame = 0
        self._last_light_update_frame = -1
        self._seen_in_lane.clear()
        self._seen_in_lane_during_red.clear()
        self._entered_intersection.clear()
        self._intersection_red_frames.clear()
        self._legal_crosser.clear()
        self._violation_reported.clear()
