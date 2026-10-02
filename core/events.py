from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import time


@dataclass
class FrameTelemetry:
    """Telemetry data generated on each processed video frame."""
    frame_number: int
    timestamp: float           # Video timestamp in seconds
    fps: float
    vehicle_count: int         # Vehicles present in current frame
    total_tracked: int         # Cumulative unique vehicle IDs seen so far
    light_state: str           # 'red', 'green', or 'unknown'
    violations_count: int      # Cumulative violations confirmed so far

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ViolationEvent:
    """Structured event generated when a vehicle commits a red-light violation."""
    camera_id: str
    track_id: int
    vehicle_class: str
    class_id: int
    frame_number: int
    timestamp: str             # e.g., "14.20s" or ISO format
    light_state: str           # Must be 'red'
    bbox: List[float]          # [x1, y1, x2, y2]
    confidence: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    snapshot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
