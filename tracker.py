from pathlib import Path
import numpy as np
import supervision as sv
from boxmot import BotSORT


class VehicleTracker:
    def __init__(self, track_thresh, match_thresh, track_buffer, device='cuda'):
        # BoT-SORT: adds camera-motion compensation + Re-ID appearance features on top of ByteTrack's
        # IoU matching — significantly better for dense overlapping scenes (e.g. Vietnamese motorcycles)
        self.tracker = BotSORT(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Lightweight Re-ID model (~1MB auto-downloaded)
            device=device,
            half=False,
            track_high_thresh=track_thresh,
            track_low_thresh=0.10,         # Low-conf detections used for re-association of lost tracks
            new_track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self.tracked_objects = {}  # Store info about tracked vehicles

    def update(self, detections, frame):
        """
        Update tracker with new detections and the current frame (needed for Re-ID appearance features).
        Returns: sv.Detections with tracker_id populated
        """
        if len(detections) == 0:
            dets = np.empty((0, 6))
        else:
            dets = np.array(detections, dtype=np.float32)  # [x1, y1, x2, y2, conf, cls]

        # BoT-SORT returns: [x1, y1, x2, y2, track_id, conf, cls, det_index]
        tracks = self.tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0:
            return sv.Detections.empty()

        xyxy        = tracks[:, :4].astype(np.float32)
        tracker_ids = tracks[:, 4].astype(int)
        confidences = tracks[:, 5].astype(np.float32)
        class_ids   = tracks[:, 6].astype(int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
            tracker_id=tracker_ids,
        )

        # Update internal registry
        for i, tracker_id in enumerate(tracker_ids):
            if tracker_id not in self.tracked_objects:
                self.tracked_objects[tracker_id] = {
                    'class_id': class_ids[i],
                    'first_seen': True,
                }
            else:
                self.tracked_objects[tracker_id]['first_seen'] = False

        return sv_detections

    def get_total_tracked(self):
        """Get total number of unique vehicles tracked so far."""
        return len(self.tracked_objects)
