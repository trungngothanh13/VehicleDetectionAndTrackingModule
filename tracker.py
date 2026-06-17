import supervision as sv
import numpy as np

class VehicleTracker:
    def __init__(self, track_thresh, match_thresh, track_buffer):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh
        )
        self.tracked_objects = {}  # Store info about tracked vehicles
        
    def update(self, detections):
        """
        Update tracker with new detections
        Returns: tracked detections with IDs
        """
        if len(detections) == 0:
            # Return empty supervision Detections object
            empty_detections = sv.Detections.empty()
            return self.tracker.update_with_detections(empty_detections)
        
        # Convert detections to supervision format
        detections_array = np.array(detections)
        xyxy = detections_array[:, :4]
        confidence = detections_array[:, 4]
        class_id = detections_array[:, 5].astype(int)
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Update tracker
        tracked = self.tracker.update_with_detections(sv_detections)
        
        # Update tracked vehicles info
        for i, tracker_id in enumerate(tracked.tracker_id):
            if tracker_id not in self.tracked_objects:
                self.tracked_objects[tracker_id] = {
                    'class_id': tracked.class_id[i],
                    'first_seen': True
                }
            else:
                self.tracked_objects[tracker_id]['first_seen'] = False
        
        return tracked
    
    def get_total_tracked(self):
        """Get total number of unique vehicles tracked"""
        return len(self.tracked_objects)
