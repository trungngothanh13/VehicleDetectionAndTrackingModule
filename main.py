import cv2
import os
import threading
from queue import Queue
from config import Config
from zone_drawer import ZoneDrawer
from trackers import create_tracker


class FrameReader:
    """
    Reads video frames on a background thread so CPU decode never blocks GPU inference.
    Keeps a small buffer of pre-decoded frames ready.
    """
    def __init__(self, cap, buffer_size=8):
        self.cap = cap
        self.queue = Queue(maxsize=buffer_size)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            ret, frame = self.cap.read()
            self.queue.put((ret, frame))
            if not ret:
                break

    def read(self):
        return self.queue.get()


def process_video():
    config = Config()

    if config.ENABLE_ZONE_DRAWER:
        ZoneDrawer().draw_zones(config.INPUT_VIDEO)
        return

    from visualizer import Visualizer
    from zone_checker import ZoneChecker
    from traffic_light_detector import TrafficLightDetector

    # Ensure output directory exists
    output_dir = getattr(config, 'OUTPUT_DIR', 'output')
    os.makedirs(output_dir, exist_ok=True)
    if config.SAVE_OUTPUT_VIDEO and os.path.dirname(config.OUTPUT_VIDEO):
        os.makedirs(os.path.dirname(config.OUTPUT_VIDEO), exist_ok=True)
    if getattr(config, 'SAVE_VIOLATION_LOG', True) and os.path.dirname(config.VIOLATION_LOG):
        os.makedirs(os.path.dirname(config.VIOLATION_LOG), exist_ok=True)

    # Instantiate the selected tracker from config
    tracker = create_tracker(config)
    visualizer = Visualizer(config.DETECTION_CLASSES)

    zone_checker = None
    traffic_light_detector = None
    violations = set()
    violation_records = []

    if os.path.exists(config.ZONES_FILE):
        import json
        with open(config.ZONES_FILE, 'r') as f:
            zones_data = json.load(f)
        zone_checker = ZoneChecker(config.ZONES_FILE)
        if zones_data.get('traffic_lights') and len(zones_data['traffic_lights']) >= 2:
            traffic_light_detector = TrafficLightDetector(zones_data['traffic_lights'])

    # Open video
    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {config.INPUT_VIDEO}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{'='*60}")
    print(f"Processing:     {config.INPUT_VIDEO}")
    print(f"Resolution:     {width}x{height} @ {fps} FPS")
    print(f"Active Tracker: {config.TRACKER_TYPE.upper()} (Model: {config.MODEL_NAME})")
    print(f"Output Video:   {config.OUTPUT_VIDEO if config.SAVE_OUTPUT_VIDEO else 'Disabled'}")
    print(f"Violation Log:  {config.VIOLATION_LOG if getattr(config, 'SAVE_VIOLATION_LOG', True) else 'Disabled'}")
    print(f"{'='*60}\n")

    out = None
    if config.SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (width, height))

    # FrameReader runs on a background thread
    reader = FrameReader(cap)
    frame_count = 0

    while True:
        ret, frame = reader.read()
        if not ret:
            break

        frame_count += 1

        # Detect + track using configured tracker
        tracked = tracker.track(frame)

        # Traffic light state
        light_state = 'unknown'
        if traffic_light_detector:
            light_state = traffic_light_detector.detect(frame)

        # Zone violation checks
        if zone_checker and tracked.tracker_id is not None:
            light_is_red = (light_state == 'red')
            for i in range(len(tracked)):
                class_id = int(tracked.class_id[i])
                if class_id not in config.VIOLATION_CLASS_IDS:
                    continue
                tracker_id = int(tracked.tracker_id[i])
                bbox = tracked.xyxy[i]
                if zone_checker.check_lane_to_intersection(tracker_id, bbox, light_is_red):
                    if tracker_id not in violations:
                        violations.add(tracker_id)
                        class_name = tracker.get_class_name(class_id)
                        timestamp_sec = f"{frame_count / fps:.2f}s"
                        violation_records.append({
                            'id': tracker_id,
                            'frame': frame_count,
                            'timestamp': timestamp_sec,
                            'class': class_name
                        })
                        print(f"🚨 [VIOLATION] Vehicle #{tracker_id} ({class_name}) at Frame {frame_count} ({timestamp_sec})")

            frame = visualizer.draw_zones(frame, zone_checker.lanes, zone_checker.intersection)

        # Draw detections and violation status
        frame = visualizer.draw_detections(frame, tracked, tracker, violation_ids=violations)

        current_count = len(tracked) if tracked.tracker_id is not None else 0
        total_tracked = tracker.get_total_tracked()

        visualizer.set_violations_count(len(violations))
        visualizer.set_light_state(light_state)
        frame = visualizer.draw_statistics(frame, current_count, total_tracked, frame_count)

        if out:
            out.write(frame)

        if config.SHOW_LIVE_PREVIEW:
            cv2.imshow('Vehicle Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Vehicles={current_count}, Tracked={total_tracked}, Violations={len(violations)}, Light={light_state}")

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Write violation report to .txt file
    if getattr(config, 'SAVE_VIOLATION_LOG', True):
        with open(config.VIOLATION_LOG, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VEHICLE RED-LIGHT VIOLATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Input Video:      {config.INPUT_VIDEO}\n")
            f.write(f"Active Tracker:   {config.TRACKER_TYPE}\n")
            f.write(f"Detector Model:   {config.MODEL_NAME}\n")
            f.write(f"Total Frames:     {frame_count} (FPS: {fps})\n")
            f.write(f"Total Vehicles:   {tracker.get_total_tracked()}\n")
            f.write(f"Total Violations: {len(violation_records)}\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Vehicle ID':<12} | {'Frame':<8} | {'Timestamp':<12} | {'Class':<12}\n")
            f.write("-" * 60 + "\n")
            for r in violation_records:
                f.write(f"{r['id']:<12} | {r['frame']:<8} | {r['timestamp']:<12} | {r['class']:<12}\n")
            f.write("=" * 60 + "\n")

    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total unique vehicles tracked: {tracker.get_total_tracked()}")
    print(f"Total violations logged: {len(violation_records)}")
    if config.SAVE_OUTPUT_VIDEO:
        print(f"Output video saved: {config.OUTPUT_VIDEO}")
    if getattr(config, 'SAVE_VIOLATION_LOG', True):
        print(f"Violation report saved: {config.VIOLATION_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_video()