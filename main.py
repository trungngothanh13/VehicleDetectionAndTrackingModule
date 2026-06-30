import cv2
import os
from config import Config
from vehicle_model import VehicleModel
from visualizer import Visualizer
from zone_checker import ZoneChecker
from zone_drawer import ZoneDrawer
from traffic_light_detector import TrafficLightDetector

def process_video():
    config = Config()

    if config.ENABLE_ZONE_DRAWER:
        ZoneDrawer().draw_zones(config.INPUT_VIDEO)
        return

    model = VehicleModel(
        config.MODEL_NAME,
        config.DETECTION_CLASSES,
        config.CONFIDENCE_THRESHOLD,
        config.IMGSZ,
    )
    visualizer = Visualizer(config.DETECTION_CLASSES)

    zone_checker = None
    traffic_light_detector = None
    violations = set()
    if os.path.exists(config.ZONES_FILE):
        import json
        with open(config.ZONES_FILE, 'r') as f:
            zones_data = json.load(f)
        zone_checker = ZoneChecker(config.ZONES_FILE)
        if zones_data.get('traffic_lights') and len(zones_data['traffic_lights']) >= 2:
            traffic_light_detector = TrafficLightDetector(zones_data['traffic_lights'])

    # Open video
    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing: {config.INPUT_VIDEO}")
    print(f"Resolution: {width}x{height} @ {fps} FPS\n")

    out = None
    if config.SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect + track in one pass (Ultralytics BoT-SORT)
        tracked = model.track(frame)

        # Traffic light state
        light_state = 'unknown'
        if traffic_light_detector:
            light_state = traffic_light_detector.detect(frame)

        # Draw detections
        frame = visualizer.draw_detections(frame, tracked, model)

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
                    violations.add(tracker_id)

            frame = visualizer.draw_zones(frame, zone_checker.lanes, zone_checker.intersection)

        current_count = len(tracked) if tracked.tracker_id is not None else 0
        total_tracked = model.get_total_tracked()

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
            print(f"Frame {frame_count}: Current={current_count}, Total Tracked={total_tracked}, Violations={len(violations)}, Light={light_state}")

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total unique vehicles tracked: {model.get_total_tracked()}")
    if config.SAVE_OUTPUT_VIDEO:
        print(f"Output saved: {config.OUTPUT_VIDEO}")
    print(f"{'='*50}")


if __name__ == "__main__":
    process_video()