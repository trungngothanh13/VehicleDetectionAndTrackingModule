import os
import argparse
from core.config import Config
from core.pipeline import TrafficPipeline
from ingestion import VideoFileReader, RTSPStreamReader
from spatial import ZoneChecker
from traffic_light import TrafficLightDetector
from visualization import Visualizer
from trackers import create_tracker
from sinks import FileSink, WebSocketSink, PreviewSink
from zone_drawer import ZoneDrawer


def parse_args():
    parser = argparse.ArgumentParser(description="VDAT: Vehicle Detection, Tracking, and Red-Light Violation Pipeline")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input video file path or RTSP stream URL")
    parser.add_argument("--tracker", "-t", type=str, default=None, choices=["botsort", "bytetrack", "deepsort"], help="Tracking algorithm")
    parser.add_argument("--model", "-m", type=str, default=None, help="Detector model weights file (e.g. yolo26l.pt)")
    parser.add_argument("--max-frames", "-n", type=int, default=None, help="Limit number of frames to process (for fast testing)")
    parser.add_argument("--preview", action="store_true", help="Display live OpenCV window preview")
    parser.add_argument("--no-video", action="store_true", help="Disable saving output MP4 video")
    parser.add_argument("--ws", action="store_true", help="Enable WebSocket streaming to api.VDAT")
    parser.add_argument("--ws-url", type=str, default=None, help="WebSocket URL for api.VDAT backend")
    parser.add_argument("--draw-zones", action="store_true", help="Launch interactive zone drawing GUI tool, then exit")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Apply CLI overrides to Config
    if args.input:
        config.INPUT_VIDEO = args.input
    if args.tracker:
        config.TRACKER_TYPE = args.tracker
    if args.model:
        config.MODEL_NAME = args.model
    if args.preview:
        config.SHOW_LIVE_PREVIEW = True
    if args.no_video:
        config.SAVE_OUTPUT_VIDEO = False
    if args.ws:
        config.ENABLE_WEBSOCKET_STREAM = True
    if args.ws_url:
        config.API_WS_URL = args.ws_url
    if args.draw_zones:
        config.ENABLE_ZONE_DRAWER = True
    if args.max_frames is not None:
        config.MAX_FRAMES = args.max_frames

    # If zone drawer requested, run GUI tool and exit
    if config.ENABLE_ZONE_DRAWER:
        ZoneDrawer().draw_zones(config.INPUT_VIDEO)
        return

    # 1. Source Reader (File vs RTSP)
    input_source = config.INPUT_VIDEO
    is_live_stream = str(input_source).startswith(("rtsp://", "http://", "https://"))

    if is_live_stream:
        reader = RTSPStreamReader(input_source)
    else:
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Input video file not found: {input_source}")
        reader = VideoFileReader(input_source)

    # 2. Components
    tracker = create_tracker(config)
    visualizer = Visualizer(config.DETECTION_CLASSES)

    # 3. Spatial & Light Detection
    zone_checker = None
    light_detector = None
    if os.path.exists(config.ZONES_FILE):
        zone_checker = ZoneChecker(config.ZONES_FILE)
        light_detector = TrafficLightDetector.from_file(config.ZONES_FILE)

    # 4. Sinks
    sinks = []

    # File Sink (MP4 video, snapshots, violation report)
    file_sink = FileSink(
        output_video_path=config.OUTPUT_VIDEO,
        violation_log_path=config.VIOLATION_LOG,
        evidence_dir=getattr(config, 'EVIDENCE_DIR', 'output/evidence'),
        save_video=config.SAVE_OUTPUT_VIDEO,
        save_log=config.SAVE_VIOLATION_LOG,
        save_evidence=getattr(config, 'SAVE_EVIDENCE_SNAPSHOTS', True),
    )
    sinks.append(file_sink)

    # Optional WebSocket Sink (api.VDAT streaming)
    if config.ENABLE_WEBSOCKET_STREAM:
        ws_sink = WebSocketSink(
            ws_url=config.API_WS_URL,
            camera_id=config.CAMERA_ID,
            jpeg_quality=getattr(config, 'JPEG_QUALITY', 75),
        )
        sinks.append(ws_sink)

    # Optional Desktop Preview Sink
    if config.SHOW_LIVE_PREVIEW:
        sinks.append(PreviewSink())

    # 5. Pipeline Execution
    pipeline = TrafficPipeline(
        reader=reader,
        tracker=tracker,
        visualizer=visualizer,
        sinks=sinks,
        config=config,
        light_detector=light_detector,
        zone_checker=zone_checker,
        max_frames=config.MAX_FRAMES,
    )

    pipeline.run()


if __name__ == "__main__":
    main()