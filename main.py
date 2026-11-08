import os
import argparse
from core.config import Config
from core.pipeline import TrafficPipeline
from ingestion import (
    VideoFileReader,
    RTSPStreamReader,
    SimulatedCameraReader,
    WebcamReader,
    WebSocketFrameReader,
)
from spatial import ZoneChecker
from traffic_light import TrafficLightDetector
from visualization import Visualizer
from trackers import create_tracker
from sinks import FileSink, WebSocketSink, PreviewSink
from zone_drawer import ZoneDrawer


def parse_args():
    parser = argparse.ArgumentParser(
        description="VDAT: Vehicle Detection, Tracking, and Red-Light Violation Pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input source: video file path, RTSP URL, 'webcam' / camera index (e.g. '0'), or 'ws' (web-fed video/webcam stream)",
    )
    parser.add_argument(
        "--webcam",
        "-w",
        nargs="?",
        const="0",
        default=None,
        help="Use laptop/USB hardware camera as input (optional index/path, defaults to 0)",
    )
    parser.add_argument(
        "--web-input",
        action="store_true",
        help="Ingest live video footage or laptop webcam frames streamed from the VDAT Web UI over WebSocket",
    )
    parser.add_argument(
        "--tracker",
        "-t",
        type=str,
        default=None,
        choices=["botsort", "bytetrack", "deepsort"],
        help="Tracking algorithm",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Detector model weights file (e.g. yolo26l.pt)",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=None,
        help="Limit number of frames to process (for fast testing)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live OpenCV window preview",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable saving output MP4 video",
    )
    parser.add_argument(
        "--ws",
        action="store_true",
        help="Enable WebSocket streaming to api.VDAT",
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default=None,
        help="WebSocket URL for api.VDAT backend",
    )
    parser.add_argument(
        "--simulate-camera",
        "-s",
        action="store_true",
        help="Simulate live camera ingestion (real-time FPS pacing, frame dropping)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop input video endlessly when simulating camera",
    )
    parser.add_argument(
        "--draw-zones",
        action="store_true",
        help="Launch interactive zone drawing GUI tool, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Apply CLI overrides to Config
    if args.input:
        config.INPUT_VIDEO = args.input
    if args.webcam is not None:
        config.INPUT_VIDEO = f"webcam:{args.webcam}"
    if args.web_input:
        config.INPUT_VIDEO = "ws"
        config.ENABLE_WEBSOCKET_STREAM = True
    if args.tracker:
        config.TRACKER_TYPE = args.tracker
    if args.model:
        config.MODEL_NAME = args.model
    if args.preview:
        config.SHOW_LIVE_PREVIEW = True
    if args.no_video:
        config.SAVE_OUTPUT_VIDEO = True if False else False
    if args.ws:
        config.ENABLE_WEBSOCKET_STREAM = True
    if args.ws_url:
        config.API_WS_URL = args.ws_url
    if args.simulate_camera:
        config.SIMULATE_CAMERA = True
    if args.loop:
        config.LOOP_VIDEO = True
    if args.draw_zones:
        config.ENABLE_ZONE_DRAWER = True
    if args.max_frames is not None:
        config.MAX_FRAMES = args.max_frames

    # If zone drawer requested, run GUI tool and exit
    if config.ENABLE_ZONE_DRAWER:
        ZoneDrawer().draw_zones(config.INPUT_VIDEO)
        return

    # 1. Source Reader Selection (Web-Fed Stream vs Laptop Webcam vs RTSP vs Simulated Camera vs File)
    input_source = str(config.INPUT_VIDEO).strip()
    is_ws_ingest = input_source.lower() in ("ws", "web", "websocket")
    is_webcam = (
        input_source.isdigit()
        or input_source.lower() == "webcam"
        or input_source.lower().startswith("webcam:")
        or input_source.startswith("/dev/video")
    )
    is_live_stream = input_source.startswith(("rtsp://", "http://", "https://"))

    ingest_mode = "local"
    if is_ws_ingest:
        print("[Ingestion] Web-Fed Stream Mode: Waiting for video/webcam frames from VDAT Web UI over WebSocket")
        config.ENABLE_WEBSOCKET_STREAM = True
        ingest_mode = "ws"
        reader = WebSocketFrameReader(width=854, height=480, fps=30)
    elif is_webcam:
        if input_source.lower().startswith("webcam:"):
            cam_dev = input_source.split(":", 1)[1] or "0"
        elif input_source.lower() == "webcam":
            cam_dev = "0"
        else:
            cam_dev = input_source
        print(f"[Ingestion] Opening laptop/hardware camera source: {cam_dev}")
        ingest_mode = "webcam"
        reader = WebcamReader(device=cam_dev, target_width=854, target_height=480, target_fps=30)
    elif is_live_stream:
        ingest_mode = "rtsp"
        reader = RTSPStreamReader(input_source)
    elif config.SIMULATE_CAMERA:
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Input video file not found: {input_source}")
        print(f"[Ingestion] Simulating live camera from {input_source} (Loop: {config.LOOP_VIDEO})")
        ingest_mode = "simulated_camera"
        reader = SimulatedCameraReader(input_source, loop=config.LOOP_VIDEO)
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

    file_sink = FileSink(
        output_video_path=config.OUTPUT_VIDEO,
        violation_log_path=config.VIOLATION_LOG,
        evidence_dir=getattr(config, 'EVIDENCE_DIR', 'output/evidence'),
        save_video=config.SAVE_OUTPUT_VIDEO and not is_ws_ingest,
        save_log=config.SAVE_VIOLATION_LOG,
        save_evidence=getattr(config, 'SAVE_EVIDENCE_SNAPSHOTS', True),
    )
    sinks.append(file_sink)

    ws_sink = None
    if config.ENABLE_WEBSOCKET_STREAM:
        ws_sink = WebSocketSink(
            ws_url=config.API_WS_URL,
            camera_id=config.CAMERA_ID,
            jpeg_quality=getattr(config, 'JPEG_QUALITY', 65),
            target_fps=getattr(config, 'STREAM_TARGET_FPS', 15.0),
            max_width=getattr(config, 'STREAM_MAX_WIDTH', 960),
            evidence_dir=config.EVIDENCE_DIR,
            zones_file=config.ZONES_FILE,
            on_ingest_frame=reader.push_frame if isinstance(reader, WebSocketFrameReader) else None,
            ingest_mode=ingest_mode,
        )
        sinks.append(ws_sink)

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

    if ws_sink is not None:
        ws_sink.on_zone_update = pipeline.reload_zones

        def handle_remote_source_switch(cmd: dict) -> None:
            source_type = cmd.get("sourceType", "video_file")
            source_val = str(cmd.get("sourceValue", "test_1.mp4")).strip()
            loop_flag = bool(cmd.get("loop", True))

            if source_type in ("web_stream", "ws", "laptop_webcam_web", "custom_video_web"):
                if isinstance(pipeline.reader, WebSocketFrameReader) and pipeline._pending_reader is None:
                    ws_sink.on_ingest_frame = pipeline.reader.push_frame
                    ws_sink.ingest_mode = "ws"
                    print(f"🎬 [AI Source] Active WebSocketFrameReader ready for Web-Fed Stream: {source_val}")
                else:
                    new_ws_reader = WebSocketFrameReader(width=854, height=480, fps=30)
                    ws_sink.on_ingest_frame = new_ws_reader.push_frame
                    ws_sink.ingest_mode = "ws"
                    pipeline.request_reader_switch(new_ws_reader, f"Web-Fed Stream ({source_val})")
            elif source_type in ("webcam", "laptop_webcam_host"):
                dev = source_val if source_val else "0"
                ws_sink.on_ingest_frame = None
                ws_sink.ingest_mode = "webcam"
                new_cam_reader = WebcamReader(device=dev, target_width=854, target_height=480, target_fps=30)
                pipeline.request_reader_switch(new_cam_reader, f"Host Laptop Camera ({dev})")
            elif source_type == "server_video":
                safe_name = os.path.basename(source_val)
                if os.path.exists(safe_name):
                    ws_sink.on_ingest_frame = None
                    ws_sink.ingest_mode = "simulated_camera"
                    new_sim_reader = SimulatedCameraReader(safe_name, loop=loop_flag)
                    pipeline.request_reader_switch(new_sim_reader, f"Server Video ({safe_name})")

        ws_sink.on_switch_source = handle_remote_source_switch

    pipeline.run()


if __name__ == "__main__":
    main()
