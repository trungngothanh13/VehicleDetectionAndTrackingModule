import os
import shutil
import threading
import argparse
from typing import List, Optional
from ultralytics import YOLO
from core.config import Config
from core.model_resolver import apply_model_classes, resolve_profile_model
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
from trackers.base_tracker import BatchModelRunner
from sinks import FileSink, WebSocketSink, PreviewSink
from zone_drawer import ZoneDrawer


def parse_args():
    parser = argparse.ArgumentParser(
        description="VDAT: Vehicle Detection, Tracking, and Red-Light Violation Pipeline"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera IDs to run concurrently on one shared YOLO model (e.g. --cameras cam-01 cam-02 cam-03)",
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
        help="Detector model weights file override (default: fine-tuned weights/best.pt, or yolo26l.pt with --legacy)",
    )
    parser.add_argument(
        "--legacy",
        "--stock",
        action="store_true",
        dest="legacy",
        help="Start in legacy stock COCO YOLO26l mode (yolo26l.pt) instead of fine-tuned VDAT weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold override (default: 0.45 for trained, 0.30 for legacy stock)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO inference input size (default: 640)",
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


def resolve_camera_ids(args, config: Config) -> List[str]:
    """Parses --cameras (space or comma separated), defaulting to 6 web slots (cam-01..cam-06) in --web-input mode."""
    if args.cameras:
        ids: List[str] = []
        for item in args.cameras:
            for part in str(item).split(","):
                cleaned = part.strip()
                if cleaned and cleaned not in ids:
                    ids.append(cleaned)
        if ids:
            return ids

    input_source = str(config.INPUT_VIDEO).strip().lower()
    if args.web_input or input_source in ("ws", "web", "websocket"):
        return ["cam-01", "cam-02", "cam-03", "cam-04", "cam-05", "cam-06"]

    return [config.CAMERA_ID]


def resolve_camera_zones_file(camera_id: str, default_zones_file: str) -> str:
    """Each camera session maintains its own isolated zones JSON file under output/zones/."""
    zones_dir = os.path.join("output", "zones")
    os.makedirs(zones_dir, exist_ok=True)
    cam_zones_path = os.path.join(zones_dir, f"{camera_id}.json")
    if not os.path.exists(cam_zones_path) and camera_id == "cam-01" and os.path.exists(default_zones_file):
        shutil.copyfile(default_zones_file, cam_zones_path)
    return cam_zones_path


def purge_temp_videos_dir(temp_dir: str) -> None:
    """Deletes any stale temporary uploaded video files on startup or shutdown."""
    if not temp_dir or not os.path.exists(temp_dir):
        return
    for name in os.listdir(temp_dir):
        full_path = os.path.join(temp_dir, name)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass


def build_session_reader(camera_id: str, config: Config, is_primary: bool):
    """Builds the initial frame reader for a single camera session."""
    input_source = str(config.INPUT_VIDEO).strip()
    is_ws_ingest = input_source.lower() in ("ws", "web", "websocket")

    # Additional camera slots wait for WebSocket frames when started alongside a primary local source
    if is_ws_ingest or not is_primary:
        print(f"[{camera_id}] Ingestion: WebSocketFrameReader ready (waiting for Web UI frames @ 15 FPS)")
        return WebSocketFrameReader(width=854, height=480, fps=15), "ws", True

    is_webcam = (
        input_source.isdigit()
        or input_source.lower() == "webcam"
        or input_source.lower().startswith("webcam:")
        or input_source.startswith("/dev/video")
    )
    is_live_stream = input_source.startswith(("rtsp://", "http://", "https://"))

    if is_webcam:
        if input_source.lower().startswith("webcam:"):
            cam_dev = input_source.split(":", 1)[1] or "0"
        elif input_source.lower() == "webcam":
            cam_dev = "0"
        else:
            cam_dev = input_source
        print(f"[{camera_id}] Opening laptop/hardware camera source: {cam_dev}")
        return WebcamReader(device=cam_dev, target_width=854, target_height=480, target_fps=30), "webcam", False

    if is_live_stream:
        return RTSPStreamReader(input_source), "rtsp", False

    if config.SIMULATE_CAMERA:
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Input video file not found: {input_source}")
        print(f"[{camera_id}] Simulating live camera from {input_source} (Loop: {config.LOOP_VIDEO})")
        return SimulatedCameraReader(input_source, loop=config.LOOP_VIDEO), "simulated_camera", False

    if not os.path.exists(input_source):
        raise FileNotFoundError(f"Input video file not found: {input_source}")
    return VideoFileReader(input_source), "local", False


def create_camera_session(
    camera_id: str,
    config: Config,
    shared_model: Optional[YOLO],
    model_lock: threading.Lock,
    is_primary: bool,
    enable_preview: bool,
    on_switch_model=None,
    on_switch_tracker=None,
) -> TrafficPipeline:
    """Instantiates an isolated per-camera pipeline sharing one GPU YOLO model."""
    reader, ingest_mode, is_ws_ingest = build_session_reader(camera_id, config, is_primary=is_primary)

    tracker = create_tracker(
        config,
        shared_model=shared_model,
        model_lock=model_lock,
        camera_id=camera_id,
    )
    visualizer = Visualizer(config.DETECTION_CLASSES)

    cam_zones_file = resolve_camera_zones_file(camera_id, config.ZONES_FILE)
    zone_checker = None
    light_detector = None
    if os.path.exists(cam_zones_file):
        zone_checker = ZoneChecker(cam_zones_file)
        light_detector = TrafficLightDetector.from_file(cam_zones_file)

    sinks = []
    output_video_path = (
        os.path.join(config.OUTPUT_DIR, f"{config.TRACKER_TYPE}_{camera_id}.mp4")
        if config.SAVE_OUTPUT_VIDEO and not is_ws_ingest
        else None
    )
    violation_log_path = (
        os.path.join(config.OUTPUT_DIR, f"{config.TRACKER_TYPE}_{camera_id}.txt")
        if config.SAVE_VIOLATION_LOG and not is_ws_ingest
        else None
    )

    file_sink = FileSink(
        output_video_path=output_video_path,
        violation_log_path=violation_log_path,
        evidence_dir=getattr(config, 'EVIDENCE_DIR', 'output/evidence'),
        save_video=config.SAVE_OUTPUT_VIDEO and not is_ws_ingest,
        save_log=config.SAVE_VIOLATION_LOG and not is_ws_ingest,
        save_evidence=getattr(config, 'SAVE_EVIDENCE_SNAPSHOTS', True),
    )
    sinks.append(file_sink)

    ws_sink = None
    if config.ENABLE_WEBSOCKET_STREAM:
        ws_sink = WebSocketSink(
            ws_url=config.API_WS_URL,
            camera_id=camera_id,
            jpeg_quality=getattr(config, 'JPEG_QUALITY', 66),
            target_fps=getattr(config, 'STREAM_TARGET_FPS', 24.0),
            max_width=getattr(config, 'STREAM_MAX_WIDTH', 854),
            preview_interval_sec=getattr(config, 'PREVIEW_INTERVAL_SEC', 60.0),
            evidence_dir=config.EVIDENCE_DIR,
            zones_file=cam_zones_file,
            temp_video_dir=getattr(config, 'TEMP_VIDEO_DIR', 'temp_videos'),
            on_ingest_frame=reader.push_frame if isinstance(reader, WebSocketFrameReader) else None,
            ingest_mode=ingest_mode,
        )
        if isinstance(reader, WebSocketFrameReader):
            reader._on_drop_frame = ws_sink.send_frame_ack
        sinks.append(ws_sink)

    if enable_preview:
        sinks.append(PreviewSink())

    pipeline = TrafficPipeline(
        reader=reader,
        tracker=tracker,
        visualizer=visualizer,
        sinks=sinks,
        config=config,
        light_detector=light_detector,
        zone_checker=zone_checker,
        max_frames=config.MAX_FRAMES,
        camera_id=camera_id,
    )

    if ws_sink is not None:
        ws_sink.on_zone_update = pipeline.reload_zones

        def handle_remote_source_switch(cmd: dict) -> None:
            source_type = cmd.get("sourceType", "video_file")
            source_val = str(cmd.get("sourceValue", "test_1.mp4")).strip()
            loop_flag = bool(cmd.get("loop", True))

            if source_type == "uploaded_video":
                local_path = str(cmd.get("localPath", "")).strip()
                if not local_path or not os.path.exists(local_path):
                    ws_sink.send_source_status(
                        "error",
                        source_type="uploaded_video",
                        label=source_val,
                        error=f"Downloaded video file missing: {local_path}",
                    )
                    return

                def handle_eof() -> None:
                    idle_reader = WebSocketFrameReader(
                        width=854,
                        height=480,
                        fps=15,
                        on_drop_frame=ws_sink.send_frame_ack,
                    )
                    ws_sink.on_ingest_frame = idle_reader.push_frame
                    ws_sink.ingest_mode = "ws"
                    pipeline.request_reader_switch(idle_reader, f"Standby (Video Ended: {source_val})")
                    ws_sink.send_source_status("ended", source_type="uploaded_video", label=source_val)

                new_sim_reader = SimulatedCameraReader(
                    local_path,
                    loop=loop_flag,
                    target_width=854,
                    target_height=480,
                    delete_on_release=True,
                    on_eof=handle_eof,
                )
                ws_sink.on_ingest_frame = None
                ws_sink.ingest_mode = "uploaded_video"
                pipeline.request_reader_switch(new_sim_reader, f"Uploaded Video ({source_val})")
            elif source_type in ("web_stream", "ws", "laptop_webcam_web", "custom_video_web"):
                if isinstance(pipeline.reader, WebSocketFrameReader) and pipeline._pending_reader is None:
                    ws_sink.on_ingest_frame = pipeline.reader.push_frame
                    ws_sink.ingest_mode = "ws"
                    pipeline.reset_state()
                    print(f"🎬 [{camera_id}] Active WebSocketFrameReader reset & ready for Web-Fed Stream: {source_val}")
                else:
                    new_ws_reader = WebSocketFrameReader(
                        width=854,
                        height=480,
                        fps=15,
                        on_drop_frame=ws_sink.send_frame_ack,
                    )
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

        def handle_stop_source() -> None:
            if isinstance(pipeline.reader, WebSocketFrameReader) and pipeline._pending_reader is None:
                ws_sink.on_ingest_frame = pipeline.reader.push_frame
                ws_sink.ingest_mode = "ws"
                pipeline.reset_state()
            else:
                idle_reader = WebSocketFrameReader(
                    width=854,
                    height=480,
                    fps=15,
                    on_drop_frame=ws_sink.send_frame_ack,
                )
                ws_sink.on_ingest_frame = idle_reader.push_frame
                ws_sink.ingest_mode = "ws"
                pipeline.request_reader_switch(idle_reader, "Idle (Stopped)")

        def handle_set_loop(loop_val: bool) -> None:
            if hasattr(pipeline.reader, 'set_loop'):
                pipeline.reader.set_loop(loop_val)

        ws_sink.on_switch_source = handle_remote_source_switch
        ws_sink.on_stop_source = handle_stop_source
        ws_sink.on_set_loop = handle_set_loop
        ws_sink.on_switch_model = on_switch_model
        ws_sink.on_switch_tracker = on_switch_tracker

    return pipeline


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
    if args.legacy:
        config.MODEL_PROFILE = "stock"
    if args.model:
        config.MODEL_NAME = args.model
        # Infer profile label when user explicitly passes --model
        if "yolo26l.pt" in os.path.basename(args.model).lower() and not args.legacy:
            config.MODEL_PROFILE = "stock"
    else:
        resolved_profile, resolved_weights, profile_conf = resolve_profile_model(config, config.MODEL_PROFILE)
        config.MODEL_PROFILE = resolved_profile
        config.MODEL_NAME = resolved_weights
        if args.conf is None and not os.getenv("CONFIDENCE_THRESHOLD"):
            config.CONFIDENCE_THRESHOLD = profile_conf

    if args.conf is not None:
        config.CONFIDENCE_THRESHOLD = args.conf
    if args.imgsz is not None:
        config.IMGSZ = args.imgsz
    purge_temp_videos_dir(getattr(config, 'TEMP_VIDEO_DIR', 'temp_videos'))
    if args.preview:
        config.SHOW_LIVE_PREVIEW = True
    if args.no_video:
        config.SAVE_OUTPUT_VIDEO = False
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

    camera_ids = resolve_camera_ids(args, config)
    if len(camera_ids) > 1:
        config.ENABLE_WEBSOCKET_STREAM = True

    print(
        f"🧠 [AI Core] Loading shared YOLO model '{config.MODEL_NAME}' "
        f"(profile={config.MODEL_PROFILE}, imgsz={config.IMGSZ}, conf={config.CONFIDENCE_THRESHOLD}) "
        f"with Micro-Batcher for cameras: {camera_ids}"
    )
    raw_model = YOLO(config.MODEL_NAME)
    resolved_classes = apply_model_classes(config, getattr(raw_model, "names", None))
    print(f"🏷️  [AI Core] Active vehicle classes resolved from model: {resolved_classes}")
    batch_runner = BatchModelRunner(
        raw_model,
        max_batch_size=max(8, len(camera_ids)),
        max_wait_sec=0.008,
        warmup_imgsz=config.IMGSZ,
    )
    model_lock = threading.Lock()
    model_cache = {config.MODEL_PROFILE: (raw_model, config.MODEL_NAME)}
    switch_lock = threading.Lock()
    sessions: List[TrafficPipeline] = []

    def handle_live_model_switch(requested_profile: str) -> None:
        """
        Zero-downtime live toggle between Fine-Tuned VDAT (5-class best.pt) and
        Legacy Stock COCO YOLO26l (yolo26l.pt) across all active camera sessions.
        """
        with switch_lock:
            target_profile, target_weights, target_conf = resolve_profile_model(config, requested_profile)
            if args.conf is not None:
                target_conf = args.conf

            try:
                if target_profile in model_cache:
                    next_model, loaded_weights = model_cache[target_profile]
                else:
                    print(f"📦 [AI Core] Loading '{target_profile}' model weights: {target_weights} ...")
                    next_model = YOLO(target_weights)
                    loaded_weights = target_weights
                    model_cache[target_profile] = (next_model, loaded_weights)

                next_classes = apply_model_classes(config, getattr(next_model, "names", None))
                config.MODEL_PROFILE = target_profile
                config.MODEL_NAME = loaded_weights
                config.CONFIDENCE_THRESHOLD = target_conf

                batch_runner.swap_model(next_model, label=f"{target_profile} ({os.path.basename(loaded_weights)})")

                for sess in sessions:
                    sess.tracker.target_classes = dict(next_classes)
                    sess.tracker.confidence_threshold = target_conf
                    sess.tracker.model = next_model
                    sess.tracker.request_reset()
                    sess.visualizer.update_classes(next_classes)

                status_payload = {
                    "profile": target_profile,
                    "modelName": os.path.basename(loaded_weights),
                    "confidence": target_conf,
                    "classes": {str(k): v for k, v in next_classes.items()},
                    "switching": False,
                }
                for sess in sessions:
                    for sink in sess.sinks:
                        if isinstance(sink, WebSocketSink):
                            sink.send_model_status(status_payload)
                            break
                    break

                print(
                    f"✅ [AI Core] Switched ALL {len(sessions)} camera(s) to '{target_profile}' "
                    f"({os.path.basename(loaded_weights)}, conf={target_conf}, classes={next_classes})"
                )
            except Exception as exc:
                print(f"❌ [AI Core] Failed to switch model to '{requested_profile}': {exc}")

    def handle_live_tracker_switch(requested_tracker: str) -> None:
        """Hot-swaps the tracking algorithm (bytetrack / botsort / deepsort) across all active sessions."""
        with switch_lock:
            normalized = str(requested_tracker or "bytetrack").strip().lower()
            if normalized not in ("bytetrack", "botsort", "deepsort"):
                print(f"⚠️  [AI Core] Ignoring unsupported tracker '{requested_tracker}'")
                return
            try:
                config.TRACKER_TYPE = normalized
                for sess in sessions:
                    new_tracker = create_tracker(
                        config,
                        shared_model=batch_runner,
                        model_lock=model_lock,
                        camera_id=sess.camera_id,
                    )
                    sess.request_tracker_switch(new_tracker)

                status_payload = {
                    "tracker": normalized,
                    "switching": False,
                }
                for sess in sessions:
                    for sink in sess.sinks:
                        if isinstance(sink, WebSocketSink):
                            sink.send_tracker_status(status_payload)
                            break
                    break
                print(f"✅ [AI Core] Switched ALL {len(sessions)} camera(s) tracker -> '{normalized}'")
            except Exception as exc:
                print(f"❌ [AI Core] Failed to switch tracker to '{requested_tracker}': {exc}")

    if len(camera_ids) == 1:
        try:
            pipeline = create_camera_session(
                camera_id=camera_ids[0],
                config=config,
                shared_model=batch_runner,
                model_lock=model_lock,
                is_primary=True,
                enable_preview=config.SHOW_LIVE_PREVIEW,
                on_switch_model=handle_live_model_switch,
                on_switch_tracker=handle_live_tracker_switch,
            )
            sessions.append(pipeline)
            pipeline.run()
        finally:
            batch_runner.stop()
            purge_temp_videos_dir(getattr(config, 'TEMP_VIDEO_DIR', 'temp_videos'))
        return

    # Multi-camera concurrent sessions sharing the single loaded YOLO model via BatchModelRunner
    threads: List[threading.Thread] = []
    for idx, cam_id in enumerate(camera_ids):
        session = create_camera_session(
            camera_id=cam_id,
            config=config,
            shared_model=batch_runner,
            model_lock=model_lock,
            is_primary=(idx == 0),
            enable_preview=False,
            on_switch_model=handle_live_model_switch,
            on_switch_tracker=handle_live_tracker_switch,
        )
        sessions.append(session)
        t = threading.Thread(target=session.run, name=f"CameraSession-{cam_id}", daemon=True)
        threads.append(t)
        t.start()

    print(f"🚀 [AI Core] {len(sessions)} concurrent camera session(s) active on 1 micro-batched YOLO model.")
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n🛑 [AI Core] Shutting down all camera sessions...")
        for s in sessions:
            s.stop()
        for t in threads:
            t.join(timeout=2.0)
    finally:
        batch_runner.stop()
        purge_temp_videos_dir(getattr(config, 'TEMP_VIDEO_DIR', 'temp_videos'))


if __name__ == "__main__":
    main()
