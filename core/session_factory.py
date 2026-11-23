import os
import shutil
import threading
from typing import Optional

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
    shared_model: Optional[object],
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
        ws_sink.on_set_bypass_zones = pipeline.set_bypass_zones
        ws_sink.on_switch_model = on_switch_model
        ws_sink.on_switch_tracker = on_switch_tracker

    return pipeline
