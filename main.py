import os
import threading
import argparse
from typing import List
from ultralytics import YOLO
from core.config import Config
from core.model_resolver import apply_model_classes, resolve_profile_model
from core.pipeline import TrafficPipeline
from core.session_factory import create_camera_session, purge_temp_videos_dir
from trackers import create_tracker
from trackers.base_tracker import BatchModelRunner
from sinks import WebSocketSink
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
