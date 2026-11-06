import cv2
import os
import json
import base64
import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional
import numpy as np
import requests

try:
    import websockets.sync.client as ws_sync
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from sinks.base_sink import BaseSink
from core.events import FrameTelemetry, ViolationEvent


class WebSocketSink(BaseSink):
    """
    Asynchronously streams processed video frames (JPEG) and violation alerts
    over a persistent WebSocket connection to api.VDAT.
    Uses a background worker thread with a bounded queue to prevent network latency
    from blocking GPU inference.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:5000/ws/inference",
        camera_id: str = "cam-01",
        jpeg_quality: int = 65,
        target_fps: float = 15.0,
        max_width: int = 960,
        queue_size: int = 2,
        reconnect_interval: int = 5,
        evidence_dir: str = "output/evidence",
        zones_file: str = "zones.json",
        on_zone_update: Optional[ Any ] = None,
        on_ingest_frame: Optional[ Any ] = None,
        on_switch_source: Optional[ Any ] = None,
        ingest_mode: str = "local",
    ):
        self.ws_url = ws_url
        self.camera_id = camera_id
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.max_width = max_width
        self.reconnect_interval = reconnect_interval
        self.evidence_dir = evidence_dir
        self.zones_file = zones_file
        self.on_zone_update = on_zone_update
        self.on_ingest_frame = on_ingest_frame
        self.on_switch_source = on_switch_source
        self.ingest_mode = ingest_mode
        self._pipeline_info: Dict[str, Any] = {}

        self._queue: Queue = Queue(maxsize=queue_size)
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        self._last_frame_time = 0.0
        self._min_frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        # Derive HTTP upload URL from WebSocket URL (e.g. ws://... -> http://.../api/evidence/upload)
        http_scheme = "https" if self.ws_url.startswith("wss://") else "http"
        host_part = self.ws_url.split("://")[1].split("/")[0]
        self._upload_url = f"{http_scheme}://{host_part}/api/evidence/upload"

        if not WEBSOCKETS_AVAILABLE:
            print("[WebSocketSink] Warning: 'websockets' package not available. WebSocket stream disabled.")

    def on_start(self, pipeline_info: Dict[str, Any]) -> None:
        if not WEBSOCKETS_AVAILABLE:
            return

        self._pipeline_info = pipeline_info
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

        # Send handshake / start packet
        self._enqueue({
            "type": "camera_init",
            "cameraId": self.camera_id,
            "ingestMode": self.ingest_mode,
            "pipeline": {
                "tracker": pipeline_info.get("tracker_type"),
                "model": pipeline_info.get("model_name"),
                "fps": pipeline_info.get("fps"),
                "width": pipeline_info.get("width"),
                "height": pipeline_info.get("height"),
                "ingestMode": self.ingest_mode,
            },
            "timestamp": time.time(),
        })

    def on_frame(self, frame: np.ndarray, telemetry: FrameTelemetry) -> None:
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return

        now = time.time()
        # Throttle stream FPS to target_fps to prevent network congestion
        if self._min_frame_interval > 0 and (now - self._last_frame_time) < self._min_frame_interval:
            return
        self._last_frame_time = now

        # Downscale frame for lightweight live web preview
        h, w = frame.shape[:2]
        if self.max_width and w > self.max_width:
            new_h = int(h * (self.max_width / w))
            preview_frame = cv2.resize(frame, (self.max_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            preview_frame = frame

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', preview_frame, self._encode_params)
        if not ret:
            return

        b64_image = base64.b64encode(buffer).decode('utf-8')
        message = {
            "type": "frame",
            "cameraId": self.camera_id,
            "telemetry": telemetry.to_dict(),
            "image": b64_image,
        }
        self._enqueue(message)

    def on_violation(self, event: ViolationEvent, snapshot: np.ndarray) -> None:
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return

        ret, buffer = cv2.imencode('.jpg', snapshot, self._encode_params)
        b64_snapshot = base64.b64encode(buffer).decode('utf-8') if ret else None

        message = {
            "type": "violation",
            "cameraId": self.camera_id,
            "event": event.to_dict(),
            "evidenceImage": b64_snapshot,
        }
        self._enqueue(message)

    def _enqueue(self, message: Dict[str, Any]) -> None:
        try:
            # If queue is full, discard oldest frame so only the freshest frame is sent
            while self._queue.full():
                try:
                    dropped = self._queue.get_nowait()
                    # Always preserve violations & clip ready notifications
                    if dropped.get('type') in ('violation', 'evidence_clip_ready'):
                        self._queue.put_nowait(dropped)
                        break
                except Empty:
                    break
            self._queue.put_nowait(message)
        except Exception:
            pass

    def _reader_loop(self, ws: Any) -> None:
        """Background thread listening for incoming server commands (e.g. on-demand clip requests)."""
        while not self._stopped:
            try:
                raw_msg = ws.recv(timeout=1.0)
                if not raw_msg:
                    continue
                data = json.loads(raw_msg)
                msg_type = data.get("type")

                if msg_type == "request_clip_upload":
                    filename = data.get("filename")
                    if filename:
                        t = threading.Thread(
                            target=self._upload_clip_worker,
                            args=(filename,),
                            daemon=True
                        )
                        t.start()
                elif msg_type == "camera_ack":
                    print(f"[WebSocketSink] Server acknowledged camera '{self.camera_id}'")
                    zones = data.get("zones")
                    if zones:
                        self._apply_remote_zones(zones, source="camera_ack")
                elif msg_type == "zone_update":
                    zones = data.get("zones")
                    if zones:
                        self._apply_remote_zones(zones, source="zone_update")
                elif msg_type == "ingest_frame":
                    b64_img = data.get("image")
                    if b64_img and callable(self.on_ingest_frame):
                        try:
                            raw_bytes = base64.b64decode(b64_img)
                            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if decoded is not None:
                                self.on_ingest_frame(decoded)
                        except Exception:
                            pass
                elif msg_type == "switch_source":
                    if callable(self.on_switch_source):
                        try:
                            self.on_switch_source(data)
                        except Exception as e:
                            print(f"[WebSocketSink] Error switching source: {e}")
            except TimeoutError:
                continue
            except Exception:
                # Connection dropped or closed; exit reader thread
                break

    def _apply_remote_zones(self, zones_cfg: Dict[str, Any], source: str = "remote") -> None:
        """Persists updated zones from api.VDAT to disk and triggers live reload."""
        try:
            lanes = zones_cfg.get("lanes") or []
            intersection = zones_cfg.get("intersection") or []
            light_box = zones_cfg.get("lightBox") or zones_cfg.get("traffic_lights") or []
            payload = {
                "lanes": lanes,
                "intersection": intersection,
                "traffic_lights": light_box if isinstance(light_box, list) else [],
            }
            with open(self.zones_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(
                f"[WebSocketSink] Synced zones from {source}: "
                f"{len(lanes)} lane(s), {len(intersection)} intersection pts, "
                f"{len(payload['traffic_lights'])} light point(s)"
            )
            if callable(self.on_zone_update):
                self.on_zone_update(self.zones_file)
        except Exception as e:
            print(f"[WebSocketSink] Failed to apply remote zones: {e}")

    def _upload_clip_worker(
        self,
        filename: str,
        track_id: Optional[int] = None,
        frame_number: Optional[int] = None,
    ) -> None:
        """Uploads a requested or newly encoded evidence clip to api.VDAT in the background."""
        candidates = [
            os.path.join(self.evidence_dir, filename),
            os.path.join("output", "evidence", filename),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "evidence", filename)),
        ]
        clip_path = None
        for c in candidates:
            if os.path.exists(c):
                clip_path = c
                break

        if not clip_path:
            print(f"[WebSocketSink] Clip '{filename}' not found on disk")
            return

        print(f"[WebSocketSink] Uploading evidence clip '{filename}' to {self._upload_url}...")
        try:
            with open(clip_path, "rb") as f:
                files = {"file": (filename, f, "video/mp4")}
                data: Dict[str, Any] = {"cameraId": self.camera_id}
                if track_id is not None:
                    data["trackId"] = str(track_id)
                if frame_number is not None:
                    data["frameNumber"] = str(frame_number)
                res = requests.post(self._upload_url, files=files, data=data, timeout=60)
                if res.status_code in (200, 201):
                    print(f"[WebSocketSink] Evidence clip '{filename}' synced for streaming.")
                else:
                    print(f"[WebSocketSink] Evidence upload failed ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"[WebSocketSink] Error uploading evidence clip '{filename}': {e}")

    def _worker_loop(self) -> None:
        """Background worker thread that manages the WebSocket client connection."""
        while not self._stopped:
            try:
                with ws_sync.connect(self.ws_url) as ws:
                    print(f"[WebSocketSink] Connected to {self.ws_url}")

                    # Start dedicated reader thread to listen for server commands
                    reader_thread = threading.Thread(
                        target=self._reader_loop,
                        args=(ws,),
                        daemon=True
                    )
                    reader_thread.start()

                    while not self._stopped:
                        try:
                            msg = self._queue.get(timeout=0.5)
                            ws.send(json.dumps(msg))
                        except Empty:
                            continue
            except Exception as e:
                if not self._stopped:
                    # Connection failed or disconnected; wait and retry
                    time.sleep(self.reconnect_interval)

    def on_evidence_clip(self, event: ViolationEvent, clip_path: str) -> None:
        """Notify the API that an evidence clip is ready and proactively upload it for instant streaming."""
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return

        filename = os.path.basename(clip_path)
        message = {
            "type": "evidence_clip_ready",
            "cameraId": self.camera_id,
            "event": event.to_dict(),
            "filename": filename,
            "clipPath": clip_path,
        }
        self._enqueue(message)

        # Proactively push the encoded MP4 to api.VDAT in background so web clients can stream immediately
        t = threading.Thread(
            target=self._upload_clip_worker,
            args=(filename, event.track_id, event.frame_number),
            daemon=True,
        )
        t.start()

    def on_complete(self) -> None:
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


