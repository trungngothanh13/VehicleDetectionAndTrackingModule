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
    ):
        self.ws_url = ws_url
        self.camera_id = camera_id
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.max_width = max_width
        self.reconnect_interval = reconnect_interval
        self.evidence_dir = evidence_dir

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

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

        # Send handshake / start packet
        self._enqueue({
            "type": "camera_init",
            "cameraId": self.camera_id,
            "pipeline": {
                "tracker": pipeline_info.get("tracker_type"),
                "model": pipeline_info.get("model_name"),
                "fps": pipeline_info.get("fps"),
                "width": pipeline_info.get("width"),
                "height": pipeline_info.get("height"),
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
            except TimeoutError:
                continue
            except Exception:
                # Connection dropped or closed; exit reader thread
                break

    def _upload_clip_worker(self, filename: str) -> None:
        """Uploads a requested evidence clip to api.VDAT in the background without blocking streaming."""
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
            print(f"[WebSocketSink] ⚠️ On-demand clip '{filename}' not found on disk")
            return

        print(f"[WebSocketSink] 📤 Uploading on-demand proof clip '{filename}' to {self._upload_url}...")
        try:
            with open(clip_path, "rb") as f:
                files = {"file": (filename, f, "video/mp4")}
                data = {"cameraId": self.camera_id}
                res = requests.post(self._upload_url, files=files, data=data, timeout=60)
                if res.status_code in (200, 201):
                    print(f"[WebSocketSink] ✅ Proof clip '{filename}' uploaded successfully!")
                else:
                    print(f"[WebSocketSink] ❌ Proof upload failed ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"[WebSocketSink] ❌ Error uploading proof clip '{filename}': {e}")

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
        """Notify the API that an evidence clip is ready on ThunderCompute disk."""
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return

        message = {
            "type": "evidence_clip_ready",
            "cameraId": self.camera_id,
            "event": event.to_dict(),
            "filename": os.path.basename(clip_path),
            "clipPath": clip_path,
        }
        self._enqueue(message)

    def on_complete(self) -> None:
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


