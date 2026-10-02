import cv2
import json
import base64
import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional
import numpy as np

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
        jpeg_quality: int = 75,
        queue_size: int = 30,
        reconnect_interval: int = 5,
    ):
        self.ws_url = ws_url
        self.camera_id = camera_id
        self.jpeg_quality = jpeg_quality
        self.reconnect_interval = reconnect_interval

        self._queue: Queue = Queue(maxsize=queue_size)
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

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

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, self._encode_params)
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
            # If queue is full, drop oldest non-violation message to keep stream fresh
            if self._queue.full():
                try:
                    dropped = self._queue.get_nowait()
                    # Preserve violations; re-enqueue if it was a violation
                    if dropped.get('type') == 'violation':
                        self._queue.put_nowait(dropped)
                except Empty:
                    pass
            self._queue.put_nowait(message)
        except Exception:
            pass

    def _worker_loop(self) -> None:
        """Background worker thread that manages the WebSocket client connection."""
        while not self._stopped:
            try:
                with ws_sync.connect(self.ws_url) as ws:
                    print(f"[WebSocketSink] Connected to {self.ws_url}")
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

    def on_complete(self) -> None:
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
