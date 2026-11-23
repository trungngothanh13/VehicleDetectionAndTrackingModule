import cv2
import os
import json
import base64
import struct
import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional, Tuple
import numpy as np
import requests

try:
    import websockets.sync.client as ws_sync
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from sinks.base_sink import BaseSink
from core.events import FrameTelemetry, ViolationEvent


def _pack_binary_frame(header: Dict[str, Any], payload: bytes) -> bytes:
    """Packs [4-byte uint32 BE headerLen][UTF-8 JSON header][raw JPEG bytes]."""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + payload


def _unpack_binary_frame(raw: bytes) -> Optional[Tuple[Dict[str, Any], bytes]]:
    """Unpacks [4-byte uint32 BE headerLen][UTF-8 JSON header][raw JPEG bytes]."""
    if len(raw) < 4:
        return None
    header_len = struct.unpack(">I", raw[:4])[0]
    if header_len <= 0 or 4 + header_len > len(raw):
        return None
    try:
        header = json.loads(raw[4:4 + header_len].decode("utf-8"))
        return header, raw[4 + header_len:]
    except Exception:
        return None


class WebSocketSink(BaseSink):
    """
    Asynchronously streams processed video frames (binary JPEG when focused,
    lightweight JSON telemetry when unfocused) and violation alerts over a
    persistent WebSocket connection to api.VDAT.
    Moves JPEG encoding completely off the inference thread onto the background
    sender thread so GPU inference never blocks on encoding or network I/O.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:5000/ws/inference",
        camera_id: str = "cam-01",
        jpeg_quality: int = 60,
        target_fps: float = 15.0,
        max_width: int = 854,
        queue_size: int = 2,
        reconnect_interval: int = 5,
        evidence_dir: str = "output/evidence",
        zones_file: str = "zones.json",
        temp_video_dir: str = "temp_videos",
        on_zone_update: Optional[Any] = None,
        on_ingest_frame: Optional[Any] = None,
        on_switch_source: Optional[Any] = None,
        on_stop_source: Optional[Any] = None,
        on_set_loop: Optional[Any] = None,
        on_switch_model: Optional[Any] = None,
        on_switch_tracker: Optional[Any] = None,
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
        self.temp_video_dir = temp_video_dir
        self.on_zone_update = on_zone_update
        self.on_ingest_frame = on_ingest_frame
        self.on_switch_source = on_switch_source
        self.on_stop_source = on_stop_source
        self.on_set_loop = on_set_loop
        self.on_switch_model = on_switch_model
        self.on_switch_tracker = on_switch_tracker
        self.ingest_mode = ingest_mode
        self._pipeline_info: Dict[str, Any] = {}

        # Only encode & upload JPEG images back over SSH when a web client focuses on this camera
        self._send_image: bool = (self.camera_id == "cam-01")

        # Separate lossless queue for violations, clip alerts, and credit ACKs
        self._priority_queue: Queue = Queue()
        # Single-slot latest-frame handoff (encoded on background worker thread, not inference thread)
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[Tuple[Optional[np.ndarray], Dict[str, Any]]] = None
        self._pending_acked_frames = 0
        self._last_received_seq = 0
        self._wake_event = threading.Event()

        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        self._last_frame_time = 0.0
        self._min_frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._ingested_frames = 0
        self._sent_frames = 0

        # Derive HTTP base & upload URL from WebSocket URL (e.g. ws://... -> http://...)
        http_scheme = "https" if self.ws_url.startswith("wss://") else "http"
        host_part = self.ws_url.split("://")[1].split("/")[0]
        self._api_base_url = f"{http_scheme}://{host_part}"
        self._upload_url = f"{self._api_base_url}/api/evidence/upload"

        if not WEBSOCKETS_AVAILABLE:
            print("[WebSocketSink] Warning: 'websockets' package not available. WebSocket stream disabled.")

    @property
    def needs_annotated_frame(self) -> bool:
        return self._send_image

    def send_frame_ack(self, count: int = 1) -> None:
        """Immediately sends a lightweight credit ACK so upstream sliding windows never stall on dropped frames."""
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return
        self._enqueue_priority({
            "type": "frame_ack",
            "cameraId": self.camera_id,
            "ackSeq": self._last_received_seq,
            "count": max(1, int(count)),
        })

    def send_source_status(
        self,
        status: str,
        source_type: str = "",
        label: str = "",
        error: Optional[str] = None,
    ) -> None:
        """Notifies api.VDAT / web clients when an AI-hosted video starts, stops, ends, or errors."""
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return
        self._enqueue_priority({
            "type": "source_status",
            "cameraId": self.camera_id,
            "status": status,
            "sourceType": source_type,
            "label": label,
            "error": error,
            "timestamp": time.time(),
        })

    def _build_camera_init_packet(self) -> Dict[str, Any]:
        pipeline_info = self._pipeline_info or {}
        return {
            "type": "camera_init",
            "cameraId": self.camera_id,
            "ingestMode": self.ingest_mode,
            "pipeline": {
                "tracker": pipeline_info.get("tracker_type"),
                "model": pipeline_info.get("model_name"),
                "modelProfile": pipeline_info.get("model_profile", "trained"),
                "confidence": pipeline_info.get("confidence"),
                "fps": pipeline_info.get("fps"),
                "width": pipeline_info.get("width"),
                "height": pipeline_info.get("height"),
                "ingestMode": self.ingest_mode,
            },
            "timestamp": time.time(),
        }

    def on_start(self, pipeline_info: Dict[str, Any]) -> None:
        if not WEBSOCKETS_AVAILABLE:
            return

        self._pipeline_info = pipeline_info
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def on_frame(self, frame: np.ndarray, telemetry: FrameTelemetry) -> None:
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return

        # In local/file/RTSP/uploaded_video mode, only pace above target_fps so inference cadence stays smooth
        if self.ingest_mode != "ws":
            now = time.monotonic()
            if self._min_frame_interval > 0 and (now - self._last_frame_time) < self._min_frame_interval:
                return
            self._last_frame_time = now

        frame_ref = frame if self._send_image else None
        tel_dict = telemetry.to_dict()
        with self._frame_lock:
            self._latest_frame = (frame_ref, tel_dict)
            self._pending_acked_frames += 1
        self._wake_event.set()

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
        self._enqueue_priority(message)

    def send_model_status(self, status_payload: Dict[str, Any]) -> None:
        """Broadcasts active model profile status back to api.VDAT after a live hot-swap."""
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return
        self._pipeline_info["model_name"] = status_payload.get("modelName", self._pipeline_info.get("model_name"))
        self._pipeline_info["model_profile"] = status_payload.get("profile", self._pipeline_info.get("model_profile"))
        self._pipeline_info["confidence"] = status_payload.get("confidence", self._pipeline_info.get("confidence"))
        self._enqueue_priority({
            "type": "model_status",
            "cameraId": self.camera_id,
            **status_payload,
            "timestamp": time.time(),
        })

    def send_tracker_status(self, status_payload: Dict[str, Any]) -> None:
        """Broadcasts active tracking engine status back to api.VDAT after a live hot-swap."""
        if not WEBSOCKETS_AVAILABLE or self._stopped:
            return
        self._pipeline_info["tracker_type"] = status_payload.get("tracker", self._pipeline_info.get("tracker_type"))
        self._enqueue_priority({
            "type": "tracker_status",
            "cameraId": self.camera_id,
            **status_payload,
            "timestamp": time.time(),
        })

    def _enqueue_priority(self, message: Dict[str, Any]) -> None:
        try:
            self._priority_queue.put_nowait(message)
            self._wake_event.set()
        except Exception:
            pass

    def _parse_inbound(self, raw_msg: Any) -> Optional[Tuple[Dict[str, Any], Optional[bytes]]]:
        if isinstance(raw_msg, (bytes, bytearray, memoryview)):
            return _unpack_binary_frame(bytes(raw_msg))
        if isinstance(raw_msg, str) and raw_msg:
            try:
                return json.loads(raw_msg), None
            except Exception:
                return None
        return None

    def _handle_control_message(self, data: Dict[str, Any]) -> None:
        """Handles non-frame WebSocket messages from api.VDAT."""
        msg_type = data.get("type")

        if msg_type == "set_stream_focus":
            send_img = bool(data.get("sendImage", False))
            if self._send_image != send_img:
                self._send_image = send_img
                print(
                    f"👁️ [{self.camera_id}] Stream image focus -> "
                    f"{'ON (binary JPEG)' if send_img else 'OFF (telemetry only)'}"
                )
        elif msg_type == "request_clip_upload":
            filename = data.get("filename")
            if filename:
                t = threading.Thread(
                    target=self._upload_clip_worker,
                    args=(filename,),
                    daemon=True,
                )
                t.start()
        elif msg_type == "delete_evidence_clip":
            filename = data.get("filename")
            if filename:
                self._delete_local_clip(filename)
        elif msg_type == "camera_ack":
            print(
                f"✅ [AI Engine] Registered with api.VDAT for camera '{self.camera_id}' "
                f"(ingest_mode={self.ingest_mode})"
            )
            zones = data.get("zones")
            if zones:
                self._apply_remote_zones(zones, source="camera_ack")
        elif msg_type == "zone_update":
            zones = data.get("zones")
            if zones is not None:
                self._apply_remote_zones(zones, source="zone_update")
        elif msg_type == "switch_source":
            source_type = data.get("sourceType")
            print(
                f"🎬 [{self.camera_id}] Switching input source -> "
                f"type='{source_type}', value='{data.get('sourceValue')}'"
            )
            if source_type == "uploaded_video":
                threading.Thread(
                    target=self._download_and_switch_uploaded_video,
                    args=(data,),
                    daemon=True,
                ).start()
            elif callable(self.on_switch_source):
                try:
                    self.on_switch_source(data)
                except Exception as e:
                    print(f"[WebSocketSink] Error switching source: {e}")
        elif msg_type == "stop_source":
            print(f"🛑 [{self.camera_id}] Received stop_source request from web client")
            if callable(self.on_stop_source):
                try:
                    self.on_stop_source()
                except Exception as e:
                    print(f"[WebSocketSink] Error stopping source: {e}")
            self.send_source_status("stopped")
        elif msg_type == "set_source_loop":
            loop_val = bool(data.get("loop", True))
            if callable(self.on_set_loop):
                try:
                    self.on_set_loop(loop_val)
                except Exception as e:
                    print(f"[WebSocketSink] Error updating loop flag: {e}")
        elif msg_type == "switch_model":
            target_profile = str(data.get("profile", "trained")).strip().lower()
            print(f"🔁 [{self.camera_id}] Received live model switch request -> profile='{target_profile}'")
            if callable(self.on_switch_model):
                threading.Thread(
                    target=self.on_switch_model,
                    args=(target_profile,),
                    daemon=True,
                ).start()
        elif msg_type == "switch_tracker":
            target_tracker = str(data.get("tracker", "bytetrack")).strip().lower()
            print(f"🧭 [{self.camera_id}] Received live tracker switch request -> tracker='{target_tracker}'")
            if callable(self.on_switch_tracker):
                threading.Thread(
                    target=self.on_switch_tracker,
                    args=(target_tracker,),
                    daemon=True,
                ).start()

    def _process_ingest_frame(
        self,
        data: Dict[str, Any],
        jpeg_bytes: Optional[bytes] = None,
    ) -> None:
        """Decodes the latest ingested frame (binary JPEG or base64 JSON) and pushes it to the reader."""
        # Auto-switch pipeline to web_stream if it was started without --web-input
        if not callable(self.on_ingest_frame) and callable(self.on_switch_source):
            try:
                print("🔄 [AI Ingestion] Web frame detected — switching AI reader to web_stream...")
                self.on_switch_source({
                    "sourceType": "web_stream",
                    "sourceValue": data.get("sourceType", "web_stream"),
                })
                self.ingest_mode = "ws"
            except Exception as e:
                print(f"[WebSocketSink] Auto-switch to web_stream failed: {e}")

        if not callable(self.on_ingest_frame):
            return

        try:
            if jpeg_bytes is not None:
                if not jpeg_bytes:
                    return
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            else:
                b64_img = data.get("image")
                if not b64_img:
                    return
                raw_bytes = base64.b64decode(b64_img)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)

            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                self._ingested_frames += 1
                if self._ingested_frames == 1 or self._ingested_frames % 60 == 0:
                    print(
                        f"📥 [{self.camera_id}] AI Ingestion frame #{self._ingested_frames} received "
                        f"({decoded.shape[1]}x{decoded.shape[0]})"
                    )
                source_frame_id = data.get("frameNumber")
                try:
                    self.on_ingest_frame(decoded, source_frame_id)
                except TypeError:
                    self.on_ingest_frame(decoded)
        except Exception as e:
            print(f"[{self.camera_id}] Error decoding ingest_frame: {e}")

    def _download_and_switch_uploaded_video(self, data: Dict[str, Any]) -> None:
        """Downloads a staged video file from api.VDAT into gitignored temp_videos/ and starts local playback."""
        download_path = str(data.get("downloadPath", "")).strip()
        filename = os.path.basename(str(data.get("filename") or data.get("sourceValue") or "video.mp4"))
        label = str(data.get("sourceValue") or filename)
        if not download_path:
            self.send_source_status("error", source_type="uploaded_video", label=label, error="Missing downloadPath")
            return

        os.makedirs(self.temp_video_dir, exist_ok=True)
        safe_cam = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in self.camera_id)
        final_path = os.path.join(self.temp_video_dir, f"{safe_cam}_{filename}")
        part_path = final_path + ".part"
        url = f"{self._api_base_url}{download_path}"
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}cleanup=1"

        try:
            print(f"⬇️  [{self.camera_id}] Pulling staged video '{label}' from {url}...")
            with requests.get(url, stream=True, timeout=(10, 300)) as resp:
                resp.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=262144):
                        if self._stopped:
                            raise InterruptedError("WebSocketSink stopped during video download")
                        if chunk:
                            f.write(chunk)
            os.replace(part_path, final_path)
            if callable(self.on_switch_source):
                self.on_switch_source({**data, "localPath": final_path})
            self.send_source_status("playing", source_type="uploaded_video", label=label)
        except Exception as exc:
            print(f"❌ [{self.camera_id}] Failed to pull staged video '{label}': {exc}")
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
            self.send_source_status("error", source_type="uploaded_video", label=label, error=str(exc))

    def _reader_loop(self, ws: Any) -> None:
        """Background thread listening for incoming server commands and draining stale ingest frames."""
        while not self._stopped:
            try:
                raw_msg = ws.recv(timeout=1.0)
                if not raw_msg:
                    continue
                parsed = self._parse_inbound(raw_msg)
                if parsed is None:
                    continue
                data, jpeg_bytes = parsed
                msg_type = data.get("type")

                if msg_type not in ("ingest_frame", "source_frame"):
                    self._handle_control_message(data)
                    continue

                seq = data.get("seq")
                if isinstance(seq, int) and seq > self._last_received_seq:
                    self._last_received_seq = seq

                # Collect any immediately buffered burst frames (e.g. WAN TCP jitter burst)
                # Keep up to the newest 2 frames for smooth tracker motion; drop older excess
                burst: list = [(data, jpeg_bytes)]
                skipped_frames = 0
                while not self._stopped:
                    try:
                        queued_raw = ws.recv(timeout=0)
                    except TimeoutError:
                        break
                    if not queued_raw:
                        continue
                    queued_parsed = self._parse_inbound(queued_raw)
                    if queued_parsed is None:
                        continue
                    queued_data, queued_jpeg = queued_parsed
                    queued_type = queued_data.get("type")
                    if queued_type in ("ingest_frame", "source_frame"):
                        q_seq = queued_data.get("seq")
                        if isinstance(q_seq, int) and q_seq > self._last_received_seq:
                            self._last_received_seq = q_seq
                        burst.append((queued_data, queued_jpeg))
                    else:
                        if queued_type == "switch_source":
                            skipped_frames += len(burst)
                            burst.clear()
                        self._handle_control_message(queued_data)

                if len(burst) > 2:
                    skipped_frames += len(burst) - 2
                    burst = burst[-2:]

                if skipped_frames > 0:
                    self._enqueue_priority({
                        "type": "frame_ack",
                        "cameraId": self.camera_id,
                        "ackSeq": self._last_received_seq,
                        "count": skipped_frames,
                    })

                for frame_meta, frame_jpeg in burst:
                    self._process_ingest_frame(frame_meta, frame_jpeg)
            except TimeoutError:
                continue
            except Exception:
                # Connection dropped or closed; exit reader thread
                break

    def _delete_local_clip(self, filename: str) -> None:
        """Deletes local MP4 evidence file when a violation is dismissed by an officer."""
        safe_name = os.path.basename(filename)
        candidates = [
            os.path.join(self.evidence_dir, safe_name),
            os.path.join("output", "evidence", safe_name),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "evidence", safe_name)),
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    os.remove(c)
                    print(f"[{self.camera_id}] Deleted dismissed local clip: {safe_name}")
                except OSError:
                    pass

    def _apply_remote_zones(self, zones_cfg: Dict[str, Any], source: str = "remote") -> None:
        """Persists updated ZoneProfile from api.VDAT to disk and triggers live reload."""
        try:
            lanes = zones_cfg.get("lanes") or []
            intersection = zones_cfg.get("intersection") or []
            exit_zones = zones_cfg.get("exitZones") or zones_cfg.get("exit_zones") or []
            light_box = zones_cfg.get("lightBox") or zones_cfg.get("traffic_lights") or []
            crossing_mode = bool(zones_cfg.get("crossingMode") or zones_cfg.get("crossing_mode") or False)
            raw_clearance = zones_cfg.get("clearanceSeconds")
            if raw_clearance is None:
                raw_clearance = zones_cfg.get("clearance_seconds")
            clearance_seconds = float(raw_clearance if raw_clearance is not None else 3.0)

            payload = {
                "lanes": lanes,
                "intersection": intersection,
                "exit_zones": exit_zones,
                "traffic_lights": light_box if isinstance(light_box, list) else [],
                "crossing_mode": crossing_mode,
                "clearance_seconds": clearance_seconds,
            }
            os.makedirs(os.path.dirname(self.zones_file) or ".", exist_ok=True)
            with open(self.zones_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(
                f"[{self.camera_id}] Synced zones from {source}: "
                f"{len(lanes)} approach lane(s), {len(intersection)} intersection pts, "
                f"{len(exit_zones)} exit zone(s), {len(payload['traffic_lights'])} light point(s), "
                f"crossing_mode={crossing_mode}"
            )
            if callable(self.on_zone_update):
                self.on_zone_update(self.zones_file)
        except Exception as e:
            print(f"[{self.camera_id}] Failed to apply remote zones: {e}")

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

    def _send_latest_frame(
        self,
        ws: Any,
        frame_obj: Optional[np.ndarray],
        tel_dict: Dict[str, Any],
        acked_frames: int,
    ) -> None:
        """Encodes crisp binary JPEG on the sender thread when focused; otherwise sends compact JSON telemetry."""
        ack_seq = self._last_received_seq
        if frame_obj is not None and self._send_image:
            h, w = frame_obj.shape[:2]
            if self.max_width and w > self.max_width:
                new_h = int(h * (self.max_width / w))
                preview = cv2.resize(frame_obj, (self.max_width, new_h), interpolation=cv2.INTER_AREA)
            else:
                preview = frame_obj

            ret, buffer = cv2.imencode('.jpg', preview, self._encode_params)
            if ret:
                packet = _pack_binary_frame(
                    {
                        "type": "frame",
                        "cameraId": self.camera_id,
                        "telemetry": tel_dict,
                        "ackSeq": ack_seq,
                        "ackedFrames": acked_frames,
                    },
                    buffer.tobytes(),
                )
                ws.send(packet)
                mode_label = "binary JPEG"
            else:
                ws.send(
                    json.dumps(
                        {
                            "type": "frame",
                            "cameraId": self.camera_id,
                            "telemetry": tel_dict,
                            "image": None,
                            "ackSeq": ack_seq,
                            "ackedFrames": acked_frames,
                        },
                        separators=(",", ":"),
                    )
                )
                mode_label = "telemetry"
        else:
            ws.send(
                json.dumps(
                    {
                        "type": "frame",
                        "cameraId": self.camera_id,
                        "telemetry": tel_dict,
                        "image": None,
                        "ackSeq": ack_seq,
                        "ackedFrames": acked_frames,
                    },
                    separators=(",", ":"),
                )
            )
            mode_label = "telemetry"

        self._sent_frames += 1
        if self._sent_frames == 1 or self._sent_frames % 90 == 0:
            print(
                f"📤 [{self.camera_id}] AI Output #{self._sent_frames} "
                f"({mode_label}) "
                f"(vehicles={tel_dict.get('vehicle_count', 0)}, "
                f"light={tel_dict.get('light_state', 'unknown')}, "
                f"fps={tel_dict.get('fps', 0)})"
            )

    def _worker_loop(self) -> None:
        """Background worker thread that manages the WebSocket client connection."""
        while not self._stopped:
            try:
                with ws_sync.connect(
                    self.ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=2,
                    max_size=16 * 1024 * 1024,
                    max_queue=32,
                    compression=None,
                ) as ws:
                    print(f"🔌 [WebSocketSink] Connected to {self.ws_url}")
                    self._last_received_seq = 0

                    # Register camera and receive latest ZoneProfile & initial stream focus state
                    ws.send(json.dumps(self._build_camera_init_packet()))

                    reader_thread = threading.Thread(
                        target=self._reader_loop,
                        args=(ws,),
                        daemon=True,
                    )
                    reader_thread.start()

                    while not self._stopped:
                        self._wake_event.wait(timeout=0.5)
                        self._wake_event.clear()

                        # 1. Flush priority messages (violations, evidence clips, frame_ack)
                        while not self._stopped:
                            try:
                                prio_msg = self._priority_queue.get_nowait()
                                ws.send(json.dumps(prio_msg, separators=(",", ":")))
                            except Empty:
                                break

                        # 2. Send latest frame/telemetry slot if updated
                        frame_slot = None
                        acked_frames = 0
                        with self._frame_lock:
                            if self._latest_frame is not None:
                                frame_slot = self._latest_frame
                                acked_frames = max(1, self._pending_acked_frames)
                                self._latest_frame = None
                                self._pending_acked_frames = 0

                        if frame_slot is not None:
                            self._send_latest_frame(ws, frame_slot[0], frame_slot[1], acked_frames)
            except Exception as e:
                if not self._stopped:
                    print(f"⚠️ [{self.camera_id}] WebSocket connection lost ({e}). Reconnecting in {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)

    def on_evidence_clip(self, event: ViolationEvent, clip_path: str) -> None:
        """Notify api.VDAT that an evidence clip is ready. Uploaded on-demand when streamed by an officer."""
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
        self._enqueue_priority(message)

    def on_complete(self) -> None:
        self._stopped = True
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
