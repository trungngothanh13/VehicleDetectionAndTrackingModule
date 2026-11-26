"""TEMPORARY TABLE 6.7 BENCHMARK LOGGER — DELETE AFTER THESIS EVALUATION."""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class Table67BenchmarkLogger:
    """Thread-safe rolling telemetry collector for multi-camera Table 6.7 benchmarking."""

    def __init__(self, log_interval_sec: float = 10.0, max_samples: int = 2000) -> None:
        self._lock = threading.Lock()
        self._log_interval_sec = log_interval_sec
        self._last_log_mono = time.monotonic()

        # (timestamp, active_streams, batch_size, gpu_ms)
        self._batches: Deque[Tuple[float, int, int, float]] = deque(maxlen=max_samples)
        # (timestamp, camera_id, e2e_ms)
        self._frames: Deque[Tuple[float, str, float]] = deque(maxlen=max_samples * 2)
        self._camera_last_seen: Dict[str, float] = {}
        self._latest_active_streams: int = 0

    def reset(self) -> None:
        """Reset collected samples (useful for tests)."""
        with self._lock:
            self._batches.clear()
            self._frames.clear()
            self._camera_last_seen.clear()
            self._latest_active_streams = 0
            self._last_log_mono = time.monotonic()

    def record_batch(self, active_streams: int, batch_size: int, gpu_ms: float) -> None:
        """Record a YOLO micro-batch inference sample."""
        now = time.monotonic()
        with self._lock:
            self._latest_active_streams = max(0, int(active_streams))
            self._batches.append((now, int(active_streams), int(batch_size), float(gpu_ms)))

    def record_camera_frame(self, camera_id: str, e2e_ms: float) -> None:
        """Record end-to-end per-camera pipeline latency and emit periodic snapshots."""
        now = time.monotonic()
        snapshot: Optional[str] = None
        with self._lock:
            self._frames.append((now, camera_id, float(e2e_ms)))
            self._camera_last_seen[camera_id] = now
            if (now - self._last_log_mono) >= self._log_interval_sec and self._batches:
                self._last_log_mono = now
                snapshot = self._format_snapshot_locked(now)
        if snapshot is not None:
            print(snapshot, flush=True)

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = (len(sorted_vals) - 1) * (pct / 100.0)
        lo = int(idx)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    @staticmethod
    def _get_vram_stats() -> str:
        parts: list[str] = []
        try:
            import torch

            if torch.cuda.is_available():
                alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                parts.append(f"PyTorch alloc={alloc_mb:.0f}MB / reserved={res_mb:.0f}MB")
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
            ).strip()
            if out:
                first_line = out.splitlines()[0]
                used, total, util = [x.strip() for x in first_line.split(",")[:3]]
                parts.append(f"nvidia-smi={used}/{total}MB (GPU util={util}%)")
        except Exception:
            pass

        return " | ".join(parts) if parts else "N/A (no CUDA / nvidia-smi)"

    def _format_snapshot_locked(self, now: float) -> str:
        # Prune stale cameras inactive for > 5 seconds
        active_cams = [cid for cid, ts in self._camera_last_seen.items() if (now - ts) <= 5.0]
        cam_count = max(self._latest_active_streams, len(active_cams), 1 if self._frames else 0)

        gpu_times = sorted(b[3] for b in self._batches)
        batch_sizes = [b[2] for b in self._batches]
        total_batch_frames = sum(batch_sizes)
        total_gpu_ms = sum(gpu_times)

        avg_batch = (total_batch_frames / len(batch_sizes)) if batch_sizes else 0.0
        gpu_mean = (total_gpu_ms / len(gpu_times)) if gpu_times else 0.0
        gpu_p50 = self._percentile(gpu_times, 50.0)
        gpu_p95 = self._percentile(gpu_times, 95.0)
        gpu_cap_fps = (total_batch_frames * 1000.0 / total_gpu_ms) if total_gpu_ms > 0 else 0.0

        e2e_times = sorted(f[2] for f in self._frames)
        e2e_mean = (sum(e2e_times) / len(e2e_times)) if e2e_times else 0.0
        e2e_p50 = self._percentile(e2e_times, 50.0)
        e2e_p95 = self._percentile(e2e_times, 95.0)

        # Effective FPS per camera over the frame window
        eff_fps_total = 0.0
        if len(self._frames) >= 2:
            span = self._frames[-1][0] - self._frames[0][0]
            if span > 0.1:
                eff_fps_total = len(self._frames) / span
        eff_fps_per_cam = (eff_fps_total / cam_count) if cam_count > 0 else 0.0

        vram_str = self._get_vram_stats()

        lines = [
            "==================== [TABLE 6.7 BENCHMARK SNAPSHOT] ====================",
            f"Concurrent Active Streams : {cam_count} (observed cameras={len(active_cams)}, batcher_active={self._latest_active_streams})",
            f"GPU Micro-Batch (n={len(gpu_times):>4}) : avg_batch={avg_batch:.2f} | mean={gpu_mean:.2f}ms | P50={gpu_p50:.2f}ms | P95={gpu_p95:.2f}ms",
            f"GPU Capacity FPS          : {gpu_cap_fps:.1f} FPS",
            f"Effective Pipeline FPS    : total={eff_fps_total:.1f} FPS | per_camera={eff_fps_per_cam:.2f} FPS/cam",
            f"Pipeline AI E2E (n={len(e2e_times):>4})  : mean={e2e_mean:.2f}ms | P50={e2e_p50:.2f}ms | P95={e2e_p95:.2f}ms",
            f"VRAM / GPU Utilization    : {vram_str}",
            "==========================================================================",
        ]
        return "\n".join(lines)

    def format_snapshot(self) -> str:
        """Return a formatted snapshot string on demand."""
        with self._lock:
            return self._format_snapshot_locked(time.monotonic())

    def log_final_snapshot(self, min_interval_sec: float = 2.0) -> None:
        """Print a final snapshot if enough time has elapsed since the last log."""
        now = time.monotonic()
        snapshot: Optional[str] = None
        with self._lock:
            if self._batches and (now - self._last_log_mono) >= min_interval_sec:
                self._last_log_mono = now
                snapshot = self._format_snapshot_locked(now)
        if snapshot is not None:
            print(snapshot, flush=True)


table67_logger = Table67BenchmarkLogger()
