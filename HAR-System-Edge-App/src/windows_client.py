"""
Phase 4: HTTP client for window ingest — POST to /v1/windows/ingest from a worker thread.
Short timeouts (connect ~0.5–1s, read 1–2s); bounded queue with drop policy; non-blocking enqueue.
Saves last payload to /tmp/last_window.json and logs keypoints shape (T, K, C) for 422 debugging.
"""

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from src.window_schema import WindowPayload

_LOG = logging.getLogger(__name__)
LAST_WINDOW_PATH = os.environ.get("LAST_WINDOW_DEBUG_PATH", "/tmp/last_window.json")

DROP_POLICY_OLDEST: Literal["oldest"] = "oldest"
DROP_POLICY_NEWEST: Literal["newest"] = "newest"


@dataclass
class WindowsConfig:
    """Config for windows ingest: URL, path, API key, short timeouts, queue size, drop policy."""

    cloud_base_url: str
    cloud_windows_path: str = "/v1/windows/ingest"
    api_key: str = ""
    connect_timeout_sec: float = 0.5
    read_timeout_sec: float = 2.0
    verify_tls: bool = True
    max_queue_size: int = 500
    drop_policy: Literal["oldest", "newest"] = DROP_POLICY_OLDEST

    def get_api_key_from_env(self) -> str:
        if self.api_key:
            return self.api_key
        return os.environ.get("CLOUD_API_KEY", "")


class WindowsSender:
    """POST JSON window payload to cloud_windows_path. Returns True on 2xx."""

    def __init__(self, config: WindowsConfig):
        self.config = config
        base = config.cloud_base_url.rstrip("/")
        path = config.cloud_windows_path if config.cloud_windows_path.startswith("/") else "/" + config.cloud_windows_path
        self.url = base + path

    def send(self, payload: Dict[str, Any]) -> bool:
        # Normalize for cloud: ts_start_ms, ts_end_ms as int
        out = dict(payload)
        if "ts_start_ms" in out and out["ts_start_ms"] is not None:
            out["ts_start_ms"] = int(round(float(out["ts_start_ms"])))
        if "ts_end_ms" in out and out["ts_end_ms"] is not None:
            out["ts_end_ms"] = int(round(float(out["ts_end_ms"])))
        if "track_id" in out and out["track_id"] is not None:
            out["track_id"] = int(out["track_id"])
        if "window_size" in out and out["window_size"] is not None:
            out["window_size"] = int(out["window_size"])

        # Debug: save last window and log dimensions (T=30, K=17, C=3)
        try:
            with open(LAST_WINDOW_PATH, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            kp = out.get("keypoints") or []
            T = len(kp)
            K = len(kp[0]) if T else 0
            C = len(kp[0][0]) if K else 0
            _LOG.info(
                "[window ingest] last payload saved to %s | keypoints shape T=%s K=%s C=%s (expected 30 17 3)",
                LAST_WINDOW_PATH, T, K, C,
            )
        except OSError as e:
            _LOG.debug("Could not write last_window.json: %s", e)

        api_key = self.config.get_api_key_from_env()
        body = json.dumps(out, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        if api_key:
            req.add_header("X-API-Key", api_key)
        timeout_sec = max(0.1, self.config.connect_timeout_sec + self.config.read_timeout_sec)
        try:
            if not self.config.verify_tls:
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
                    code = resp.getcode()
            else:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    code = resp.getcode()
            return 200 <= code < 300
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read()
            except Exception:
                pass
            msg = body.decode("utf-8", errors="replace") if body else str(e)
            _LOG.warning("[window ingest] HTTP %s | response: %s", e.code, msg[:500])
            if e.code == 422:
                _LOG.warning("[window ingest] 422 detail (full): %s", msg)
            return False
        except (urllib.error.URLError, OSError, TimeoutError):
            return False


class WindowsSendQueue:
    """
    Bounded queue for window payloads; worker thread sends in background.
    enqueue() is non-blocking; when full, drop oldest or newest per policy.
    Counters: windows_sent, windows_failed, windows_dropped, windows_queue_depth_max.
    """

    def __init__(
        self,
        config: WindowsConfig,
        sender: WindowsSender,
        counters: Dict[str, Any],
    ):
        self.config = config
        self.sender = sender
        self.counters = counters
        self._max_size = config.max_queue_size
        self._drop_policy = config.drop_policy
        self._deque: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def enqueue(self, payload: Dict[str, Any]) -> None:
        """Non-blocking: append payload; if at capacity drop one (oldest or newest) then append."""
        with self._lock:
            while len(self._deque) >= self._max_size:
                if self._drop_policy == DROP_POLICY_OLDEST:
                    self._deque.popleft()
                else:
                    self._deque.pop()
                self.counters["windows_dropped"] = self.counters.get("windows_dropped", 0) + 1
            self._deque.append(payload)
            depth = len(self._deque)
            self.counters["windows_queue_depth_max"] = max(
                self.counters.get("windows_queue_depth_max", 0), depth
            )

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._deque)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            payload = None
            with self._lock:
                if self._deque:
                    payload = self._deque.popleft()
            if payload is not None:
                if self.sender.send(payload):
                    self.counters["windows_sent"] = self.counters.get("windows_sent", 0) + 1
                else:
                    self.counters["windows_failed"] = self.counters.get("windows_failed", 0) + 1
                    self.counters["windows_dropped"] = self.counters.get("windows_dropped", 0) + 1
            else:
                self._stop.wait(timeout=0.05)
        return None

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
