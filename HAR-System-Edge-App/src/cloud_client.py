"""
Phase 3: HTTP transport and in-memory queue for cloud event streaming.

- CloudConfig: base URL, path, api_key, timeout, retries, backoff (constant), verify_tls.
- CloudSender: POST JSON to endpoint with X-API-Key or Authorization header; returns bool.
- CloudSendQueue: bounded queue; drop oldest when full (policy configurable); drain_one with retry.
Single transport in Phase 3: HTTP only. WS/SSE can be added later.
Uses stdlib urllib.request; no new dependency.
"""

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Literal, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DROP_POLICY_OLDEST: Literal["oldest"] = "oldest"
DROP_POLICY_NEWEST: Literal["newest"] = "newest"


@dataclass
class CloudConfig:
    """Cloud send configuration."""

    cloud_base_url: str
    cloud_ingest_path: str = "/v1/edge/events"
    api_key: str = ""
    timeout_ms: int = 5000
    max_retries: int = 2
    backoff_seconds: float = 0.5  # constant backoff between retries
    verify_tls: bool = True
    compression: Optional[str] = None  # e.g. "gzip"; None = no compression
    max_queue_size: int = 1000
    drop_policy: Literal["oldest", "newest"] = DROP_POLICY_OLDEST

    def get_api_key_from_env(self) -> str:
        """Use CLOUD_API_KEY env if api_key not set."""
        if self.api_key:
            return self.api_key
        return os.environ.get("CLOUD_API_KEY", "")


# ---------------------------------------------------------------------------
# CloudSender (HTTP POST)
# ---------------------------------------------------------------------------
class CloudSender:
    """
    HTTP client that POSTs JSON payloads to cloud_ingest_path.
    On timeout or 4xx/5xx returns False; caller/queue handles retry and drop.
    """

    def __init__(self, config: CloudConfig):
        self.config = config
        base = config.cloud_base_url.rstrip("/")
        path = config.cloud_ingest_path if config.cloud_ingest_path.startswith("/") else "/" + config.cloud_ingest_path
        self.url = base + path

    def send(self, payload: Dict[str, Any]) -> bool:
        """
        POST payload as JSON. Returns True on 2xx, False on timeout or error.
        Uses X-API-Key header if api_key is set; otherwise no auth header.
        """
        api_key = self.config.get_api_key_from_env()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.config.compression == "gzip":
            import gzip
            body = gzip.compress(body)
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        if api_key:
            req.add_header("X-API-Key", api_key)
        if self.config.compression == "gzip":
            req.add_header("Content-Encoding", "gzip")

        timeout_sec = max(0.1, self.config.timeout_ms / 1000.0)
        try:
            # Optional: disable SSL verify if verify_tls is False
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
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
            return False


# ---------------------------------------------------------------------------
# Queue with drop policy and drain with retry
# ---------------------------------------------------------------------------
class CloudSendQueue:
    """
    Bounded in-memory queue. When full, drop one (oldest or newest per drop_policy).
    drain_one() pops one payload, sends with retries; updates counters.
    Counters: events_sent, events_failed, events_dropped, queue_depth_max (caller provides mutable container).
    """

    def __init__(
        self,
        config: CloudConfig,
        sender: CloudSender,
        counters: Dict[str, Any],
    ):
        self.config = config
        self.sender = sender
        self.counters = counters  # events_sent, events_failed, events_dropped, queue_depth_max
        # Unbounded deque; we enforce max ourselves so we can apply drop_policy (oldest vs newest)
        self._max_size = config.max_queue_size
        self._drop_policy = config.drop_policy
        self._deque_inner: Deque[Dict[str, Any]] = deque()  # unbounded; we enforce max ourselves

    def enqueue(self, payload: Dict[str, Any]) -> None:
        """Enqueue one payload. If at capacity, drop one (oldest or newest) then append. Updates queue_depth_max."""
        while len(self._deque_inner) >= self._max_size:
            if self._drop_policy == DROP_POLICY_OLDEST:
                self._deque_inner.popleft()
            else:
                self._deque_inner.pop()
            self.counters["events_dropped"] = self.counters.get("events_dropped", 0) + 1
        self._deque_inner.append(payload)
        depth = len(self._deque_inner)
        self.counters["queue_depth_max"] = max(self.counters.get("queue_depth_max", 0), depth)

    def queue_depth(self) -> int:
        return len(self._deque_inner)

    def drain_one(self) -> bool:
        """
        Pop one payload, send with retries. On success: events_sent += 1. On failure after retries: events_failed += 1, events_dropped += 1.
        Returns True if one item was sent successfully, False if queue empty or send failed after retries.
        """
        if not self._deque_inner:
            return False
        payload = self._deque_inner.popleft()
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            if self.sender.send(payload):
                self.counters["events_sent"] = self.counters.get("events_sent", 0) + 1
                return True
            if attempt < self.config.max_retries:
                time.sleep(self.config.backoff_seconds)
        self.counters["events_failed"] = self.counters.get("events_failed", 0) + 1
        self.counters["events_dropped"] = self.counters.get("events_dropped", 0) + 1
        return False
