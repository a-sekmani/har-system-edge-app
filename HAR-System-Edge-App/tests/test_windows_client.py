"""Unit tests for Phase 4 windows client: WindowsConfig, WindowsSender, WindowsSendQueue."""

from unittest.mock import patch

import pytest

from src.windows_client import (
    DROP_POLICY_OLDEST,
    DROP_POLICY_NEWEST,
    WindowsConfig,
    WindowsSender,
    WindowsSendQueue,
)


class TestWindowsConfig:
    """WindowsConfig defaults and get_api_key_from_env."""

    def test_get_api_key_from_env_when_empty(self):
        cfg = WindowsConfig(cloud_base_url="http://example.com", api_key="")
        with patch.dict("os.environ", {"CLOUD_API_KEY": "dev-key"}, clear=False):
            assert cfg.get_api_key_from_env() == "dev-key"
        with patch.dict("os.environ", {}, clear=False):
            assert cfg.get_api_key_from_env() == ""

    def test_get_api_key_uses_explicit_first(self):
        cfg = WindowsConfig(cloud_base_url="http://example.com", api_key="explicit")
        with patch.dict("os.environ", {"CLOUD_API_KEY": "env-key"}, clear=False):
            assert cfg.get_api_key_from_env() == "explicit"


class TestWindowsSender:
    """WindowsSender URL construction and send (with mocked urlopen)."""

    def test_url_built_from_base_and_path(self):
        cfg = WindowsConfig(cloud_base_url="http://192.168.1.106:8000", cloud_windows_path="/v1/windows/ingest")
        sender = WindowsSender(cfg)
        assert "192.168.1.106" in sender.url
        assert "/v1/windows/ingest" in sender.url

    def test_url_path_without_leading_slash_gets_prefixed(self):
        cfg = WindowsConfig(cloud_base_url="http://x", cloud_windows_path="v1/windows/ingest")
        sender = WindowsSender(cfg)
        assert sender.url.endswith("/v1/windows/ingest")

    def test_send_returns_true_on_2xx(self):
        cfg = WindowsConfig(cloud_base_url="http://localhost", api_key="")
        sender = WindowsSender(cfg)
        payload = {"id": "w1", "created_at": "2026-01-01T12:00:00.000Z", "device_id": "d", "camera_id": "c",
                   "session_id": "s", "track_id": 1, "ts_start_ms": 0, "ts_end_ms": 1000, "fps": 30,
                   "window_size": 30, "keypoints": [[[0.0, 0.0, 0.0]] * 17] * 30}
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__ = lambda self: self
            m.return_value.__exit__ = lambda *a: None
            m.return_value.getcode = lambda: 200
            result = sender.send(payload)
        assert result is True

    def test_send_returns_false_on_connection_error(self):
        cfg = WindowsConfig(cloud_base_url="http://invalid.nonexistent.local", api_key="")
        sender = WindowsSender(cfg)
        payload = {"id": "w1", "created_at": "2026-01-01T12:00:00.000Z", "device_id": "d", "camera_id": "c",
                   "session_id": "s", "track_id": 1, "ts_start_ms": 0, "ts_end_ms": 1000, "fps": 30,
                   "window_size": 30, "keypoints": [[[0.0, 0.0, 0.0]] * 17] * 30}
        result = sender.send(payload)
        assert result is False


class TestWindowsSendQueue:
    """WindowsSendQueue enqueue, drop policy, counters."""

    def test_enqueue_when_full_drops_oldest_and_increments_dropped(self):
        counters = {"windows_sent": 0, "windows_failed": 0, "windows_dropped": 0, "windows_queue_depth_max": 0}
        cfg = WindowsConfig(cloud_base_url="http://x", max_queue_size=2, drop_policy=DROP_POLICY_OLDEST)
        sender = WindowsSender(cfg)
        queue = WindowsSendQueue(cfg, sender, counters)
        queue.enqueue({"id": "a"})
        queue.enqueue({"id": "b"})
        assert queue.queue_depth() == 2
        queue.enqueue({"id": "c"})
        assert queue.queue_depth() == 2
        assert counters["windows_dropped"] == 1
        assert counters["windows_queue_depth_max"] == 2

    def test_enqueue_when_full_drop_newest_drops_newest(self):
        counters = {"windows_sent": 0, "windows_failed": 0, "windows_dropped": 0, "windows_queue_depth_max": 0}
        cfg = WindowsConfig(cloud_base_url="http://x", max_queue_size=2, drop_policy=DROP_POLICY_NEWEST)
        sender = WindowsSender(cfg)
        queue = WindowsSendQueue(cfg, sender, counters)
        queue.enqueue({"id": "a"})
        queue.enqueue({"id": "b"})
        queue.enqueue({"id": "c"})
        assert queue.queue_depth() == 2
        assert counters["windows_dropped"] == 1
        with queue._lock:
            ids = [p.get("id") for p in list(queue._deque)]
        assert ids == ["a", "c"]

    def test_queue_depth_max_updated_on_enqueue(self):
        counters = {"windows_sent": 0, "windows_failed": 0, "windows_dropped": 0, "windows_queue_depth_max": 0}
        cfg = WindowsConfig(cloud_base_url="http://x", max_queue_size=10)
        sender = WindowsSender(cfg)
        queue = WindowsSendQueue(cfg, sender, counters)
        queue.enqueue({"id": "1"})
        assert counters["windows_queue_depth_max"] == 1
        queue.enqueue({"id": "2"})
        assert counters["windows_queue_depth_max"] == 2
