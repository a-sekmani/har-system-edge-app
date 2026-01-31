"""
Unit tests for Phase 3 cloud client: CloudConfig, CloudSender, CloudSendQueue (queue drop policy, retry).
"""
from unittest.mock import MagicMock, patch

import pytest

from src.cloud_client import (
    CloudConfig,
    CloudSender,
    CloudSendQueue,
    DROP_POLICY_OLDEST,
    DROP_POLICY_NEWEST,
)


class TestCloudConfig:
    """CloudConfig defaults and get_api_key_from_env."""

    def test_get_api_key_from_env_when_empty(self):
        """When api_key is empty, get_api_key_from_env returns CLOUD_API_KEY env or ''."""
        cfg = CloudConfig(cloud_base_url="https://api.example.com", api_key="")
        with patch.dict("os.environ", {"CLOUD_API_KEY": "secret"}, clear=False):
            assert cfg.get_api_key_from_env() == "secret"
        with patch.dict("os.environ", {}, clear=False):
            assert cfg.get_api_key_from_env() == ""

    def test_get_api_key_uses_explicit_first(self):
        """When api_key is set, get_api_key_from_env returns it."""
        cfg = CloudConfig(cloud_base_url="https://x.com", api_key="explicit")
        with patch.dict("os.environ", {"CLOUD_API_KEY": "env"}, clear=False):
            assert cfg.get_api_key_from_env() == "explicit"


class TestCloudSender:
    """CloudSender.send: POST with correct URL/headers; failure returns False."""

    def test_send_returns_false_on_connection_error(self):
        """When POST fails (e.g. connection error), send returns False."""
        cfg = CloudConfig(cloud_base_url="http://invalid.example.local", api_key="", timeout_ms=100)
        sender = CloudSender(cfg)
        result = sender.send({"event_type": "frame_event", "source": {}, "frame": {}, "persons": []})
        assert result is False

    def test_send_posts_to_base_url_plus_path(self):
        """Sender builds URL from base_url and ingest_path."""
        cfg = CloudConfig(
            cloud_base_url="https://api.example.com",
            cloud_ingest_path="/v1/edge/events",
            api_key="",
        )
        sender = CloudSender(cfg)
        assert "/v1/edge/events" in sender.url
        assert "api.example.com" in sender.url


class TestCloudSendQueueDropPolicy:
    """Queue drops one when full; drop_policy oldest vs newest."""

    def test_enqueue_when_full_drops_oldest_and_increments_dropped(self):
        """When queue is at max size, enqueue drops oldest and increments events_dropped."""
        counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
        cfg = CloudConfig(
            cloud_base_url="http://x",
            max_queue_size=2,
            drop_policy=DROP_POLICY_OLDEST,
        )
        sender = CloudSender(cfg)
        queue = CloudSendQueue(cfg, sender, counters)
        queue.enqueue({"id": 1})
        queue.enqueue({"id": 2})
        assert queue.queue_depth() == 2
        queue.enqueue({"id": 3})
        assert queue.queue_depth() == 2
        assert counters["events_dropped"] == 1
        # First item (1) was dropped; queue has 2, 3
        assert [q.get("id") for q in queue._deque_inner] == [2, 3]

    def test_enqueue_when_full_drop_newest_drops_newest(self):
        """When drop_policy is newest, enqueue at capacity drops the newest (last) and appends."""
        counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
        cfg = CloudConfig(
            cloud_base_url="http://x",
            max_queue_size=2,
            drop_policy=DROP_POLICY_NEWEST,
        )
        sender = CloudSender(cfg)
        queue = CloudSendQueue(cfg, sender, counters)
        queue.enqueue({"id": 1})
        queue.enqueue({"id": 2})
        queue.enqueue({"id": 3})
        assert counters["events_dropped"] == 1
        assert queue.queue_depth() == 2
        assert [q.get("id") for q in queue._deque_inner] == [1, 3]

    def test_queue_depth_max_updated_on_enqueue(self):
        """queue_depth_max in counters is updated when enqueueing."""
        counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
        cfg = CloudConfig(cloud_base_url="http://x", max_queue_size=10)
        sender = CloudSender(cfg)
        queue = CloudSendQueue(cfg, sender, counters)
        queue.enqueue({})
        assert counters["queue_depth_max"] == 1
        queue.enqueue({})
        assert counters["queue_depth_max"] == 2


class TestCloudSendQueueDrainRetry:
    """drain_one: on send failure after max_retries, events_failed and events_dropped incremented."""

    def test_drain_one_empty_returns_false(self):
        """drain_one on empty queue returns False."""
        counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
        cfg = CloudConfig(cloud_base_url="http://x")
        sender = CloudSender(cfg)
        queue = CloudSendQueue(cfg, sender, counters)
        assert queue.drain_one() is False

    def test_drain_one_success_increments_events_sent(self):
        """When send returns True, drain_one increments events_sent."""
        counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
        cfg = CloudConfig(cloud_base_url="http://x")
        mock_sender = MagicMock()
        mock_sender.send = MagicMock(return_value=True)
        queue = CloudSendQueue(cfg, mock_sender, counters)
        queue.enqueue({"event_type": "frame_event"})
        result = queue.drain_one()
        assert result is True
        assert counters["events_sent"] == 1
        assert counters["events_failed"] == 0
        assert counters["events_dropped"] == 0

    def test_drain_one_failure_after_retries_increments_failed_and_dropped(self):
        """When send always returns False, after max_retries events_failed and events_dropped increment."""
        counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
        cfg = CloudConfig(cloud_base_url="http://x", max_retries=1, backoff_seconds=0)
        mock_sender = MagicMock()
        mock_sender.send = MagicMock(return_value=False)
        queue = CloudSendQueue(cfg, mock_sender, counters)
        queue.enqueue({"event_type": "frame_event"})
        result = queue.drain_one()
        assert result is False
        assert counters["events_sent"] == 0
        assert counters["events_failed"] == 1
        assert counters["events_dropped"] == 1
