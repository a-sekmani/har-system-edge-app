# Phase 3 — Cloud Streaming

Phase 3 streams tracks and 17 keypoints (no images) from the edge to a configurable cloud endpoint. When `--enable-cloud` is false, Phase 1 and Phase 2 behaviour and counters are unchanged.

## Cloud Event schema (JSON)

- **event_type**: "frame_event"
- **source**: device_id, session_id, model, tracking_source
- **frame**: frame_index, ts_monotonic_ms or ts_unix_ms, image_w, image_h, optional fps_current, fps_avg
- **persons**: list with track_id (int), bbox_xyxy, optional score, keypoints (17; name, x, y, c), keypoints_format "coco17", coords "pixel"

No image data is sent.

## Transport

HTTP POST to base URL + ingest path. Auth via X-API-Key header (or env CLOUD_API_KEY). Configurable timeout, retries, TLS verification.

## Queue and drop policy

In-memory queue with max_queue_size. When full: drop oldest (default) or drop newest (--drop-policy). On send failure: retry up to max_retries; then drop and increment events_failed/events_dropped. Sending does not block the pipeline.

## Rate control

`--send-every-n-frames` (default 1) controls how often an event is built and enqueued.

## Phase 3 counters

events_built, events_sent, events_failed, events_dropped, queue_depth, queue_depth_max.

## CLI flags (Phase 3)

--enable-cloud, --cloud-url, --cloud-api-key, --cloud-ingest-path (default /v1/edge/events), --send-every-n-frames (default 1), --max-queue-size (default 1000), --send-timeout-ms (default 5000), --max-retries (default 2), --drop-policy (default oldest), --dry-run, --no-verify-tls (TLS verification on by default).

## Acceptance test

```bash
python acceptance_tests/test_phase3.py
```

Checks: (1) enable_cloud false → Phase 3 counters zero; (2) dry-run criteria; (3) send_every_n_frames=2; (4) local HTTP sink; (5) invalid URL → events_failed > 0.
