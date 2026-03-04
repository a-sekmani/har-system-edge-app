# Phase 4 — Windows Ingest

Phase 4 sends sliding windows of keypoints (30 frames per track) as a single payload for HAR inference in the cloud.

## Window schema (JSON payload)

- **id**: UUID for the window
- **created_at**: Single date-time in ISO 8601 with timezone. Format: YYYY-MM-DDTHH:mm:ss[.sss](Z|+00:00|±HH:mm). The edge uses UTC with Z and 3 decimal places (e.g. 2026-02-24T11:32:05.123Z), generated at window creation/send time so the cloud shows correct Date and Time in Recent Windows.
- **device_id**, **camera_id**, **session_id**, **track_id**, **ts_start_ms**, **ts_end_ms**, **fps**, **window_size**
- **keypoints**: [T][17][3] — T frames, 17 COCO keypoints, 3 values (x, y, confidence) normalized to [0,1]. Missing keypoints: [0.0, 0.0, 0.0].

Keypoint normalization: x_norm = x_pixel / image_w (clamped), y_norm = y_pixel / image_h (clamped).

## CLI flags (Phase 4)

--cloud-mode frames|windows, --cloud-windows-path (default /v1/windows/ingest), --window-size (30), --window-stride (30), --window-max-buffers (50), --max-windows-queue-size (500), --windows-drop-policy (oldest|newest).

## Phase 4 counters

windows_built, windows_sent, windows_failed, windows_dropped, windows_queue_depth, windows_queue_depth_max.

## Example

```bash
python src/har_pose_app.py --input rpi --no-display \
  --enable-cloud --cloud-mode windows \
  --cloud-url http://192.168.1.105:8000 --cloud-api-key dev-key
```

## Acceptance test

```bash
python acceptance_tests/test_phase4.py
```

(1) Dry-run: windows_built > 0, windows_sent == 0; (2) Local HTTP sink: windows_sent > 0, keypoints [30][17][3]; (3) Invalid URL: windows_failed > 0.
