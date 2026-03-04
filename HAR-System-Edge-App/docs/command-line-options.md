# Command-Line Options Reference

Quick reference. See phase and feature docs for details.

**Input/Display:** --input, --no-display, --show-fps, --log-pose-summary, --dump-frames

**Tracking:** --tracking-source, --max-missing-frames, --iou-threshold, --min-bbox-area, --min-bbox-height, --min-pose-confidence, --log-tracking-summary

**Cloud:** --enable-cloud, --cloud-url, --cloud-api-key, --cloud-mode, --cloud-ingest-path, --cloud-windows-path, --send-every-n-frames, --max-queue-size, --send-timeout-ms, --max-retries, --drop-policy, --dry-run, --verify-tls, --no-verify-tls

**Windows:** --window-size, --window-stride, --window-max-buffers, --max-windows-queue-size (default 500), --windows-drop-policy, --window-attach-person

**Face:** --enable-face, --face-gallery-url, --face-gallery-cache (persistent gallery dir; default: project `face_gallery/`; sync from cloud only when cloud updated_at is newer), --face-det-size default 256, --face-max-faces default 1, --face-skip-frames default 10, and related face options. See [Face recognition](face-recognition.md) for sync behavior (updated_at only, never overwrite on failure).

**Dataset export:** --export-skeleton, --video-dir, --export-out, --export-format, --max-videos, --skip-existing

Run with --help for the full list.
