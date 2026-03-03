# Dataset Export Mode

Export COCO-17 keypoints from video files to JSONL files for HAR model training. Uses the same Hailo pipeline but saves keypoints locally instead of sending to the cloud.

## CLI flags

--export-skeleton, --video-dir PATH, --export-out PATH, --export-format jsonl|json, --max-videos N, --skip-existing 0|1.

## Output

Each video produces one .skeleton.jsonl file: line 1 meta (name, action_id, fps, frame_count, dimensions, schema_version); lines 2+ per-frame (frame_index, timestamp, persons with normalized keypoints). Output structure: export_out/A009/..., summary.csv.

## Example

```bash
python src/har_pose_app.py --export-skeleton \
  --video-dir /data/ntu_rgb --export-out /data/ntu_skeleton \
  --max-videos 100 --skip-existing 1 --no-display
```

## Mock cloud server (E2E)

For Phase 3 and Phase 4 testing (not Dataset Export). Accepts POST on any path, returns 200. Stdlib only.

**Terminal 1:** `python tools/mock_cloud_server.py --port 9999`

**Terminal 2:** `python src/har_pose_app.py --input rpi --no-display --show-fps --enable-cloud --cloud-url http://127.0.0.1:9999 --send-every-n-frames 10`

Expected: events_sent > 0, events_failed == 0, FPS near 30.
