# Phase 2 — Tracking (Track IDs)

Every person in a FrameEvent has a **track_id** (int). Keypoints and validation are unchanged; a missing track_id does not make the frame invalid.

## Track ID

- **track_id in PersonPose**: Required; use -1 (TRACK_ID_UNKNOWN) only when tracking cannot assign an id.
- **Source**: `metadata` — use Hailo HAILO_UNIQUE_ID when available; otherwise fallback tracker. `fallback` — IoU-based tracker for all detections.
- **Lifecycle**: Tracks expire after max_missing_frames (or optional max_track_age_seconds). Re-appearing detections get a new id after expiry.

## Detection filter (reduce ghost tracks)

Before tracking, detections can be filtered: `--min-bbox-area A`, `--min-bbox-height H`, `--min-pose-confidence C` (0–1, average keypoint confidence). This helps when the model outputs extra ghost detections.

## Phase 2 counters

| Counter | Meaning |
|--------|--------|
| unique_track_ids | Distinct track ids ever seen |
| new_tracks_created | First-time track IDs (metadata or fallback) |
| tracks_ended | Tracks removed after max_missing_frames / max_track_age |
| id_switch_suspected | Heuristic: same frame ids but assignment flip |
| multi_person_frames | Frames with >= 2 persons |
| detections_total | Raw person detections (before filter) |
| filtered_detections_total | Detections excluded by filter options |

## CLI flags (Phase 2)

- `--tracking-source metadata|fallback` (default: metadata)
- `--max-missing-frames N` (default: 15)
- `--iou-threshold X` (default: 0.3)
- `--min-bbox-area A`, `--min-bbox-height H`, `--min-pose-confidence C`
- `--log-tracking-summary`

## Acceptance test

```bash
python test_phase2.py
```

Runs with `--no-display` and `--tracking-source fallback` for 30 seconds. Single-person criteria: frame_events >= 95% of total_frames, invalid_caps/validate == 0, frames_with_persons >= 30, unique_track_ids <= 2, id_switch_suspected == 0.
