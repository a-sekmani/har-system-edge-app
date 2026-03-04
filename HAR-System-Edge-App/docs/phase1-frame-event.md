# Phase 1 — Raw Pose Data (FrameEvent)

The app produces one **FrameEvent** per frame: frame number, timestamp, image size, and a list of **PersonPose** (bbox in pixels, detection confidence, 17 keypoints in COCO order). There is no tracking (track_id) or cloud upload in Phase 1.

## Data model

- **COCO-17 keypoint order** (indices 0–16): nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
- **Missing keypoint sentinel**: `[-1, -1, 0.0]` (same type for all keypoints; no null).
- Each person has `bbox` `[x1, y1, x2, y2]` in pixels, `bbox_conf`, and exactly 17 keypoints `[x, y, c]`.
- Invalid frames are skipped and counted; the app does not crash.

## Acceptance test

```bash
python acceptance_tests/test_phase1.py
```

The script runs the app with `--no-display` for 30 seconds, parses counters from the log, and exits 0 (PASS) or 1 (FAIL). One person in view is recommended.

**Phase 1 passes only if all of the following hold:**

| Condition | Requirement |
|-----------|-------------|
| frame_events | >= 0.95 * total_frames |
| invalid_caps | == 0 |
| invalid_validate | == 0 |
| frames_keypoints_len_not_17 | == 0 |
| frames_with_persons | >= MIN_PERSON_FRAMES (30) |
| frames_with_landmarks | >= 0.8 * frames_with_persons |

**Summary counters** (logged every FPS interval and at exit): frames_with_persons, frames_no_persons, persons_total, frames_with_landmarks, frames_keypoints_len_not_17.
