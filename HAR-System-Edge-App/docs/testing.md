# Testing

## Unit tests (pytest)

From HAR-System-Edge-App (parent directory must provide hailo_apps on PYTHONPATH):

```bash
cd HAR-System-Edge-App
pytest tests/ -v
```

Or from repository root: `pytest HAR-System-Edge-App/tests/ -v`

**Coverage:** test_parser (parser and Phase 2/Face flags), test_fps_tracker, test_har_user_data, test_callback, test_har_pose_app, test_frame_event, test_pose_extraction, test_validation, test_phase1_logic (parse_counters, FPS Person line), test_tracker, test_phase2_logic, test_cloud_schema, test_cloud_client, test_window_schema (keypoints_to_17x3_normalized pad/sentinel, WindowPayload, created_at ISO 8601, fps clamp, to_dict int rounding), test_window_assembler (non-overlap, per-track, eviction, empty persons, two persons per event, image missing dimensions), test_windows_client (WindowsConfig, WindowsSender URL/send, WindowsSendQueue drop policy and counters), test_phase4_logic, test_phase_face_fps_windows (face worker, gallery URL, enable-face, FPS Person), tests/face (gallery_client, gallery_store, recognizer_match, tracker_binding).

## Acceptance tests

- **Phase 0:** `python test_phase0.py` - baseline run
- **Phase 1:** `python test_phase1.py` - 30 s, parse counters, PASS/FAIL (frame_events, invalid_*, frames_with_persons, frames_with_landmarks, keypoints_len_not_17)
- **Phase 2:** `python test_phase2.py` - 30 s, fallback tracking, single-person criteria (unique_track_ids, id_switch_suspected)
- **Phase 3:** `python test_phase3.py` - enable_cloud false, dry-run, send_every_n_frames, local HTTP sink, invalid URL
- **Phase 4:** `python test_phase4.py` - dry-run, local HTTP sink (windows_sent, keypoints shape), invalid URL

## Acceptance criteria summary

Application runs without errors for 5-10 minutes; FPS shown or logged; no errors in logs. Phase 1: frame_events >= 95% of total_frames, invalid_caps/validate 0, keypoints_len_not_17 0, frames_with_persons >= 30, frames_with_landmarks >= 80% of frames_with_persons. Phase 2: same plus unique_track_ids <= 2, id_switch_suspected == 0. Phase 3: dry-run and live send criteria. Phase 4: windows_sent > 0, keypoints [30][17][3], windows_failed == 0.
