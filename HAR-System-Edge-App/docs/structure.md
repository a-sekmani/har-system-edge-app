# Project Structure

```
HAR-System-Edge-App/
├── src/
│   ├── frame_event.py            # FrameEvent, PersonPose (with track_id), COCO-17, validation
│   ├── tracker.py                # Phase 2: TrackingConfig, FallbackTracker, get_metadata_track_id
│   ├── cloud_schema.py           # Phase 3: Cloud Event schema, build_cloud_payload()
│   ├── cloud_client.py           # Phase 3: CloudSender (HTTP POST), CloudSendQueue, retry, drop policy
│   ├── window_schema.py          # Phase 4: WindowPayload schema, keypoints normalization, created_at ISO 8601
│   ├── window_assembler.py       # Phase 4: WindowAssembler, buffer per track, non-overlap windows
│   ├── windows_client.py         # Phase 4: WindowsSender (HTTP POST), WindowsSendQueue
│   ├── skeleton_exporter.py      # Dataset Export: COCO-17 keypoints to JSONL files
│   ├── face/                     # Face recognition (gallery, recognizer, tracker binding)
│   │   ├── gallery_client.py     # GET face-gallery and updated_at (version endpoint) from cloud
│   │   ├── gallery_store.py      # Load/save gallery cache
│   │   ├── recognizer.py         # InsightFace detect + embedding + match
│   │   ├── tracker_binding.py    # Bind face identity to pose track_id
│   │   └── schemas.py            # FaceDetection, FaceGallery, PersonAttachment
│   └── har_pose_app.py           # Main application (Phase 1–4 + Dataset Export + Face)
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── test_parser.py
│   ├── test_fps_tracker.py
│   ├── test_har_user_data.py
│   ├── test_callback.py
│   ├── test_har_pose_app.py
│   ├── test_frame_event.py
│   ├── test_pose_extraction.py
│   ├── test_validation.py
│   ├── test_phase1_logic.py
│   ├── test_tracker.py           # Phase 2: tracker core tests
│   ├── test_phase2_logic.py       # Phase 2: parse_counters, check_phase2_conditions
│   ├── test_cloud_schema.py      # Phase 3: payload schema
│   ├── test_cloud_client.py       # Phase 3: queue, retry, CloudSender
│   ├── test_window_schema.py     # Phase 4: WindowPayload, created_at ISO 8601, keypoints normalization
│   ├── test_window_assembler.py  # Phase 4: WindowAssembler
│   ├── test_windows_client.py    # Phase 4: WindowsConfig, WindowsSender, WindowsSendQueue
│   ├── test_phase4_logic.py      # Phase 4: parse_counters (windows_*)
│   ├── test_phase_face_fps_windows.py  # Face worker, gallery URL, FPS Person
│   └── face/                     # Face recognition unit tests
│       ├── test_gallery_client.py
│       ├── test_gallery_store.py
│       ├── test_recognizer_match.py
│       └── test_tracker_binding.py
├── scripts/
│   ├── request_face_gallery.py   # Request face gallery only; save to local cache
│   └── check_cloud_gallery.py    # Request gallery updated_at + full gallery; print summary
├── docs/                         # Documentation (this folder)
├── test_phase0.py                # Phase 0 acceptance test
├── test_phase1.py                # Phase 1 acceptance test
├── test_phase2.py                # Phase 2 acceptance test
├── test_phase3.py                # Phase 3 acceptance test
├── test_phase4.py                # Phase 4 acceptance test
├── tools/
│   └── mock_cloud_server.py      # Mock HTTP server for E2E (Phase 3/4)
├── pytest.ini
├── README.md
└── requirements.txt
```

See [Testing](testing.md) for what each test module covers.
