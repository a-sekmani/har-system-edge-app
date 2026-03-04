# HAR-System-Edge-App

Edge application for the HAR (Human Activity Recognition) system. Uses the hailo-apps library for pose estimation from a Raspberry Pi camera. Supports baseline capture (Phase 0), raw frame events (Phase 1), tracking (Phase 2), cloud streaming (Phase 3), window ingest (Phase 4), optional face recognition, and dataset export.

## Documentation

Full documentation is in the **[docs](docs/README.md)** folder:

| Topic | Document |
|-------|----------|
| Setup, requirements, installation | [docs/setup.md](docs/setup.md) |
| Project structure | [docs/structure.md](docs/structure.md) |
| Phase 0 - Baseline | [docs/phase0.md](docs/phase0.md) |
| Phase 1 - Frame events | [docs/phase1-frame-event.md](docs/phase1-frame-event.md) |
| Phase 2 - Tracking | [docs/phase2-tracking.md](docs/phase2-tracking.md) |
| Phase 3 - Cloud streaming | [docs/phase3-cloud-streaming.md](docs/phase3-cloud-streaming.md) |
| Phase 4 - Windows ingest | [docs/phase4-windows.md](docs/phase4-windows.md) |
| Face recognition | [docs/face-recognition.md](docs/face-recognition.md) |
| Scripts (no camera) | [docs/scripts.md](docs/scripts.md) |
| Dataset export, mock server | [docs/dataset-export.md](docs/dataset-export.md) |
| CLI options reference | [docs/command-line-options.md](docs/command-line-options.md) |
| Testing | [docs/testing.md](docs/testing.md) |

## Quick start

**Requirements:** Python 3.10+, hailo-apps, Raspberry Pi with camera, Hailo device. See [docs/setup.md](docs/setup.md).

```bash
cd HAR-System-Edge-App
source ../venv_hailo_apps/bin/activate   # or your venv
python src/har_pose_app.py --input rpi --no-display --show-fps
```

**With cloud windows and face recognition:**

```bash
python src/har_pose_app.py --input rpi --no-display \
  --enable-cloud --cloud-mode windows --enable-face \
  --cloud-url http://192.168.1.105:8000 --cloud-api-key dev-key
```

## Acceptance tests

From `HAR-System-Edge-App`: `python acceptance_tests/test_phase0.py`, `python acceptance_tests/test_phase1.py`, `python acceptance_tests/test_phase2.py`, `python acceptance_tests/test_phase3.py`, `python acceptance_tests/test_phase4.py`. See [docs/testing.md](docs/testing.md).

## Unit tests

```bash
cd HAR-System-Edge-App
pytest tests/ -v
```

See [docs/testing.md](docs/testing.md) for coverage and criteria.
