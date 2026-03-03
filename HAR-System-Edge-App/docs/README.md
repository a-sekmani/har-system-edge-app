# HAR-System-Edge-App Documentation

Full documentation for the HAR (Human Activity Recognition) edge application. The app uses the hailo-apps library for pose estimation from a Raspberry Pi camera and supports baseline capture, tracking, cloud streaming, window ingest, and optional face recognition.

## Documentation Index

- [Setup and requirements](setup.md)
- [Project structure](structure.md)
- [Phase 0 - Baseline](phase0.md) - Run with/without display, acceptance test
- [Phase 1 - Frame events](phase1-frame-event.md)
- [Phase 2 - Tracking](phase2-tracking.md)
- [Phase 3 - Cloud streaming](phase3-cloud-streaming.md)
- [Phase 4 - Windows ingest](phase4-windows.md)
- [Face recognition](face-recognition.md)
- [Scripts](scripts.md)
- [Dataset export](dataset-export.md)
- [Command-line options](command-line-options.md)
- [Testing](testing.md)

## Quick start

From the project root (HAR-System-Edge-App):

```bash
python src/har_pose_app.py --input rpi --show-fps
python src/har_pose_app.py --input rpi --no-display --show-fps
```

See [Setup and requirements](setup.md) for dependencies.
