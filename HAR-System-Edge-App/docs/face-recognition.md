# Face Recognition (Optional)

With --enable-face, the app can fetch a face gallery from the cloud, run face detection and recognition on CPU (InsightFace), and bind identities to pose tracks. It can run with local cache only. Windows can include a person object (person_id, name, face_conf, source, verified_at_ms).

**Gallery URL priority:** --face-gallery-url, --cloud-url, FACE_GALLERY_URL, CLOUD_URL. Request gallery without camera: `python scripts/request_face_gallery.py`.

**Face thread:** Runs in a dedicated background thread (queue maxsize=1) so the pipeline stays at about 30 FPS. FPS log includes "Person: name or Unknown"; separate line "Persons on screen: ...". Names are not drawn on video.

**Defaults:** --face-skip-frames 10, --face-det-size 256, --face-max-faces 1, --face-sim-threshold 0.45.

## Gallery sync behavior

- **Update decision:** Based only on **updated_at** (ISO 8601). The app does not use a version number. On each run and periodically (e.g. every 1 minute), it GETs the version endpoint; if the cloud returns an **updated_at** newer than the local copy, it fetches the full gallery and updates the cache. If the date is the same or older, no update is performed.
- **Never overwrite on failure:** If the request for updated_at fails, or the full gallery fetch fails, or any exception occurs, the app **keeps the existing local gallery** and logs that the local copy was unchanged. The local gallery is always preserved; a failed sync never clears or replaces it with empty data.
- **Cloud API:** GET .../v1/face-gallery/version should return **updated_at** (ISO 8601; e.g. `2026-03-04T14:00:00Z`). The full gallery GET .../v1/face-gallery returns persons and embeddings; it may also include updated_at. POST windows may require X-API-Key (--cloud-api-key). Window payload can include "person" (--window-attach-person auto|never|always).

## Example

```bash
python src/har_pose_app.py --input rpi --no-display --enable-cloud --cloud-mode windows \
  --enable-face --face-gallery-url http://192.168.1.105:8000 \
  --cloud-url http://192.168.1.105:8000 --cloud-api-key dev-key --window-attach-person auto
```

Using `--face-gallery-url` (or `--cloud-url` when the gallery is on the same host) is required so the app can fetch the face gallery; use `--enable-face` so persons are recognized and attached to windows.

**Debug:** --log-face-summary; last window in /tmp/last_window.json.
