# Face Recognition (Optional)

With --enable-face, the app can fetch a face gallery from the cloud, run face detection and recognition on CPU (InsightFace), and bind identities to pose tracks. It can run with local cache only. Windows can include a person object (person_id, name, face_conf, source, verified_at_ms).

Gallery URL priority: --face-gallery-url, --cloud-url, FACE_GALLERY_URL, CLOUD_URL. Gallery is fetched on each run when URL is set; cache updated when version differs. Request gallery without camera: python scripts/request_face_gallery.py.

Face runs in a dedicated background thread (queue maxsize=1) so the pipeline stays at about 30 FPS. FPS log includes "Person: name or Unknown"; separate line "Persons on screen: ...". Names are not drawn on video.

Defaults: --face-skip-frames 10, --face-det-size 256, --face-max-faces 1.

Cloud: GET .../v1/face-gallery/version and .../v1/face-gallery; POST windows may require X-API-Key (--cloud-api-key). Window payload can include "person" (--window-attach-person auto|never|always).

Example: python src/har_pose_app.py --input rpi --no-display --enable-cloud --cloud-mode windows --enable-face --cloud-url http://192.168.1.106:8000 --cloud-api-key dev-key --window-attach-person auto

Debug: --log-face-summary; last window in /tmp/last_window.json.
