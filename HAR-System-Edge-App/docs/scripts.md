# Scripts (No Camera)

## request_face_gallery.py

Fetches gallery **updated_at** from the version endpoint (GET .../v1/face-gallery/version; cloud returns updated_at, ISO 8601) and the full gallery from the cloud, then saves to local cache. Does not start the camera or the main app.

```bash
cd HAR-System-Edge-App
python scripts/request_face_gallery.py [base_url]
# Or: FACE_GALLERY_URL=http://... python scripts/request_face_gallery.py
# Or: CLOUD_URL=http://... python scripts/request_face_gallery.py
```

Defaults: base_url from env FACE_GALLERY_URL or CLOUD_URL or http://192.168.1.105:8000; gallery dir from env FACE_GALLERY_CACHE or project `face_gallery/` (persistent; updated from cloud only when cloud updated_at is newer).

## check_cloud_gallery.py

Requests face gallery **updated_at** (from the version endpoint) and full gallery from the cloud and prints a summary. Does not write to cache.

```bash
python scripts/check_cloud_gallery.py [base_url] [api_key]
```

Defaults: base_url=http://192.168.1.105:8000, api_key=dev-key.
