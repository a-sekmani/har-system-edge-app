#!/usr/bin/env python3
"""
Request face gallery info from cloud and print results.
Run from HAR-System-Edge-App: python scripts/check_cloud_gallery.py [base_url] [api_key]
Defaults: base_url=http://192.168.1.106:8000, api_key=dev-key
"""
import json
import sys
import urllib.error
import urllib.request

def main():
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.106:8000").rstrip("/")
    api_key = sys.argv[2] if len(sys.argv) > 2 else "dev-key"
    version_path = "/v1/face-gallery/version"
    gallery_path = "/v1/face-gallery"
    timeout = 10.0

    print("=== Request face gallery info from cloud ===\n")
    print(f"Base URL: {base_url}")
    print(f"API Key:  {api_key}\n")

    # 1) Version endpoint (returns updated_at for update decision)
    version_url = base_url + version_path
    print(f"[1] GET {version_url}")
    try:
        req = urllib.request.Request(version_url, method="GET")
        req.add_header("Accept", "application/json")
        if api_key:
            req.add_header("X-API-Key", api_key)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            print(f"    Status: {resp.status}")
            print(f"    Response: {json.dumps(data, ensure_ascii=False, indent=2)}")
            updated_at = data.get("updated_at", data.get("created_at"))
            print(f"    → updated_at = {updated_at} (used for gallery update decision)\n")
    except urllib.error.URLError as e:
        print(f"    Error: {e}\n")
        return
    except Exception as e:
        print(f"    Error: {e}\n")
        return

    # 2) Full gallery
    gallery_url = base_url + gallery_path
    print(f"[2] GET {gallery_url}")
    try:
        req = urllib.request.Request(gallery_url, method="GET")
        req.add_header("Accept", "application/json")
        if api_key:
            req.add_header("X-API-Key", api_key)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            print(f"    Status: {resp.status}")
            # Cloud format: gallery_version, embedding_dim, threshold, people
            people = data.get("people", data.get("persons", []))
            out = {
                "gallery_version": data.get("gallery_version", data.get("version")),
                "embedding_dim": data.get("embedding_dim"),
                "threshold": data.get("threshold"),
                "updated_at": data.get("updated_at", data.get("created_at")),
                "people": [
                    {
                        "person_id": p.get("person_id"),
                        "name": p.get("name"),
                        "embeddings_count": len(p.get("embeddings", [])),
                        "embedding_dims": len(p["embeddings"][0]) if p.get("embeddings") else 0,
                    }
                    for p in people
                ],
            }
            print(f"    Response (summary): {json.dumps(out, ensure_ascii=False, indent=2)}")
            print(f"    → Total people: {len(people)}")
            total_embs = sum(len(p.get("embeddings", [])) for p in people)
            print(f"    → Total embeddings: {total_embs}\n")
    except urllib.error.URLError as e:
        print(f"    Error: {e}\n")
        return
    except Exception as e:
        print(f"    Error: {e}\n")
        return

    print("=== Done ===")

if __name__ == "__main__":
    main()
