#!/usr/bin/env python3
"""
Mock cloud server for Phase 3 and Phase 4 E2E testing: receives POST, returns 200.
- Phase 3: POST /v1/edge/events (frame events).
- Phase 4: POST /v1/windows/ingest (windows); prints path, content-length, device_id, track_id, window_size, count.
No extra dependencies (stdlib only).
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock


_received = 0
_lock = Lock()


class _MockCloudHandler(BaseHTTPRequestHandler):
    """Accepts POST on any path; returns 200. For /v1/windows/ingest logs window fields."""

    def do_POST(self):
        global _received
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length else b""
        with _lock:
            _received += 1
            n = _received
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "received": n}).encode("utf-8"))

        # Log path and content-length for all
        msg = "POST {} -> 200 (received={}, content-length={})".format(
            self.path, n, content_length
        )
        if self.path.rstrip("/").endswith("/v1/windows/ingest") and body_bytes:
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                device_id = data.get("device_id", "")
                track_id = data.get("track_id", "")
                window_size = data.get("window_size", 0)
                msg += " | device_id={}, track_id={}, window_size={}".format(
                    device_id, track_id, window_size
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        print(msg)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Mock cloud server for Phase 3 (frame events) and Phase 4 (windows ingest)."
    )
    parser.add_argument("--port", type=int, default=9999, help="Port (default: 9999)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), _MockCloudHandler)
    print("Mock cloud server listening on http://{}:{} (POST any path -> 200)".format(args.host, args.port))
    print("Phase 3: --enable-cloud --cloud-mode frames --cloud-url http://{}:{}".format(args.host, args.port))
    print("Phase 4: --enable-cloud --cloud-mode windows --cloud-url http://{}:{}".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReceived {} POST(s) total.".format(_received))
        server.shutdown()


if __name__ == "__main__":
    main()
