#!/usr/bin/env python3
"""
Mock cloud server for Phase 3 E2E testing: receives POST, returns 200 immediately.
Use to verify send/receive end-to-end and that FPS stays high when the backend is fast.
No extra dependencies (stdlib only).
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock


# Shared counter (thread-safe)
_received = 0
_lock = Lock()


class _MockCloudHandler(BaseHTTPRequestHandler):
    """Accepts POST on any path; returns 200 with {ok: true, received: N}."""

    def do_POST(self):
        global _received
        # Consume request body so client doesn't block
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        with _lock:
            _received += 1
            n = _received
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = json.dumps({"ok": True, "received": n}).encode("utf-8")
        self.wfile.write(body)
        print("POST {} -> 200 (received={})".format(self.path, n))

    def log_message(self, format, *args):
        pass  # quiet by default


def main():
    parser = argparse.ArgumentParser(description="Mock cloud server for Phase 3 E2E.")
    parser.add_argument("--port", type=int, default=9999, help="Port to listen on (default: 9999)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), _MockCloudHandler)
    print("Mock cloud server listening on http://{}:{} (POST any path -> 200)".format(args.host, args.port))
    print("Run app with: --enable-cloud --cloud-url http://{}:{} --send-every-n-frames 10".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReceived {} POST(s) total.".format(_received))
        server.shutdown()


if __name__ == "__main__":
    main()
