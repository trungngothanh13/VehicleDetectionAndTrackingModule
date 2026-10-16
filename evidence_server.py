"""
Evidence HTTP Server
====================
A lightweight HTTP server that serves evidence video clips and snapshots
from the local `output/evidence/` directory. Runs independently of the
inference WebSocket connection, so video proofs are always available
for download by the backend (api.VDAT) on demand.

Usage:
    python evidence_server.py [--port 8080] [--evidence-dir output/evidence]

The server is also auto-started by main.py when --ws is enabled.
"""

import os
import argparse
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
import threading
import json


class EvidenceRequestHandler(SimpleHTTPRequestHandler):
    """Serves evidence files with proper MIME types and CORS headers."""

    def __init__(self, *args, evidence_dir: str = "output/evidence", **kwargs):
        self.evidence_dir = os.path.abspath(evidence_dir)
        super().__init__(*args, directory=self.evidence_dir, **kwargs)

    def end_headers(self):
        # Allow cross-origin requests from any backend
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # /health endpoint for readiness checks
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = json.dumps({
                "status": "ok",
                "evidence_dir": self.evidence_dir,
                "files": len([
                    f for f in os.listdir(self.evidence_dir)
                    if os.path.isfile(os.path.join(self.evidence_dir, f))
                ]) if os.path.exists(self.evidence_dir) else 0,
            })
            self.wfile.write(response.encode())
            return

        # /list endpoint to list available evidence files
        if self.path == "/list":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            files = []
            if os.path.exists(self.evidence_dir):
                for f in sorted(os.listdir(self.evidence_dir)):
                    fp = os.path.join(self.evidence_dir, f)
                    if os.path.isfile(fp):
                        files.append({
                            "filename": f,
                            "size": os.path.getsize(fp),
                        })
            self.wfile.write(json.dumps({"files": files}).encode())
            return

        # Serve files from evidence_dir — sanitize path to prevent traversal
        # SimpleHTTPRequestHandler already serves from self.directory
        super().do_GET()

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"📦 [Evidence Server] {args[0]}")


def start_evidence_server(port: int = 8080, evidence_dir: str = "output/evidence", daemon: bool = True):
    """
    Start the evidence HTTP server.
    
    Args:
        port: Port to listen on (default: 8080)
        evidence_dir: Path to evidence directory (default: output/evidence)
        daemon: If True, runs in a background thread. If False, blocks.
    
    Returns:
        The HTTPServer instance (if daemon=True) or None (blocks forever if daemon=False).
    """
    os.makedirs(evidence_dir, exist_ok=True)

    handler = partial(EvidenceRequestHandler, evidence_dir=evidence_dir)
    server = HTTPServer(("0.0.0.0", port), handler)

    print(f"📦 [Evidence Server] Serving evidence clips from '{evidence_dir}' on port {port}")
    print(f"📦 [Evidence Server] Health check: http://0.0.0.0:{port}/health")
    print(f"📦 [Evidence Server] File listing: http://0.0.0.0:{port}/list")

    if daemon:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
    else:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence HTTP File Server")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--evidence-dir", "-d", type=str, default="output/evidence", help="Evidence directory path")
    args = parser.parse_args()

    start_evidence_server(port=args.port, evidence_dir=args.evidence_dir, daemon=False)
