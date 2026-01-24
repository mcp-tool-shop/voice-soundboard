"""
Claude Collaborate - Voice-enabled collaboration server.

A space where Claude can talk while brainstorming in different web UI environments.

Usage:
    python server.py

Then open http://localhost:8877 in your browser.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8877
DIRECTORY = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def end_headers(self):
        # Add CORS headers for voice API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()


def main():
    os.chdir(DIRECTORY)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print()
        print("=" * 55)
        print("  Claude Collaborate")
        print("  Voice-enabled collaboration environment")
        print("=" * 55)
        print()
        print(f"  Main UI:     http://localhost:{PORT}")
        print(f"  Voice API:   http://localhost:8080 (voice-soundboard)")
        print(f"  Playground:  http://localhost:8765 (claude-adventures)")
        print()
        print("  Press Ctrl+C to stop")
        print("=" * 55)
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
