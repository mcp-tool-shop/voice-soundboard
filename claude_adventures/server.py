"""
Claude Adventures - Standalone Server
A fun interactive playground for AI exploration.

Usage:
    python server.py

Then open http://localhost:8765 in your browser.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8765
DIRECTORY = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    os.chdir(DIRECTORY)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print()
        print("=" * 50)
        print("  Claude Adventures")
        print("=" * 50)
        print()
        print(f"  Server running at http://localhost:{PORT}")
        print()
        print("  Press Ctrl+C to stop")
        print("=" * 50)
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
