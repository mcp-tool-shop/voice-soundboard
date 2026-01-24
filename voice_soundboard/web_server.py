"""
Unified Voice Soundboard Web Server.

Provides:
- Static file serving for the web UI
- REST API for speech generation
- Serves audio files
- Claude Collaborate UI and WebSocket bridge
- Claude Adventures sandbox environments

Usage:
    python -m voice_soundboard.web_server

Then open http://YOUR_IP:8080 on your phone/tablet.

Unified Routes:
    /              - Voice Soundboard main page
    /studio        - Voice Studio v2.0
    /collaborate   - Claude Collaborate (unified sandbox environment)
    /adventures    - Claude Adventures (creative lab)
    /ws            - WebSocket bridge for real-time communication
"""

import asyncio
import io
import json
import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

from aiohttp import web
import aiohttp

from voice_soundboard.engine import VoiceEngine
from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.effects import get_effect, list_effects
from voice_soundboard.audio import play_audio

logger = logging.getLogger(__name__)

# WebSocket bridge state
connected_ws_clients: Set[web.WebSocketResponse] = set()
MESSAGE_FILE = Path(__file__).parent.parent / "claude_collaborate" / "messages.jsonl"
CLAUDE_RESPONSE_FILE = Path(__file__).parent.parent / "claude_collaborate" / "claude_responses.jsonl"

# Global engine (lazy loaded)
_engine: Optional[VoiceEngine] = None


def get_engine() -> VoiceEngine:
    """Get or create voice engine."""
    global _engine
    if _engine is None:
        _engine = VoiceEngine()
    return _engine


def get_local_ip() -> str:
    """Get local IP address for network access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# Routes
async def index_handler(request: web.Request) -> web.Response:
    """Serve the main HTML page."""
    web_dir = Path(__file__).parent / "web"
    index_path = web_dir / "index.html"

    if index_path.exists():
        return web.FileResponse(index_path)
    else:
        return web.Response(text="Web UI not found", status=404)


async def studio_handler(request: web.Request) -> web.Response:
    """Serve the Voice Studio page."""
    web_dir = Path(__file__).parent / "web" / "studio"
    index_path = web_dir / "index.html"

    if index_path.exists():
        return web.FileResponse(index_path)
    else:
        return web.Response(text="Voice Studio not found", status=404)


async def playground_handler(request: web.Request) -> web.Response:
    """Serve Claude's Playground - an interactive sandbox for AI exploration."""
    web_dir = Path(__file__).parent / "web" / "playground"
    index_path = web_dir / "index.html"

    if index_path.exists():
        return web.FileResponse(index_path)
    else:
        return web.Response(text="Claude's Playground not found", status=404)


async def collaborate_handler(request: web.Request) -> web.Response:
    """Serve Claude Collaborate - unified sandbox environment."""
    collaborate_dir = Path(__file__).parent.parent / "claude_collaborate"
    index_path = collaborate_dir / "index.html"

    if index_path.exists():
        return web.FileResponse(index_path)
    else:
        return web.Response(text="Claude Collaborate not found", status=404)


async def collaborate_file_handler(request: web.Request) -> web.Response:
    """Serve Claude Collaborate static files (whiteboard.html, etc.)."""
    filename = request.match_info.get("filename", "")
    collaborate_dir = Path(__file__).parent.parent / "claude_collaborate"
    file_path = collaborate_dir / filename

    # Security: prevent directory traversal
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(collaborate_dir.resolve())):
            return web.Response(text="Forbidden", status=403)
    except Exception:
        return web.Response(text="Invalid path", status=400)

    if file_path.exists() and file_path.is_file():
        return web.FileResponse(file_path)
    else:
        return web.Response(text=f"File not found: {filename}", status=404)


async def adventures_handler(request: web.Request) -> web.Response:
    """Serve Claude Adventures - creative lab environments."""
    adventures_dir = Path(__file__).parent.parent / "claude_adventures"
    index_path = adventures_dir / "index.html"

    if index_path.exists():
        return web.FileResponse(index_path)
    else:
        return web.Response(text="Claude Adventures not found", status=404)


async def adventures_file_handler(request: web.Request) -> web.Response:
    """Serve Claude Adventures static files."""
    filename = request.match_info.get("filename", "")
    adventures_dir = Path(__file__).parent.parent / "claude_adventures"
    file_path = adventures_dir / filename

    # Security: prevent directory traversal
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(adventures_dir.resolve())):
            return web.Response(text="Forbidden", status=403)
    except Exception:
        return web.Response(text="Invalid path", status=400)

    if file_path.exists() and file_path.is_file():
        return web.FileResponse(file_path)
    else:
        return web.Response(text=f"File not found: {filename}", status=404)


# =============================================================================
# WebSocket Bridge for Claude Collaborate
# =============================================================================

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections from browser for real-time Claude communication."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    connected_ws_clients.add(ws)
    logger.info(f"WebSocket client connected. Total: {len(connected_ws_clients)}")

    # Send welcome message
    await ws.send_json({
        "type": "connected",
        "message": "Connected to Claude Collaborate Bridge",
        "timestamp": datetime.now().isoformat()
    })

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await handle_ws_message(ws, data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        connected_ws_clients.discard(ws)
        logger.info(f"WebSocket client disconnected. Total: {len(connected_ws_clients)}")

    return ws


async def handle_ws_message(ws: web.WebSocketResponse, data: dict):
    """Process incoming WebSocket messages from browser."""
    msg_type = data.get("type", "unknown")
    timestamp = datetime.now().isoformat()

    if msg_type == "user_message":
        # User sent a message to Claude
        message = {
            "type": "user_message",
            "content": data.get("content", ""),
            "timestamp": timestamp,
            "source": "claude_collaborate"
        }

        # Ensure directory exists
        MESSAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Append to message file for Claude Code to read
        with open(MESSAGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

        # Acknowledge receipt
        await ws.send_json({
            "type": "message_received",
            "timestamp": timestamp,
            "content": data.get("content", "")[:50] + "..."
        })

        logger.info(f"User message: {data.get('content', '')[:80]}")

    elif msg_type == "ping":
        await ws.send_json({"type": "pong", "timestamp": timestamp})


async def broadcast_to_ws_clients(message: dict):
    """Send message to all connected WebSocket clients."""
    if connected_ws_clients:
        await asyncio.gather(
            *[client.send_json(message) for client in connected_ws_clients if not client.closed],
            return_exceptions=True
        )


async def ws_respond_handler(request: web.Request) -> web.Response:
    """HTTP endpoint for Claude Code to send responses to browser via WebSocket."""
    try:
        data = await request.json()
        message = {
            "type": "claude_response",
            "content": data.get("content", ""),
            "timestamp": datetime.now().isoformat()
        }

        # Broadcast to all connected browser clients
        await broadcast_to_ws_clients(message)

        # Also save to file as backup
        CLAUDE_RESPONSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CLAUDE_RESPONSE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

        return web.json_response({"status": "sent", "clients": len(connected_ws_clients)})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def ws_messages_handler(request: web.Request) -> web.Response:
    """HTTP endpoint for Claude Code to read user messages."""
    messages = []
    if MESSAGE_FILE.exists():
        try:
            with open(MESSAGE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        messages.append(json.loads(line))
            # Clear the file after reading
            MESSAGE_FILE.write_text("")
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    return web.json_response({"messages": messages, "count": len(messages)})


async def ws_status_handler(request: web.Request) -> web.Response:
    """WebSocket bridge status endpoint."""
    return web.json_response({
        "status": "ok",
        "connected_clients": len(connected_ws_clients),
        "timestamp": datetime.now().isoformat()
    })


async def captures_handler(request: web.Request) -> web.Response:
    """List captured screenshots and recordings."""
    from datetime import datetime

    captures_dir = Path(__file__).parent.parent / "captures"
    captures_dir.mkdir(exist_ok=True)

    captures = []
    for f in captures_dir.glob("*"):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm"]:
            stat = f.stat()
            captures.append({
                "name": f.name,
                "path": str(f).replace("\\", "/"),
                "type": "screenshot" if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"] else "recording",
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })

    # Sort by creation time, newest first
    captures.sort(key=lambda x: x["created"], reverse=True)

    return web.json_response({
        "captures_dir": str(captures_dir).replace("\\", "/"),
        "count": len(captures),
        "captures": captures[:50]  # Return last 50
    })


async def manifest_handler(request: web.Request) -> web.Response:
    """Serve PWA manifest for 'Add to Home Screen' functionality."""
    manifest = {
        "name": "Voice Soundboard",
        "short_name": "Voice SB",
        "description": "AI-powered text-to-speech soundboard",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1a2e",
        "theme_color": "#e94560",
        "icons": [
            {
                "src": "/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    return web.json_response(manifest)


async def voices_handler(request: web.Request) -> web.Response:
    """List available voices."""
    return web.json_response(KOKORO_VOICES)


async def presets_handler(request: web.Request) -> web.Response:
    """List available presets."""
    return web.json_response(VOICE_PRESETS)


async def effects_handler(request: web.Request) -> web.Response:
    """List available sound effects."""
    effects = list_effects()
    return web.json_response(effects)


async def speak_handler(request: web.Request) -> web.Response:
    """Generate speech from text and return audio."""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = data.get("text", "").strip()
    if not text:
        return web.json_response({"error": "No text provided"}, status=400)

    voice = data.get("voice", "af_bella")
    speed = data.get("speed", 1.0)
    preset = data.get("preset")
    play = data.get("play", False)

    try:
        engine = get_engine()
        result = engine.speak(
            text=text,
            voice=voice,
            speed=speed,
            preset=preset if preset else None,
        )

        # Optionally play on server
        if play:
            await asyncio.to_thread(play_audio, result.audio_path)

        # Read audio file and return it
        audio_data = result.audio_path.read_bytes()

        return web.Response(
            body=audio_data,
            content_type="audio/wav",
            headers={
                "X-Voice-Used": result.voice_used,
                "X-Duration": str(result.duration_seconds),
                "X-Realtime-Factor": str(result.realtime_factor),
            }
        )
    except Exception as e:
        logger.error(f"Speech generation error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def speak_json_handler(request: web.Request) -> web.Response:
    """Generate speech and return metadata (not audio)."""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = data.get("text", "").strip()
    if not text:
        return web.json_response({"error": "No text provided"}, status=400)

    voice = data.get("voice", "af_bella")
    speed = data.get("speed", 1.0)
    preset = data.get("preset")
    play = data.get("play", True)  # Default to playing on server

    try:
        engine = get_engine()
        result = engine.speak(
            text=text,
            voice=voice,
            speed=speed,
            preset=preset if preset else None,
        )

        # Play on server if requested
        if play:
            await asyncio.to_thread(play_audio, result.audio_path)

        return web.json_response({
            "success": True,
            "file": str(result.audio_path),
            "voice": result.voice_used,
            "duration": result.duration_seconds,
            "realtime_factor": result.realtime_factor,
        })
    except Exception as e:
        logger.error(f"Speech generation error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def effect_handler(request: web.Request) -> web.Response:
    """Play a sound effect."""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    effect_name = data.get("effect", "").strip()
    if not effect_name:
        return web.json_response({"error": "No effect specified"}, status=400)

    play = data.get("play", True)

    try:
        result = get_effect(effect_name)

        if play:
            await asyncio.to_thread(play_audio, result["path"])

        return web.json_response({
            "success": True,
            "effect": effect_name,
            "file": str(result["path"]),
        })
    except Exception as e:
        logger.error(f"Effect error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def health_handler(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()

    # CORS middleware for mobile browsers
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)

        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    app.middlewares.append(cors_middleware)

    # =========================================================================
    # Voice Soundboard Routes
    # =========================================================================
    app.router.add_get("/", index_handler)
    app.router.add_get("/studio", studio_handler)
    app.router.add_get("/playground", playground_handler)
    app.router.add_get("/manifest.json", manifest_handler)
    app.router.add_get("/api/voices", voices_handler)
    app.router.add_get("/api/presets", presets_handler)
    app.router.add_get("/api/effects", effects_handler)
    app.router.add_post("/speak", speak_handler)
    app.router.add_post("/api/speak", speak_json_handler)
    app.router.add_post("/api/effect", effect_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/api/captures", captures_handler)

    # =========================================================================
    # Claude Collaborate Routes (unified sandbox environment)
    # =========================================================================
    app.router.add_get("/collaborate", collaborate_handler)
    app.router.add_get("/collaborate/{filename:.*}", collaborate_file_handler)

    # =========================================================================
    # Claude Adventures Routes (creative lab)
    # =========================================================================
    app.router.add_get("/adventures", adventures_handler)
    app.router.add_get("/adventures/{filename:.*}", adventures_file_handler)

    # =========================================================================
    # WebSocket Bridge Routes (real-time communication)
    # =========================================================================
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/api/ws/respond", ws_respond_handler)
    app.router.add_get("/api/ws/messages", ws_messages_handler)
    app.router.add_get("/api/ws/status", ws_status_handler)

    # =========================================================================
    # Static Files
    # =========================================================================
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.router.add_static("/static/", web_dir, name="static")
        # Also serve studio static files
        studio_dir = web_dir / "studio"
        if studio_dir.exists():
            app.router.add_static("/studio/static/", studio_dir, name="studio_static")

    return app


async def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    local_ip = get_local_ip()
    print(f"""
================================================================
           Unified Voice Soundboard Server
================================================================

  Voice Soundboard:   http://localhost:{port}/
  Voice Studio:       http://localhost:{port}/studio
  Claude Collaborate: http://localhost:{port}/collaborate
  Claude Adventures:  http://localhost:{port}/adventures
  WebSocket Bridge:   ws://localhost:{port}/ws

  Network: http://{local_ip}:{port}

================================================================
API Endpoints:
  - Voice:    POST /api/speak, /api/effect
  - Bridge:   GET  /api/ws/messages, /api/ws/status
              POST /api/ws/respond
  - Health:   GET  /health
================================================================
""")

    # Run forever
    await asyncio.Event().wait()


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Soundboard Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
