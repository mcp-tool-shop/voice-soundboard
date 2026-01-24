"""
Simple HTTP web server for mobile access to Voice Soundboard.

Provides:
- Static file serving for the web UI
- REST API for speech generation
- Serves audio files

Usage:
    python -m voice_soundboard.web_server

Then open http://YOUR_IP:8080 on your phone/tablet.
"""

import asyncio
import io
import json
import logging
import socket
from pathlib import Path
from typing import Optional

from aiohttp import web

from voice_soundboard.engine import VoiceEngine
from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.effects import get_effect, list_effects
from voice_soundboard.audio import play_audio

logger = logging.getLogger(__name__)

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

    # Routes
    app.router.add_get("/", index_handler)
    app.router.add_get("/studio", studio_handler)
    app.router.add_get("/manifest.json", manifest_handler)
    app.router.add_get("/api/voices", voices_handler)
    app.router.add_get("/api/presets", presets_handler)
    app.router.add_get("/api/effects", effects_handler)
    app.router.add_post("/speak", speak_handler)
    app.router.add_post("/api/speak", speak_json_handler)
    app.router.add_post("/api/effect", effect_handler)
    app.router.add_get("/health", health_handler)

    # Static files
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
    print(f"\n{'='*50}")
    print(f"  Voice Soundboard Web Server")
    print(f"{'='*50}")
    print(f"\n  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print(f"\n  Open the Network URL on your phone/tablet!")
    print(f"{'='*50}\n")

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
