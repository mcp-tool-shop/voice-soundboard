"""
WebSocket API for Voice Soundboard.

Provides real-time bidirectional communication with security hardening:
- Origin validation (CSWSH protection)
- Optional API key authentication
- Rate limiting
- Connection limits
- Input validation
- TLS support

Example client usage:
    import websockets
    import json

    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({
            "action": "speak",
            "text": "Hello world",
            "voice": "af_bella"
        }))
        response = await ws.recv()
        print(json.loads(response))
"""

from __future__ import annotations

import asyncio
import json
import base64
import logging
import os
import ssl
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

from voice_soundboard.engine import VoiceEngine
from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.audio import play_audio, stop_playback
from voice_soundboard.effects import get_effect, list_effects
from voice_soundboard.ssml import parse_ssml
from voice_soundboard.emotions import get_emotion_voice_params, apply_emotion_to_text, EMOTIONS
from voice_soundboard.streaming import StreamingEngine
from voice_soundboard.security import (
    WebSocketSecurityManager,
    validate_text_input,
    validate_speed,
    safe_error_message,
    get_security_config,
    SecurityConfig,
)

# Configure logging - don't log sensitive data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice-ws")


@dataclass
class WSResponse:
    """Standard WebSocket response format."""
    success: bool
    action: str
    data: dict
    error: Optional[str] = None
    request_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class VoiceWebSocketServer:
    """
    Secure WebSocket server for voice soundboard.

    Security features:
    - Origin validation (prevents CSWSH)
    - Optional API key authentication
    - Rate limiting per client
    - Connection limits
    - Input validation
    - TLS/SSL support

    Supports actions:
    - speak: Generate speech from text
    - speak_stream: Stream audio chunks as they generate
    - speak_realtime: Generate and play immediately
    - effect: Play a sound effect
    - stop: Stop current playback
    - list_voices: Get available voices
    - list_presets: Get available presets
    - list_emotions: Get available emotions
    - list_effects: Get available effects
    - status: Get server status
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        api_key: Optional[str] = None,
        allowed_origins: Optional[set[str]] = None,
        max_connections: int = 100,
        ssl_cert: Optional[str] = None,
        ssl_key: Optional[str] = None,
    ):
        """
        Initialize the WebSocket server.

        Args:
            host: Host to bind to
            port: Port to listen on
            api_key: Optional API key for authentication (env: VOICE_API_KEY)
            allowed_origins: Set of allowed Origin headers
            max_connections: Maximum concurrent connections
            ssl_cert: Path to SSL certificate (enables wss://)
            ssl_key: Path to SSL private key
        """
        self.host = host
        self.port = port

        # Security manager
        self._security = WebSocketSecurityManager(
            allowed_origins=allowed_origins,
            api_key=api_key or os.getenv("VOICE_API_KEY"),
            max_connections=max_connections,
        )

        # SSL/TLS setup
        self._ssl_context: Optional[ssl.SSLContext] = None
        if ssl_cert and ssl_key:
            try:
                self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                self._ssl_context.load_cert_chain(ssl_cert, ssl_key)
                logger.info("TLS enabled")
            except ssl.SSLError as e:
                logger.error("Failed to load SSL certificate/key: %s", e)
                raise ValueError(f"SSL configuration error: {e}") from e
            except FileNotFoundError as e:
                logger.error("SSL certificate or key file not found: %s", e)
                raise ValueError(f"SSL file not found: {e}") from e

        # Engine instances (lazy loaded)
        self._engine: Optional[VoiceEngine] = None
        self._streaming_engine: Optional[StreamingEngine] = None

        # Client tracking
        self._clients: dict[str, WebSocketServerProtocol] = {}
        self._is_running = False

        # Config
        self._config = get_security_config()

    def _get_engine(self) -> VoiceEngine:
        """Lazy-load voice engine."""
        if self._engine is None:
            logger.info("Loading voice engine...")
            try:
                self._engine = VoiceEngine()
                logger.info("Voice engine loaded")
            except Exception as e:
                logger.error("Failed to load voice engine: %s", e)
                raise RuntimeError(f"Voice engine initialization failed: {e}") from e
        return self._engine

    def _get_streaming_engine(self) -> StreamingEngine:
        """Lazy-load streaming engine."""
        if self._streaming_engine is None:
            try:
                self._streaming_engine = StreamingEngine()
                logger.debug("Streaming engine loaded")
            except Exception as e:
                logger.error("Failed to load streaming engine: %s", e)
                raise RuntimeError(f"Streaming engine initialization failed: {e}") from e
        return self._streaming_engine

    def _get_client_id(self, ws: WebSocketServerProtocol) -> str:
        """Get unique client identifier."""
        return str(id(ws))

    async def _send_response(
        self,
        ws: WebSocketServerProtocol,
        success: bool,
        action: str,
        data: dict,
        error: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Send a JSON response to the client."""
        response = WSResponse(
            success=success,
            action=action,
            data=data,
            error=error,
            request_id=request_id,
        )
        try:
            await ws.send(response.to_json())
        except websockets.exceptions.ConnectionClosed:
            logger.debug("Could not send response: connection closed")

    async def _send_error(
        self,
        ws: WebSocketServerProtocol,
        action: str,
        error: str,
        request_id: Optional[str] = None,
    ):
        """Send an error response."""
        await self._send_response(ws, False, action, {}, error, request_id)

    async def handle_speak(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Handle speak action - generate speech and return file path."""
        try:
            # Validate text input
            text = validate_text_input(
                params.get("text", ""),
                max_length=self._config.max_text_length
            )
        except ValueError as e:
            await self._send_error(ws, "speak", str(e), request_id)
            return

        voice = params.get("voice")
        preset = params.get("preset")
        speed = params.get("speed", 1.0)
        emotion = params.get("emotion")
        play = params.get("play", False)
        return_audio = params.get("return_audio", False)

        try:
            # Validate speed
            speed = validate_speed(speed)

            # Apply emotion if specified
            if emotion:
                emotion_params = get_emotion_voice_params(emotion, voice, speed)
                voice = emotion_params["voice"]
                speed = emotion_params["speed"]
                text = apply_emotion_to_text(text, emotion)

            engine = self._get_engine()
            result = engine.speak(
                text=text,
                voice=voice,
                preset=preset,
                speed=speed,
            )

            data = {
                "file_path": str(result.audio_path),
                "voice": result.voice_used,
                "duration": result.duration_seconds,
                "realtime_factor": result.realtime_factor,
            }

            # Optionally return base64 audio
            if return_audio:
                audio_data, sr = sf.read(str(result.audio_path))
                audio_bytes = audio_data.tobytes()
                data["audio_base64"] = base64.b64encode(audio_bytes).decode()
                data["sample_rate"] = sr

            # Optionally play
            if play:
                await asyncio.to_thread(play_audio, result.audio_path)
                data["played"] = True

            # Log without sensitive content
            logger.info(f"Generated speech: {len(text)} chars, voice={result.voice_used}")
            await self._send_response(ws, True, "speak", data, request_id=request_id)

        except Exception as e:
            logger.exception("Error in speak")
            await self._send_error(ws, "speak", safe_error_message(e), request_id)

    async def handle_speak_stream(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Handle streaming speech - send audio chunks as they generate."""
        try:
            text = validate_text_input(
                params.get("text", ""),
                max_length=self._config.max_text_length
            )
        except ValueError as e:
            await self._send_error(ws, "speak_stream", str(e), request_id)
            return

        voice = params.get("voice")
        preset = params.get("preset")
        speed = validate_speed(params.get("speed", 1.0))
        emotion = params.get("emotion")

        try:
            # Apply emotion
            if emotion:
                emotion_params = get_emotion_voice_params(emotion, voice, speed)
                voice = emotion_params["voice"]
                speed = emotion_params["speed"]
                text = apply_emotion_to_text(text, emotion)

            engine = self._get_streaming_engine()
            chunk_index = 0
            total_samples = 0
            start_time = time.time()

            # Send start message
            await self._send_response(
                ws, True, "speak_stream_start",
                {"chars": len(text)},
                request_id=request_id
            )

            async for chunk in engine.stream(text, voice=voice, preset=preset, speed=speed):
                if chunk.is_final:
                    break

                # Send chunk as base64
                audio_bytes = chunk.samples.tobytes()
                await ws.send(json.dumps({
                    "action": "speak_stream_chunk",
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "sample_rate": chunk.sample_rate,
                    "samples": len(chunk.samples),
                    "audio_base64": base64.b64encode(audio_bytes).decode(),
                }))

                total_samples += len(chunk.samples)
                chunk_index += 1

            # Send completion message
            duration = total_samples / 24000 if total_samples > 0 else 0
            await self._send_response(
                ws, True, "speak_stream_end",
                {
                    "total_chunks": chunk_index,
                    "total_duration": duration,
                    "generation_time": time.time() - start_time,
                },
                request_id=request_id
            )

        except Exception as e:
            logger.exception("Error in speak_stream")
            await self._send_error(ws, "speak_stream", safe_error_message(e), request_id)

    async def handle_speak_realtime(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Handle realtime speech - generate and play immediately."""
        try:
            text = validate_text_input(
                params.get("text", ""),
                max_length=self._config.max_text_length
            )
        except ValueError as e:
            await self._send_error(ws, "speak_realtime", str(e), request_id)
            return

        voice = params.get("voice")
        preset = params.get("preset")
        speed = validate_speed(params.get("speed", 1.0))
        emotion = params.get("emotion")

        try:
            from voice_soundboard.streaming import stream_realtime

            # Apply emotion
            if emotion:
                emotion_params = get_emotion_voice_params(emotion, voice, speed)
                voice = emotion_params["voice"]
                speed = emotion_params["speed"]
                text = apply_emotion_to_text(text, emotion)

            # Notify start
            await self._send_response(
                ws, True, "speak_realtime_start",
                {"chars": len(text)},
                request_id=request_id
            )

            # Stream with realtime playback
            result = await stream_realtime(
                text=text,
                voice=voice,
                preset=preset,
                speed=speed,
            )

            await self._send_response(
                ws, True, "speak_realtime",
                {
                    "duration": result.total_duration,
                    "chunks": result.total_chunks,
                    "voice": result.voice_used,
                    "generation_time": result.generation_time,
                },
                request_id=request_id
            )

        except Exception as e:
            logger.exception("Error in speak_realtime")
            await self._send_error(ws, "speak_realtime", safe_error_message(e), request_id)

    async def handle_speak_ssml(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Handle SSML speech."""
        try:
            ssml = validate_text_input(
                params.get("ssml", ""),
                max_length=self._config.max_ssml_length,
                field_name="ssml"
            )
        except ValueError as e:
            await self._send_error(ws, "speak_ssml", str(e), request_id)
            return

        voice = params.get("voice")
        preset = params.get("preset")
        play = params.get("play", False)

        try:
            text, ssml_params = parse_ssml(ssml)
            if ssml_params.voice and not voice:
                voice = ssml_params.voice
            speed = ssml_params.speed

            engine = self._get_engine()
            result = engine.speak(
                text=text,
                voice=voice,
                preset=preset,
                speed=speed,
            )

            data = {
                "file_path": str(result.audio_path),
                "voice": result.voice_used,
                "duration": result.duration_seconds,
                "speed": speed,
            }

            if play:
                await asyncio.to_thread(play_audio, result.audio_path)
                data["played"] = True

            await self._send_response(ws, True, "speak_ssml", data, request_id=request_id)

        except Exception as e:
            logger.exception("Error in speak_ssml")
            await self._send_error(ws, "speak_ssml", safe_error_message(e), request_id)

    async def handle_effect(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Handle sound effect playback."""
        effect_name = params.get("effect", "")
        if not effect_name:
            await self._send_error(ws, "effect", "Missing 'effect' parameter", request_id)
            return

        try:
            effect = get_effect(effect_name)
            await asyncio.to_thread(effect.play)

            await self._send_response(
                ws, True, "effect",
                {"effect": effect_name, "duration": effect.duration},
                request_id=request_id
            )

        except ValueError as e:
            await self._send_error(ws, "effect", str(e), request_id)
        except Exception as e:
            logger.exception("Error in effect")
            await self._send_error(ws, "effect", safe_error_message(e), request_id)

    async def handle_stop(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Handle stop playback."""
        try:
            stop_playback()
            await self._send_response(
                ws, True, "stop", {"stopped": True}, request_id=request_id
            )
        except Exception as e:
            await self._send_error(ws, "stop", safe_error_message(e), request_id)

    async def handle_list_voices(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """List available voices."""
        filter_gender = params.get("gender")
        filter_accent = params.get("accent")

        voices = []
        for voice_id, info in sorted(KOKORO_VOICES.items()):
            if filter_gender and info.get("gender") != filter_gender:
                continue
            if filter_accent and info.get("accent") != filter_accent:
                continue
            voices.append({
                "id": voice_id,
                "name": info["name"],
                "gender": info["gender"],
                "accent": info["accent"],
                "style": info["style"],
            })

        await self._send_response(
            ws, True, "list_voices", {"voices": voices, "count": len(voices)},
            request_id=request_id
        )

    async def handle_list_presets(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """List available presets."""
        presets = []
        for name, config in VOICE_PRESETS.items():
            presets.append({
                "name": name,
                "voice": config["voice"],
                "speed": config.get("speed", 1.0),
                "description": config.get("description", ""),
            })

        await self._send_response(
            ws, True, "list_presets", {"presets": presets},
            request_id=request_id
        )

    async def handle_list_emotions(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """List available emotions."""
        emotions = []
        for name, params_obj in EMOTIONS.items():
            emotions.append({
                "name": name,
                "speed": params_obj.speed,
                "voice": params_obj.voice_preference,
            })

        await self._send_response(
            ws, True, "list_emotions", {"emotions": emotions},
            request_id=request_id
        )

    async def handle_list_effects(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """List available effects."""
        effects = list_effects()
        await self._send_response(
            ws, True, "list_effects", {"effects": effects},
            request_id=request_id
        )

    async def handle_status(
        self,
        ws: WebSocketServerProtocol,
        params: dict,
        request_id: Optional[str] = None,
    ):
        """Get server status."""
        await self._send_response(
            ws, True, "status",
            {
                "server": "voice-soundboard",
                "version": "0.1.0",
                "clients": self._security.connection_count,
                "max_clients": self._security.max_connections,
                "engine_loaded": self._engine is not None,
                "tls_enabled": self._ssl_context is not None,
            },
            request_id=request_id
        )

    async def handle_message(self, ws: WebSocketServerProtocol, message: str):
        """Route incoming messages to appropriate handlers."""
        client_id = self._get_client_id(ws)

        # SECURITY: Check rate limit
        if not self._security.check_rate_limit(client_id):
            await self._send_error(ws, "error", "Rate limit exceeded. Try again later.")
            return

        # Parse JSON
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await self._send_error(ws, "error", "Invalid JSON")
            return

        # Validate message size
        if len(message) > self._config.max_message_size:
            await self._send_error(ws, "error", "Message too large")
            return

        action = data.get("action", "")
        params = data.get("params", data)  # Allow params at top level or nested
        request_id = data.get("request_id")

        handlers = {
            "speak": self.handle_speak,
            "speak_stream": self.handle_speak_stream,
            "speak_realtime": self.handle_speak_realtime,
            "speak_ssml": self.handle_speak_ssml,
            "effect": self.handle_effect,
            "stop": self.handle_stop,
            "list_voices": self.handle_list_voices,
            "list_presets": self.handle_list_presets,
            "list_emotions": self.handle_list_emotions,
            "list_effects": self.handle_list_effects,
            "status": self.handle_status,
        }

        handler = handlers.get(action)
        if handler:
            await handler(ws, params, request_id)
        else:
            await self._send_error(
                ws, "error",
                f"Unknown action: {action}",
                request_id
            )

    async def connection_handler(self, ws: WebSocketServerProtocol):
        """Handle a WebSocket connection with security checks."""
        client_id = self._get_client_id(ws)

        # SECURITY: Check connection limit
        if not self._security.can_accept_connection():
            logger.warning(f"Connection rejected: at capacity ({self._security.connection_count})")
            await ws.close(1013, "Server at capacity")
            return

        # SECURITY: Validate Origin header (CSWSH protection)
        origin = ws.request_headers.get("Origin", "")
        if not self._security.validate_origin(origin):
            await ws.close(1008, "Origin not allowed")
            return

        # SECURITY: Check API key if required
        # API key can be in query string: ws://host:port?key=xxx
        api_key = None
        if ws.request.path and "?" in ws.request.path:
            query = ws.request.path.split("?", 1)[1]
            for param in query.split("&"):
                if param.startswith("key="):
                    api_key = param[4:]
                    break

        if not self._security.validate_api_key(api_key):
            logger.warning(f"Connection rejected: invalid API key from {origin or 'no-origin'}")
            await ws.close(1008, "Invalid API key")
            return

        # Register connection
        self._security.add_connection(client_id)
        self._clients[client_id] = ws
        logger.info(f"Client connected: {client_id} (total: {self._security.connection_count})")

        try:
            # Send welcome message
            await self._send_response(
                ws, True, "connected",
                {
                    "message": "Connected to Voice Soundboard",
                    "client_id": client_id,
                    "rate_limit": f"{self._config.rate_limit_requests} req/{self._config.rate_limit_window}s",
                }
            )

            async for message in ws:
                await self.handle_message(ws, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.exception(f"Error with client {client_id}")
        finally:
            self._security.remove_connection(client_id)
            self._clients.pop(client_id, None)
            logger.info(f"Client removed: {client_id} (remaining: {self._security.connection_count})")

    async def start(self):
        """Start the WebSocket server."""
        protocol = "wss" if self._ssl_context else "ws"
        logger.info(f"Starting WebSocket server on {protocol}://{self.host}:{self.port}")

        if self._security.api_key:
            logger.info("API key authentication enabled")
        else:
            logger.warning("No API key configured - authentication disabled")

        self._is_running = True

        try:
            async with serve(
                self.connection_handler,
                self.host,
                self.port,
                ssl=self._ssl_context,
                max_size=self._config.max_message_size,
                ping_interval=30,  # Keep-alive ping every 30s
                ping_timeout=10,   # Close if no pong in 10s
            ):
                logger.info(f"WebSocket server running on {protocol}://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever
        except OSError as e:
            logger.error("Failed to start WebSocket server: %s", e)
            raise RuntimeError(f"Could not bind to {self.host}:{self.port}: {e}") from e
        except Exception as e:
            logger.error("WebSocket server error: %s", e)
            raise

    def run(self):
        """Run the server (blocking)."""
        asyncio.run(self.start())


def create_server(
    host: str = "localhost",
    port: int = 8765,
    api_key: Optional[str] = None,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
) -> VoiceWebSocketServer:
    """Create a WebSocket server instance."""
    return VoiceWebSocketServer(
        host=host,
        port=port,
        api_key=api_key,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
    )


async def main():
    """Run the WebSocket server."""
    server = VoiceWebSocketServer(
        host=os.getenv("VOICE_WS_HOST", "localhost"),
        port=int(os.getenv("VOICE_WS_PORT", "8765")),
        api_key=os.getenv("VOICE_API_KEY"),
        ssl_cert=os.getenv("VOICE_SSL_CERT"),
        ssl_key=os.getenv("VOICE_SSL_KEY"),
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
