"""
Additional coverage tests - Batch 29: WebSocket Server.

Tests for voice_soundboard/websocket_server.py.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict


# =============================================================================
# WSResponse Tests
# =============================================================================

class TestWSResponse:
    """Tests for WSResponse dataclass."""

    def test_response_creation(self):
        """Test creating WSResponse."""
        from voice_soundboard.websocket_server import WSResponse

        response = WSResponse(
            success=True,
            action="speak",
            data={"file_path": "/tmp/test.wav"},
        )

        assert response.success is True
        assert response.action == "speak"
        assert response.data == {"file_path": "/tmp/test.wav"}
        assert response.error is None
        assert response.request_id is None

    def test_response_with_error(self):
        """Test WSResponse with error."""
        from voice_soundboard.websocket_server import WSResponse

        response = WSResponse(
            success=False,
            action="speak",
            data={},
            error="Something went wrong",
            request_id="req-123",
        )

        assert response.success is False
        assert response.error == "Something went wrong"
        assert response.request_id == "req-123"

    def test_to_json(self):
        """Test WSResponse JSON serialization."""
        from voice_soundboard.websocket_server import WSResponse

        response = WSResponse(
            success=True,
            action="status",
            data={"version": "1.0"},
        )

        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert parsed["action"] == "status"
        assert parsed["data"]["version"] == "1.0"


# =============================================================================
# VoiceWebSocketServer Initialization Tests
# =============================================================================

class TestVoiceWebSocketServerInit:
    """Tests for VoiceWebSocketServer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()

        assert server.host == "localhost"
        assert server.port == 8765
        assert server._engine is None
        assert server._streaming_engine is None
        assert server._is_running is False

    def test_custom_init(self):
        """Test custom initialization."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer(
            host="0.0.0.0",
            port=9000,
            max_connections=50,
        )

        assert server.host == "0.0.0.0"
        assert server.port == 9000
        assert server._security.max_connections == 50

    def test_api_key_from_param(self):
        """Test API key from parameter."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer(api_key="test-key")
        assert server._security.api_key == "test-key"

    def test_api_key_from_env(self):
        """Test API key from environment variable."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        with patch.dict("os.environ", {"VOICE_API_KEY": "env-key"}):
            server = VoiceWebSocketServer()
            assert server._security.api_key == "env-key"

    def test_allowed_origins(self):
        """Test allowed origins configuration."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        origins = {"http://localhost:3000", "https://myapp.com"}
        server = VoiceWebSocketServer(allowed_origins=origins)

        assert server._security.allowed_origins == origins


# =============================================================================
# VoiceWebSocketServer Engine Tests
# =============================================================================

class TestVoiceWebSocketServerEngines:
    """Tests for engine lazy loading."""

    def test_get_engine_lazy_load(self):
        """Test lazy loading of voice engine."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()

        with patch("voice_soundboard.websocket_server.VoiceEngine") as MockEngine:
            mock_instance = Mock()
            MockEngine.return_value = mock_instance

            engine = server._get_engine()

            MockEngine.assert_called_once()
            assert engine == mock_instance
            assert server._engine == mock_instance

    def test_get_engine_cached(self):
        """Test engine is cached after first load."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_engine = Mock()
        server._engine = mock_engine

        engine = server._get_engine()
        assert engine == mock_engine

    def test_get_streaming_engine_lazy_load(self):
        """Test lazy loading of streaming engine."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()

        with patch("voice_soundboard.websocket_server.StreamingEngine") as MockEngine:
            mock_instance = Mock()
            MockEngine.return_value = mock_instance

            engine = server._get_streaming_engine()

            MockEngine.assert_called_once()
            assert engine == mock_instance

    def test_get_client_id(self):
        """Test client ID generation."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = Mock()

        client_id = server._get_client_id(mock_ws)
        assert isinstance(client_id, str)


# =============================================================================
# VoiceWebSocketServer Response Tests
# =============================================================================

class TestVoiceWebSocketServerResponses:
    """Tests for response sending methods."""

    @pytest.mark.asyncio
    async def test_send_response(self):
        """Test sending a response."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server._send_response(
            mock_ws,
            success=True,
            action="test",
            data={"key": "value"},
            request_id="req-1",
        )

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["action"] == "test"
        assert sent_data["request_id"] == "req-1"

    @pytest.mark.asyncio
    async def test_send_response_connection_closed(self):
        """Test sending response when connection is closed."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer
        import websockets

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = websockets.exceptions.ConnectionClosed(None, None)

        # Should not raise
        await server._send_response(mock_ws, True, "test", {})

    @pytest.mark.asyncio
    async def test_send_error(self):
        """Test sending an error response."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server._send_error(mock_ws, "test", "Something failed", "req-1")

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert sent_data["error"] == "Something failed"


# =============================================================================
# Handler Tests
# =============================================================================

class TestHandleSpeak:
    """Tests for handle_speak method."""

    @pytest.mark.asyncio
    async def test_handle_speak_missing_text(self):
        """Test speak with missing text."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak(mock_ws, {}, "req-1")

        # Should send error about missing text
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_handle_speak_success(self):
        """Test successful speak."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        mock_result = Mock()
        mock_result.audio_path = "/tmp/test.wav"
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 0.8

        mock_engine = Mock()
        mock_engine.speak.return_value = mock_result
        server._engine = mock_engine

        await server.handle_speak(
            mock_ws,
            {"text": "Hello world"},
            "req-1",
        )

        mock_ws.send.assert_called()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["voice"] == "af_bella"

    @pytest.mark.asyncio
    async def test_handle_speak_with_emotion(self):
        """Test speak with emotion."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        mock_result = Mock()
        mock_result.audio_path = "/tmp/test.wav"
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 0.8

        mock_engine = Mock()
        mock_engine.speak.return_value = mock_result
        server._engine = mock_engine

        await server.handle_speak(
            mock_ws,
            {"text": "Hello", "emotion": "happy"},
            "req-1",
        )

        mock_ws.send.assert_called()


class TestHandleEffect:
    """Tests for handle_effect method."""

    @pytest.mark.asyncio
    async def test_handle_effect_missing_param(self):
        """Test effect with missing parameter."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_effect(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Missing" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_effect_success(self):
        """Test successful effect playback."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        mock_effect = Mock()
        mock_effect.duration = 0.5

        with patch("voice_soundboard.websocket_server.get_effect", return_value=mock_effect):
            await server.handle_effect(mock_ws, {"effect": "chime"}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["effect"] == "chime"


class TestHandleStop:
    """Tests for handle_stop method."""

    @pytest.mark.asyncio
    async def test_handle_stop(self):
        """Test stop playback."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        with patch("voice_soundboard.websocket_server.stop_playback"):
            await server.handle_stop(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["stopped"] is True


class TestHandleListVoices:
    """Tests for handle_list_voices method."""

    @pytest.mark.asyncio
    async def test_handle_list_voices(self):
        """Test listing voices."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_voices(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "voices" in sent_data["data"]
        assert "count" in sent_data["data"]

    @pytest.mark.asyncio
    async def test_handle_list_voices_with_filter(self):
        """Test listing voices with gender filter."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_voices(
            mock_ws,
            {"gender": "female"},
            "req-1",
        )

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True


class TestHandleListPresets:
    """Tests for handle_list_presets method."""

    @pytest.mark.asyncio
    async def test_handle_list_presets(self):
        """Test listing presets."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_presets(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "presets" in sent_data["data"]


class TestHandleListEmotions:
    """Tests for handle_list_emotions method."""

    @pytest.mark.asyncio
    async def test_handle_list_emotions(self):
        """Test listing emotions."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_emotions(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "emotions" in sent_data["data"]


class TestHandleListEffects:
    """Tests for handle_list_effects method."""

    @pytest.mark.asyncio
    async def test_handle_list_effects(self):
        """Test listing effects."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_effects(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "effects" in sent_data["data"]


class TestHandleStatus:
    """Tests for handle_status method."""

    @pytest.mark.asyncio
    async def test_handle_status(self):
        """Test status request."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_status(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["server"] == "voice-soundboard"
        assert "clients" in sent_data["data"]
        assert "engine_loaded" in sent_data["data"]


# =============================================================================
# Message Handling Tests
# =============================================================================

class TestHandleMessage:
    """Tests for handle_message method."""

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self):
        """Test handling invalid JSON."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_message(mock_ws, "not valid json")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Invalid JSON" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_unknown_action(self):
        """Test handling unknown action."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_message(mock_ws, json.dumps({"action": "unknown_action"}))

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Unknown action" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_rate_limit(self):
        """Test rate limiting."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        with patch.object(server._security, "check_rate_limit", return_value=False):
            await server.handle_message(mock_ws, json.dumps({"action": "status"}))

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Rate limit" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_too_large(self):
        """Test message size limit."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        server._config.max_message_size = 100
        mock_ws = AsyncMock()

        large_message = json.dumps({"action": "speak", "text": "x" * 1000})
        await server.handle_message(mock_ws, large_message)

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "too large" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_handler(self):
        """Test message routes to correct handler."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        with patch.object(server, "handle_status") as mock_handler:
            mock_handler.return_value = None

            await server.handle_message(
                mock_ws,
                json.dumps({"action": "status", "request_id": "test-123"})
            )

            mock_handler.assert_called_once()


# =============================================================================
# Connection Handler Tests
# =============================================================================

class TestConnectionHandler:
    """Tests for connection_handler method."""

    @pytest.mark.asyncio
    async def test_connection_rejected_at_capacity(self):
        """Test connection rejected when at capacity."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        with patch.object(server._security, "can_accept_connection", return_value=False):
            await server.connection_handler(mock_ws)

        mock_ws.close.assert_called_with(1013, "Server at capacity")

    @pytest.mark.asyncio
    async def test_connection_rejected_invalid_origin(self):
        """Test connection rejected for invalid origin."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer(allowed_origins={"http://allowed.com"})
        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": "http://evil.com"}

        with patch.object(server._security, "can_accept_connection", return_value=True):
            with patch.object(server._security, "validate_origin", return_value=False):
                await server.connection_handler(mock_ws)

        mock_ws.close.assert_called_with(1008, "Origin not allowed")

    @pytest.mark.asyncio
    async def test_connection_rejected_invalid_api_key(self):
        """Test connection rejected for invalid API key."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer(api_key="secret-key")
        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": ""}
        mock_ws.request = Mock()
        mock_ws.request.path = "/?key=wrong-key"

        with patch.object(server._security, "can_accept_connection", return_value=True):
            with patch.object(server._security, "validate_origin", return_value=True):
                with patch.object(server._security, "validate_api_key", return_value=False):
                    await server.connection_handler(mock_ws)

        mock_ws.close.assert_called_with(1008, "Invalid API key")


# =============================================================================
# Speak Stream Handler Tests
# =============================================================================

class TestHandleSpeakStream:
    """Tests for handle_speak_stream method."""

    @pytest.mark.asyncio
    async def test_handle_speak_stream_missing_text(self):
        """Test speak_stream with missing text."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak_stream(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False


class TestHandleSpeakRealtime:
    """Tests for handle_speak_realtime method."""

    @pytest.mark.asyncio
    async def test_handle_speak_realtime_missing_text(self):
        """Test speak_realtime with missing text."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak_realtime(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False


class TestHandleSpeakSSML:
    """Tests for handle_speak_ssml method."""

    @pytest.mark.asyncio
    async def test_handle_speak_ssml_missing_ssml(self):
        """Test speak_ssml with missing ssml."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak_ssml(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateServer:
    """Tests for create_server function."""

    def test_create_server_defaults(self):
        """Test create_server with defaults."""
        from voice_soundboard.websocket_server import create_server

        server = create_server()

        assert server.host == "localhost"
        assert server.port == 8765

    def test_create_server_custom(self):
        """Test create_server with custom params."""
        from voice_soundboard.websocket_server import create_server

        server = create_server(
            host="0.0.0.0",
            port=9000,
            api_key="test-key",
        )

        assert server.host == "0.0.0.0"
        assert server.port == 9000
        assert server._security.api_key == "test-key"


# =============================================================================
# SSL/TLS Tests
# =============================================================================

class TestSSLConfiguration:
    """Tests for SSL/TLS configuration."""

    def test_ssl_context_created(self):
        """Test SSL context is created with valid certs."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer
        import tempfile
        import os

        # Create dummy cert files
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "cert.pem")
            key_path = os.path.join(tmpdir, "key.pem")

            # Create dummy files (won't actually work as SSL certs)
            with open(cert_path, "w") as f:
                f.write("dummy cert")
            with open(key_path, "w") as f:
                f.write("dummy key")

            # This will fail because the certs are invalid, but we test the path
            with pytest.raises(ValueError, match="SSL"):
                VoiceWebSocketServer(ssl_cert=cert_path, ssl_key=key_path)

    def test_ssl_file_not_found(self):
        """Test SSL file not found error."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        with pytest.raises(ValueError, match="SSL file not found"):
            VoiceWebSocketServer(
                ssl_cert="/nonexistent/cert.pem",
                ssl_key="/nonexistent/key.pem",
            )


# =============================================================================
# Engine Error Tests
# =============================================================================

class TestEngineErrors:
    """Tests for engine initialization errors."""

    def test_get_engine_error(self):
        """Test engine initialization error."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()

        with patch("voice_soundboard.websocket_server.VoiceEngine") as MockEngine:
            MockEngine.side_effect = Exception("Failed to load")

            with pytest.raises(RuntimeError, match="Voice engine initialization failed"):
                server._get_engine()

    def test_get_streaming_engine_error(self):
        """Test streaming engine initialization error."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()

        with patch("voice_soundboard.websocket_server.StreamingEngine") as MockEngine:
            MockEngine.side_effect = Exception("Failed to load")

            with pytest.raises(RuntimeError, match="Streaming engine initialization failed"):
                server._get_streaming_engine()


# =============================================================================
# API Key Extraction Tests
# =============================================================================

class TestApiKeyExtraction:
    """Tests for API key extraction from query string."""

    @pytest.mark.asyncio
    async def test_api_key_from_query_string(self):
        """Test API key extraction from query string."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer(api_key="correct-key")
        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": ""}
        mock_ws.request = Mock()
        mock_ws.request.path = "/?key=correct-key"

        with patch.object(server._security, "can_accept_connection", return_value=True):
            with patch.object(server._security, "validate_origin", return_value=True):
                with patch.object(server._security, "validate_api_key", return_value=True) as mock_validate:
                    # Mock the async iteration to stop immediately
                    mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

                    await server.connection_handler(mock_ws)

                    mock_validate.assert_called_with("correct-key")

    @pytest.mark.asyncio
    async def test_api_key_with_multiple_params(self):
        """Test API key extraction with multiple query params."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer(api_key="secret")
        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": ""}
        mock_ws.request = Mock()
        mock_ws.request.path = "/?foo=bar&key=secret&baz=qux"

        with patch.object(server._security, "can_accept_connection", return_value=True):
            with patch.object(server._security, "validate_origin", return_value=True):
                with patch.object(server._security, "validate_api_key", return_value=True) as mock_validate:
                    mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

                    await server.connection_handler(mock_ws)

                    mock_validate.assert_called_with("secret")
