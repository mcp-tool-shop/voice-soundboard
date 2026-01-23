"""
Tests for WebSocket Server (websocket_server.py).

Tests cover:
- Server initialization and configuration
- WSResponse dataclass
- Message handling and routing
- Security validation (origin, API key, rate limiting)
- Handler methods (mocked)
- Connection management
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict

from voice_soundboard.websocket_server import (
    WSResponse,
    VoiceWebSocketServer,
    create_server,
)


class TestWSResponse:
    """Tests for WSResponse dataclass."""

    def test_response_structure(self):
        """Test WSResponse has correct fields."""
        response = WSResponse(
            success=True,
            action="speak",
            data={"duration": 1.5},
            error=None,
            request_id="req-123",
        )

        assert response.success is True
        assert response.action == "speak"
        assert response.data == {"duration": 1.5}
        assert response.error is None
        assert response.request_id == "req-123"

    def test_response_to_json(self):
        """Test WSResponse serializes to JSON correctly."""
        response = WSResponse(
            success=True,
            action="speak",
            data={"voice": "af_bella"},
            request_id="test-id"
        )

        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert parsed["action"] == "speak"
        assert parsed["data"]["voice"] == "af_bella"
        assert parsed["request_id"] == "test-id"

    def test_response_with_error(self):
        """Test WSResponse with error message."""
        response = WSResponse(
            success=False,
            action="speak",
            data={},
            error="Text is required",
        )

        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is False
        assert parsed["error"] == "Text is required"

    def test_response_defaults(self):
        """Test WSResponse default values."""
        response = WSResponse(success=True, action="test", data={})

        assert response.error is None
        assert response.request_id is None


class TestVoiceWebSocketServerInit:
    """Tests for VoiceWebSocketServer initialization."""

    def test_server_init_defaults(self):
        """Test server initializes with default values."""
        server = VoiceWebSocketServer()

        assert server.host == "localhost"
        assert server.port == 8765
        assert server._engine is None
        assert server._streaming_engine is None
        assert server._is_running is False
        assert server._ssl_context is None

    def test_server_init_custom_host_port(self):
        """Test server with custom host and port."""
        server = VoiceWebSocketServer(host="0.0.0.0", port=9000)

        assert server.host == "0.0.0.0"
        assert server.port == 9000

    def test_server_init_with_api_key(self):
        """Test server with API key configured."""
        server = VoiceWebSocketServer(api_key="test-secret-key")

        assert server._security.api_key == "test-secret-key"

    def test_server_init_with_max_connections(self):
        """Test server with custom max connections."""
        server = VoiceWebSocketServer(max_connections=50)

        assert server._security.max_connections == 50

    def test_server_security_manager_created(self):
        """Test that security manager is initialized."""
        server = VoiceWebSocketServer()

        assert server._security is not None
        assert hasattr(server._security, 'validate_origin')
        assert hasattr(server._security, 'validate_api_key')


class TestServerEngineLazyLoading:
    """Tests for lazy engine loading."""

    def test_engine_none_initially(self):
        """Test engine is None before first use."""
        server = VoiceWebSocketServer()

        assert server._engine is None
        assert server._streaming_engine is None

    @patch('voice_soundboard.websocket_server.VoiceEngine')
    def test_get_engine_creates_instance(self, mock_engine_class):
        """Test _get_engine creates VoiceEngine on first call."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        server = VoiceWebSocketServer()
        result = server._get_engine()

        mock_engine_class.assert_called_once()
        assert result == mock_engine
        assert server._engine == mock_engine

    @patch('voice_soundboard.websocket_server.VoiceEngine')
    def test_get_engine_reuses_instance(self, mock_engine_class):
        """Test _get_engine returns same instance on subsequent calls."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        server = VoiceWebSocketServer()
        engine1 = server._get_engine()
        engine2 = server._get_engine()

        # Should only be called once
        mock_engine_class.assert_called_once()
        assert engine1 is engine2

    @patch('voice_soundboard.websocket_server.StreamingEngine')
    def test_get_streaming_engine_creates_instance(self, mock_streaming_class):
        """Test _get_streaming_engine creates instance."""
        mock_engine = Mock()
        mock_streaming_class.return_value = mock_engine

        server = VoiceWebSocketServer()
        result = server._get_streaming_engine()

        mock_streaming_class.assert_called_once()
        assert result == mock_engine


class TestServerClientId:
    """Tests for client ID generation."""

    def test_get_client_id_returns_string(self):
        """Test client ID is a string."""
        server = VoiceWebSocketServer()
        mock_ws = Mock()

        client_id = server._get_client_id(mock_ws)

        assert isinstance(client_id, str)

    def test_get_client_id_unique_per_connection(self):
        """Test different connections get different IDs."""
        server = VoiceWebSocketServer()
        mock_ws1 = Mock()
        mock_ws2 = Mock()

        id1 = server._get_client_id(mock_ws1)
        id2 = server._get_client_id(mock_ws2)

        assert id1 != id2


class TestServerSendResponse:
    """Tests for _send_response method."""

    @pytest.mark.asyncio
    async def test_send_response_success(self):
        """Test sending a success response."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server._send_response(
            mock_ws,
            success=True,
            action="speak",
            data={"duration": 1.5},
            request_id="req-1"
        )

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["action"] == "speak"
        assert sent_data["data"]["duration"] == 1.5

    @pytest.mark.asyncio
    async def test_send_error(self):
        """Test sending an error response."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server._send_error(mock_ws, "speak", "Text required", "req-2")

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert sent_data["error"] == "Text required"


class TestHandleMessage:
    """Tests for message routing in handle_message."""

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self):
        """Test handling invalid JSON message."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_message(mock_ws, "not valid json")

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Invalid JSON" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_unknown_action(self):
        """Test handling unknown action."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        message = json.dumps({"action": "unknown_action"})
        await server.handle_message(mock_ws, message)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Unknown action" in sent_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_handler(self):
        """Test message routes to correct handler."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        # Mock the list_voices handler
        server.handle_list_voices = AsyncMock()

        message = json.dumps({"action": "list_voices"})
        await server.handle_message(mock_ws, message)

        server.handle_list_voices.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_rate_limit(self):
        """Test rate limiting in message handling."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        # Exhaust rate limit
        server._security.check_rate_limit = Mock(return_value=False)

        message = json.dumps({"action": "list_voices"})
        await server.handle_message(mock_ws, message)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Rate limit" in sent_data["error"]


class TestHandleListVoices:
    """Tests for handle_list_voices handler."""

    @pytest.mark.asyncio
    async def test_list_voices_returns_all(self):
        """Test list_voices returns all voices."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_voices(mock_ws, {}, "req-1")

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "voices" in sent_data["data"]
        assert sent_data["data"]["count"] > 0

    @pytest.mark.asyncio
    async def test_list_voices_filter_gender(self):
        """Test list_voices with gender filter."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_voices(mock_ws, {"gender": "female"}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        for voice in sent_data["data"]["voices"]:
            assert voice["gender"] == "female"


class TestHandleListPresets:
    """Tests for handle_list_presets handler."""

    @pytest.mark.asyncio
    async def test_list_presets_returns_all(self):
        """Test list_presets returns all presets."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_presets(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "presets" in sent_data["data"]
        assert len(sent_data["data"]["presets"]) > 0


class TestHandleListEmotions:
    """Tests for handle_list_emotions handler."""

    @pytest.mark.asyncio
    async def test_list_emotions_returns_all(self):
        """Test list_emotions returns all emotions."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_emotions(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "emotions" in sent_data["data"]
        assert len(sent_data["data"]["emotions"]) > 0


class TestHandleListEffects:
    """Tests for handle_list_effects handler."""

    @pytest.mark.asyncio
    async def test_list_effects_returns_all(self):
        """Test list_effects returns all effects."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_effects(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "effects" in sent_data["data"]


class TestHandleStatus:
    """Tests for handle_status handler."""

    @pytest.mark.asyncio
    async def test_status_returns_info(self):
        """Test status returns server information."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_status(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["server"] == "voice-soundboard"
        assert "version" in sent_data["data"]
        assert "clients" in sent_data["data"]
        assert "engine_loaded" in sent_data["data"]


class TestHandleStop:
    """Tests for handle_stop handler."""

    @pytest.mark.asyncio
    @patch('voice_soundboard.websocket_server.stop_playback')
    async def test_stop_calls_stop_playback(self, mock_stop):
        """Test stop handler calls stop_playback."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_stop(mock_ws, {}, "req-1")

        mock_stop.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["stopped"] is True


class TestHandleEffect:
    """Tests for handle_effect handler."""

    @pytest.mark.asyncio
    async def test_effect_missing_name(self):
        """Test effect handler with missing effect name."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_effect(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "Missing" in sent_data["error"]

    @pytest.mark.asyncio
    @patch('voice_soundboard.websocket_server.get_effect')
    async def test_effect_plays_effect(self, mock_get_effect):
        """Test effect handler plays the effect."""
        mock_effect = Mock()
        mock_effect.duration = 0.5
        mock_get_effect.return_value = mock_effect

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_effect(mock_ws, {"effect": "chime"}, "req-1")

        mock_get_effect.assert_called_with("chime")
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert sent_data["data"]["effect"] == "chime"


class TestHandleSpeak:
    """Tests for handle_speak handler."""

    @pytest.mark.asyncio
    async def test_speak_missing_text(self):
        """Test speak handler with missing text."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_speak_empty_text(self):
        """Test speak handler with empty text."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak(mock_ws, {"text": ""}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    @patch('voice_soundboard.websocket_server.VoiceEngine')
    async def test_speak_success(self, mock_engine_class):
        """Test speak handler with valid text."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 10.0
        mock_engine.speak.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak(mock_ws, {"text": "Hello world"}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True
        assert "file_path" in sent_data["data"]
        assert sent_data["data"]["voice"] == "af_bella"


class TestHandleSpeakSSML:
    """Tests for handle_speak_ssml handler."""

    @pytest.mark.asyncio
    async def test_speak_ssml_missing_ssml(self):
        """Test speak_ssml handler with missing ssml."""
        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_speak_ssml(mock_ws, {}, "req-1")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False


class TestCreateServer:
    """Tests for create_server factory function."""

    def test_create_server_defaults(self):
        """Test create_server with defaults."""
        server = create_server()

        assert isinstance(server, VoiceWebSocketServer)
        assert server.host == "localhost"
        assert server.port == 8765

    def test_create_server_custom_params(self):
        """Test create_server with custom parameters."""
        server = create_server(
            host="0.0.0.0",
            port=9000,
            api_key="secret",
        )

        assert server.host == "0.0.0.0"
        assert server.port == 9000
        assert server._security.api_key == "secret"


class TestConnectionHandler:
    """Tests for connection_handler security checks."""

    @pytest.mark.asyncio
    async def test_connection_rejected_at_capacity(self):
        """Test connection rejected when at capacity."""
        server = VoiceWebSocketServer(max_connections=1)
        server._security.add_connection("existing")

        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": "http://localhost"}
        mock_ws.request = Mock()
        mock_ws.request.path = "/"

        await server.connection_handler(mock_ws)

        mock_ws.close.assert_called_once()
        args = mock_ws.close.call_args[0]
        assert args[0] == 1013  # Server at capacity code

    @pytest.mark.asyncio
    async def test_connection_rejected_invalid_origin(self):
        """Test connection rejected for invalid origin."""
        server = VoiceWebSocketServer()

        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": "http://evil.com"}
        mock_ws.request = Mock()
        mock_ws.request.path = "/"

        await server.connection_handler(mock_ws)

        mock_ws.close.assert_called_once()
        args = mock_ws.close.call_args[0]
        assert args[0] == 1008  # Policy violation code

    @pytest.mark.asyncio
    async def test_connection_rejected_invalid_api_key(self):
        """Test connection rejected for invalid API key."""
        server = VoiceWebSocketServer(api_key="correct-key")

        mock_ws = AsyncMock()
        mock_ws.request_headers = {"Origin": "http://localhost"}
        mock_ws.request = Mock()
        mock_ws.request.path = "/?key=wrong-key"

        await server.connection_handler(mock_ws)

        mock_ws.close.assert_called_once()
        args = mock_ws.close.call_args[0]
        assert args[0] == 1008


class TestSSLConfiguration:
    """Tests for SSL/TLS configuration."""

    def test_ssl_disabled_by_default(self):
        """Test SSL is disabled by default."""
        server = VoiceWebSocketServer()
        assert server._ssl_context is None

    @patch('ssl.SSLContext')
    def test_ssl_enabled_with_cert_and_key(self, mock_ssl_context_class):
        """Test SSL is enabled when cert and key provided."""
        mock_context = Mock()
        mock_ssl_context_class.return_value = mock_context

        server = VoiceWebSocketServer(
            ssl_cert="cert.pem",
            ssl_key="key.pem"
        )

        mock_context.load_cert_chain.assert_called_once_with("cert.pem", "key.pem")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
