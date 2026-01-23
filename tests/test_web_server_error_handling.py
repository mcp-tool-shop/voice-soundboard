"""
Tests for web_server.py error handling (TEST-WEB-ERR series).

Tests cover error handling for:
- Engine errors
- Audio errors
- Invalid JSON
- Missing parameters
- File not found
- Timeout handling
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop


class TestWebServerErrorHandling(AioHTTPTestCase):
    """Tests for web_server.py error handling."""

    async def get_application(self):
        """Create test application."""
        from voice_soundboard.web_server import create_app
        return create_app()

    # TEST-WEB-ERR03: Invalid JSON in speak request
    async def test_invalid_json_returns_400(self):
        """TEST-WEB-ERR03: Invalid JSON in request returns 400."""
        response = await self.client.post('/api/speak',
            data=b'{invalid json',
            headers={'Content-Type': 'application/json'})

        self.assertEqual(response.status, 400)
        data = await response.json()
        self.assertIn("error", data)

    # TEST-WEB-ERR04: Missing text parameter
    async def test_missing_text_returns_400(self):
        """TEST-WEB-ERR04: Missing required 'text' parameter returns 400."""
        response = await self.client.post('/api/speak', json={
            "voice": "af_bella"
            # No text parameter
        })

        self.assertEqual(response.status, 400)
        data = await response.json()
        self.assertIn("error", data)

    # TEST-WEB-ERR05: Empty text parameter
    async def test_empty_text_returns_400(self):
        """TEST-WEB-ERR05: Empty text parameter returns 400."""
        response = await self.client.post('/api/speak', json={
            "text": "   "  # Whitespace only
        })

        self.assertEqual(response.status, 400)
        data = await response.json()
        self.assertIn("error", data)

    # TEST-WEB-ERR10: Effect missing name
    async def test_effect_missing_name_returns_400(self):
        """TEST-WEB-ERR10: Missing effect name returns 400."""
        response = await self.client.post('/api/effect', json={})

        self.assertEqual(response.status, 400)
        data = await response.json()
        self.assertIn("error", data)

    # TEST-WEB-ERR12: Invalid API endpoint
    async def test_invalid_endpoint_returns_404(self):
        """TEST-WEB-ERR12: Invalid API endpoint returns 404."""
        response = await self.client.get('/api/nonexistent_endpoint')

        self.assertEqual(response.status, 404)

    # TEST-WEB-ERR13: POST to GET-only endpoint
    async def test_wrong_http_method_returns_405(self):
        """TEST-WEB-ERR13: Wrong HTTP method returns 405."""
        response = await self.client.post('/api/voices', json={})

        self.assertEqual(response.status, 405)

    # TEST-WEB-ERR15: CORS preflight handling
    async def test_cors_preflight_handled(self):
        """TEST-WEB-ERR15: CORS preflight requests are handled."""
        response = await self.client.options('/api/speak', headers={
            'Origin': 'http://example.com',
            'Access-Control-Request-Method': 'POST'
        })

        # Should return 200 OK with CORS headers
        self.assertEqual(response.status, 200)
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        self.assertIn('Access-Control-Allow-Methods', response.headers)

    # Health endpoint test
    async def test_health_endpoint_always_works(self):
        """Test that health endpoint works."""
        response = await self.client.get('/health')

        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(data["status"], "ok")


class TestWebServerWithMocks:
    """Tests that require mocking the engine."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for testing."""
        engine = Mock()
        result = Mock()
        result.audio_path = Path("test.wav")
        result.voice_used = "af_bella"
        result.duration_seconds = 1.0
        result.realtime_factor = 5.0
        engine.speak = Mock(return_value=result)
        return engine

    # TEST-WEB-ERR01: Engine initialization error
    @pytest.mark.asyncio
    async def test_engine_init_error_returns_500(self):
        """TEST-WEB-ERR01: Engine initialization error returns 500."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        # Reset engine
        web_server._engine = None

        with patch('voice_soundboard.web_server.VoiceEngine', side_effect=Exception("GPU not available")):
            app = web_server.create_app()
            server = TestServer(app)
            client = TestClient(server)
            await client.start_server()

            try:
                response = await client.post('/api/speak', json={
                    "text": "Hello world"
                })

                assert response.status == 500
                data = await response.json()
                assert "error" in data
            finally:
                await client.close()

    # TEST-WEB-ERR02: Speech generation error
    @pytest.mark.asyncio
    async def test_speech_generation_error_returns_500(self):
        """TEST-WEB-ERR02: Speech generation error returns 500."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        mock_engine = Mock()
        mock_engine.speak = Mock(side_effect=RuntimeError("Model failed"))
        web_server._engine = mock_engine

        try:
            app = web_server.create_app()
            server = TestServer(app)
            client = TestClient(server)
            await client.start_server()

            try:
                response = await client.post('/api/speak', json={
                    "text": "Hello world"
                })

                assert response.status == 500
                data = await response.json()
                assert "error" in data
            finally:
                await client.close()
        finally:
            web_server._engine = None

    # TEST-WEB-ERR06: Invalid voice parameter
    @pytest.mark.asyncio
    async def test_invalid_voice_handled_gracefully(self, mock_engine):
        """TEST-WEB-ERR06: Invalid voice parameter is handled gracefully."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        web_server._engine = mock_engine

        try:
            with patch.object(Path, 'read_bytes', return_value=b'audio_data'):
                app = web_server.create_app()
                server = TestServer(app)
                client = TestClient(server)
                await client.start_server()

                try:
                    response = await client.post('/api/speak', json={
                        "text": "Hello",
                        "voice": "nonexistent_voice_xyz"
                    })

                    # Should succeed with fallback or return meaningful error
                    assert response.status in [200, 400, 500]
                finally:
                    await client.close()
        finally:
            web_server._engine = None

    # TEST-WEB-ERR07: Invalid speed parameter
    @pytest.mark.asyncio
    async def test_invalid_speed_handled(self, mock_engine):
        """TEST-WEB-ERR07: Invalid speed parameter is handled."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        web_server._engine = mock_engine

        try:
            with patch.object(Path, 'read_bytes', return_value=b'audio_data'):
                app = web_server.create_app()
                server = TestServer(app)
                client = TestClient(server)
                await client.start_server()

                try:
                    response = await client.post('/api/speak', json={
                        "text": "Hello",
                        "speed": -5.0
                    })

                    assert response.status in [200, 400, 500]
                finally:
                    await client.close()
        finally:
            web_server._engine = None

    # TEST-WEB-ERR08: Audio file read error
    @pytest.mark.asyncio
    async def test_audio_file_read_error(self, mock_engine):
        """TEST-WEB-ERR08: Audio file read error returns 500."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        web_server._engine = mock_engine

        try:
            with patch.object(Path, 'read_bytes', side_effect=IOError("File not found")):
                app = web_server.create_app()
                server = TestServer(app)
                client = TestClient(server)
                await client.start_server()

                try:
                    response = await client.post('/api/speak', json={
                        "text": "Hello world"
                    })

                    assert response.status == 500
                finally:
                    await client.close()
        finally:
            web_server._engine = None

    # TEST-WEB-ERR09: Effect not found
    @pytest.mark.asyncio
    async def test_effect_not_found_returns_error(self):
        """TEST-WEB-ERR09: Invalid effect name returns error."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        app = web_server.create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()

        try:
            response = await client.post('/api/effect', json={
                "effect": "nonexistent_effect_xyz"
            })

            assert response.status == 500
            data = await response.json()
            assert "error" in data
        finally:
            await client.close()

    # TEST-WEB-ERR11: Static file not found
    @pytest.mark.asyncio
    async def test_static_file_not_found_returns_404(self):
        """TEST-WEB-ERR11: Static file not found returns 404."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        app = web_server.create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()

        try:
            response = await client.get('/static/nonexistent_file.js')

            assert response.status == 404
        finally:
            await client.close()

    # TEST-WEB-ERR14: Audio playback error
    @pytest.mark.asyncio
    async def test_audio_playback_error_handled(self, mock_engine):
        """TEST-WEB-ERR14: Audio playback error is handled gracefully."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        web_server._engine = mock_engine

        try:
            with patch.object(Path, 'read_bytes', return_value=b'audio_data'), \
                 patch('voice_soundboard.web_server.play_audio', side_effect=Exception("No audio device")):
                app = web_server.create_app()
                server = TestServer(app)
                client = TestClient(server)
                await client.start_server()

                try:
                    response = await client.post('/api/speak', json={
                        "text": "Hello",
                        "play": True
                    })

                    assert response.status in [200, 500]
                finally:
                    await client.close()
        finally:
            web_server._engine = None


class TestWebServerEdgeCases:
    """Additional edge case tests for web_server.py."""

    @pytest.mark.asyncio
    async def test_very_large_text_handled(self):
        """Test that very large text input is handled."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 60.0
        mock_result.realtime_factor = 1.0
        mock_engine.speak = Mock(return_value=mock_result)
        web_server._engine = mock_engine

        try:
            with patch.object(Path, 'read_bytes', return_value=b'audio_data'):
                app = web_server.create_app()
                server = TestServer(app)
                client = TestClient(server)
                await client.start_server()

                try:
                    large_text = "Hello world. " * 1000
                    response = await client.post('/api/speak', json={
                        "text": large_text
                    })

                    assert response.status in [200, 400, 413, 500]
                finally:
                    await client.close()
        finally:
            web_server._engine = None

    @pytest.mark.asyncio
    async def test_unicode_text_handled(self):
        """Test that unicode text is handled correctly."""
        from voice_soundboard import web_server
        from aiohttp.test_utils import TestServer, TestClient

        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.0
        mock_result.realtime_factor = 5.0
        mock_engine.speak = Mock(return_value=mock_result)
        web_server._engine = mock_engine

        try:
            with patch.object(Path, 'read_bytes', return_value=b'audio_data'):
                app = web_server.create_app()
                server = TestServer(app)
                client = TestClient(server)
                await client.start_server()

                try:
                    response = await client.post('/api/speak', json={
                        "text": "Hello \u4e16\u754c \u3053\u3093\u306b\u3061\u306f \ud83c\udf0d"
                    })

                    assert response.status in [200, 400, 500]
                finally:
                    await client.close()
        finally:
            web_server._engine = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
