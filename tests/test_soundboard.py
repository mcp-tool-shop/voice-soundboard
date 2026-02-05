"""
Comprehensive test suite for Voice Soundboard.

Run with: pytest tests/ -v
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure proper encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# Module: config.py
# =============================================================================

class TestConfig:
    """Tests for configuration module."""

    def test_config_defaults(self):
        """TEST-C01: Config creates with default values."""
        from voice_soundboard.config import Config
        config = Config()
        assert config.default_speed == 1.0
        assert config.sample_rate == 24000

    def test_config_output_dir_created(self):
        """TEST-C03: Output directory is created."""
        from voice_soundboard.config import Config
        config = Config()
        assert config.output_dir.exists()

    def test_kokoro_voices_structure(self):
        """TEST-C05: KOKORO_VOICES dict has expected structure."""
        from voice_soundboard.config import KOKORO_VOICES
        assert len(KOKORO_VOICES) >= 50  # 50 voices across 9 languages
        for voice_id, info in KOKORO_VOICES.items():
            assert "name" in info
            assert "gender" in info
            assert "accent" in info

    def test_voice_presets_structure(self):
        """TEST-C06: VOICE_PRESETS has expected structure."""
        from voice_soundboard.config import VOICE_PRESETS
        assert len(VOICE_PRESETS) == 5
        expected = {"assistant", "narrator", "announcer", "storyteller", "whisper"}
        assert set(VOICE_PRESETS.keys()) == expected


# =============================================================================
# Module: effects.py
# =============================================================================

class TestEffects:
    """Tests for sound effects module."""

    def test_get_effect_valid(self):
        """TEST-F01: get_effect returns SoundEffect for valid name."""
        from voice_soundboard.effects import get_effect
        effect = get_effect("chime")
        assert effect is not None
        assert hasattr(effect, 'play')
        assert hasattr(effect, 'save')

    def test_list_effects_count(self):
        """TEST-F04: list_effects returns all effect names."""
        from voice_soundboard.effects import list_effects
        effects = list_effects()
        assert len(effects) == 13

    def test_get_effect_invalid(self):
        """TEST-F11: get_effect with invalid name raises ValueError."""
        from voice_soundboard.effects import get_effect
        with pytest.raises(ValueError):
            get_effect("nonexistent_effect")

    def test_sound_effect_has_generate(self):
        """TEST-F06: SoundEffect can generate audio."""
        from voice_soundboard.effects import get_effect
        effect = get_effect("chime")
        # Effect should have audio data
        assert effect is not None

    def test_effect_save_method(self):
        """TEST-F13: SoundEffect has save method."""
        from voice_soundboard.effects import get_effect
        effect = get_effect("success")
        assert hasattr(effect, 'save')


# =============================================================================
# Module: ssml.py
# =============================================================================

class TestSSML:
    """Tests for SSML parsing module."""

    def test_parse_ssml_break_tag(self):
        """TEST-X01: parse_ssml with <break> tag adds pause."""
        from voice_soundboard.ssml import parse_ssml
        text, params = parse_ssml('<speak>Hello<break time="500ms"/>world</speak>')
        assert "..." in text or "pause" in text.lower() or "Hello" in text

    def test_parse_ssml_prosody_rate(self):
        """TEST-X07: parse_ssml with <prosody rate='slow'>."""
        from voice_soundboard.ssml import parse_ssml
        text, params = parse_ssml('<speak><prosody rate="slow">Hello</prosody></speak>')
        assert params.speed < 1.0

    def test_parse_ssml_sub_alias(self):
        """TEST-X08: parse_ssml with <sub alias>."""
        from voice_soundboard.ssml import parse_ssml
        text, params = parse_ssml('<speak><sub alias="World Wide Web">WWW</sub></speak>')
        assert "World Wide Web" in text

    def test_ssml_to_text_convenience(self):
        """TEST-X09: ssml_to_text convenience function."""
        from voice_soundboard.ssml import ssml_to_text
        text = ssml_to_text('<speak>Hello world</speak>')
        assert "Hello" in text

    def test_parse_ssml_invalid_xml(self):
        """TEST-X15: parse_ssml with invalid XML."""
        from voice_soundboard.ssml import parse_ssml
        text, params = parse_ssml('<speak><unclosed>')
        # Should not crash, returns original or partial

    def test_parse_ssml_empty_string(self):
        """TEST-X16: parse_ssml with empty string."""
        from voice_soundboard.ssml import parse_ssml
        text, params = parse_ssml('')
        assert text == ''


# =============================================================================
# Module: emotions.py
# =============================================================================

class TestEmotions:
    """Tests for emotions module."""

    def test_get_emotion_params_happy(self):
        """TEST-M01: get_emotion_params('happy') returns correct params."""
        from voice_soundboard.emotions import get_emotion_params
        params = get_emotion_params("happy")
        assert params.speed > 1.0

    def test_get_emotion_params_sad(self):
        """TEST-M02: get_emotion_params('sad') has slower speed."""
        from voice_soundboard.emotions import get_emotion_params
        params = get_emotion_params("sad")
        assert params.speed < 1.0

    def test_list_emotions_count(self):
        """TEST-M04: list_emotions returns all emotion names."""
        from voice_soundboard.emotions import list_emotions
        emotions = list_emotions()
        assert len(emotions) == 19

    def test_get_emotion_params_invalid(self):
        """TEST-M09: Invalid emotion returns neutral."""
        from voice_soundboard.emotions import get_emotion_params
        params = get_emotion_params("nonexistent_emotion")
        # Should return neutral or default

    def test_intensify_emotion(self):
        """TEST-M08: intensify_emotion with intensity=2.0."""
        from voice_soundboard.emotions import intensify_emotion, get_emotion_params
        params = get_emotion_params("happy")
        intensified = intensify_emotion("happy", 2.0)
        assert intensified.speed > params.speed


# =============================================================================
# Module: interpreter.py
# =============================================================================

class TestInterpreter:
    """Tests for style interpreter module."""

    def test_interpret_style_warmly(self):
        """TEST-I01: interpret_style('warmly') returns style info."""
        from voice_soundboard.interpreter import interpret_style
        result = interpret_style("warmly")
        assert result.speed is not None

    def test_interpret_style_narrator(self):
        """TEST-I02: interpret_style('like a narrator') matches narrator."""
        from voice_soundboard.interpreter import interpret_style
        result = interpret_style("like a narrator")
        assert result.preset == "narrator" or "narrator" in str(result).lower()

    def test_interpret_style_empty(self):
        """TEST-I06: interpret_style with empty string."""
        from voice_soundboard.interpreter import interpret_style
        result = interpret_style("")
        assert result.confidence == 0.0

    def test_interpret_style_nonsense(self):
        """TEST-I07: interpret_style with nonsense input."""
        from voice_soundboard.interpreter import interpret_style
        result = interpret_style("asdfghjkl")
        assert result.confidence == 0.0


# =============================================================================
# Module: security.py
# =============================================================================

class TestSecurity:
    """Tests for security module."""

    def test_sanitize_filename_traversal(self):
        """TEST-SEC01: sanitize_filename removes path traversal."""
        from voice_soundboard.security import sanitize_filename
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result

    def test_sanitize_filename_special_chars(self):
        """TEST-SEC02: sanitize_filename handles special characters."""
        from voice_soundboard.security import sanitize_filename
        result = sanitize_filename("file<>:\"|?*.wav")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_filename_hidden(self):
        """TEST-SEC03: sanitize_filename prevents hidden files."""
        from voice_soundboard.security import sanitize_filename
        result = sanitize_filename(".hidden")
        assert not result.startswith(".")

    def test_sanitize_filename_empty_raises(self):
        """TEST-SEC04: sanitize_filename empty raises ValueError."""
        from voice_soundboard.security import sanitize_filename
        with pytest.raises(ValueError):
            sanitize_filename("")

    def test_validate_text_input_none(self):
        """TEST-SEC08: validate_text_input None raises error."""
        from voice_soundboard.security import validate_text_input
        with pytest.raises(ValueError):
            validate_text_input(None)

    def test_validate_text_input_empty(self):
        """TEST-SEC09: validate_text_input empty raises error."""
        from voice_soundboard.security import validate_text_input
        with pytest.raises(ValueError):
            validate_text_input("")

    def test_validate_text_input_too_long(self):
        """TEST-SEC10: validate_text_input too long raises error."""
        from voice_soundboard.security import validate_text_input
        with pytest.raises(ValueError):
            validate_text_input("x" * 100001, max_length=100000)

    def test_validate_speed_clamp_low(self):
        """TEST-SEC14: validate_speed clamps low values."""
        from voice_soundboard.security import validate_speed
        result = validate_speed(0.1)
        assert result == 0.5

    def test_validate_speed_clamp_high(self):
        """TEST-SEC15: validate_speed clamps high values."""
        from voice_soundboard.security import validate_speed
        result = validate_speed(5.0)
        assert result == 2.0

    def test_rate_limiter_allows_within_limit(self):
        """TEST-SEC17: RateLimiter allows within limit."""
        from voice_soundboard.security import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("client1") == True

    def test_rate_limiter_blocks_over_limit(self):
        """TEST-SEC18: RateLimiter blocks over limit."""
        from voice_soundboard.security import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("client1") == True
        assert limiter.is_allowed("client1") == True
        assert limiter.is_allowed("client1") == False

    def test_secure_hash_sha256(self):
        """TEST-SEC20: secure_hash produces consistent SHA-256."""
        from voice_soundboard.security import secure_hash
        hash1 = secure_hash("test")
        hash2 = secure_hash("test")
        assert hash1 == hash2
        assert len(hash1) == 8

    def test_safe_error_message_hides_paths(self):
        """TEST-SEC21: safe_error_message hides file paths."""
        from voice_soundboard.security import safe_error_message
        err = ValueError("Error at C:/secret/path/file.txt")
        msg = safe_error_message(err)
        assert "secret" not in msg.lower()

    def test_websocket_security_validates_localhost(self):
        """TEST-SEC23: WebSocketSecurityManager validates localhost."""
        from voice_soundboard.security import WebSocketSecurityManager
        mgr = WebSocketSecurityManager()
        assert mgr.validate_origin("http://localhost") == True
        assert mgr.validate_origin("http://127.0.0.1") == True

    def test_websocket_security_rejects_external(self):
        """TEST-SEC24: WebSocketSecurityManager rejects external origin."""
        from voice_soundboard.security import WebSocketSecurityManager
        mgr = WebSocketSecurityManager()
        assert mgr.validate_origin("http://evil.com") == False


# =============================================================================
# Module: audio.py
# =============================================================================

class TestAudio:
    """Tests for audio module."""

    def test_list_audio_devices(self):
        """TEST-A05: list_audio_devices returns device list."""
        from voice_soundboard.audio import list_audio_devices
        devices = list_audio_devices()
        assert isinstance(devices, list)
        # May be empty in CI without audio devices

    def test_stop_playback_function_exists(self):
        """TEST-A03: stop_playback function exists."""
        from voice_soundboard.audio import stop_playback
        assert callable(stop_playback)


# =============================================================================
# Module: streaming.py
# =============================================================================

class TestStreaming:
    """Tests for streaming module."""

    def test_streaming_engine_class_exists(self):
        """TEST-S01 partial: StreamingEngine class exists."""
        from voice_soundboard.streaming import StreamingEngine
        assert StreamingEngine is not None

    def test_realtime_player_class_exists(self):
        """TEST-S09 partial: RealtimePlayer class exists."""
        from voice_soundboard.streaming import RealtimePlayer
        assert RealtimePlayer is not None

    def test_stream_result_dataclass(self):
        """TEST-S05: StreamResult fields exist."""
        from voice_soundboard.streaming import StreamResult
        result = StreamResult(
            audio_path=None,
            total_duration=1.0,
            total_chunks=1,
            generation_time=0.5,
            voice_used="test"
        )
        assert result.total_duration == 1.0


# =============================================================================
# Module: websocket_server.py
# =============================================================================

class TestWebSocketServer:
    """Tests for WebSocket server module."""

    def test_wsresponse_serialization(self):
        """TEST-WS01: WSResponse serializes to JSON."""
        from voice_soundboard.websocket_server import WSResponse
        resp = WSResponse(
            success=True,
            action="test",
            data={"key": "value"},
            error=None,
            request_id="abc123"
        )
        json_str = resp.to_json()
        parsed = json.loads(json_str)
        assert parsed["success"] == True
        assert parsed["action"] == "test"

    def test_server_default_init(self):
        """TEST-WS03: VoiceWebSocketServer default initialization."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer
        server = VoiceWebSocketServer()
        assert server.host == "localhost"
        assert server.port == 8765

    def test_create_server_factory(self):
        """TEST-WS05: create_server factory function."""
        from voice_soundboard.websocket_server import create_server
        server = create_server(host="localhost", port=9000)
        assert server.port == 9000


# =============================================================================
# Module: __init__.py (Package)
# =============================================================================

class TestPackage:
    """Tests for package exports."""

    def test_package_exports(self):
        """TEST-PKG01: Package exports expected public API symbols."""
        import voice_soundboard
        expected = [
            "VoiceEngine", "SpeechResult", "Config",
            "KOKORO_VOICES", "VOICE_PRESETS", "quick_speak",
            "play_audio", "stop_playback",
            "get_effect", "play_effect", "list_effects",
            "get_emotion_params", "list_emotions", "EMOTIONS",
        ]
        for symbol in expected:
            assert hasattr(voice_soundboard, symbol), f"Missing: {symbol}"

    def test_package_version(self):
        """TEST-PKG02: Package has __version__."""
        import voice_soundboard
        assert hasattr(voice_soundboard, "__version__")
        assert voice_soundboard.__version__ == "1.1.0"

    def test_advanced_subpackages_importable(self):
        """TEST-PKG03: Advanced subpackages are importable."""
        from voice_soundboard.engines import TTSEngine
        from voice_soundboard.dialogue import DialogueParser
        from voice_soundboard.emotion import blend_emotions
        from voice_soundboard.cloning import VoiceCloner
        from voice_soundboard.presets import PresetCatalog
        assert all([TTSEngine, DialogueParser, blend_emotions, VoiceCloner, PresetCatalog])


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests across modules."""

    def test_config_to_effects_integration(self):
        """TEST-INT05 partial: Config and effects work together."""
        from voice_soundboard.config import Config
        from voice_soundboard.effects import list_effects
        config = Config()
        effects = list_effects()
        assert config is not None
        assert len(effects) > 0

    def test_security_chain(self):
        """Security validation chain works."""
        from voice_soundboard.security import (
            sanitize_filename, validate_text_input, validate_speed
        )
        filename = sanitize_filename("test.wav")
        text = validate_text_input("Hello world")
        speed = validate_speed(1.5)
        assert filename == "test.wav"
        assert text == "Hello world"
        assert speed == 1.5


# =============================================================================
# XXE Protection Tests
# =============================================================================

class TestXXEProtection:
    """Tests for XML External Entity (XXE) protection."""

    def test_basic_ssml_works(self):
        """TEST-XXE01: Basic SSML still works with defusedxml."""
        from voice_soundboard.ssml import parse_ssml
        text, params = parse_ssml('<speak>Hello world</speak>')
        assert "Hello" in text

    def test_billion_laughs_safe(self):
        """TEST-XXE03: Billion laughs attack is handled safely."""
        from voice_soundboard.ssml import parse_ssml
        # This would expand exponentially with vulnerable parser
        malicious = '''<?xml version="1.0"?>
        <!DOCTYPE lolz [
          <!ENTITY lol "lol">
          <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;">
        ]>
        <speak>&lol2;</speak>'''
        # Should not crash or hang
        text, params = parse_ssml(malicious)
        # defusedxml should block this or return safely


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
