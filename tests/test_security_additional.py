"""
Additional tests for Security module (security.py).

Tests cover:
- Security configuration
- Path sanitization edge cases
- Input validation edge cases
- Rate limiter behavior
- WebSocket security manager
- Safe error handling
"""

import pytest
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
import concurrent.futures

from voice_soundboard.security import (
    SecurityConfig,
    get_security_config,
    sanitize_filename,
    validate_output_path,
    safe_join_path,
    validate_text_input,
    validate_voice_id,
    validate_speed,
    validate_json_message,
    RateLimiter,
    get_rate_limiter,
    SafeError,
    safe_error_message,
    WebSocketSecurityManager,
    secure_hash,
)


class TestSecurityConfig:
    """Tests for SecurityConfig dataclass."""

    def test_config_default_values(self):
        """TEST-SEC25-29: Config has correct defaults."""
        config = SecurityConfig()

        assert config.max_text_length == 10000
        assert config.max_ssml_length == 15000
        assert config.max_filename_length == 100
        assert config.max_message_size == 65536  # 64KB
        assert config.connection_timeout == 30
        assert config.idle_timeout == 300  # 5 minutes

    def test_config_rate_limits(self):
        """Test rate limiting defaults."""
        config = SecurityConfig()

        assert config.rate_limit_requests == 60
        assert config.rate_limit_window == 60

    def test_config_allowed_dirs_initialized(self):
        """Test that allowed directories are initialized."""
        config = SecurityConfig()

        assert config.allowed_output_dirs is not None
        assert len(config.allowed_output_dirs) > 0


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_remove_path_components(self):
        """Test removal of path components."""
        assert sanitize_filename("foo/bar/baz.txt") == "baz.txt"
        assert sanitize_filename("foo\\bar\\baz.txt") == "baz.txt"

    def test_remove_null_bytes(self):
        """Test removal of null bytes."""
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_replace_special_chars(self):
        """Test replacement of special characters."""
        result = sanitize_filename("file<>:\"|?*.txt")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_prevent_hidden_files(self):
        """TEST-SEC03: Prevent hidden files."""
        result = sanitize_filename(".hidden")
        assert not result.startswith(".")

    def test_max_length_enforced(self):
        """TEST-SEC05: Max length is enforced."""
        long_name = "a" * 200 + ".txt"
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50

    def test_preserves_extension_on_truncation(self):
        """Test that extension is preserved when truncating."""
        long_name = "a" * 200 + ".wav"
        result = sanitize_filename(long_name, max_length=50)
        assert result.endswith(".wav")

    def test_empty_raises_error(self):
        """TEST-SEC04: Empty filename raises error."""
        with pytest.raises(ValueError):
            sanitize_filename("")

    def test_only_special_chars_raises(self):
        """Test that filename with only special chars raises."""
        with pytest.raises(ValueError):
            sanitize_filename("...")


class TestValidateTextInput:
    """Tests for validate_text_input function."""

    def test_none_raises_error(self):
        """TEST-SEC08: None input raises error."""
        with pytest.raises(ValueError) as exc_info:
            validate_text_input(None)

        assert "None" in str(exc_info.value)

    def test_empty_raises_error(self):
        """TEST-SEC09: Empty input raises error."""
        with pytest.raises(ValueError) as exc_info:
            validate_text_input("")

        assert "empty" in str(exc_info.value).lower()

    def test_too_long_raises_error(self):
        """TEST-SEC10: Too long input raises error."""
        with pytest.raises(ValueError) as exc_info:
            validate_text_input("x" * 20000)

        assert "too long" in str(exc_info.value).lower()

    def test_strips_whitespace(self):
        """TEST-SEC11: Input is stripped of whitespace."""
        result = validate_text_input("  hello  ")
        assert result == "hello"

    def test_whitespace_only_raises(self):
        """TEST-SEC38: Whitespace-only string is rejected."""
        with pytest.raises(ValueError):
            validate_text_input("   ")

    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        with pytest.raises(ValueError):
            validate_text_input("hello", max_length=3)


class TestValidateVoiceId:
    """Tests for validate_voice_id function."""

    def test_valid_format(self):
        """TEST-SEC12: Valid voice ID is accepted."""
        assert validate_voice_id("af_bella") == "af_bella"
        assert validate_voice_id("bm_george") == "bm_george"

    def test_invalid_format_raises(self):
        """TEST-SEC13: Invalid format raises error."""
        with pytest.raises(ValueError):
            validate_voice_id("invalid")

    def test_uppercase_rejected(self):
        """TEST-SEC39: Uppercase letters are rejected."""
        with pytest.raises(ValueError):
            validate_voice_id("AF_Bella")

    def test_double_underscore_rejected(self):
        """TEST-SEC40: Multiple underscores are rejected."""
        with pytest.raises(ValueError):
            validate_voice_id("af__bella")

    def test_empty_raises(self):
        """Test empty voice ID raises."""
        with pytest.raises(ValueError):
            validate_voice_id("")


class TestValidateSpeed:
    """Tests for validate_speed function."""

    def test_clamp_low(self):
        """TEST-SEC14: Speed below 0.5 is clamped."""
        assert validate_speed(0.1) == 0.5

    def test_clamp_high(self):
        """TEST-SEC15: Speed above 2.0 is clamped."""
        assert validate_speed(5.0) == 2.0

    def test_valid_speed_unchanged(self):
        """Test valid speed is unchanged."""
        assert validate_speed(1.5) == 1.5

    def test_int_converted_to_float(self):
        """Test integer is converted to float."""
        result = validate_speed(1)
        assert isinstance(result, float)
        assert result == 1.0


class TestValidateJsonMessage:
    """Tests for validate_json_message function."""

    def test_missing_fields_raises(self):
        """TEST-SEC16: Missing required fields raises error."""
        with pytest.raises(ValueError) as exc_info:
            validate_json_message(
                {"foo": "bar"},
                required_fields=["action", "text"]
            )

        assert "Missing" in str(exc_info.value)

    def test_required_field_none_accepted(self):
        """TEST-SEC41: Required field set to None is accepted (presence only)."""
        result = validate_json_message(
            {"action": None, "text": "hello"},
            required_fields=["action", "text"]
        )
        assert result["action"] is None

    def test_extra_fields_allowed(self):
        """TEST-SEC42: Extra unknown fields are allowed."""
        result = validate_json_message(
            {"action": "speak", "text": "hello", "extra": "field"},
            required_fields=["action"]
        )
        assert "extra" in result

    def test_non_dict_raises(self):
        """Test non-dict input raises."""
        with pytest.raises(ValueError):
            validate_json_message(
                ["list", "not", "dict"],
                required_fields=["action"]
            )

    def test_size_limit(self):
        """Test size limit is enforced."""
        big_data = {"data": "x" * 1000}
        with pytest.raises(ValueError) as exc_info:
            validate_json_message(big_data, required_fields=[], max_size=100)

        assert "too large" in str(exc_info.value).lower()


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_within_limit(self):
        """TEST-SEC17: RateLimiter allows within limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.is_allowed("client1") is True

    def test_blocks_over_limit(self):
        """TEST-SEC18: RateLimiter blocks over limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

    def test_get_remaining(self):
        """TEST-SEC19: get_remaining returns correct count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        assert limiter.get_remaining("client1") == 3

    def test_reset_clears_client(self):
        """TEST-SEC30: reset() clears client data."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")  # Should be blocked now

        limiter.reset("client1")

        assert limiter.is_allowed("client1") is True

    def test_clear_all_clears_all(self):
        """TEST-SEC31: clear_all() clears all clients."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client2")

        limiter.clear_all()

        assert limiter.get_remaining("client1") == 2
        assert limiter.get_remaining("client2") == 2

    def test_is_allowed_rate_limiting(self):
        """TEST-SEC43: RateLimiter.is_allowed() rate limiting."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # Use all requests
        for _ in range(3):
            limiter.is_allowed("client")

        # Should be blocked
        assert limiter.is_allowed("client") is False

    def test_concurrent_access(self):
        """TEST-SEC44: RateLimiter concurrent access."""
        limiter = RateLimiter(max_requests=100, window_seconds=60)

        def make_requests():
            for _ in range(50):
                limiter.is_allowed("concurrent_client")

        # Run requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_requests) for _ in range(5)]
            for f in futures:
                f.result()

        # Check that rate limiting kicked in after 100 requests
        remaining = limiter.get_remaining("concurrent_client")
        assert remaining == 0


class TestWebSocketSecurityManager:
    """Tests for WebSocketSecurityManager class."""

    def test_validates_localhost(self):
        """TEST-SEC23: Validates localhost origin."""
        mgr = WebSocketSecurityManager()

        assert mgr.validate_origin("http://localhost") is True
        assert mgr.validate_origin("http://127.0.0.1") is True

    def test_rejects_external(self):
        """TEST-SEC24: Rejects external origin."""
        mgr = WebSocketSecurityManager()

        assert mgr.validate_origin("http://evil.com") is False

    def test_validates_with_port(self):
        """TEST-SEC45: Origin validation with ports."""
        mgr = WebSocketSecurityManager()

        assert mgr.validate_origin("http://localhost:8080") is True
        assert mgr.validate_origin("http://127.0.0.1:3000") is True

    def test_connection_tracking(self):
        """TEST-SEC46: Connection tracking works."""
        mgr = WebSocketSecurityManager()

        assert mgr.connection_count == 0

        mgr.add_connection("client1")
        assert mgr.connection_count == 1

        mgr.add_connection("client2")
        assert mgr.connection_count == 2

        mgr.remove_connection("client1")
        assert mgr.connection_count == 1

    def test_max_connections(self):
        """Test max connections is enforced."""
        mgr = WebSocketSecurityManager(max_connections=2)

        mgr.add_connection("client1")
        mgr.add_connection("client2")

        assert mgr.can_accept_connection() is False

    def test_api_key_validation(self):
        """Test API key validation."""
        mgr = WebSocketSecurityManager(api_key="secret123")

        assert mgr.validate_api_key("secret123") is True
        assert mgr.validate_api_key("wrong") is False
        assert mgr.validate_api_key(None) is False
        assert mgr.validate_api_key("") is False

    def test_no_api_key_allows_all(self):
        """Test no API key required when not configured."""
        mgr = WebSocketSecurityManager(api_key=None)

        assert mgr.validate_api_key(None) is True
        assert mgr.validate_api_key("anything") is True

    def test_check_rate_limit(self):
        """Test rate limiting through manager."""
        mgr = WebSocketSecurityManager()

        # Should allow requests
        for _ in range(10):
            assert mgr.check_rate_limit("client1") is True


class TestSafeErrorMessage:
    """Tests for safe_error_message function."""

    def test_hides_file_paths(self):
        """TEST-SEC21: safe_error_message hides file paths."""
        err = ValueError("Error at C:/secret/path/file.txt")
        msg = safe_error_message(err)

        assert "secret" not in msg.lower()
        assert "[path]" in msg

    def test_hides_file_not_found_path(self):
        """TEST-SEC32: Hides FileNotFoundError path."""
        err = FileNotFoundError("File not found: C:/secret/file.txt")
        msg = safe_error_message(err)

        assert "secret" not in msg.lower()
        assert "not found" in msg.lower()

    def test_hides_permission_error_details(self):
        """TEST-SEC33: Hides PermissionError details."""
        err = PermissionError("Access denied to C:/secret/file.txt")
        msg = safe_error_message(err)

        assert "secret" not in msg.lower()
        assert "permission denied" in msg.lower()

    def test_hides_timeout_details(self):
        """TEST-SEC34: Hides TimeoutError details."""
        err = TimeoutError("Connection to 192.168.1.1:8080 timed out")
        msg = safe_error_message(err)

        assert "192.168" not in msg
        assert "timed out" in msg.lower()

    def test_safe_error_returns_safe_message(self):
        """TEST-SEC22: SafeError returns safe message."""
        err = SafeError(
            safe_message="Something went wrong",
            internal_message="Detailed internal error info"
        )
        msg = safe_error_message(err)

        assert msg == "Something went wrong"
        assert "internal" not in msg.lower()


class TestSecureHash:
    """Tests for secure_hash function."""

    def test_consistent_hash(self):
        """TEST-SEC20: secure_hash produces consistent SHA-256."""
        hash1 = secure_hash("test")
        hash2 = secure_hash("test")

        assert hash1 == hash2

    def test_default_length(self):
        """Test default hash length is 8."""
        result = secure_hash("test")
        assert len(result) == 8

    def test_custom_length(self):
        """Test custom hash length."""
        result = secure_hash("test", length=16)
        assert len(result) == 16

    def test_different_data_different_hash(self):
        """Test different data produces different hash."""
        hash1 = secure_hash("data1")
        hash2 = secure_hash("data2")

        assert hash1 != hash2


class TestSafeJoinPath:
    """Tests for safe_join_path function."""

    def test_valid_path(self, tmp_path):
        """TEST-SEC07: safe_join_path works for valid paths."""
        result = safe_join_path(tmp_path, "output.wav")

        assert result.parent == tmp_path
        assert result.name == "output.wav"

    def test_sanitizes_filename(self, tmp_path):
        """Test that filename is sanitized."""
        result = safe_join_path(tmp_path, "file<>:name.wav")

        # Should not contain special chars
        assert "<" not in str(result)


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    def test_valid_path_in_allowed_dir(self):
        """Test path in allowed directory is accepted."""
        config = SecurityConfig()

        # Create the allowed dir if needed
        allowed_dir = config.allowed_output_dirs[0]
        allowed_dir.mkdir(parents=True, exist_ok=True)

        test_path = allowed_dir / "test.wav"
        result = validate_output_path(test_path)

        assert result is not None

    def test_path_outside_allowed_raises(self, tmp_path):
        """Test path outside allowed directories raises."""
        with pytest.raises(ValueError) as exc_info:
            validate_output_path(
                tmp_path / "file.wav",
                allowed_dirs=[Path("C:/only/this/dir")]
            )

        assert "Path traversal" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
