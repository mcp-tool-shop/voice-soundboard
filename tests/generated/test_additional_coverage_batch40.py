"""
Additional coverage tests - Batch 40: Error Handling & Security Coverage.

Comprehensive tests for:
- voice_soundboard/errors.py
- voice_soundboard/security.py
- Edge cases and error handling throughout the codebase
"""

import pytest
import json
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Error Response Tests
# =============================================================================

class TestErrorResponse:
    """Tests for ErrorResponse class."""

    def test_error_response_creation(self):
        """Test creating an ErrorResponse."""
        from voice_soundboard.errors import ErrorResponse, ErrorCode

        response = ErrorResponse(
            code=ErrorCode.INVALID_INPUT,
            message="Test error message",
        )
        assert response.code == ErrorCode.INVALID_INPUT
        assert response.message == "Test error message"

    def test_error_response_to_dict(self):
        """Test ErrorResponse to_dict method."""
        from voice_soundboard.errors import ErrorResponse, ErrorCode

        response = ErrorResponse(
            code=ErrorCode.NOT_FOUND,
            message="Resource not found",
            hint="Check the ID",
        )
        data = response.to_dict()

        assert data["ok"] is False
        assert data["error"]["code"] == "not_found"
        assert data["error"]["message"] == "Resource not found"
        assert data["error"]["hint"] == "Check the ID"

    def test_error_response_to_json(self):
        """Test ErrorResponse to_json method."""
        from voice_soundboard.errors import ErrorResponse, ErrorCode

        response = ErrorResponse(
            code=ErrorCode.OPERATION_FAILED,
            message="Operation failed",
        )
        json_str = response.to_json()

        data = json.loads(json_str)
        assert data["ok"] is False

    def test_error_response_to_text_content(self):
        """Test ErrorResponse to_text_content method."""
        from voice_soundboard.errors import ErrorResponse, ErrorCode

        response = ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Internal error",
        )
        content = response.to_text_content()

        assert len(content) == 1
        assert content[0].type == "text"


# =============================================================================
# Success Response Tests
# =============================================================================

class TestSuccessResponse:
    """Tests for SuccessResponse class."""

    def test_success_response_creation(self):
        """Test creating a SuccessResponse."""
        from voice_soundboard.errors import SuccessResponse

        response = SuccessResponse(
            data={"result": "value"},
            message="Operation completed",
        )
        assert response.data == {"result": "value"}
        assert response.message == "Operation completed"

    def test_success_response_to_dict(self):
        """Test SuccessResponse to_dict method."""
        from voice_soundboard.errors import SuccessResponse

        response = SuccessResponse(
            data={"voices": ["voice1", "voice2"]},
        )
        data = response.to_dict()

        assert data["ok"] is True
        assert data["data"]["voices"] == ["voice1", "voice2"]

    def test_success_response_to_json(self):
        """Test SuccessResponse to_json method."""
        from voice_soundboard.errors import SuccessResponse

        response = SuccessResponse(data={"test": True})
        json_str = response.to_json()

        data = json.loads(json_str)
        assert data["ok"] is True

    def test_success_response_no_data(self):
        """Test SuccessResponse with no data."""
        from voice_soundboard.errors import SuccessResponse

        response = SuccessResponse(message="Done")
        data = response.to_dict()

        assert data["ok"] is True
        assert "data" not in data
        assert data["message"] == "Done"


# =============================================================================
# Error Helper Functions Tests
# =============================================================================

class TestErrorHelperFunctions:
    """Tests for error helper functions."""

    def test_error_response_function(self):
        """Test error_response function."""
        from voice_soundboard.errors import error_response, ErrorCode

        result = error_response(
            code=ErrorCode.MISSING_REQUIRED,
            message="Missing parameter",
        )
        assert len(result) == 1

    def test_error_response_with_hint(self):
        """Test error_response with custom hint."""
        from voice_soundboard.errors import error_response, ErrorCode

        result = error_response(
            code=ErrorCode.INVALID_FORMAT,
            message="Invalid format",
            hint="Use correct format",
        )
        content = json.loads(result[0].text)
        assert content["error"]["hint"] == "Use correct format"

    def test_success_response_function(self):
        """Test success_response function."""
        from voice_soundboard.errors import success_response

        result = success_response(data={"success": True})
        content = json.loads(result[0].text)
        assert content["ok"] is True

    def test_missing_param_helper(self):
        """Test missing_param helper function."""
        from voice_soundboard.errors import missing_param

        result = missing_param("voice_id")
        content = json.loads(result[0].text)
        assert "voice_id" in content["error"]["message"]

    def test_not_found_helper(self):
        """Test not_found helper function."""
        from voice_soundboard.errors import not_found

        result = not_found("voice", "my_voice")
        content = json.loads(result[0].text)
        assert "my_voice" in content["error"]["message"]

    def test_not_found_different_types(self):
        """Test not_found with different resource types."""
        from voice_soundboard.errors import not_found, ErrorCode

        for resource_type in ["voice", "effect", "preset", "file", "session"]:
            result = not_found(resource_type, f"test_{resource_type}")
            content = json.loads(result[0].text)
            assert content["ok"] is False

    def test_invalid_value_helper(self):
        """Test invalid_value helper function."""
        from voice_soundboard.errors import invalid_value

        result = invalid_value("speed", 5.0, "must be between 0.5 and 2.0")
        content = json.loads(result[0].text)
        assert "speed" in content["error"]["message"]


# =============================================================================
# Exception to Error Tests
# =============================================================================

class TestExceptionToError:
    """Tests for exception_to_error function."""

    def test_exception_value_error(self):
        """Test converting ValueError."""
        from voice_soundboard.errors import exception_to_error

        exc = ValueError("Invalid value")
        result = exception_to_error(exc)
        content = json.loads(result[0].text)
        assert content["error"]["code"] == "invalid_input"

    def test_exception_file_not_found(self):
        """Test converting FileNotFoundError."""
        from voice_soundboard.errors import exception_to_error

        exc = FileNotFoundError("File not found")
        result = exception_to_error(exc)
        content = json.loads(result[0].text)
        assert content["error"]["code"] == "file_not_found"

    def test_exception_permission_error(self):
        """Test converting PermissionError."""
        from voice_soundboard.errors import exception_to_error

        exc = PermissionError("Access denied")
        result = exception_to_error(exc)
        content = json.loads(result[0].text)
        assert content["error"]["code"] == "access_denied"

    def test_exception_timeout_error(self):
        """Test converting TimeoutError."""
        from voice_soundboard.errors import exception_to_error

        exc = TimeoutError("Timeout")
        result = exception_to_error(exc)
        content = json.loads(result[0].text)
        assert content["error"]["code"] == "operation_failed"

    def test_exception_import_error(self):
        """Test converting ImportError."""
        from voice_soundboard.errors import exception_to_error

        exc = ImportError("Module not found")
        result = exception_to_error(exc)
        content = json.loads(result[0].text)
        assert content["error"]["code"] == "dependency_missing"

    def test_exception_with_context(self):
        """Test exception conversion with context."""
        from voice_soundboard.errors import exception_to_error

        exc = ValueError("Bad value")
        result = exception_to_error(exc, context="During synthesis")
        content = json.loads(result[0].text)
        assert "During synthesis" in content["error"]["message"]

    def test_exception_path_sanitization(self):
        """Test that paths are sanitized in error messages."""
        from voice_soundboard.errors import exception_to_error

        exc = ValueError("File not found: C:/Users/test/secret/file.txt")
        result = exception_to_error(exc)
        content = json.loads(result[0].text)
        # Path should be sanitized
        assert "C:/Users" not in content["error"]["message"]


# =============================================================================
# Security - Filename Sanitization Tests
# =============================================================================

class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_simple_filename(self):
        """Test sanitizing a simple filename."""
        from voice_soundboard.security import sanitize_filename

        result = sanitize_filename("test.wav")
        assert result == "test.wav"

    def test_sanitize_path_traversal(self):
        """Test sanitizing path traversal attempts."""
        from voice_soundboard.security import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_sanitize_special_characters(self):
        """Test sanitizing special characters."""
        from voice_soundboard.security import sanitize_filename

        result = sanitize_filename("test<>:*?|file.wav")
        # Special characters should be replaced
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_hidden_file(self):
        """Test that hidden files are not allowed."""
        from voice_soundboard.security import sanitize_filename

        result = sanitize_filename(".hidden")
        assert not result.startswith(".")

    def test_sanitize_null_bytes(self):
        """Test removing null bytes."""
        from voice_soundboard.security import sanitize_filename

        result = sanitize_filename("test\x00.wav")
        assert "\x00" not in result

    def test_sanitize_max_length(self):
        """Test filename length limit."""
        from voice_soundboard.security import sanitize_filename

        long_name = "a" * 200 + ".wav"
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50

    def test_sanitize_empty_filename(self):
        """Test empty filename raises error."""
        from voice_soundboard.security import sanitize_filename

        with pytest.raises(ValueError):
            sanitize_filename("")

    def test_sanitize_invalid_only(self):
        """Test filename with only invalid characters."""
        from voice_soundboard.security import sanitize_filename

        with pytest.raises(ValueError):
            sanitize_filename("***")


# =============================================================================
# Security - Path Validation Tests
# =============================================================================

class TestPathValidation:
    """Tests for path validation functions."""

    def test_safe_join_path(self, tmp_path):
        """Test safe path joining."""
        from voice_soundboard.security import safe_join_path

        result = safe_join_path(tmp_path, "test.wav")
        assert result.parent == tmp_path

    def test_safe_join_path_traversal(self, tmp_path):
        """Test that path traversal is blocked."""
        from voice_soundboard.security import safe_join_path

        with pytest.raises(ValueError):
            safe_join_path(tmp_path, "../../../etc/passwd")


# =============================================================================
# Security - Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_text_input_valid(self):
        """Test validating valid text."""
        from voice_soundboard.security import validate_text_input

        result = validate_text_input("Hello world")
        assert result == "Hello world"

    def test_validate_text_input_strips(self):
        """Test that text is stripped."""
        from voice_soundboard.security import validate_text_input

        result = validate_text_input("  Hello  ")
        assert result == "Hello"

    def test_validate_text_input_empty(self):
        """Test that empty text raises error."""
        from voice_soundboard.security import validate_text_input

        with pytest.raises(ValueError):
            validate_text_input("")

    def test_validate_text_input_too_long(self):
        """Test that overly long text raises error."""
        from voice_soundboard.security import validate_text_input

        with pytest.raises(ValueError):
            validate_text_input("a" * 20000, max_length=10000)

    def test_validate_text_input_none(self):
        """Test that None raises error."""
        from voice_soundboard.security import validate_text_input

        with pytest.raises(ValueError):
            validate_text_input(None)

    def test_validate_voice_id_valid(self):
        """Test validating valid voice ID."""
        from voice_soundboard.security import validate_voice_id

        result = validate_voice_id("af_bella")
        assert result == "af_bella"

    def test_validate_voice_id_invalid_format(self):
        """Test invalid voice ID format."""
        from voice_soundboard.security import validate_voice_id

        with pytest.raises(ValueError):
            validate_voice_id("invalid")

    def test_validate_speed_valid(self):
        """Test validating valid speed."""
        from voice_soundboard.security import validate_speed

        result = validate_speed(1.5)
        assert result == 1.5

    def test_validate_speed_clamped(self):
        """Test speed clamping."""
        from voice_soundboard.security import validate_speed

        assert validate_speed(0.1) == 0.5  # Min clamp
        assert validate_speed(5.0) == 2.0  # Max clamp

    def test_validate_json_message(self):
        """Test validating JSON message."""
        from voice_soundboard.security import validate_json_message

        result = validate_json_message(
            {"action": "speak", "text": "Hello"},
            required_fields=["action"]
        )
        assert result["action"] == "speak"

    def test_validate_json_message_missing_field(self):
        """Test JSON message with missing required field."""
        from voice_soundboard.security import validate_json_message

        with pytest.raises(ValueError):
            validate_json_message(
                {"text": "Hello"},
                required_fields=["action"]
            )


# =============================================================================
# Security - Rate Limiting Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_allows_requests(self):
        """Test that requests within limit are allowed."""
        from voice_soundboard.security import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)
        client_id = "test_client"

        for _ in range(5):
            assert limiter.is_allowed(client_id) is True

    def test_rate_limiter_blocks_excess(self):
        """Test that excess requests are blocked."""
        from voice_soundboard.security import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)
        client_id = "test_client"

        for _ in range(3):
            limiter.is_allowed(client_id)

        # 4th request should be blocked
        assert limiter.is_allowed(client_id) is False

    def test_rate_limiter_get_remaining(self):
        """Test getting remaining requests."""
        from voice_soundboard.security import RateLimiter

        limiter = RateLimiter(max_requests=10, window_seconds=60)
        client_id = "test_client"

        assert limiter.get_remaining(client_id) == 10

        limiter.is_allowed(client_id)
        assert limiter.get_remaining(client_id) == 9

    def test_rate_limiter_reset(self):
        """Test resetting rate limit for client."""
        from voice_soundboard.security import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)
        client_id = "test_client"

        for _ in range(3):
            limiter.is_allowed(client_id)

        limiter.reset(client_id)
        assert limiter.is_allowed(client_id) is True

    def test_rate_limiter_multiple_clients(self):
        """Test rate limiter with multiple clients."""
        from voice_soundboard.security import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Each client has independent limits
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client2") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client2") is True

    def test_rate_limiter_clear_all(self):
        """Test clearing all rate limit data."""
        from voice_soundboard.security import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")

        limiter.clear_all()

        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client2") is True


# =============================================================================
# Security - SafeError Tests
# =============================================================================

class TestSafeError:
    """Tests for SafeError exception class."""

    def test_safe_error_creation(self):
        """Test creating a SafeError."""
        from voice_soundboard.security import SafeError

        error = SafeError(
            safe_message="Something went wrong",
            internal_message="Detailed error at line 42"
        )
        assert error.safe_message == "Something went wrong"
        assert error.internal_message == "Detailed error at line 42"

    def test_safe_error_str(self):
        """Test SafeError string representation."""
        from voice_soundboard.security import SafeError

        error = SafeError("Safe message")
        assert str(error) == "Safe message"

    def test_safe_error_message_function(self):
        """Test safe_error_message function."""
        from voice_soundboard.security import safe_error_message, SafeError

        error = SafeError("User-safe message")
        result = safe_error_message(error)
        assert result == "User-safe message"

    def test_safe_error_message_value_error(self):
        """Test safe_error_message with ValueError."""
        from voice_soundboard.security import safe_error_message

        error = ValueError("Invalid parameter")
        result = safe_error_message(error)
        assert "Invalid parameter" in result

    def test_safe_error_message_path_sanitization(self):
        """Test that paths are sanitized."""
        from voice_soundboard.security import safe_error_message

        error = ValueError("File not found: C:/secret/path/file.txt")
        result = safe_error_message(error)
        assert "C:/secret" not in result


# =============================================================================
# Security - WebSocket Security Manager Tests
# =============================================================================

class TestWebSocketSecurityManager:
    """Tests for WebSocketSecurityManager class."""

    def test_security_manager_creation(self):
        """Test creating a WebSocketSecurityManager."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()
        assert manager is not None

    def test_validate_origin_localhost(self):
        """Test validating localhost origin."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()
        assert manager.validate_origin("http://localhost") is True
        assert manager.validate_origin("http://127.0.0.1") is True

    def test_validate_origin_with_port(self):
        """Test validating localhost with port."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()
        assert manager.validate_origin("http://localhost:8080") is True

    def test_validate_origin_blocked(self):
        """Test blocking unknown origins."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()
        assert manager.validate_origin("http://evil.com") is False

    def test_validate_origin_none(self):
        """Test that no origin is allowed (non-browser)."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()
        assert manager.validate_origin(None) is True

    def test_validate_api_key_correct(self):
        """Test validating correct API key."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager(api_key="secret123")
        assert manager.validate_api_key("secret123") is True

    def test_validate_api_key_incorrect(self):
        """Test rejecting incorrect API key."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager(api_key="secret123")
        assert manager.validate_api_key("wrong") is False

    def test_validate_api_key_no_key_required(self):
        """Test when no API key is required."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager(api_key=None)
        assert manager.validate_api_key(None) is True

    def test_connection_management(self):
        """Test connection tracking."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager(max_connections=2)

        assert manager.can_accept_connection() is True
        manager.add_connection("client1")
        assert manager.connection_count == 1

        manager.add_connection("client2")
        assert manager.can_accept_connection() is False

        manager.remove_connection("client1")
        assert manager.can_accept_connection() is True

    def test_check_rate_limit(self):
        """Test rate limit checking."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()
        # Should allow requests
        assert manager.check_rate_limit("client1") is True


# =============================================================================
# Security - Hash Function Tests
# =============================================================================

class TestSecureHash:
    """Tests for secure_hash function."""

    def test_secure_hash(self):
        """Test secure hash generation."""
        from voice_soundboard.security import secure_hash

        result = secure_hash("test data")
        assert len(result) == 8
        assert isinstance(result, str)

    def test_secure_hash_custom_length(self):
        """Test secure hash with custom length."""
        from voice_soundboard.security import secure_hash

        result = secure_hash("test", length=16)
        assert len(result) == 16

    def test_secure_hash_consistency(self):
        """Test that same input produces same hash."""
        from voice_soundboard.security import secure_hash

        hash1 = secure_hash("same input")
        hash2 = secure_hash("same input")
        assert hash1 == hash2

    def test_secure_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        from voice_soundboard.security import secure_hash

        hash1 = secure_hash("input1")
        hash2 = secure_hash("input2")
        assert hash1 != hash2


# =============================================================================
# Edge Cases - Empty/None Inputs
# =============================================================================

class TestEdgeCasesEmptyInputs:
    """Tests for empty/None input handling."""

    def test_normalize_text_empty(self):
        """Test normalizing empty text."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("")
        assert result == ""

    def test_normalize_text_whitespace(self):
        """Test normalizing whitespace-only text."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("   \n\t   ")
        assert result == ""


# =============================================================================
# Edge Cases - Very Long Inputs
# =============================================================================

class TestEdgeCasesLongInputs:
    """Tests for very long input handling."""

    def test_number_to_words_large(self):
        """Test converting very large numbers."""
        from voice_soundboard.normalizer import number_to_words

        result = number_to_words(999999999999)
        assert "billion" in result

    def test_text_input_validation_long(self):
        """Test validation of very long text."""
        from voice_soundboard.security import validate_text_input

        # Should raise for extremely long input
        with pytest.raises(ValueError):
            validate_text_input("a" * 100000, max_length=50000)


# =============================================================================
# Edge Cases - Special Characters
# =============================================================================

class TestEdgeCasesSpecialChars:
    """Tests for special character handling."""

    def test_normalize_unicode(self):
        """Test normalizing Unicode text."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Hello \u2014 World")  # em-dash
        assert isinstance(result, str)

    def test_sanitize_unicode_filename(self):
        """Test sanitizing Unicode in filenames."""
        from voice_soundboard.security import sanitize_filename

        result = sanitize_filename("file\u2019s.wav")  # smart quote
        assert isinstance(result, str)
