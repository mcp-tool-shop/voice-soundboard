"""
Tests for the structured error response module.
"""

import json
import pytest
from unittest.mock import patch

from voice_soundboard.errors import (
    ErrorCode,
    ErrorResponse,
    SuccessResponse,
    error_response,
    success_response,
    exception_to_error,
    missing_param,
    not_found,
    invalid_value,
    ERROR_HINTS,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_are_strings(self):
        """All error codes should be string values."""
        for code in ErrorCode:
            assert isinstance(code.value, str)

    def test_error_code_uniqueness(self):
        """All error codes should be unique."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))

    def test_common_codes_exist(self):
        """Common error codes should exist."""
        assert ErrorCode.MISSING_REQUIRED
        assert ErrorCode.INVALID_INPUT
        assert ErrorCode.NOT_FOUND
        assert ErrorCode.INTERNAL_ERROR


class TestErrorResponse:
    """Tests for ErrorResponse dataclass."""

    def test_to_dict_basic(self):
        """Basic error response should have ok=false and error dict."""
        resp = ErrorResponse(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid value provided",
        )
        result = resp.to_dict()

        assert result["ok"] is False
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["message"] == "Invalid value provided"
        assert "hint" not in result["error"]

    def test_to_dict_with_hint(self):
        """Error response with hint should include hint in error dict."""
        resp = ErrorResponse(
            code=ErrorCode.MISSING_REQUIRED,
            message="Parameter 'text' is required",
            hint="Provide a non-empty text value.",
        )
        result = resp.to_dict()

        assert result["error"]["hint"] == "Provide a non-empty text value."

    def test_to_json(self):
        """JSON output should be valid JSON."""
        resp = ErrorResponse(
            code=ErrorCode.NOT_FOUND,
            message="Voice not found",
        )
        json_str = resp.to_json()

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["ok"] is False
        assert parsed["error"]["code"] == "not_found"

    def test_to_text_content(self):
        """Should return MCP TextContent list."""
        resp = ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Something went wrong",
        )
        contents = resp.to_text_content()

        assert len(contents) == 1
        assert contents[0].type == "text"

        parsed = json.loads(contents[0].text)
        assert parsed["ok"] is False


class TestSuccessResponse:
    """Tests for SuccessResponse dataclass."""

    def test_to_dict_with_data(self):
        """Success response should have ok=true and data."""
        resp = SuccessResponse(
            data={"voices": ["af_bella", "af_sky"], "count": 2},
            message="Found 2 voices",
        )
        result = resp.to_dict()

        assert result["ok"] is True
        assert result["message"] == "Found 2 voices"
        assert result["data"]["count"] == 2
        assert len(result["data"]["voices"]) == 2

    def test_to_dict_without_data(self):
        """Success response without data should still work."""
        resp = SuccessResponse(message="Operation completed")
        result = resp.to_dict()

        assert result["ok"] is True
        assert result["message"] == "Operation completed"
        assert "data" not in result

    def test_to_json(self):
        """JSON output should be valid JSON."""
        resp = SuccessResponse(data={"file": "/path/to/audio.wav"})
        json_str = resp.to_json()

        parsed = json.loads(json_str)
        assert parsed["ok"] is True
        assert parsed["data"]["file"] == "/path/to/audio.wav"


class TestErrorResponseFunction:
    """Tests for error_response() helper function."""

    def test_returns_text_content(self):
        """Should return MCP TextContent list."""
        result = error_response(
            code=ErrorCode.INVALID_INPUT,
            message="Bad input",
        )

        assert len(result) == 1
        assert result[0].type == "text"

    def test_logs_internal_details(self):
        """Should log internal details server-side only."""
        with patch("voice_soundboard.errors.logger") as mock_logger:
            error_response(
                code=ErrorCode.INTERNAL_ERROR,
                message="An error occurred",
                internal_details="Stack trace: ...",
            )
            mock_logger.error.assert_called_once()

    def test_uses_default_hint(self):
        """Should use default hint from ERROR_HINTS if none provided."""
        result = error_response(
            code=ErrorCode.MISSING_REQUIRED,
            message="Parameter 'text' is required",
        )

        parsed = json.loads(result[0].text)
        assert "hint" in parsed["error"]
        assert parsed["error"]["hint"] == ERROR_HINTS[ErrorCode.MISSING_REQUIRED]

    def test_custom_hint_overrides_default(self):
        """Custom hint should override default hint."""
        result = error_response(
            code=ErrorCode.MISSING_REQUIRED,
            message="Parameter 'text' is required",
            hint="Custom hint here",
        )

        parsed = json.loads(result[0].text)
        assert parsed["error"]["hint"] == "Custom hint here"


class TestSuccessResponseFunction:
    """Tests for success_response() helper function."""

    def test_returns_text_content(self):
        """Should return MCP TextContent list."""
        result = success_response(
            data={"result": "ok"},
            message="Done",
        )

        assert len(result) == 1
        assert result[0].type == "text"

        parsed = json.loads(result[0].text)
        assert parsed["ok"] is True


class TestExceptionToError:
    """Tests for exception_to_error() function."""

    def test_valueerror_sanitizes_paths(self):
        """ValueError should sanitize Windows paths."""
        exc = ValueError("Invalid file at C:\\Users\\test\\secret.txt")
        result = exception_to_error(exc)

        parsed = json.loads(result[0].text)
        assert "C:\\" not in parsed["error"]["message"]
        assert "[path]" in parsed["error"]["message"]

    def test_filenotfounderror(self):
        """FileNotFoundError should return file_not_found code."""
        exc = FileNotFoundError("audio.wav")
        result = exception_to_error(exc)

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "file_not_found"
        assert "not found" in parsed["error"]["message"].lower()

    def test_permissionerror(self):
        """PermissionError should return access_denied code."""
        exc = PermissionError("Access denied")
        result = exception_to_error(exc)

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "access_denied"

    def test_importerror(self):
        """ImportError should return dependency_missing code."""
        exc = ImportError("No module named 'chatterbox'")
        result = exception_to_error(exc)

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "dependency_missing"

    def test_with_context(self):
        """Context should be prepended to message."""
        exc = ValueError("Invalid speed")
        result = exception_to_error(exc, context="Speech generation")

        parsed = json.loads(result[0].text)
        assert parsed["error"]["message"].startswith("Speech generation:")

    def test_logs_full_exception(self):
        """Should log full exception server-side."""
        with patch("voice_soundboard.errors.logger") as mock_logger:
            exc = RuntimeError("Internal failure")
            exception_to_error(exc)
            mock_logger.exception.assert_called()


class TestHelperFunctions:
    """Tests for shorthand helper functions."""

    def test_missing_param(self):
        """missing_param() should create proper error."""
        result = missing_param("text")

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "missing_required"
        assert "'text'" in parsed["error"]["message"]

    def test_not_found_voice(self):
        """not_found() should use correct code for voices."""
        result = not_found("voice", "unknown_voice")

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "voice_not_found"
        assert "unknown_voice" in parsed["error"]["message"]

    def test_not_found_effect(self):
        """not_found() should use correct code for effects."""
        result = not_found("effect", "unknown_effect")

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "effect_not_found"

    def test_not_found_generic(self):
        """not_found() should use generic code for unknown types."""
        result = not_found("widget", "my_widget")

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "not_found"

    def test_invalid_value(self):
        """invalid_value() should create proper error."""
        result = invalid_value("speed", -5.0, "must be positive")

        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "invalid_input"
        assert "'speed'" in parsed["error"]["message"]
        assert "must be positive" in parsed["error"]["message"]
        assert "-5.0" in parsed["error"]["hint"]
