"""
Structured error responses for MCP/server APIs.

Provides consistent error formatting that:
- Hides internal details (paths, stack traces) from clients
- Includes machine-readable error codes
- Logs full details server-side
- Provides actionable hints where possible
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from mcp.types import TextContent

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Machine-readable error codes for API responses."""

    # Validation errors (4xx-style)
    MISSING_REQUIRED = "missing_required"
    INVALID_INPUT = "invalid_input"
    INVALID_FORMAT = "invalid_format"
    VALUE_OUT_OF_RANGE = "value_out_of_range"

    # Resource errors
    NOT_FOUND = "not_found"
    FILE_NOT_FOUND = "file_not_found"
    VOICE_NOT_FOUND = "voice_not_found"
    EFFECT_NOT_FOUND = "effect_not_found"
    PRESET_NOT_FOUND = "preset_not_found"

    # Permission/security errors
    ACCESS_DENIED = "access_denied"
    PATH_TRAVERSAL = "path_traversal"
    RATE_LIMITED = "rate_limited"

    # Dependency/availability errors
    DEPENDENCY_MISSING = "dependency_missing"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    FEATURE_UNAVAILABLE = "feature_unavailable"

    # Operation errors
    OPERATION_FAILED = "operation_failed"
    PLAYBACK_FAILED = "playback_failed"
    SYNTHESIS_FAILED = "synthesis_failed"
    CONVERSION_FAILED = "conversion_failed"
    CLONING_FAILED = "cloning_failed"

    # State errors
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_ALREADY_EXISTS = "session_already_exists"
    INVALID_STATE = "invalid_state"

    # Unknown/internal
    INTERNAL_ERROR = "internal_error"
    UNKNOWN_TOOL = "unknown_tool"


# Hints for common errors to help users resolve issues
ERROR_HINTS: dict[ErrorCode, str] = {
    ErrorCode.MISSING_REQUIRED: "Check the tool parameters documentation for required fields.",
    ErrorCode.DEPENDENCY_MISSING: "Install the optional dependency: pip install voice-soundboard[<extra>]",
    ErrorCode.ENGINE_UNAVAILABLE: "The voice engine may need to be initialized first.",
    ErrorCode.VOICE_NOT_FOUND: "Use list_voices to see available voice IDs.",
    ErrorCode.EFFECT_NOT_FOUND: "Use list_effects to see available sound effects.",
    ErrorCode.PRESET_NOT_FOUND: "Use list_presets or list_voice_presets to see available presets.",
    ErrorCode.PATH_TRAVERSAL: "Use only relative filenames without path separators.",
    ErrorCode.ACCESS_DENIED: "Ensure the file path is within allowed directories.",
    ErrorCode.RATE_LIMITED: "Wait before making more requests.",
}


@dataclass
class ErrorResponse:
    """Structured error response for MCP tools."""

    code: ErrorCode
    message: str
    hint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {
            "ok": False,
            "error": {
                "code": self.code.value,
                "message": self.message,
            }
        }
        if self.hint:
            result["error"]["hint"] = self.hint
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_text_content(self) -> list[TextContent]:
        """Convert to MCP TextContent list."""
        return [TextContent(type="text", text=self.to_json())]


@dataclass
class SuccessResponse:
    """Structured success response for MCP tools."""

    data: Any = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"ok": True}
        if self.message:
            result["message"] = self.message
        if self.data is not None:
            result["data"] = self.data
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_text_content(self) -> list[TextContent]:
        """Convert to MCP TextContent list."""
        return [TextContent(type="text", text=self.to_json())]


def error_response(
    code: ErrorCode,
    message: str,
    hint: Optional[str] = None,
    internal_details: Optional[str] = None,
) -> list[TextContent]:
    """
    Create a structured error response.

    Args:
        code: Error code for machine-readable classification
        message: User-safe error message
        hint: Optional actionable hint for resolution
        internal_details: Optional details to log server-side only

    Returns:
        MCP TextContent list with structured JSON error
    """
    # Log internal details if provided
    if internal_details:
        logger.error("Error [%s]: %s | Internal: %s", code.value, message, internal_details)
    else:
        logger.warning("Error [%s]: %s", code.value, message)

    # Use default hint if none provided
    if hint is None:
        hint = ERROR_HINTS.get(code)

    return ErrorResponse(code=code, message=message, hint=hint).to_text_content()


def success_response(
    data: Any = None,
    message: Optional[str] = None,
) -> list[TextContent]:
    """
    Create a structured success response.

    Args:
        data: Response data (will be JSON serialized)
        message: Optional success message

    Returns:
        MCP TextContent list with structured JSON response
    """
    return SuccessResponse(data=data, message=message).to_text_content()


def exception_to_error(
    exc: Exception,
    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
    context: Optional[str] = None,
) -> list[TextContent]:
    """
    Convert an exception to a structured error response.

    Hides internal details (paths, stack traces) from the response
    while logging them server-side.

    Args:
        exc: The exception to convert
        default_code: Error code if exception type isn't recognized
        context: Optional context about what operation failed

    Returns:
        MCP TextContent list with structured JSON error
    """
    import re
    from voice_soundboard.security import SafeError

    # Map exception types to error codes
    exc_type = type(exc)
    code = default_code
    message = "An error occurred"
    hint = None

    if isinstance(exc, SafeError):
        # SafeError already has a safe message
        code = ErrorCode.OPERATION_FAILED
        message = exc.safe_message

    elif isinstance(exc, ValueError):
        code = ErrorCode.INVALID_INPUT
        # ValueError messages are usually safe, but sanitize paths
        message = re.sub(r'[A-Za-z]:[/\\][^\s]+', '[path]', str(exc))

    elif isinstance(exc, FileNotFoundError):
        code = ErrorCode.FILE_NOT_FOUND
        message = "Requested file not found"
        hint = "Check that the file path is correct and the file exists."

    elif isinstance(exc, PermissionError):
        code = ErrorCode.ACCESS_DENIED
        message = "Permission denied"

    elif isinstance(exc, TimeoutError):
        code = ErrorCode.OPERATION_FAILED
        message = "Operation timed out"
        hint = "Try again or reduce input size."

    elif isinstance(exc, ImportError):
        code = ErrorCode.DEPENDENCY_MISSING
        message = "Required component not installed"
        # Extract package name if present
        exc_str = str(exc)
        if "voice-soundboard[" in exc_str:
            hint = exc_str
        else:
            hint = ERROR_HINTS.get(code)

    elif isinstance(exc, RuntimeError):
        code = ErrorCode.ENGINE_UNAVAILABLE
        message = "Engine initialization failed"

    elif isinstance(exc, KeyError):
        code = ErrorCode.NOT_FOUND
        message = f"Key not found: {exc}"

    # Add context to message if provided
    if context:
        message = f"{context}: {message}"

    # Log full exception details server-side
    logger.exception("Exception during operation: %s", exc)

    return error_response(
        code=code,
        message=message,
        hint=hint,
        internal_details=str(exc),
    )


def missing_param(param_name: str) -> list[TextContent]:
    """Shorthand for missing required parameter error."""
    return error_response(
        code=ErrorCode.MISSING_REQUIRED,
        message=f"Required parameter '{param_name}' is missing",
    )


def not_found(resource_type: str, resource_id: str) -> list[TextContent]:
    """Shorthand for resource not found error."""
    code_map = {
        "voice": ErrorCode.VOICE_NOT_FOUND,
        "effect": ErrorCode.EFFECT_NOT_FOUND,
        "preset": ErrorCode.PRESET_NOT_FOUND,
        "file": ErrorCode.FILE_NOT_FOUND,
        "session": ErrorCode.SESSION_NOT_FOUND,
    }
    code = code_map.get(resource_type, ErrorCode.NOT_FOUND)

    return error_response(
        code=code,
        message=f"{resource_type.title()} '{resource_id}' not found",
    )


def invalid_value(param_name: str, value: Any, reason: str) -> list[TextContent]:
    """Shorthand for invalid parameter value error."""
    return error_response(
        code=ErrorCode.INVALID_INPUT,
        message=f"Invalid value for '{param_name}': {reason}",
        hint=f"Provided value: {value!r}",
    )
