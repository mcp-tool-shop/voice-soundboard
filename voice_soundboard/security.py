"""
Security utilities for Voice Soundboard.

Provides:
- Path sanitization (prevent traversal attacks)
- Input validation (length limits, schema validation)
- Rate limiting
- Safe error handling
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("voice-security")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Input limits
    max_text_length: int = 10000  # ~5 minutes of audio
    max_ssml_length: int = 15000  # SSML can be longer due to tags
    max_filename_length: int = 100
    max_message_size: int = 65536  # 64KB for WebSocket messages

    # Rate limiting
    rate_limit_requests: int = 60  # requests per window
    rate_limit_window: int = 60  # seconds

    # Connection limits
    max_connections: int = 100
    connection_timeout: int = 30  # seconds
    idle_timeout: int = 300  # 5 minutes

    # Allowed paths (resolved at runtime)
    allowed_output_dirs: list[Path] = None

    def __post_init__(self):
        if self.allowed_output_dirs is None:
            self.allowed_output_dirs = [
                Path("F:/AI/voice-soundboard/output").resolve(),
                Path("F:/AI/voice-soundboard/cache").resolve(),
            ]


# Global config instance
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get or create security config singleton."""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


# =============================================================================
# Path Sanitization
# =============================================================================

def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Args:
        filename: The filename to sanitize
        max_length: Maximum allowed length

    Returns:
        Safe filename with only alphanumeric, dots, dashes, underscores

    Raises:
        ValueError: If filename is empty or invalid after sanitization
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Remove any path components (prevent traversal)
    filename = os.path.basename(filename)

    # Remove null bytes and other dangerous characters
    filename = filename.replace('\x00', '')

    # Only allow safe characters: alphanumeric, dots, dashes, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

    # Prevent hidden files
    filename = filename.lstrip('.')

    # Prevent empty result
    if not filename:
        raise ValueError("Filename contains no valid characters")

    # Enforce length limit
    if len(filename) > max_length:
        # Keep extension if present
        name, ext = os.path.splitext(filename)
        max_name_len = max_length - len(ext)
        if max_name_len > 0:
            filename = name[:max_name_len] + ext
        else:
            filename = filename[:max_length]

    return filename


def validate_output_path(
    output_path: Path,
    allowed_dirs: Optional[list[Path]] = None
) -> Path:
    """
    Validate that an output path is within allowed directories.

    Args:
        output_path: The path to validate
        allowed_dirs: List of allowed parent directories

    Returns:
        Resolved, validated path

    Raises:
        ValueError: If path is outside allowed directories
    """
    config = get_security_config()
    allowed_dirs = allowed_dirs or config.allowed_output_dirs

    # Resolve to absolute path
    resolved = output_path.resolve()

    # Check if path is within any allowed directory
    for allowed_dir in allowed_dirs:
        allowed_resolved = allowed_dir.resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue

    raise ValueError(
        f"Path traversal detected: {output_path} is outside allowed directories"
    )


def safe_join_path(base_dir: Path, filename: str) -> Path:
    """
    Safely join a base directory with a filename.

    Args:
        base_dir: The base directory
        filename: The filename to join

    Returns:
        Safe, validated path

    Raises:
        ValueError: If resulting path is outside base_dir
    """
    # Sanitize the filename first
    safe_filename = sanitize_filename(filename)

    # Join and resolve
    result = (base_dir / safe_filename).resolve()

    # Verify it's still within base_dir
    base_resolved = base_dir.resolve()
    try:
        result.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {filename} escapes {base_dir}"
        )

    return result


# =============================================================================
# Input Validation
# =============================================================================

def validate_text_input(
    text: str,
    max_length: Optional[int] = None,
    field_name: str = "text"
) -> str:
    """
    Validate text input.

    Args:
        text: The text to validate
        max_length: Maximum allowed length (uses config default if None)
        field_name: Name of field for error messages

    Returns:
        Validated text (stripped)

    Raises:
        ValueError: If text is invalid
    """
    config = get_security_config()
    max_length = max_length or config.max_text_length

    if text is None:
        raise ValueError(f"{field_name} cannot be None")

    if not isinstance(text, str):
        raise ValueError(f"{field_name} must be a string")

    text = text.strip()

    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    if len(text) > max_length:
        raise ValueError(
            f"{field_name} too long: {len(text)} > {max_length} characters"
        )

    return text


def validate_voice_id(voice: str) -> str:
    """
    Validate a voice ID.

    Args:
        voice: The voice ID to validate

    Returns:
        Validated voice ID

    Raises:
        ValueError: If voice ID is invalid format
    """
    if not voice:
        raise ValueError("Voice ID cannot be empty")

    # Voice IDs should match pattern: xx_name (e.g., af_bella, bm_george)
    if not re.match(r'^[a-z]{2}_[a-z]+$', voice):
        raise ValueError(
            f"Invalid voice ID format: {voice}. "
            "Expected format: xx_name (e.g., af_bella)"
        )

    return voice


def validate_speed(speed: float) -> float:
    """
    Validate and clamp speed parameter.

    Args:
        speed: The speed value

    Returns:
        Clamped speed between 0.5 and 2.0
    """
    if not isinstance(speed, (int, float)):
        raise ValueError("Speed must be a number")

    return max(0.5, min(2.0, float(speed)))


def validate_json_message(
    data: dict,
    required_fields: list[str],
    max_size: Optional[int] = None
) -> dict:
    """
    Validate a JSON message.

    Args:
        data: The parsed JSON data
        required_fields: List of required field names
        max_size: Maximum serialized size (optional)

    Returns:
        Validated data

    Raises:
        ValueError: If validation fails
    """
    import json

    if not isinstance(data, dict):
        raise ValueError("Message must be a JSON object")

    # Check required fields
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    # Check size if specified
    if max_size:
        size = len(json.dumps(data))
        if size > max_size:
            raise ValueError(f"Message too large: {size} > {max_size} bytes")

    return data


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.

    Tracks requests per client and enforces limits.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60
    ):
        """
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock_times: dict[str, float] = {}

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request is allowed for a client.

        Args:
            client_id: Unique client identifier

        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()

        # Check if client is in lockout
        if client_id in self._lock_times:
            if now < self._lock_times[client_id]:
                return False
            else:
                del self._lock_times[client_id]

        # Remove old requests outside window
        cutoff = now - self.window_seconds
        self._requests[client_id] = [
            t for t in self._requests[client_id]
            if t > cutoff
        ]

        # Check if under limit
        if len(self._requests[client_id]) >= self.max_requests:
            # Add lockout for repeated violations
            self._lock_times[client_id] = now + self.window_seconds
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False

        # Record this request
        self._requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        now = time.time()
        cutoff = now - self.window_seconds
        current = len([
            t for t in self._requests.get(client_id, [])
            if t > cutoff
        ])
        return max(0, self.max_requests - current)

    def reset(self, client_id: str):
        """Reset rate limit for a client."""
        self._requests.pop(client_id, None)
        self._lock_times.pop(client_id, None)

    def clear_all(self):
        """Clear all rate limit data."""
        self._requests.clear()
        self._lock_times.clear()


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        config = get_security_config()
        _rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window
        )
    return _rate_limiter


# =============================================================================
# Safe Error Handling
# =============================================================================

class SafeError(Exception):
    """Exception with safe, user-visible message."""

    def __init__(self, safe_message: str, internal_message: str = None):
        self.safe_message = safe_message
        self.internal_message = internal_message or safe_message
        super().__init__(self.internal_message)


def safe_error_message(error: Exception) -> str:
    """
    Convert an exception to a safe error message.

    Hides internal details like file paths and stack traces.

    Args:
        error: The exception

    Returns:
        Safe message string
    """
    if isinstance(error, SafeError):
        return error.safe_message

    if isinstance(error, ValueError):
        # ValueError messages are usually safe
        msg = str(error)
        # But sanitize any paths
        msg = re.sub(r'[A-Za-z]:[/\\][^\s]+', '[path]', msg)
        return msg

    if isinstance(error, FileNotFoundError):
        return "Requested file not found"

    if isinstance(error, PermissionError):
        return "Permission denied"

    if isinstance(error, TimeoutError):
        return "Operation timed out"

    # Generic message for unknown errors
    logger.exception("Internal error occurred")
    return "An internal error occurred"


# =============================================================================
# WebSocket Security
# =============================================================================

class WebSocketSecurityManager:
    """
    Security manager for WebSocket connections.

    Handles:
    - Origin validation
    - API key authentication
    - Connection tracking
    - Rate limiting
    """

    def __init__(
        self,
        allowed_origins: Optional[set[str]] = None,
        api_key: Optional[str] = None,
        max_connections: int = 100
    ):
        """
        Args:
            allowed_origins: Set of allowed Origin headers (None = localhost only)
            api_key: Required API key (None = no auth required)
            max_connections: Maximum concurrent connections
        """
        self.allowed_origins = allowed_origins or {
            "http://localhost",
            "http://127.0.0.1",
            "https://localhost",
            "https://127.0.0.1",
            "null",  # For file:// origins
        }
        self.api_key = api_key or os.getenv("VOICE_API_KEY")
        self.max_connections = max_connections
        self._connections: set[str] = set()
        self._rate_limiter = RateLimiter()

    def validate_origin(self, origin: Optional[str]) -> bool:
        """
        Validate the Origin header.

        Args:
            origin: The Origin header value

        Returns:
            True if origin is allowed
        """
        # No origin header (non-browser clients) - allow
        if not origin:
            return True

        # Check against allowlist
        # Also check with ports (localhost:8080, etc.)
        for allowed in self.allowed_origins:
            if origin == allowed or origin.startswith(allowed + ":"):
                return True

        logger.warning(f"Rejected connection from origin: {origin}")
        return False

    def validate_api_key(self, provided_key: Optional[str]) -> bool:
        """
        Validate API key.

        Args:
            provided_key: The provided API key

        Returns:
            True if valid or no key required
        """
        # No key configured = no auth required
        if not self.api_key:
            return True

        if not provided_key:
            return False

        # Constant-time comparison to prevent timing attacks
        import hmac
        return hmac.compare_digest(self.api_key, provided_key)

    def can_accept_connection(self) -> bool:
        """Check if a new connection can be accepted."""
        return len(self._connections) < self.max_connections

    def add_connection(self, client_id: str):
        """Register a new connection."""
        self._connections.add(client_id)

    def remove_connection(self, client_id: str):
        """Unregister a connection."""
        self._connections.discard(client_id)
        self._rate_limiter.reset(client_id)

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        return self._rate_limiter.is_allowed(client_id)

    @property
    def connection_count(self) -> int:
        """Current number of connections."""
        return len(self._connections)


# =============================================================================
# Secure Hash Functions
# =============================================================================

def secure_hash(data: str, length: int = 8) -> str:
    """
    Generate a secure hash of data.

    Uses SHA-256 instead of MD5.

    Args:
        data: Data to hash
        length: Length of hash to return

    Returns:
        Hex hash string
    """
    return hashlib.sha256(data.encode()).hexdigest()[:length]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Config
    "SecurityConfig",
    "get_security_config",
    # Path sanitization
    "sanitize_filename",
    "validate_output_path",
    "safe_join_path",
    # Input validation
    "validate_text_input",
    "validate_voice_id",
    "validate_speed",
    "validate_json_message",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    # Error handling
    "SafeError",
    "safe_error_message",
    # WebSocket security
    "WebSocketSecurityManager",
    # Hashing
    "secure_hash",
]
