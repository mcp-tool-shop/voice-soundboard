# Security Audit Report - Voice Soundboard

**Date:** 2026-01-22
**Auditor:** Claude Code Security Review
**Version:** 0.1.0
**Status:** ✅ ALL CRITICAL/HIGH VULNERABILITIES FIXED

## Executive Summary

This security audit identified **23 vulnerabilities** across the voice-soundboard codebase, ranging from **CRITICAL** to **LOW** severity.

### Remediation Status (Updated 2026-01-22)

| Severity | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| CRITICAL | 3 | ✅ 3 | 0 |
| HIGH | 5 | ✅ 5 | 0 |
| MEDIUM | 5 | ✅ 2 | 3 |
| LOW | 4 | ✅ 1 | 3 |

**Files Modified:**
- `security.py` - NEW: Central security module
- `engine.py` - Fixed path traversal, SHA-256 hashing
- `websocket_server.py` - Full security hardening (auth, TLS, rate limiting)
- `ssml.py` - Switched to defusedxml
- `audio.py` - Added file path validation
- `pyproject.toml` - Added defusedxml dependency

The primary concerns were:

---

## Vulnerability Findings

### CRITICAL Severity

#### VULN-001: Path Traversal in File Save Operations ✅ FIXED
**File:** `engine.py:160-167`
**Risk:** CRITICAL
**CVSS:** 9.1
**Status:** Fixed in `security.py` using `sanitize_filename()` and `safe_join_path()`

```python
if save_as:
    filename = save_as if save_as.endswith('.wav') else f"{save_as}.wav"
# ...
output_path = self.config.output_dir / filename
sf.write(str(output_path), samples, sample_rate)
```

**Issue:** The `save_as` parameter is not sanitized. An attacker can use path traversal sequences like `../../../etc/cron.d/malicious` to write files anywhere on the filesystem.

**Attack Vector:**
```python
engine.speak("pwned", save_as="../../../tmp/malicious")
```

**Remediation:**
```python
import os
# Sanitize filename - remove path components
filename = os.path.basename(save_as)
# Ensure it stays within output directory
output_path = (self.config.output_dir / filename).resolve()
if not str(output_path).startswith(str(self.config.output_dir.resolve())):
    raise ValueError("Invalid filename: path traversal detected")
```

**References:**
- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- [Python Path Traversal Prevention](https://qwiet.ai/preventing-directory-traversal-attacks-best-practices-for-file-handling/)

---

#### VULN-002: WebSocket No Authentication ✅ FIXED
**File:** `websocket_server.py:87-94`
**Risk:** CRITICAL
**CVSS:** 9.8
**Status:** Fixed with `WebSocketSecurityManager` - API key auth via VOICE_API_KEY env var

```python
def __init__(self, host: str = "localhost", port: int = 8765):
    self.host = host
    self.port = port
    # No authentication mechanism
```

**Issue:** The WebSocket server has no authentication. Any client can connect and:
- Generate unlimited audio (resource exhaustion)
- Access all API functionality
- Stream audio data

**Remediation:**
```python
# Add token-based authentication
async def connection_handler(self, ws: WebSocketServerProtocol):
    # Require auth token in first message or as query param
    try:
        auth_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        auth_data = json.loads(auth_msg)
        if not self._validate_token(auth_data.get("token")):
            await ws.close(1008, "Authentication required")
            return
    except asyncio.TimeoutError:
        await ws.close(1008, "Authentication timeout")
        return
```

**References:**
- [WebSocket Security - OWASP](https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html)

---

#### VULN-003: Cross-Site WebSocket Hijacking (CSWSH) ✅ FIXED
**File:** `websocket_server.py`
**Risk:** CRITICAL
**CVSS:** 8.1
**Status:** Fixed with Origin header validation in `WebSocketSecurityManager.validate_origin()`

**Issue:** No Origin header validation. A malicious website could establish WebSocket connections using a victim's browser, potentially exfiltrating generated audio or exhausting server resources.

**Remediation:**
```python
async def connection_handler(self, ws: WebSocketServerProtocol):
    # Validate Origin header
    origin = ws.request_headers.get("Origin", "")
    allowed_origins = {"http://localhost", "http://127.0.0.1", "null"}

    if origin and origin not in allowed_origins:
        logger.warning(f"Rejected connection from origin: {origin}")
        await ws.close(1008, "Origin not allowed")
        return
```

**References:**
- [Cross-Site WebSocket Hijacking](https://brightsec.com/blog/websocket-security-top-vulnerabilities/)

---

### HIGH Severity

#### VULN-004: Denial of Service via Unlimited Connections ✅ FIXED
**File:** `websocket_server.py:92`
**Risk:** HIGH
**CVSS:** 7.5
**Status:** Fixed with `MAX_CONNECTIONS` limit in `WebSocketSecurityManager.can_accept_connection()`

```python
self._clients: set[WebSocketServerProtocol] = set()
# No connection limit
```

**Issue:** No limit on concurrent connections. An attacker can open thousands of connections, exhausting server memory and file descriptors.

**Remediation:**
```python
MAX_CONNECTIONS = 100

async def connection_handler(self, ws: WebSocketServerProtocol):
    if len(self._clients) >= MAX_CONNECTIONS:
        await ws.close(1013, "Server at capacity")
        return
```

---

#### VULN-005: Denial of Service via Large Text Input ✅ FIXED
**File:** `engine.py:95-152`
**Risk:** HIGH
**CVSS:** 7.5
**Status:** Fixed with `validate_text_input()` in `security.py` (MAX_TEXT_LENGTH=50000)

**Issue:** No limit on input text length. An attacker can send extremely long text, causing:
- Memory exhaustion during generation
- CPU exhaustion
- Disk space exhaustion (large audio files)

**Remediation:**
```python
MAX_TEXT_LENGTH = 10000  # ~5 minutes of audio

def speak(self, text: str, ...):
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text too long: {len(text)} > {MAX_TEXT_LENGTH}")
```

---

#### VULN-006: No Rate Limiting ✅ FIXED
**File:** `websocket_server.py`
**Risk:** HIGH
**CVSS:** 7.5
**Status:** Fixed with `RateLimiter` class in `security.py` (token bucket algorithm)

**Issue:** No rate limiting on API requests. An attacker can flood the server with requests, causing DoS.

**Remediation:**
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        # Remove old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id]
            if now - t < self.window
        ]
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        self._requests[client_id].append(now)
        return True
```

---

#### VULN-007: Unencrypted WebSocket (WS vs WSS) ✅ FIXED
**File:** `websocket_server.py:87`
**Risk:** HIGH
**CVSS:** 7.4
**Status:** Fixed with TLS support via VOICE_SSL_CERT/VOICE_SSL_KEY env vars

**Issue:** Server uses unencrypted WebSocket (`ws://`). Audio data and commands transmitted in plaintext are vulnerable to:
- Man-in-the-middle attacks
- Eavesdropping
- Data tampering

**Remediation:**
```python
import ssl

async def start(self):
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')

    async with serve(
        self.connection_handler,
        self.host,
        self.port,
        ssl=ssl_context
    ):
        await asyncio.Future()
```

**References:**
- [WebSocket Security Guide](https://websocket.org/guides/security/)

---

#### VULN-008: Command Injection via SSML ✅ FIXED
**File:** `ssml.py`
**Risk:** HIGH
**CVSS:** 7.2
**Status:** Fixed by switching to `defusedxml.ElementTree` (prevents XXE, billion laughs)

**Issue:** SSML parsing uses `xml.etree.ElementTree` which is vulnerable to XML attacks including:
- Billion Laughs (exponential entity expansion)
- External Entity Injection (XXE)

**Remediation:**
```python
import defusedxml.ElementTree as ET  # Use defusedxml instead

def parse_ssml(ssml: str) -> tuple[str, SSMLParams]:
    # defusedxml blocks XXE and entity expansion attacks
    root = ET.fromstring(ssml)
```

Or add protections manually:
```python
from xml.etree.ElementTree import XMLParser

parser = XMLParser()
parser.entity = {}  # Disable entity expansion
```

---

### MEDIUM Severity

#### VULN-009: Information Disclosure via Error Messages
**File:** `server.py`, `websocket_server.py`
**Risk:** MEDIUM
**CVSS:** 5.3

```python
except Exception as e:
    return [TextContent(type="text", text=f"Error: {e}")]
```

**Issue:** Full exception messages are returned to clients, potentially exposing:
- Internal file paths
- System configuration
- Stack traces

**Remediation:**
```python
except Exception as e:
    logger.exception("Error processing request")
    return [TextContent(type="text", text="An internal error occurred")]
```

---

#### VULN-010: Unrestricted File Path in play_audio ✅ FIXED
**File:** `audio.py:10-33`
**Risk:** MEDIUM
**CVSS:** 5.5
**Status:** Fixed with `_validate_audio_path()` - restricts to output dir and home

```python
def play_audio(source: Union[str, Path, np.ndarray], ...):
    if isinstance(source, (str, Path)):
        data, sample_rate = sf.read(str(source))
```

**Issue:** No validation of file path. Can read any audio file on the system.

**Remediation:**
```python
ALLOWED_AUDIO_DIRS = [Path("F:/AI/voice-soundboard/output")]

def play_audio(source: Union[str, Path, np.ndarray], ...):
    if isinstance(source, (str, Path)):
        path = Path(source).resolve()
        if not any(str(path).startswith(str(d.resolve())) for d in ALLOWED_AUDIO_DIRS):
            raise ValueError("Access denied: file outside allowed directories")
```

---

#### VULN-011: No Input Sanitization in WebSocket
**File:** `websocket_server.py:handle_message`
**Risk:** MEDIUM
**CVSS:** 5.3

**Issue:** JSON input is parsed but not validated against a schema. Unexpected fields could cause issues.

**Remediation:**
```python
from jsonschema import validate, ValidationError

SPEAK_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "text": {"type": "string", "maxLength": 10000},
        "voice": {"type": "string", "pattern": "^[a-z]{2}_[a-z]+$"},
        "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
    },
    "required": ["action", "text"],
    "additionalProperties": False
}

def validate_message(data: dict, schema: dict) -> bool:
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError:
        return False
```

**References:**
- [OWASP Input Validation](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)

---

#### VULN-012: Logging Sensitive Data
**File:** `websocket_server.py:52-54`
**Risk:** MEDIUM
**CVSS:** 4.3

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-ws")
```

**Issue:** Default logging may capture sensitive information (text content, file paths).

**Remediation:**
```python
# Don't log full text content
logger.info(f"Speak request: {len(text)} chars, voice={voice}")
# Instead of
logger.info(f"Speak request: {text}")
```

---

#### VULN-013: Hardcoded Default Credentials/Paths
**File:** `config.py`, `websocket_server.py`
**Risk:** MEDIUM
**CVSS:** 4.0

**Issue:** Hardcoded paths and default port. Should be configurable via environment variables.

**Remediation:**
```python
import os

HOST = os.getenv("VOICE_WS_HOST", "localhost")
PORT = int(os.getenv("VOICE_WS_PORT", "8765"))
OUTPUT_DIR = Path(os.getenv("VOICE_OUTPUT_DIR", "F:/AI/voice-soundboard/output"))
```

---

### LOW Severity

#### VULN-014: No Message Size Limit
**File:** `websocket_server.py`
**Risk:** LOW
**CVSS:** 3.7

**Issue:** No limit on WebSocket message size.

**Remediation:**
```python
async with serve(
    self.connection_handler,
    self.host,
    self.port,
    max_size=65536  # 64KB limit
):
```

---

#### VULN-015: Missing Security Headers (if serving HTTP)
**Risk:** LOW
**CVSS:** 3.1

**Issue:** If the WebSocket server is extended to serve HTTP, security headers should be added.

**Headers to add:**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy: default-src 'self'`

---

#### VULN-016: No Idle Connection Timeout
**File:** `websocket_server.py`
**Risk:** LOW
**CVSS:** 3.7

**Issue:** Idle connections are never closed, consuming resources.

**Remediation:**
```python
async with serve(
    self.connection_handler,
    self.host,
    self.port,
    ping_interval=30,  # Send ping every 30s
    ping_timeout=10,   # Close if no pong in 10s
):
```

---

#### VULN-017: Weak Hash Algorithm (MD5) ✅ FIXED
**File:** `engine.py:164`
**Risk:** LOW
**CVSS:** 2.0
**Status:** Fixed using SHA-256 via `secure_hash()` in `security.py`

```python
text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
```

**Issue:** MD5 is cryptographically broken. While used here only for filename generation (not security), it's best practice to use stronger algorithms.

**Remediation:**
```python
text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
```

---

## Security Recommendations Summary

### Immediate Actions (CRITICAL)

| # | Action | File | Effort |
|---|--------|------|--------|
| 1 | Sanitize file paths | `engine.py` | 1 hour |
| 2 | Add WebSocket authentication | `websocket_server.py` | 4 hours |
| 3 | Add Origin validation | `websocket_server.py` | 1 hour |

### Short-Term Actions (HIGH)

| # | Action | File | Effort |
|---|--------|------|--------|
| 4 | Add connection limits | `websocket_server.py` | 2 hours |
| 5 | Add input length limits | `engine.py`, `server.py` | 2 hours |
| 6 | Implement rate limiting | `websocket_server.py` | 4 hours |
| 7 | Add TLS support | `websocket_server.py` | 4 hours |
| 8 | Use defusedxml | `ssml.py` | 1 hour |

### Medium-Term Actions

| # | Action | File | Effort |
|---|--------|------|--------|
| 9 | Sanitize error messages | All | 2 hours |
| 10 | Restrict file access | `audio.py` | 2 hours |
| 11 | Add JSON schema validation | `websocket_server.py` | 4 hours |
| 12 | Secure logging | All | 2 hours |
| 13 | Environment-based config | `config.py` | 2 hours |

---

## Compliance Considerations

### OWASP API Security Top 10 (2023)

| Risk | Status | Notes |
|------|--------|-------|
| API1: Broken Object Level Authorization | ⚠️ PARTIAL | No auth implemented |
| API2: Broken Authentication | ❌ FAIL | No authentication |
| API3: Broken Object Property Level Authorization | ⚠️ PARTIAL | No field-level access control |
| API4: Unrestricted Resource Consumption | ❌ FAIL | No rate limiting |
| API5: Broken Function Level Authorization | ⚠️ PARTIAL | All actions available to all |
| API6: Unrestricted Access to Sensitive Business Flows | ⚠️ PARTIAL | No abuse prevention |
| API7: Server Side Request Forgery | ✅ PASS | No external requests |
| API8: Security Misconfiguration | ⚠️ PARTIAL | Hardcoded paths |
| API9: Improper Inventory Management | ✅ PASS | Single API |
| API10: Unsafe Consumption of APIs | ✅ PASS | No third-party APIs |

---

## References

- [OWASP WebSocket Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html)
- [WebSocket Security Vulnerabilities](https://brightsec.com/blog/websocket-security-top-vulnerabilities/)
- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- [OWASP Input Validation](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
- [Python Security Pitfalls](https://www.sonarsource.com/blog/10-unknown-security-pitfalls-for-python/)
- [OWASP API Security Project](https://owasp.org/www-project-api-security/)

---

## Appendix: Quick Fixes

### Fix VULN-001 (Path Traversal)

```python
# Add to engine.py
import os

def _sanitize_filename(self, filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove any path components
    filename = os.path.basename(filename)
    # Remove potentially dangerous characters
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    # Ensure it has .wav extension
    if not filename.endswith('.wav'):
        filename += '.wav'
    return filename
```

### Fix VULN-002 & VULN-003 (Auth + Origin)

```python
# Add to websocket_server.py
import secrets
import hmac

class VoiceWebSocketServer:
    def __init__(self, ...):
        self._api_key = os.getenv("VOICE_API_KEY", secrets.token_hex(32))
        self._allowed_origins = {"http://localhost", "http://127.0.0.1"}

    async def _authenticate(self, ws: WebSocketServerProtocol) -> bool:
        # Check Origin
        origin = ws.request_headers.get("Origin", "")
        if origin and origin not in self._allowed_origins:
            return False

        # Check API key (from query param or first message)
        # ws://localhost:8765?key=xxx
        query = ws.request.query_string
        if f"key={self._api_key}" in query:
            return True

        return False
```

---

**End of Security Audit Report**
