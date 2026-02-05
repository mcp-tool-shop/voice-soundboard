# Security & Threat Model Summary

A reviewer should be able to assess risk by reading this page alone.
For the full vulnerability report, see [SECURITY_AUDIT.md](../SECURITY_AUDIT.md).

---

## Threat Model

### What This System Does

Voice Soundboard converts text to audio locally. It does not:
- Access the internet (all inference is local)
- Store or transmit user credentials
- Process real-time microphone input (conversion module is experimental)

### Assets Worth Protecting

| Asset | Risk | Mitigation |
|-------|------|------------|
| User's filesystem | Path traversal via filenames | `sanitize_filename()`, `safe_join_path()` in security.py |
| Cloned voice data | Unauthorized voice cloning | Consent required (`consent_given=True`), audit trail |
| Server availability | DoS via large text input | Rate limiting, max text length (10,000 chars) |
| Server availability | Connection exhaustion | Max 100 concurrent connections, idle timeout |
| SSML input | XXE injection | `defusedxml` parser (non-negotiable) |
| Error messages | Information leakage | `safe_error_message()` strips file paths |

### Attack Surfaces

| Surface | Exposed When | Protection |
|---------|-------------|------------|
| MCP server (stdio) | Running as MCP tool | Input validation, rate limiting |
| WebSocket server | `websocket_server.py` started | API key auth, origin validation, TLS |
| HTTP server | `web_server.py` started | Localhost-only by default |
| File system | Any speak() call | Output restricted to configured directory |
| CLI | User runs CLI | Standard shell escaping applies |

### Out of Scope

- Network-level attacks (the system is local-only)
- GPU memory exploits (ONNX Runtime handles this)
- Model poisoning (models are downloaded once from known sources)

---

## Voice Cloning Safeguards

Voice cloning raises ethical concerns. The safeguards are:

1. **Consent is required by default.** `CloningConfig.require_consent = True`.
   Calling `clone()` without `consent_given=True` fails immediately.

2. **Consent is audited.** `VoiceProfile` stores `consent_given`, `consent_date`,
   and `consent_notes` as metadata alongside the cloned voice.

3. **Consent notes are documented.** The API encourages documenting the source
   (e.g. `consent_notes="Self-recording for personal use"`).

4. **No watermarking yet.** `CloningConfig.add_watermark` exists but is not
   implemented. This is a known gap.

### What consent does NOT do

- It does not verify identity (it's a boolean flag, not biometric verification)
- It does not prevent re-cloning from saved audio
- It does not restrict distribution of cloned voices

These are inherent limitations of local-only software with no cloud backend.

---

## Ethical Use Statement

Voice Soundboard is built for:
- AI agent developers adding speech output to their tools
- Content creators generating voiceovers and audiobooks
- Accessibility developers building assistive technology
- Researchers studying speech synthesis and voice science

Voice Soundboard is **not built for**:
- Impersonating real people without their consent
- Generating deceptive audio (deepfakes)
- Surveillance or unauthorized voice analysis
- Bypassing voice-based authentication systems

The consent mechanism in voice cloning exists to make responsible use the default.
Disabling it requires explicit code changes (`require_consent=False`), making
the decision visible and auditable.

---

## Security Contacts

- Vulnerability reporting: See [.github/SECURITY.md](../.github/SECURITY.md)
- Response time: 48-hour acknowledgment, 7-day status update
