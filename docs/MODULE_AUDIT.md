# Module Audit

Last updated: Phase 4 (2026-02-05)

Every module has a status and a one-sentence justification.
Modules without justification get deleted.

---

## Status Key

| Status | Meaning |
|--------|---------|
| **Core** | Required for platform identity. Cannot be removed. |
| **Optional** | Useful, ships with the package, but not loaded by default. |
| **Research** | Experimental. Not on the default import path. May change or be removed. |
| **Server** | Entry point for a specific deployment mode. Not imported by other modules. |
| **Remove** | No longer justified. Deleted or archived. |

---

## Top-Level Modules

| Module | Status | Justification |
|--------|--------|---------------|
| `__init__.py` | Core | Public API surface. Defines what users see. |
| `engine.py` | Core | The VoiceEngine. Every speak() call goes through here. |
| `config.py` | Core | Configuration, voice registry, preset definitions. |
| `audio.py` | Core | Plays audio files. Required for any audible output. |
| `emotions.py` | Core | Maps emotion names to synthesis parameters. |
| `effects.py` | Core | Procedural sound effects (chime, success, error). |
| `normalizer.py` | Core | Text preprocessing (numbers, currency, abbreviations). |
| `interpreter.py` | Core | Natural language style hints to voice parameters. |
| `security.py` | Core | Path sanitization, input validation, rate limiting. |
| `exceptions.py` | Core | Domain exception hierarchy. |
| `ssml.py` | Core | SSML parsing with XXE protection. |
| `errors.py` | Core | Structured error responses for MCP/server APIs. |
| `streaming.py` | Optional | Chunked TTS generation. Only needed for real-time playback. |
| `cli.py` | Server | CLI entry point. Not imported by other modules. |
| `server.py` | Server | MCP server. Not imported by other modules. |
| `web_server.py` | Server | HTTP server for mobile access. Not imported by other modules. |
| `websocket_server.py` | Server | WebSocket server. Not imported by other modules. |
| `mcp_screenshot.py` | **Remove** | Screenshot utility unrelated to TTS. Does not belong in this package. |

---

## Packages

| Package | Status | Justification |
|---------|--------|---------------|
| `engines/` | Optional | Multi-engine support (Kokoro, Chatterbox, F5-TTS). |
| `dialogue/` | Optional | Multi-speaker dialogue synthesis. |
| `emotion/` | Optional | Advanced emotion control (VAD model, blending, curves). |
| `cloning/` | Optional | Voice cloning from audio samples. |
| `presets/` | Optional | Extended voice preset catalog with search. |
| `codecs/` | Research | Neural audio codecs for LLM integration. No production users. |
| `conversion/` | Research | Real-time voice conversion. Hardware-dependent, immature. |
| `llm/` | Research | LLM conversation pipeline. Depends on external providers. |
| `vocology/` | Research | Voice science analysis and humanization. Academic-oriented. |
| `studio/` | Research | Voice Studio editing sessions. Incomplete. |

---

## Decisions

### Removed: `mcp_screenshot.py`

Screenshot capture and screen recording have nothing to do with text-to-speech.
This module was added for browser automation support but belongs in a separate tool.
Deleted in this phase.

### Research modules stay in-tree but are gated

`codecs/`, `conversion/`, `llm/`, `vocology/`, and `studio/` remain in the repository
but are not loaded unless explicitly imported. They are documented as experimental
in `__init__.py` and `docs/API_STABILITY.md`.

### No "maybe" classifications

Every module has a definitive status. If it doesn't justify itself in one sentence,
it gets removed.
