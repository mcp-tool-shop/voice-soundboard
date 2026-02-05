# Feature Flags & Opt-In Complexity

Voice Soundboard follows a strict principle: **core features work with zero configuration**.
Advanced and experimental features exist but never appear unless explicitly requested.

---

## How Features Are Gated

| Gate | Mechanism | Example |
|------|-----------|---------|
| **Install extra** | pip optional dependency group | `pip install voice-soundboard[chatterbox]` |
| **Subpackage import** | Not re-exported from `__init__.py` | `from voice_soundboard.codecs import ...` |
| **Runtime availability** | `*_AVAILABLE` constants | `MIMI_AVAILABLE`, `DUALCODEC_AVAILABLE`, `CHATTERBOX_AVAILABLE` |
| **Constructor parameter** | Explicit opt-in argument | `CloningConfig(require_consent=True)` |

---

## Tier 1: Core (always available)

These work after `pip install voice-soundboard` + downloading Kokoro models.
No flags needed.

- `VoiceEngine.speak()` with voice, preset, emotion, style, speed
- `play_audio()`, `stop_playback()`
- Sound effects (`get_effect`, `play_effect`)
- Text normalization (automatic)
- SSML parsing
- CLI (`voice-soundboard speak "..."`)

## Tier 2: Optional (install extras, import subpackage)

Available after installing the relevant extra. Import from the subpackage.

| Feature | Install | Import From |
|---------|---------|-------------|
| Chatterbox (paralinguistic tags, 23 languages) | `[chatterbox]` | `voice_soundboard.engines.chatterbox` |
| F5-TTS (voice cloning engine) | `[f5tts]` | `voice_soundboard.engines.f5tts` |
| Multi-speaker dialogue | (included) | `voice_soundboard.dialogue` |
| Advanced emotion control | (included) | `voice_soundboard.emotion` |
| Voice cloning + library | (included) | `voice_soundboard.cloning` |
| Extended preset catalog | (included) | `voice_soundboard.presets` |
| Multi-engine support | (included) | `voice_soundboard.engines` |
| MCP server | `[mcp]` | `voice_soundboard.server` |
| WebSocket server | `[websocket]` | `voice_soundboard.websocket_server` |
| Web UI | `[web]` | `voice_soundboard.web_server` |

## Tier 3: Research (experimental, may change)

These are explicitly marked experimental. They may change or be removed between
minor releases. Import from the subpackage; they are never re-exported.

| Feature | Import From | Status |
|---------|-------------|--------|
| Neural audio codecs | `voice_soundboard.codecs` | No production users. API may change. |
| Real-time voice conversion | `voice_soundboard.conversion` | Hardware-dependent. Latency varies. |
| LLM conversation pipeline | `voice_soundboard.llm` | Depends on external LLM providers. |
| Voice science / humanization | `voice_soundboard.vocology` | Academic-oriented. Tuning needed. |
| Voice Studio editor | `voice_soundboard.studio` | Incomplete. |

---

## Rules

1. **Core features never import research modules.** If `engine.py` needs something
   from `vocology/`, it's a bug.

2. **Research modules may depend on core modules** but not on each other
   (except `llm/` depending on `codecs/` conditionally).

3. **`__init__.py` never exports research symbols.** Users must explicitly
   `from voice_soundboard.codecs import ...`.

4. **Runtime availability flags** (`*_AVAILABLE`) let code gracefully degrade
   when optional dependencies are missing.

5. **No feature flag file or registry.** Features are gated by Python's import
   system and pip extras. No `.env` flags, no config file toggles.
