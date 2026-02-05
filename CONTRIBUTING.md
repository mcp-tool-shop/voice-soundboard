# Contributing to Voice Soundboard

## Development Setup

```bash
git clone https://github.com/mcp-tool-shop-org/voice-soundboard.git
cd voice-soundboard
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

Download models (required for integration tests, not for smoke tests):

```bash
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

---

## Project Structure

```
voice_soundboard/
  engine.py           # VoiceEngine - the main entry point
  config.py           # Config dataclass, voice/preset definitions
  audio.py            # Playback utilities
  emotions.py         # Emotion -> parameter mapping
  effects.py          # Sound effects
  interpreter.py      # Natural language style -> parameters
  cli.py              # CLI entrypoint
  security.py         # Path validation, rate limiting
  engines/            # TTS backends (Kokoro, Chatterbox, F5-TTS)
  dialogue/           # Multi-speaker dialogue
  emotion/            # Advanced: VAD, blending, curves
  cloning/            # Advanced: voice cloning
  presets/            # Advanced: 50+ preset catalog
  codecs/             # Experimental: neural audio codecs
  conversion/         # Experimental: real-time voice conversion
  llm/                # Experimental: LLM conversation pipeline
  vocology/           # Experimental: voice science
  studio/             # Experimental: voice studio sessions
  server.py           # MCP server (40+ tools)
```

**Stability tiers** are documented in [docs/API_STABILITY.md](docs/API_STABILITY.md).

---

## Running Tests

```bash
# Smoke tests only (fast, no models needed, <1 second)
pytest tests/smoke/ -v

# Core tests (default, no models needed, ~20 seconds)
pytest tests/ -v

# Everything including generated coverage tests
pytest tests/ -v -m "" --no-header
```

### Test Tiers

| Tier | Directory | Speed | When to Run |
|------|-----------|-------|-------------|
| **Smoke** | `tests/smoke/` | <1s | Every commit, every PR |
| **Core** | `tests/` (top-level) | ~20s | Every PR, CI default |
| **Generated** | `tests/generated/` | ~2min | Coverage audits only |

Smoke tests verify imports, data structures, and the CLI parser -- no models required.
Core tests mock the TTS engine and test business logic.
Generated tests are auto-created for coverage; they run in CI but are skipped by default locally.

---

## How to Add a Feature

1. Check [docs/API_STABILITY.md](docs/API_STABILITY.md) -- is your change in stable, advanced, or experimental code?
2. Write the code in the appropriate module
3. Add tests in the matching tier (smoke for public API, core for logic)
4. Update the docstring if you change a public function signature
5. Run `pytest tests/smoke/ -v` before pushing

## How to Add a Preset

Presets live in `config.py` under `VOICE_PRESETS`:

```python
VOICE_PRESETS = {
    "my_preset": {
        "voice": "af_bella",     # Must be a valid KOKORO_VOICES key
        "speed": 1.0,            # 0.5 to 2.0
        "description": "Short description of the preset",
    },
}
```

Do **not** add more than 5-7 curated presets to config.py. The preset catalog in
`voice_soundboard/presets/` is for the larger collection.

## How to Add an Emotion

Emotions live in `emotions.py` under `EMOTIONS`:

```python
EMOTIONS = {
    "my_emotion": EmotionParams(
        speed=1.0,                         # Speed multiplier
        voice_preference="af_bella",       # Suggested voice (optional)
        punctuation_boost=False,           # Add emphasis punctuation
        pause_multiplier=1.0,             # Pause scaling
    ),
}
```

---

## What NOT to Modify Casually

| File | Why |
|------|-----|
| `__init__.py` | Defines the public API. Adding exports here means committing to stability. |
| `security.py` | Security-critical. Changes need careful review. |
| `server.py` | 4800+ lines, handles all MCP tools. Avoid unless you're fixing a specific handler. |
| `config.py` VOICE_PRESETS | Curated set. Don't add presets without discussion. |

---

## Good First Changes

- Fix a typo in a docstring
- Add a test to `tests/smoke/`
- Improve an error message in `engine.py`
- Add an example to `examples/`
- Add a voice emotion to `emotions.py`

---

## Commit Messages

Format: `type: brief description`

| Type | When |
|------|------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `refactor` | Code change that doesn't add features or fix bugs |

## Code Style

- Type hints on all public functions
- Docstrings on all public classes and functions
- No `print()` in library code -- use `logging.getLogger(__name__)`
- Use `sanitize_filename()` from `security.py` for any user-provided filenames

## Security

- Never commit credentials or API keys
- Validate all user inputs at system boundaries
- Report security issues privately via GitHub Security Advisories
