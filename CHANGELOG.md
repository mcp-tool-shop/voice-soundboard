# Changelog

All notable changes to Voice Soundboard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-22

### Added

#### Core Features
- **VoiceEngine** - High-quality TTS using Kokoro ONNX (82M params, 2-3x realtime)
- **50+ Voices** - American, British, Japanese, Mandarin accents with male/female options
- **5 Voice Presets** - assistant, narrator, announcer, storyteller, whisper
- **19 Emotions** - happy, sad, angry, excited, calm, and more
- **Natural Language Styles** - "say this warmly", "excitedly", "like a narrator"

#### SSML Support
- Full SSML parsing with defusedxml (XXE protected)
- `<break>` - Pauses with time attribute
- `<prosody>` - Speed/rate control
- `<emphasis>` - Stress levels
- `<say-as>` - Date, time, cardinal, telephone formatting
- `<sub>` - Pronunciation substitution

#### Streaming
- **StreamingEngine** - Low-latency chunked generation
- **Real-time playback** - Play audio as it generates
- **stream_to_file()** - Stream directly to file
- **Callbacks** - on_chunk, on_progress hooks

#### Sound Effects
- 13 built-in effects: chime, success, error, attention, click, pop, whoosh, warning, critical, info, rain, white_noise, drone
- Generate and save custom effects
- Waveform types: sine, square, triangle, sawtooth

#### MCP Integration
- 11 MCP tools for AI agent integration
- speak, speak_long, speak_ssml
- list_voices, list_presets, list_effects, list_emotions
- sound_effect, play_audio, stop_audio

#### WebSocket Server
- Real-time bidirectional API
- All speak/stream/effect actions
- JSON message protocol
- Base64 audio return option

### Security

- **Path Traversal Protection** - sanitize_filename(), safe_join_path()
- **XXE Protection** - defusedxml for SSML parsing
- **Rate Limiting** - Token bucket algorithm (60 req/min default)
- **Input Validation** - Length limits, type checking
- **WebSocket Security**:
  - Origin validation (CSWSH protection)
  - Optional API key authentication
  - TLS/SSL support
  - Connection limits (100 max)
- **Safe Error Messages** - No internal paths exposed

### Testing

- 254 tests across 9 sessions
- 98% pass rate (249 passed, 5 known minor issues)
- Full module coverage:
  - engine.py, audio.py, streaming.py
  - effects.py, ssml.py, emotions.py
  - interpreter.py, server.py, config.py
  - security.py, websocket_server.py

### Documentation

- Comprehensive README with examples
- Security audit report (SECURITY_AUDIT.md)
- Test plan with results (TEST_PLAN.md)
- API reference in docstrings

### Infrastructure

- GitHub Actions CI/CD
- PyPI publishing workflow
- Type hints throughout
- pyproject.toml configuration

## [Unreleased]

### Planned
- Voice cloning with StyleTTS2
- Background music mixing
- More emotion types
- Async MCP server

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-22 | Initial release with full feature set |
