# Changelog

All notable changes to Voice Soundboard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-22

### The "2027 Edition" - Feature Complete Release

This release represents the culmination of the full roadmap, transforming Voice Soundboard from a simple TTS tool into a comprehensive voice synthesis platform.

### Added

#### Core Features (v0.1.0)
- **VoiceEngine** - High-quality TTS using Kokoro ONNX (82M params, 2-3x realtime)
- **54+ Voices** - American, British, Japanese, Mandarin accents with male/female options
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

#### Phase 1: Chatterbox Integration (v0.2.0)
- **ChatterboxEngine** - Alternative TTS backend with expressive features
- **Paralinguistic tags** - `[laugh]`, `[sigh]`, `[gasp]`, `[cough]`, `[chuckle]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`
- **Emotion exaggeration** - 0.0 (monotone) to 1.0 (dramatic) slider
- **CFG weight control** - Pacing adjustment

#### Phase 2: Multi-Speaker Dialogue (v0.3.0)
- **DialogueParser** - Script parsing with speaker tags `[S1:name]`
- **DialogueEngine** - Multi-speaker synthesis orchestration
- **Stage directions** - `(whispering)`, `(angrily)`, `(nervously)`, etc.
- **Auto voice assignment** - Automatic distinct voice selection
- **Turn pause control** - Configurable pauses between speakers

#### Phase 3: Advanced Emotion Control (v0.4.0)
- **Word-level emotion tags** - `{happy}text{/happy}` syntax
- **VAD emotion model** - Valence-Arousal-Dominance mapping for 50+ emotions
- **Emotion blending** - Mix emotions with weights (e.g., 70% happy + 30% surprised)
- **Emotion curves** - Dynamic emotion over time with keyframes
- **Pre-built curves** - tension_build, joy_arc, suspense, revelation

#### Phase 4: Voice Cloning (v0.5.0)
- **VoiceCloner** - Clone any voice from 3-10 second sample
- **VoiceLibrary** - Store and manage cloned voices
- **Consent tracking** - Ethical voice cloning with required acknowledgment
- **Emotion-timbre separation** - Apply different emotions to cloned voices
- **Cross-language cloning** - Clone English voice, speak Chinese
- **Similar voice search** - Find voices similar to a reference

#### Phase 5: Neural Audio Codecs (v0.6.0)
- **AudioCodec** - Abstract base class for codec implementations
- **MimiCodec** - 12.5 Hz codec for LLM integration (8 codebooks, 2048 codes)
- **DualCodec** - Semantic-acoustic separation for voice conversion
- **LLMCodecBridge** - Utilities for LLM token integration
- **Token estimation** - Predict token count for audio duration

#### Phase 6: Real-Time Voice Conversion (v0.7.0)
- **VoiceConverter** - Real-time voice changing
- **LatencyMode** - ultra_low (~60ms), balanced (~150ms), high_quality (~300ms)
- **AudioDeviceManager** - Input/output device selection
- **StreamingConverter** - Continuous audio stream processing
- **RealtimeSession** - Managed conversion sessions

#### Phase 7: LLM Integration (v1.0.0)
- **speak_with_context** - Context-aware emotion selection
- **Conversation management** - Start/add/get conversation context
- **User emotion detection** - Sentiment analysis for responses
- **Response emotion selection** - Automatic appropriate emotion choice
- **LLM provider abstraction** - Support for Ollama, OpenAI, etc.

#### Text Normalization (v1.0.1)
- **Normalizer module** - Comprehensive text preprocessing for TTS
- Number to words conversion (integers, decimals, large numbers)
- Currency expansion ($100 -> "one hundred dollars")
- Abbreviation expansion (Dr., Mr., etc.)
- Acronym expansion (FBI -> "F B I")
- Emoji to text ("grinning face")
- Math symbol expansion
- HTML entity decoding
- URL/Email verbalization

#### Mobile Web Access (v1.0.1)
- **Web server** - aiohttp-based HTTP server for mobile access
- **Responsive UI** - Mobile-friendly HTML/CSS/JS interface
- **PWA support** - Add to home screen capability
- **WebSocket + REST fallback** - Reliable connectivity
- Voice selection grid with language filters
- Quick phrase buttons
- Sound effects panel

#### MCP Integration
- 40+ MCP tools for AI agent integration
- Core: speak, speak_long, speak_ssml, speak_chatterbox, speak_dialogue
- Voice: list_voices, list_presets, clone_voice, list_voice_library
- Emotion: list_emotions, blend_emotions, get_emotion_vad
- Utility: sound_effect, play_audio, stop_audio, encode_audio_tokens

#### WebSocket Server
- Real-time bidirectional API
- All speak/stream/effect/dialogue actions
- JSON message protocol
- Base64 audio return option
- Origin validation
- API key authentication
- TLS/SSL support

### Security

- **Path Traversal Protection** - sanitize_filename(), safe_join_path()
- **XXE Protection** - defusedxml for SSML parsing
- **Rate Limiting** - Token bucket algorithm (60 req/min default)
- **Input Validation** - Length limits, type checking
- **Voice Cloning Consent** - Required acknowledgment for ethical use
- **WebSocket Security**:
  - Origin validation (CSWSH protection)
  - Optional API key authentication
  - TLS/SSL support
  - Connection limits (100 max)
- **Safe Error Messages** - No internal paths exposed

### Testing

- 495+ tests across 12 sessions
- 98.4% pass rate (487 passed, 8 known minor issues)
- Full module coverage including:
  - Core: engine.py, audio.py, streaming.py
  - Features: effects.py, ssml.py, emotions.py, normalizer.py
  - Advanced: dialogue/, emotion/, cloning/, codecs/, conversion/
  - APIs: server.py, websocket_server.py, web_server.py
  - Security: security.py, comprehensive security tests

### Documentation

- Comprehensive README with examples
- Security audit report (SECURITY_AUDIT.md)
- Test plan with results (TEST_PLAN.md)
- Full roadmap (ROADMAP.md)
- Press release (PRESS_RELEASE.md)
- Contributing guide (CONTRIBUTING.md)
- API reference in docstrings
- GitHub templates (issues, PRs, security)

### Infrastructure

- GitHub Actions CI/CD
- PyPI publishing workflow
- Dependabot for security updates
- Type hints throughout
- pyproject.toml configuration

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01-22 | Feature complete "2027 Edition" |
| 0.7.0 | 2026-01-21 | Real-time voice conversion |
| 0.6.0 | 2026-01-20 | Neural audio codecs (Mimi, DualCodec) |
| 0.5.0 | 2026-01-19 | Voice cloning with library management |
| 0.4.0 | 2026-01-18 | Advanced emotion control (VAD, blending, curves) |
| 0.3.0 | 2026-01-17 | Multi-speaker dialogue synthesis |
| 0.2.0 | 2026-01-16 | Chatterbox integration (paralinguistic tags) |
| 0.1.0 | 2026-01-15 | Initial release with core features |

---

## Upgrade Guide

### From 0.x to 1.0.0

1. **Import changes**: New modules are available at the top level
   ```python
   # Old
   from voice_soundboard.dialogue.engine import DialogueEngine

   # New (also works)
   from voice_soundboard import DialogueEngine
   ```

2. **Chatterbox**: Now requires optional dependency
   ```bash
   pip install voice-soundboard[chatterbox]
   ```

3. **Mobile access**: New web server available
   ```bash
   pip install voice-soundboard[web]
   python -m voice_soundboard.web_server
   ```

4. **Voice cloning**: Requires consent acknowledgment
   ```python
   cloner.clone("sample.wav", consent_given=True)
   ```
