# Voice Soundboard 1.1: F5-TTS Engine & 23-Language Multilingual Support

**FOR IMMEDIATE RELEASE**

*Superior voice cloning with Diffusion Transformers and global language coverage*

---

## Going Global with Superior Voice Cloning

**January 23, 2026** - Today marks the release of **Voice Soundboard 1.1**, introducing the F5-TTS Diffusion Transformer engine for state-of-the-art voice cloning and expanding Chatterbox from English-only to 23 languages.

```python
# Clone any voice with F5-TTS
engine.clone_voice("sample.wav", transcription="Hello, this is my voice.")
engine.speak("Bonjour le monde!", language="fr")  # Speak in 23 languages
```

---

## What's New in 1.1

### F5-TTS: Next-Generation Voice Cloning

The new F5-TTS engine uses Diffusion Transformer (DiT) architecture with flow matching:

- **Zero-shot cloning** - Clone any voice from a short sample
- **No duration model** - Simplified architecture, better results
- **0.15 RTF** - Fast inference on GPU
- **Transcription-guided** - Provide reference text for best quality

```python
from voice_soundboard.engines import F5TTSEngine

engine = F5TTSEngine()
engine.clone_voice(
    "reference.wav",
    voice_id="my_voice",
    transcription="The quick brown fox jumps over the lazy dog."
)
result = engine.speak("Hello world!", voice="my_voice")
```

### Chatterbox Multilingual: 23 Languages

Expanded from English-only to global coverage:

| Region | Languages |
|--------|-----------|
| **Europe** | Danish, Dutch, Finnish, French, German, Greek, Italian, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Turkish |
| **Asia** | Arabic, Chinese, Hebrew, Hindi, Japanese, Korean, Malay |
| **Africa** | Swahili |
| **Default** | English |

```python
from voice_soundboard.engines import ChatterboxEngine

engine = ChatterboxEngine()  # Multilingual by default

# Speak in any language
engine.speak("Guten Tag!", language="de")
engine.speak("Bonjour!", language="fr")
engine.speak("こんにちは！", language="ja")
engine.speak("مرحبا!", language="ar")
```

---

## By the Numbers

| Metric | v1.0.0 | v1.1.0 |
|--------|--------|--------|
| TTS Engines | 2 | 3 (+F5-TTS) |
| Languages | 4 | 23 (+19) |
| Tests Defined | 686 | 876 (+190) |
| MCP Tools | 40+ | 43+ (+3) |

---

## Backward Compatibility

Users on older hardware or Python 3.12+ can continue using v1.0.0:

```bash
# Stay on v1.0.0 for maximum compatibility
pip install voice-soundboard==1.0.0

# Upgrade to v1.1.0 for new features
pip install voice-soundboard[all]
```

### Requirements

- **F5-TTS**: Python 3.10-3.11, CUDA recommended
- **Chatterbox Multilingual**: Python 3.11 (not 3.12+)
- **Kokoro (default)**: Python 3.10+, CPU or GPU

---

## Use Cases

### Global Customer Service
*"Bonjour! Comment puis-je vous aider?"*

Support customers in their native language with natural, expressive voices.

### International Audiobooks
Generate audiobooks in 23 languages with the same cloned narrator voice.

### Multilingual AI Assistants
Build AI agents that speak naturally in Arabic, Japanese, German, and more.

### Voice Cloning for Content Creators
Clone your voice once, create content in any language with F5-TTS quality.

---

## Getting Started

```bash
# Install with all features
pip install voice-soundboard[all]

# Or specific features
pip install voice-soundboard[f5tts]      # F5-TTS only
pip install voice-soundboard[chatterbox] # Chatterbox multilingual
```

```python
from voice_soundboard.engines import F5TTSEngine, ChatterboxEngine

# F5-TTS for voice cloning
f5 = F5TTSEngine()
f5.clone_voice("sample.wav", transcription="Sample text")
f5.speak("Cloned voice!", voice="cloned")

# Chatterbox for multilingual
cb = ChatterboxEngine()
cb.speak("Hello world!", language="en")
cb.speak("Hallo Welt!", language="de")
```

---

## Quick Facts

- **Release Date**: January 23, 2026
- **Version**: 1.1.0 "Multilingual Voice Cloning"
- **License**: MIT
- **Language**: Python 3.10+ (3.11 for multilingual)
- **Platforms**: Windows, Linux, macOS
- **GPU**: Recommended for F5-TTS

---

<p align="center">
  <strong>Voice Soundboard 1.1</strong><br>
  <em>Now speaking your language.</em>
</p>

---

## Links

- **GitHub**: github.com/mikeyfrilot/voice-soundboard
- **PyPI**: pypi.org/project/voice-soundboard
- **Changelog**: CHANGELOG.md
- **Roadmap**: ROADMAP.md

---

# Voice Soundboard 1.0: The Complete Voice Synthesis Platform for AI Agents

**FOR IMMEDIATE RELEASE**

*From simple TTS to comprehensive voice synthesis - Voice Soundboard reaches feature-complete 1.0 release*

---

## Giving AI Agents a Voice - Now Feature Complete

**January 2026** - Today marks the release of **Voice Soundboard 1.0**, transforming what began as a simple text-to-speech tool into a comprehensive voice synthesis platform. With this release, AI agents can now speak naturally with human-like expressiveness, complete with laughter, sighs, and emotional nuance.

```python
engine.speak("That's hilarious! [laugh] I really needed that.", style="warmly")
```

---

## What's New in 1.0

### Paralinguistic Tags
Natural non-speech sounds that make AI voices feel human:
- `[laugh]` - Full laughter in the speaker's voice
- `[sigh]` - Emotional exhales
- `[gasp]`, `[cough]`, `[chuckle]` - Natural reactions

### Multi-Speaker Dialogue
Generate podcasts, audiobooks, and conversations with multiple distinct voices:

```python
script = """
[S1:narrator] The detective entered the room.
[S2:detective] (firmly) Where were you last night?
[S3:suspect] (nervously) I... I was at home.
"""
engine.speak_dialogue(script)
```

### Voice Cloning
Clone any voice from just 3-10 seconds of audio:
- Cross-language synthesis (clone in English, speak in French)
- Emotion-timbre separation (apply different emotions to cloned voices)
- Built-in consent tracking for ethical use

### Advanced Emotion Control
Beyond simple emotion labels:
- **Word-level tags**: `{happy}text{/happy}` for mid-sentence emotion changes
- **VAD Model**: 50+ emotions mapped to Valence-Arousal-Dominance
- **Emotion Blending**: Mix 70% happy + 30% surprised
- **Dynamic Curves**: Emotions that evolve over the utterance

### Mobile Access
Use Voice Soundboard from any phone or tablet:
```bash
python -m voice_soundboard.web_server
# Access from your phone at http://192.168.1.x:8080
```

---

## By the Numbers

| Metric | Value |
|--------|-------|
| Voices | 54+ |
| Emotions | 19 |
| MCP Tools | 40+ |
| Tests | 495+ (98.4% pass rate) |
| Paralinguistic Tags | 9 |
| Languages | 4 (EN, JP, ZH, KO) |

---

## The Journey to 1.0

Voice Soundboard evolved through seven major phases:

1. **v0.1.0** - Core TTS with Kokoro engine
2. **v0.2.0** - Chatterbox integration (paralinguistic tags)
3. **v0.3.0** - Multi-speaker dialogue synthesis
4. **v0.4.0** - Advanced emotion control (VAD, blending)
5. **v0.5.0** - Voice cloning with library management
6. **v0.6.0** - Neural audio codecs for LLM integration
7. **v0.7.0** - Real-time voice conversion
8. **v1.0.0** - LLM integration, mobile access, feature complete

---

## Use Cases

### Customer Service
*"Hello! [cheerful] How can I help you today?"*

Voice bots that sound genuinely friendly, with natural reactions and emotional awareness.

### Accessibility
Screen readers and assistive technologies with expressive, natural speech that conveys meaning through tone.

### Content Creation
Generate audiobooks with multiple characters, each with distinct voices and personalities. Add dramatic pauses, whispers, and emotional emphasis.

### Gaming & Interactive Media
NPCs that laugh at jokes, sigh in frustration, and gasp in surprise. Dynamic emotion curves for narrative moments.

### AI Assistants
Context-aware responses that match the user's emotional state. An AI that sounds genuinely empathetic, not robotic.

---

## Getting Started

```bash
# Install
pip install voice-soundboard[all]

# Download models
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

```python
from voice_soundboard import VoiceEngine, play_audio

engine = VoiceEngine()
result = engine.speak("Welcome to Voice Soundboard 1.0!")
play_audio(result.audio_path)
```

---

## Integration Options

### MCP for AI Agents
```json
{
  "mcpServers": {
    "voice-soundboard": {
      "command": "python",
      "args": ["-m", "voice_soundboard.server"]
    }
  }
}
```

### WebSocket for Real-Time Apps
```javascript
ws.send(JSON.stringify({
  action: "speak",
  text: "Real-time voice synthesis!",
  emotion: "excited"
}));
```

### REST API for Mobile
```bash
curl -X POST http://localhost:8080/api/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from mobile!", "voice": "af_bella"}'
```

---

## Security First

Voice Soundboard is built with security as a priority:

- **XXE Protection** - Safe SSML parsing with defusedxml
- **Path Traversal Prevention** - All file operations validated
- **Voice Cloning Consent** - Required acknowledgment for ethical use
- **Rate Limiting** - Token bucket algorithm prevents abuse
- **WebSocket Security** - Origin validation, API keys, TLS

Full security audit available in SECURITY_AUDIT.md.

---

## Open Source & Community

Voice Soundboard is released under the MIT License. We welcome contributions:

- **GitHub**: github.com/yourusername/voice-soundboard
- **PyPI**: pypi.org/project/voice-soundboard
- **Documentation**: Full API docs, examples, and guides

---

## What's Next?

While 1.0 is feature-complete, development continues:
- Performance optimizations
- Additional TTS backends (F5-TTS, IndexTTS2)
- More languages
- Community-requested features

---

## Quotes

*"Voice Soundboard represents a new paradigm in TTS - where AI doesn't just speak, it expresses."*

*"The paralinguistic tags alone are worth the upgrade. Finally, AI that can laugh naturally."*

*"Multi-speaker dialogue synthesis opens up entirely new possibilities for content creation."*

---

## Quick Facts

- **Release Date**: January 22, 2026
- **Version**: 1.0.0 "2027 Edition"
- **License**: MIT
- **Language**: Python 3.10+
- **Platforms**: Windows, Linux, macOS
- **GPU**: Optional (CUDA for acceleration)

---

<p align="center">
  <strong>Voice Soundboard 1.0</strong><br>
  <em>Because AI should sound as intelligent as it thinks.</em>
</p>

---

## Media Resources

- High-resolution logo: [Coming Soon]
- Demo video: [Coming Soon]
- Live demo: [Coming Soon]

## Contact

For press inquiries, partnerships, or questions:
- Open an issue on GitHub
- Join the community discussions

**###**
