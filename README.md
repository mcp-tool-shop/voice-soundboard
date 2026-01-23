# Voice Soundboard

[![PyPI version](https://badge.fury.io/py/voice-soundboard.svg)](https://badge.fury.io/py/voice-soundboard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: Audited](https://img.shields.io/badge/security-audited-green.svg)](SECURITY_AUDIT.md)
[![Tests: 495+](https://img.shields.io/badge/tests-495%2B%20passing-brightgreen.svg)](TEST_PLAN.md)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

**AI-powered voice synthesis with natural language control.** Give AI agents expressive, human-like voices with 54+ voices, 19 emotions, real-time streaming, voice cloning, and multi-speaker dialogue.

```python
engine.speak("Hello! How can I help you today?", style="warmly and cheerfully")
```

---

## Highlights

| Feature | Description |
|---------|-------------|
| **Natural Language Styles** | "say this warmly", "excitedly", "like a narrator" |
| **54+ Voices** | American, British, Japanese, Mandarin accents |
| **19 Emotions** | happy, sad, angry, excited, calm, and more |
| **Paralinguistic Tags** | `[laugh]`, `[sigh]`, `[gasp]` - natural non-speech sounds |
| **Multi-Speaker Dialogue** | Generate podcasts, audiobooks, conversations |
| **Voice Cloning** | Clone any voice from 3-10 seconds of audio |
| **Real-time Streaming** | Sub-100ms latency for interactive applications |
| **Mobile Web UI** | Access from phone/tablet via responsive web interface |
| **MCP Integration** | 40+ tools for AI agent integration |
| **Security Hardened** | Path validation, rate limiting, XXE protection |

---

## Installation

```bash
# From PyPI
pip install voice-soundboard

# From source
git clone https://github.com/yourusername/voice-soundboard.git
cd voice-soundboard
pip install -e .

# With optional features
pip install voice-soundboard[websocket]   # WebSocket server
pip install voice-soundboard[web]         # Mobile web UI
pip install voice-soundboard[mcp]         # MCP server
pip install voice-soundboard[chatterbox]  # Paralinguistic tags & voice cloning
pip install voice-soundboard[all]         # Everything
```

### Model Download (Required)

```bash
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

---

## Quick Start

### Basic Speech

```python
from voice_soundboard import VoiceEngine, play_audio

engine = VoiceEngine()

# Simple speech
result = engine.speak("Hello world!")
play_audio(result.audio_path)

# With preset
result = engine.speak("Breaking news!", preset="announcer")

# With emotion
result = engine.speak("I'm so happy!", emotion="excited")

# With natural language style
result = engine.speak("Good morning!", style="warmly and cheerfully")

# With specific voice
result = engine.speak("Cheerio!", voice="bm_george")  # British male
```

### Paralinguistic Tags (Chatterbox)

```python
from voice_soundboard.engines import ChatterboxEngine

engine = ChatterboxEngine()

# Natural non-speech sounds
result = engine.speak(
    "That's hilarious! [laugh] Oh man, [sigh] I needed that.",
    emotion_exaggeration=0.7  # 0.0 = monotone, 1.0 = dramatic
)
```

Supported tags: `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[cough]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`

### Multi-Speaker Dialogue

```python
from voice_soundboard import DialogueEngine

engine = DialogueEngine()

script = """
[S1:narrator] The door creaked open slowly.
[S2:alice] Hello? Is anyone there? [gasp]
[S3:bob] (whispering) Don't go in there...
"""

result = await engine.speak_dialogue(
    script,
    voices={"narrator": "bm_george", "alice": "af_bella", "bob": "am_michael"}
)
```

### Voice Cloning

```python
from voice_soundboard import VoiceCloner

cloner = VoiceCloner()

# Clone from 3-10 second sample
my_voice = cloner.clone("my_sample.wav", consent_given=True)

# Use the cloned voice
engine.speak("Hello, this is my cloned voice!", voice=my_voice.voice_id)

# Cross-language cloning
cloner.speak("Bonjour le monde!", voice=my_voice.voice_id, language="fr")
```

### Emotion Control

```python
from voice_soundboard import blend_emotions, EmotionCurve

# Blend emotions
result = blend_emotions([("happy", 0.7), ("surprised", 0.3)])
print(result.closest_emotion)  # "pleasantly surprised"

# Dynamic emotion curves
curve = EmotionCurve()
curve.add_point(0.0, "worried")
curve.add_point(0.5, "neutral")
curve.add_point(1.0, "excited")

# Word-level emotion tags
engine.speak(
    "I was so {happy}excited{/happy} to see you, but then {sad}you left{/sad}."
)
```

### Streaming (Low Latency)

```python
from voice_soundboard import stream_realtime

# Stream with real-time playback
result = await stream_realtime(
    "This is a long text that will be streamed...",
    voice="af_bella",
    speed=1.0
)
print(f"Generated {result.total_chunks} chunks in {result.generation_time:.2f}s")
```

### SSML Support

```python
from voice_soundboard import parse_ssml

ssml = '''
<speak>
  Hello <break time="500ms"/> world!
  <prosody rate="slow">This is slower.</prosody>
  The date is <say-as interpret-as="date">2024-01-15</say-as>.
</speak>
'''

text, params = parse_ssml(ssml)
result = engine.speak(text, speed=params.speed)
```

### Sound Effects

```python
from voice_soundboard import play_effect, list_effects

# List available effects
print(list_effects())  # ['chime', 'success', 'error', 'attention', ...]

# Play an effect
play_effect("success")
```

---

## Mobile Web Interface

Access Voice Soundboard from any phone or tablet with the built-in web server:

```bash
# Start the web server
python -m voice_soundboard.web_server

# Output:
# Voice Soundboard Web Server
# Local:   http://localhost:8080
# Network: http://192.168.1.100:8080  <- Use this on your phone!
```

**Features:**
- Responsive design for phones and tablets
- PWA support (add to home screen)
- Voice selection grid with language filters
- Quick phrase buttons
- Sound effects panel
- Real-time WebSocket with REST fallback

---

## MCP Server (For AI Agents)

Voice Soundboard exposes 40+ tools for AI agents via the Model Context Protocol.

### Core Tools

| Tool | Description |
|------|-------------|
| `speak` | Generate speech with natural language control |
| `speak_long` | Stream long text efficiently |
| `speak_ssml` | Process SSML markup |
| `speak_chatterbox` | Paralinguistic tags and emotion control |
| `speak_dialogue` | Multi-speaker conversation synthesis |
| `speak_realtime` | Ultra-low latency streaming |
| `speak_with_context` | Context-aware emotion selection |

### Voice & Emotion Tools

| Tool | Description |
|------|-------------|
| `list_voices` | List 54+ voices with filtering |
| `list_presets` | Show preset configurations |
| `list_emotions` | List available emotions |
| `blend_emotions` | Mix emotions with weights |
| `clone_voice` | Clone voice from audio sample |
| `list_voice_library` | Browse cloned voices |

### Utility Tools

| Tool | Description |
|------|-------------|
| `sound_effect` | Play/save sound effects |
| `play_audio` | Play audio file |
| `stop_audio` | Stop playback |
| `encode_audio_tokens` | Convert audio to LLM tokens |
| `start_voice_conversion` | Real-time voice changing |

### Claude Desktop Configuration

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

---

## WebSocket Server

Real-time bidirectional API with security features.

```bash
# Start server
python -m voice_soundboard.websocket_server

# With authentication
VOICE_API_KEY=secret123 python -m voice_soundboard.websocket_server

# With TLS
VOICE_SSL_CERT=cert.pem VOICE_SSL_KEY=key.pem python -m voice_soundboard.websocket_server
```

### WebSocket Actions

```javascript
// Connect
const ws = new WebSocket("ws://localhost:8765?key=secret123");

// Generate speech
ws.send(JSON.stringify({
  action: "speak",
  text: "Hello world!",
  voice: "af_bella",
  play: true
}));

// Multi-speaker dialogue
ws.send(JSON.stringify({
  action: "dialogue",
  script: "[S1:alice] Hello! [S2:bob] Hi there!",
  voices: {"alice": "af_bella", "bob": "am_michael"}
}));
```

---

## Voice Presets

| Preset | Voice | Speed | Description |
|--------|-------|-------|-------------|
| assistant | af_bella | 1.0 | Friendly, helpful, conversational |
| narrator | bm_george | 0.95 | Calm, clear, documentary style |
| announcer | am_michael | 1.1 | Bold, energetic, broadcast style |
| storyteller | bf_emma | 0.9 | Expressive, varied pacing |
| whisper | af_nicole | 0.85 | Soft, intimate, gentle |

## Emotions

| Emotion | Speed | Description |
|---------|-------|-------------|
| happy | 1.1 | Upbeat, cheerful |
| sad | 0.85 | Slower, softer |
| angry | 1.15 | Intense, forceful |
| excited | 1.25 | Fast, energetic |
| calm | 0.9 | Steady, relaxed |
| ... | ... | 19 total emotions |

## Natural Language Styles

The interpreter understands:
- **Speed**: "quickly", "slowly", "carefully"
- **Tone**: "warmly", "excitedly", "calmly", "mysteriously"
- **Character**: "like a narrator", "like an announcer"
- **Gender**: "in a male voice", "in a female voice"
- **Accent**: "with a british accent", "american"

Combine them: `"cheerfully in a british accent, slowly"`

---

## Architecture

```
voice_soundboard/
├── engine.py              # VoiceEngine - core TTS synthesis
├── engines/               # TTS backends (Kokoro, Chatterbox)
├── dialogue/              # Multi-speaker dialogue synthesis
├── emotion/               # Advanced emotion control (VAD, blending)
├── cloning/               # Voice cloning & library management
├── codecs/                # Neural audio codecs (Mimi, DualCodec)
├── conversion/            # Real-time voice conversion
├── llm/                   # LLM integration utilities
├── streaming.py           # Real-time streaming
├── server.py              # MCP server (40+ tools)
├── websocket_server.py    # WebSocket API server
├── web_server.py          # Mobile web UI server
├── web/                   # HTML/CSS/JS for mobile interface
├── interpreter.py         # Natural language -> parameters
├── emotions.py            # Emotion parameters
├── ssml.py                # SSML parsing with defusedxml
├── effects.py             # Sound effect generation
├── normalizer.py          # Text normalization for TTS
├── security.py            # Security utilities
├── audio.py               # Playback with path validation
└── config.py              # Voices, presets, settings
```

---

## Security

Voice Soundboard is security-hardened:

- **Path Traversal Protection** - All file paths validated
- **XXE Protection** - Uses defusedxml for SSML parsing
- **Rate Limiting** - Token bucket algorithm
- **Input Validation** - Length limits, type checking
- **WebSocket Security** - Origin validation, API key auth, TLS support
- **Voice Cloning Consent** - Required acknowledgment for ethical use
- **Safe Error Messages** - No internal paths exposed

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for the full security audit.

---

## Requirements

- Python 3.10+
- ~350MB for models
- Optional: CUDA-capable GPU for acceleration

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=voice_soundboard
```

**Test Results**: 495+ tests, 98.4% pass rate. See [TEST_PLAN.md](TEST_PLAN.md).

---

## Use Cases

### Customer Service Bots
Give chatbots warm, professional voices that build trust with customers.

### Accessibility Tools
Create screen readers and assistive technologies with natural speech.

### Content Creation
Generate voiceovers for videos, podcasts, and presentations.

### Gaming & Interactive Media
Bring NPCs and characters to life with expressive dialogue.

### Audiobooks & Podcasts
Multi-speaker dialogue synthesis for immersive audio content.

### Smart Home Assistants
Build custom voice interfaces for IoT devices.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development roadmap.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) - High-quality TTS engine
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Paralinguistic tags
- [defusedxml](https://github.com/tiran/defusedxml) - Secure XML parsing

---

<p align="center">
  <strong>Voice Soundboard</strong> - Because AI should sound as intelligent as it thinks.
</p>
