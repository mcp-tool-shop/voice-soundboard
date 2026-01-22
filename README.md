# Voice Soundboard

[![PyPI version](https://badge.fury.io/py/voice-soundboard.svg)](https://badge.fury.io/py/voice-soundboard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: Audited](https://img.shields.io/badge/security-audited-green.svg)](SECURITY_AUDIT.md)

**AI-powered voice synthesis with natural language control.** Let AI agents speak naturally with 50+ voices, emotions, streaming, and real-time playback.

## Highlights

- **Natural Language Styles** - "say this warmly", "excitedly", "like a narrator"
- **50+ Voices** - American, British, Japanese, Mandarin accents
- **19 Emotions** - happy, sad, angry, excited, calm, and more
- **5 Presets** - assistant, narrator, announcer, storyteller, whisper
- **SSML Support** - `<break>`, `<prosody>`, `<emphasis>`, `<say-as>`
- **Real-time Streaming** - Low-latency audio generation
- **Sound Effects** - 13 built-in effects (chime, success, error, etc.)
- **MCP Integration** - AI agents can speak via tool calls
- **WebSocket API** - Real-time bidirectional communication
- **Security Hardened** - Path validation, rate limiting, XXE protection

## Installation

```bash
# From PyPI
pip install voice-soundboard

# From source
git clone https://github.com/yourusername/voice-soundboard.git
cd voice-soundboard
pip install -e .

# With optional dependencies
pip install voice-soundboard[websocket]  # WebSocket server
pip install voice-soundboard[mcp]        # MCP server
pip install voice-soundboard[all]        # Everything
```

### Model Download (Required)

```bash
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Quick Start

### Python API

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
from voice_soundboard import parse_ssml, VoiceEngine

ssml = '''
<speak>
  Hello <break time="500ms"/> world!
  <prosody rate="slow">This is slower.</prosody>
  The date is <say-as interpret-as="date">2024-01-15</say-as>.
</speak>
'''

text, params = parse_ssml(ssml)
engine = VoiceEngine()
result = engine.speak(text, speed=params.speed)
```

### Sound Effects

```python
from voice_soundboard import get_effect, play_effect, list_effects

# List available effects
print(list_effects())  # ['chime', 'success', 'error', 'attention', ...]

# Play an effect
play_effect("success")

# Save an effect
effect = get_effect("chime")
effect.save("notification.wav")
```

### Emotions

```python
from voice_soundboard import get_emotion_params, get_emotion_voice_params

# Get emotion parameters
params = get_emotion_params("excited")
print(f"Speed: {params.speed}, Voice: {params.voice_preference}")

# Apply emotion to voice
voice_params = get_emotion_voice_params("happy", voice="af_bella", speed=1.0)
```

## MCP Server (For AI Agents)

Voice Soundboard exposes tools for AI agents via the Model Context Protocol.

### Available Tools

| Tool | Description |
|------|-------------|
| `speak` | Generate speech with natural language control |
| `speak_long` | Stream long text efficiently |
| `speak_ssml` | Process SSML markup |
| `list_voices` | List 50+ voices with filtering |
| `list_presets` | Show preset configurations |
| `list_effects` | List sound effects |
| `list_emotions` | List available emotions |
| `sound_effect` | Play/save sound effects |
| `play_audio` | Play audio file |
| `stop_audio` | Stop playback |

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

// Stream speech
ws.send(JSON.stringify({
  action: "speak_stream",
  text: "Long text to stream...",
  emotion: "happy"
}));

// Play effect
ws.send(JSON.stringify({
  action: "effect",
  effect: "chime"
}));
```

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

## Security

Voice Soundboard is security-hardened:

- **Path Traversal Protection** - All file paths validated
- **XXE Protection** - Uses defusedxml for SSML parsing
- **Rate Limiting** - Token bucket algorithm
- **Input Validation** - Length limits, type checking
- **WebSocket Security** - Origin validation, API key auth, TLS support
- **Safe Error Messages** - No internal paths exposed

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for the full security audit.

## Architecture

```
voice_soundboard/
├── engine.py          # VoiceEngine - core TTS synthesis
├── streaming.py       # Real-time streaming with low latency
├── server.py          # MCP server with 11 tools
├── websocket_server.py # WebSocket API server
├── interpreter.py     # Natural language → parameters
├── emotions.py        # Emotion parameters and text modification
├── ssml.py            # SSML parsing with defusedxml
├── effects.py         # Sound effect generation
├── security.py        # Security utilities
├── audio.py           # Playback with path validation
└── config.py          # Voices, presets, settings
```

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

**Test Results**: 254 tests, 98% pass rate. See [TEST_PLAN.md](TEST_PLAN.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) - High-quality TTS engine
- [defusedxml](https://github.com/tiran/defusedxml) - Secure XML parsing
