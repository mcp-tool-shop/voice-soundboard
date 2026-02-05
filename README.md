# Voice Soundboard

[![PyPI version](https://badge.fury.io/py/voice-soundboard.svg)](https://badge.fury.io/py/voice-soundboard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

**Text-to-speech for AI agents and developers.** Generate natural-sounding audio from text with one function call.

---

## Quick Start (60 seconds)

### 1. Install

```bash
pip install voice-soundboard
```

### 2. Download models

```bash
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 3. Generate speech

```python
from voice_soundboard import VoiceEngine

engine = VoiceEngine()
result = engine.speak("Hello world!")
print(result.audio_path)  # output/af_bella_<hash>.wav
```

That's it. You now have a `.wav` file you can play with any audio player.

### Or use the CLI

```bash
voice-soundboard speak "Hello world!"
# -> output/hello_world.wav
```

---

## What This Is

Voice Soundboard is a Python library that converts text to speech. It wraps the [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) TTS engine and exposes it through:

- A **Python API** (`VoiceEngine.speak()`)
- A **CLI** (`voice-soundboard speak "..."`)
- An **MCP server** (40+ tools for AI agents like Claude)

It includes 54+ voices, 19 emotions, natural language style control, and optional advanced features like voice cloning and multi-speaker dialogue.

## What This Is NOT

- Not a voice assistant or chatbot
- Not a real-time conversation system (though it supports streaming)
- Not a cloud service -- everything runs locally on your machine
- Not a recording or audio editing tool

## Who This Is For

- **AI agent developers** who want their agents to speak (via MCP)
- **Python developers** who need text-to-speech in their applications
- **Content creators** generating voiceovers, podcasts, or audiobooks
- **Accessibility developers** building screen readers or assistive tools

---

## Core Features

```python
from voice_soundboard import VoiceEngine

engine = VoiceEngine()

# Pick a voice (54+ available)
result = engine.speak("Cheerio!", voice="bm_george")  # British male

# Use a preset
result = engine.speak("Breaking news!", preset="announcer")

# Set an emotion
result = engine.speak("I'm so happy!", emotion="excited")

# Describe the style in plain English
result = engine.speak("Good morning!", style="warmly and cheerfully")
```

### Voice Presets

| Preset | Voice | Speed | Style |
|--------|-------|-------|-------|
| assistant | af_bella | 1.0 | Friendly, conversational |
| narrator | bm_george | 0.95 | Calm, documentary |
| announcer | am_michael | 1.1 | Bold, energetic |
| storyteller | bf_emma | 0.9 | Expressive, varied |
| whisper | af_nicole | 0.85 | Soft, gentle |

### Emotions

happy, sad, angry, excited, calm, fearful, surprised, disgusted, contemptuous, tender, proud, ashamed, guilty, anxious, nostalgic, hopeful, determined, confused, amused -- 19 total.

---

## Optional Engines

Voice Soundboard supports additional TTS engines for advanced use cases. These are **not required** for basic usage.

| Engine | Install | What It Adds |
|--------|---------|--------------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | `pip install voice-soundboard[chatterbox]` | Paralinguistic tags (`[laugh]`, `[sigh]`), 23 languages, emotion exaggeration |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | `pip install voice-soundboard[f5tts]` | Zero-shot voice cloning from 3-10s audio samples |

---

## MCP Server (For AI Agents)

Voice Soundboard exposes 40+ tools via the [Model Context Protocol](https://modelcontextprotocol.io/). Add it to Claude Desktop:

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

Then ask Claude: *"Say hello in an excited voice"* -- and it will generate audio.

Install the MCP dependency: `pip install voice-soundboard[mcp]`

---

## More Examples

See [`examples/`](examples/) for runnable scripts covering every feature:

| Example | What It Shows |
|---------|---------------|
| [`hello_world.py`](examples/hello_world.py) | **Start here.** Minimal working example. |
| [`01_basic_speech.py`](examples/01_basic_speech.py) | Voices, presets, emotions, styles |
| [`02_paralinguistic_tags.py`](examples/02_paralinguistic_tags.py) | `[laugh]`, `[sigh]` with Chatterbox |
| [`03_multi_speaker_dialogue.py`](examples/03_multi_speaker_dialogue.py) | Multi-speaker conversations |
| [`04_voice_cloning.py`](examples/04_voice_cloning.py) | Clone a voice from audio |
| [`05_emotion_control.py`](examples/05_emotion_control.py) | Emotion blending and curves |

---

## Installation Options

```bash
pip install voice-soundboard              # Core (Kokoro engine)
pip install voice-soundboard[mcp]         # + MCP server for AI agents
pip install voice-soundboard[chatterbox]  # + Paralinguistic tags & 23 languages
pip install voice-soundboard[f5tts]       # + F5-TTS voice cloning
pip install voice-soundboard[websocket]   # + WebSocket server
pip install voice-soundboard[web]         # + Mobile web UI
pip install voice-soundboard[all]         # Everything
```

### Requirements

- Python 3.10+
- ~350MB for Kokoro models
- Optional: CUDA GPU for faster generation

---

## Advanced Topics

These features are stable but beyond the scope of getting started:

- [Voice cloning](examples/04_voice_cloning.py) -- clone any voice from a short audio sample
- [Multi-speaker dialogue](examples/03_multi_speaker_dialogue.py) -- generate conversations with multiple voices
- [SSML support](examples/08_ssml.py) -- fine-grained control with Speech Synthesis Markup Language
- [WebSocket server](examples/09_websocket_client.py) -- real-time bidirectional API
- [Mobile web UI](examples/README.md) -- access from phone/tablet
- [Vocology module](docs/research/vocology/) -- voice humanization, formant analysis, rhythm metrics

---

## Security

Voice Soundboard is security-hardened. See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for details.

- Path traversal protection on all file operations
- XXE protection via defusedxml for SSML parsing
- Rate limiting, input validation, safe error messages
- Voice cloning requires explicit consent acknowledgment

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run tests with:

```bash
pytest tests/ -v
```

## License

MIT -- see [LICENSE](LICENSE).

## Acknowledgments

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) -- primary TTS engine
- [Chatterbox](https://github.com/resemble-ai/chatterbox) -- paralinguistic tags & multilingual
- [F5-TTS](https://github.com/SWivid/F5-TTS) -- Diffusion Transformer voice cloning
