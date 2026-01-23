# Voice Soundboard Examples

This directory contains example scripts and demos for Voice Soundboard.

## Quick Start

```bash
# Install voice-soundboard
pip install voice-soundboard[all]

# Download models (required)
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..

# Run any example
python examples/01_basic_speech.py
```

## Python Examples

| # | File | Description |
|---|------|-------------|
| 01 | `01_basic_speech.py` | Core TTS: voices, presets, emotions, natural language styles |
| 02 | `02_paralinguistic_tags.py` | Natural sounds: [laugh], [sigh], [gasp], emotion exaggeration |
| 03 | `03_multi_speaker_dialogue.py` | Conversations, stage directions, auto voice assignment |
| 04 | `04_voice_cloning.py` | Clone and use custom voices, cross-language synthesis |
| 05 | `05_emotion_control.py` | VAD model, emotion blending, word-level tags, curves |
| 06 | `06_streaming.py` | Real-time low-latency streaming with callbacks |
| 07 | `07_sound_effects.py` | Built-in sound effects for notifications and UI |
| 08 | `08_ssml.py` | SSML markup for fine-grained control |
| 09 | `09_websocket_client.py` | Python WebSocket client for real-time API |
| 10 | `10_websocket_html_client.html` | Browser-based WebSocket client demo |
| 11 | `11_mcp_integration.py` | MCP tools demo for AI agent integration |

## Interactive Demos

### Google Colab Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcp-tool-shop/voice-soundboard/blob/main/examples/voice_soundboard_demo.ipynb)

Run the full demo in your browser without any local setup.

```
examples/voice_soundboard_demo.ipynb
```

### Mobile Web Interface
Access from any phone or tablet:

```bash
# Start the web server
python -m voice_soundboard.web_server

# Open in browser: http://localhost:8080
# Or from your phone: http://<your-ip>:8080
```

### WebSocket Browser Demo
Open the HTML file directly in your browser:

```bash
# Start the WebSocket server
python -m voice_soundboard.websocket_server

# Open the HTML client
open examples/10_websocket_html_client.html
```

### MCP Integration (Claude Desktop)
Add to your Claude Desktop config:

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

Then ask Claude to speak!

## Requirements

### Core
- Python 3.10+
- Voice Soundboard: `pip install voice-soundboard`
- Kokoro models (see Quick Start)

### Optional Dependencies
| Feature | Install Command |
|---------|-----------------|
| Chatterbox (paralinguistic tags) | `pip install voice-soundboard[chatterbox]` |
| WebSocket server | `pip install voice-soundboard[websocket]` |
| Mobile web UI | `pip install voice-soundboard[web]` |
| MCP server | `pip install voice-soundboard[mcp]` |
| Everything | `pip install voice-soundboard[all]` |

## Running All Examples

### Linux/macOS
```bash
for i in 01 02 03 04 05 06 07 08 09 11; do
    echo "Running example $i..."
    python examples/${i}_*.py
    echo ""
done
```

### Windows
```powershell
foreach ($i in 1..11) {
    $file = Get-ChildItem "examples\$($i.ToString('00'))_*.py" -ErrorAction SilentlyContinue
    if ($file) {
        Write-Host "Running $($file.Name)..."
        python $file.FullName
    }
}
```

## Example Descriptions

### 01_basic_speech.py
Demonstrates core functionality:
- Simple text-to-speech
- Voice selection (54+ voices)
- Preset usage (assistant, narrator, announcer, etc.)
- Emotion application (happy, sad, excited, etc.)
- Natural language style hints

### 02_paralinguistic_tags.py
Chatterbox integration for natural non-speech sounds:
- Paralinguistic tags: [laugh], [sigh], [gasp], [cough]
- Emotion exaggeration control (0.0 - 1.0)
- Combining tags with emotion

### 03_multi_speaker_dialogue.py
Multi-speaker conversation synthesis:
- Script parsing with speaker tags
- Stage directions (whispering, angrily, etc.)
- Automatic voice assignment
- Turn pause control

### 04_voice_cloning.py
Voice cloning features:
- Clone from 3-10 second sample
- Consent tracking for ethical use
- Cross-language synthesis
- Voice library management

### 05_emotion_control.py
Advanced emotion features:
- Word-level emotion tags {happy}text{/happy}
- VAD model (Valence-Arousal-Dominance)
- Emotion blending
- Dynamic emotion curves

### 06_streaming.py
Real-time streaming:
- Low-latency generation
- Chunk callbacks
- Progress tracking
- Stream to file

### 07_sound_effects.py
Built-in effects:
- Notification sounds (chime, success, error)
- UI sounds (click, pop, whoosh)
- Ambient sounds (rain, white noise)
- Saving effects to files

### 08_ssml.py
SSML support:
- Pauses with `<break>`
- Speed control with `<prosody>`
- Emphasis with `<emphasis>`
- Special formatting with `<say-as>`

### 09_websocket_client.py
WebSocket API client:
- Connection handling
- Speech generation
- Streaming
- Sound effects
- Authentication

### 10_websocket_html_client.html
Browser-based WebSocket demo:
- Interactive UI
- Voice selection
- Emotion control
- Multi-speaker dialogue
- Message logging

### 11_mcp_integration.py
MCP tools demonstration:
- All 40+ tools explained
- Equivalent Python code
- Configuration examples

## Creating Your Own

```python
from voice_soundboard import VoiceEngine, play_audio

engine = VoiceEngine()

# Your custom synthesis code here
result = engine.speak(
    "Hello from my custom script!",
    style="warmly and cheerfully",
    voice="af_bella"
)
play_audio(result.audio_path)
```

## Troubleshooting

### Models not found
```
Error: Model file not found
```
**Solution**: Download models to the `models/` directory (see Quick Start)

### Chatterbox not available
```
CHATTERBOX_AVAILABLE = False
```
**Solution**: `pip install voice-soundboard[chatterbox]`

### WebSocket connection refused
```
Connection refused
```
**Solution**: Start the server first: `python -m voice_soundboard.websocket_server`

### No audio playback
```
No audio device found
```
**Solution**: Check your audio output settings and permissions

## Need Help?

- Check the main [README.md](../README.md)
- See [CONTRIBUTING.md](../CONTRIBUTING.md)
- Review [SECURITY_AUDIT.md](../SECURITY_AUDIT.md)
- Open an issue on GitHub
