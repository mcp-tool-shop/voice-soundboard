# Claude Collaborate

> *Where Human Creativity Meets AI Intelligence*

Claude Collaborate is a unified sandbox environment for real-time human-AI collaboration. It brings together voice synthesis, interactive workspaces, and seamless communication in one beautiful interface.

![Claude Collaborate](https://img.shields.io/badge/Claude-Collaborate-cc785c?style=for-the-badge&logo=anthropic)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## The Vision

Imagine a workspace where you can:
- **Speak naturally** with an AI that responds with a real voice
- **Draw and brainstorm** on a shared whiteboard
- **Write code together** with instant preview
- **Play chess** and discuss strategy
- **Create content** with GitHub-ready tools

All in one place. All in real-time. All beautifully integrated.

## Features

### Unified Environment Switcher
Switch seamlessly between creative workspaces:
- **Voice Studio** - Full TTS controls with 25+ voices
- **Creative Lab** - Interactive experiments
- **Whiteboard** - Draw, sketch, brainstorm
- **Code Workshop** - HTML/CSS/JS editor with live preview
- **Chess Workshop** - Strategy and tactics playground
- **Capture Viewer** - Screenshots and recordings
- **GitHub Toolkit** - README and marketing generators

### Real-Time Communication
- **WebSocket Bridge** - Instant message delivery
- **Collapsible Chat Panel** - Maximizes your workspace
- **Voice Input** - Speak your thoughts (browser STT)
- **Voice Output** - Claude speaks back (Kokoro TTS)

### Single Server Architecture
Everything runs on one unified server:
```
http://localhost:8080/collaborate  - Main UI
http://localhost:8080/studio       - Voice Studio
http://localhost:8080/adventures   - Creative Lab
ws://localhost:8080/ws            - WebSocket Bridge
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mcp-tool-shop/claude-collaborate.git
cd claude-collaborate

# Install dependencies
pip install aiohttp

# Start the unified server
python -m voice_soundboard.web_server

# Open in browser
# http://localhost:8080/collaborate
```

## Architecture

```
Claude Collaborate
├── index.html           # Main UI with environment switcher
├── whiteboard.html      # Drawing and brainstorming
├── code-playground.html # Live HTML/CSS/JS editor
├── chess.html           # Chess analysis board
├── capture-viewer.html  # Screenshot/recording viewer
├── github-toolkit.html  # README and marketing tools
└── template.html        # Starter for new environments
```

### Server Integration

The unified `web_server.py` handles:
- Static file serving for all environments
- REST API for voice synthesis
- WebSocket bridge for real-time messaging
- Health checks and status endpoints

### WebSocket Protocol

```javascript
// Send message to Claude
{ "type": "user_message", "content": "Hello!" }

// Receive response
{ "type": "claude_response", "content": "Hi there!" }

// Connection status
{ "type": "connected", "message": "Connected to Claude Collaborate Bridge" }
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collaborate` | GET | Main Claude Collaborate UI |
| `/collaborate/{file}` | GET | Static files (whiteboard, etc.) |
| `/api/speak` | POST | Text-to-speech synthesis |
| `/api/ws/messages` | GET | Read pending user messages |
| `/api/ws/respond` | POST | Send response to browser |
| `/api/ws/status` | GET | WebSocket bridge status |
| `/health` | GET | Server health check |

## For Claude Code Users

Claude Collaborate integrates with Claude Code via the WebSocket bridge:

```bash
# Read messages from the UI
curl http://localhost:8080/api/ws/messages

# Send a response back
curl -X POST http://localhost:8080/api/ws/respond \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello from Claude!"}'
```

## Creating New Environments

1. Copy `template.html` to `your-environment.html`
2. Add it to the sidebar in `index.html`:
```html
<div class="env-item" data-url="/collaborate/your-environment.html" data-name="Your Environment">
    <div class="env-icon">...</div>
    <div class="env-info">
        <h3>Your Environment</h3>
        <p>Description</p>
    </div>
</div>
```
3. Refresh and start building!

## Requirements

- Python 3.10+
- aiohttp
- Modern browser with WebSocket support
- (Optional) Kokoro TTS for voice synthesis

## Contributing

We welcome contributions! Whether it's:
- New environment templates
- UI/UX improvements
- Bug fixes
- Documentation

Please open an issue or submit a PR.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Anthropic** - For Claude and the vision of helpful AI
- **Kokoro TTS** - Beautiful voice synthesis
- **The Community** - For pushing the boundaries of human-AI collaboration

---

<p align="center">
  <i>Built with love for the future of collaboration</i><br>
  <a href="https://github.com/mcp-tool-shop">MCP Tool Shop</a>
</p>
