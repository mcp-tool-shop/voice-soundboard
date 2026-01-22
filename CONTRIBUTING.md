# Contributing to Voice Soundboard

Thank you for your interest in contributing to Voice Soundboard! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something great together.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include reproduction steps and environment details
4. Provide error messages and tracebacks

### Suggesting Features

1. Check existing feature requests
2. Use the feature request template
3. Explain the use case and benefits
4. Consider implementation complexity

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/voice-soundboard.git
cd voice-soundboard

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e ".[dev]"

# Download models
mkdir models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Code Style

- Use type hints for all functions
- Follow PEP 8 guidelines
- Keep functions focused and small
- Write docstrings for public APIs
- Add comments for complex logic

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=voice_soundboard

# Run specific test
pytest tests/test_soundboard.py::TestSecurity -v
```

### Writing Tests

- Test files go in `tests/`
- Use descriptive test names
- Test both success and error cases
- Mock external dependencies

## Security

- Never commit credentials or API keys
- Validate all user inputs
- Use `sanitize_filename()` for file operations
- Report security issues privately

## Documentation

- Update README for new features
- Add docstrings to new functions
- Update CHANGELOG for notable changes

## Commit Messages

Format: `type: brief description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes

Examples:
```
feat: add voice cloning support
fix: handle empty SSML input
docs: update WebSocket examples
test: add security validation tests
```

## Questions?

Open a discussion or issue. We're happy to help!

---

Thank you for contributing!
