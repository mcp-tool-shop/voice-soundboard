# Voice Soundboard v1.2.0 - "Voice Science" Release

**FOR IMMEDIATE RELEASE**

January 23, 2026

---

## Make AI Voices Sound Human: Voice Soundboard Introduces Vocology Module

**Voice Soundboard v1.2.0** brings voice science to AI speech synthesis with the new **Vocology Module** - a comprehensive toolkit for analyzing and humanizing synthetic voices.

### The Problem

TTS engines produce technically accurate speech, but it often sounds robotic. Missing are the subtle imperfections that make human speech natural: micro-variations in pitch, natural breathing patterns, and rhythm variations that convey emotion and engagement.

### The Solution

The Vocology Module bridges this gap with research-backed voice manipulation:

**Voice Humanization (scored 9/10 in testing)**
- Intelligent breath insertion at phrase boundaries
- Pitch micro-jitter and drift for natural variation
- 7 emotional presets: EXCITED, CALM, TIRED, ANXIOUS, CONFIDENT, INTIMATE
- One-line API: `humanizer.humanize(audio, sample_rate=24000)`

**Formant Shifting (scored 9/10 in testing)**
- Make any voice deeper or brighter while preserving identity
- Simple ratio control: 0.9 = deeper, 1.1 = brighter
- Perfect for character voices and voice aging effects

**Rhythm Analysis**
- nPVI metrics for speech rhythm classification
- Rhythm Zone Theory (RZT) analysis
- Automatic classification: stress-timed, syllable-timed, mora-timed
- Research-grade metrics matching published linguistic studies

### Quick Example

```python
from voice_soundboard.vocology import VoiceHumanizer, FormantShifter

# Generate TTS audio (any engine)
engine = VoiceEngine()
result = engine.speak("Hello, this is a test.")

# Humanize it
humanizer = VoiceHumanizer()
audio, sr = humanizer.humanize(result.audio, sample_rate=24000)

# Make it deeper for a narrator effect
shifter = FormantShifter()
narrator_voice, sr = shifter.shift(audio, ratio=0.92, sample_rate=sr)
```

### By The Numbers

| Metric | Value |
|--------|-------|
| New Python modules | 8 |
| Documentation pages | 13 |
| New tests | 165 |
| Total tests | 1,195 |
| Humanization score | 9/10 |
| Formant shifting score | 9/10 |

### What's Next

Version 1.3.0 will introduce the **Preset Library** - curated voice characters combining humanization, formants, and emotion for instant narrator, character, and emotional voices. See the [Preset Roadmap](docs/vocology/PRESET_ROADMAP.md).

### Installation

```bash
pip install voice-soundboard --upgrade

# Or from source
git clone https://github.com/mcp-tool-shop/voice-soundboard.git
cd voice-soundboard && pip install -e .
```

### Links

- **GitHub**: https://github.com/mcp-tool-shop/voice-soundboard
- **Documentation**: https://github.com/mcp-tool-shop/voice-soundboard/tree/master/docs/vocology
- **Changelog**: https://github.com/mcp-tool-shop/voice-soundboard/blob/master/CHANGELOG.md

### About Voice Soundboard

Voice Soundboard is an open-source AI voice synthesis platform with 54+ voices, 23 languages, voice cloning, multi-speaker dialogue, and MCP integration for AI agents. Built for developers who want expressive, human-like voices in their applications.

---

**Contact**: https://github.com/mcp-tool-shop/voice-soundboard/issues

**License**: MIT
