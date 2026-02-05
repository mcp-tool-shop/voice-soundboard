# System Overview

A technical reviewer should understand Voice Soundboard in 10 minutes
by reading this page.

---

## What It Is

A Python library that converts text to speech. Runs entirely locally.
The primary engine is [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
(82M parameters, 24kHz WAV output, 2-6x real-time on GPU).

## How It Works

```
                    ┌──────────────────────────────────────────┐
                    │              VoiceEngine                  │
                    │                                          │
  "Hello world!"   │  Normalize ─> Interpret ─> Synthesize    │   hello.wav
  emotion="happy"  │     │            │             │          │
  preset="narrator" │  expand $,   resolve     Kokoro ONNX    │
                    │  URLs, etc   voice+speed  model call     │
                    │                                          │
                    └──────────────────────────────────────────┘
```

Every `speak()` call goes through three stages:

1. **Normalize** — Expand abbreviations, currency, numbers, emojis into
   speakable text. Pure function, no side effects.

2. **Interpret** — Resolve which voice and speed to use. Priority order
   (highest wins): explicit `voice=` > `preset=` > `style=` > `emotion=` > defaults.

3. **Synthesize** — Call the Kokoro ONNX model. Returns PCM float32 samples
   at 24kHz. Save to WAV.

The output is always a `SpeechResult` with an `audio_path` pointing to
a `.wav` file on disk.

## Where Extension Happens

```
voice_soundboard/
├── engine.py            Core: VoiceEngine.speak() ← start here
├── config.py            Voice/preset definitions, paths
├── emotions.py          Emotion → voice parameter mapping
├── interpreter.py       Natural language style → parameters
│
├── engines/             Alternative TTS backends
│   ├── kokoro.py        Default (82M params, fast)
│   ├── chatterbox.py    Paralinguistic tags, 23 languages
│   └── f5tts.py         Diffusion Transformer voice cloning
│
├── dialogue/            Multi-speaker dialogue synthesis
├── emotion/             Advanced: VAD model, blending, curves
├── cloning/             Voice cloning from audio samples
├── presets/             50+ curated voice presets
│
├── server.py            MCP server (40+ tools for AI agents)
├── websocket_server.py  Real-time WebSocket API
├── web_server.py        HTTP server for web/mobile access
└── cli.py               Command-line interface
```

**To add a voice:** edit `config.py` → `KOKORO_VOICES`.
**To add a preset:** edit `config.py` → `VOICE_PRESETS`.
**To add an emotion:** edit `emotions.py` → `EMOTIONS`.
**To add a TTS engine:** subclass `engines/base.py` → `TTSEngine`.

## What It Does Not Do

- No internet access. All inference is local.
- No real-time conversation (streaming exists, but not bidirectional voice chat).
- No audio editing or post-processing (except vocology module, which is experimental).
- No cloud deployment. Designed for local machines with optional GPU.

## Access Points

| How | Entry Point | Use Case |
|-----|-------------|----------|
| Python | `from voice_soundboard import VoiceEngine` | Application integration |
| CLI | `voice-soundboard speak "..."` | Scripts, automation |
| MCP | `python -m voice_soundboard.server` | AI agents (Claude, etc.) |
| WebSocket | `python -m voice_soundboard.websocket_server` | Real-time web apps |
| HTTP | `python -m voice_soundboard.web_server` | Mobile access |

## Key Numbers

| Metric | Value |
|--------|-------|
| Kokoro model size | 82M parameters, ~350MB on disk |
| Sample rate | 24,000 Hz |
| Output format | WAV (PCM float32) |
| Voices | 54+ (American, British, Japanese, Mandarin) |
| Presets | 5 curated (assistant, narrator, announcer, storyteller, whisper) |
| Emotions | 19 named |
| Realtime factor | ~2x CPU, ~6x GPU |

## Deeper Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — Frozen interfaces and invariants
- [API_STABILITY.md](API_STABILITY.md) — What's stable, what's experimental
- [MODULE_AUDIT.md](MODULE_AUDIT.md) — Every module justified
- [FEATURE_FLAGS.md](FEATURE_FLAGS.md) — Core / Optional / Research tiers
- [SECURITY_SUMMARY.md](SECURITY_SUMMARY.md) — Threat model and ethics
