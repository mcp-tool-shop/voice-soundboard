# Demos

Three canonical demos that show what Voice Soundboard does.
Each fits on one screen, uses default presets, and produces deterministic output.

## Run Them

```bash
python demos/narrator.py      # Long-form documentary narration
python demos/character.py     # Same character, three emotions
python demos/assistant.py     # Streaming low-latency response
```

## What They Show

| Demo | Feature | Preset | Output |
|------|---------|--------|--------|
| **Narrator** | Controlled long-form speech | `narrator` + `calm` | `demo_narrator.wav` |
| **Character** | Emotion variation | `af_bella` + 3 emotions | `demo_character_*.wav` |
| **Assistant** | Streaming generation | Default voice, async | `demo_assistant.wav` |

## Requirements

- Kokoro models downloaded (see main README)
- `pip install voice-soundboard`

No experimental features, no special flags, no configuration files.
