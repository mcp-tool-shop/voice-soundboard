#!/usr/bin/env python3
"""
Demo 2: Character Voice — Emotion and variation for games / storytelling.

Shows: voice selection, emotion parameter, multiple outputs.
No experimental features. Deterministic output.

    python demos/character.py
"""

from voice_soundboard import VoiceEngine

engine = VoiceEngine()

# Same character, three emotional states.
lines = [
    ("excited", "We found it! The treasure is real!"),
    ("calm",    "Listen carefully. We only get one chance at this."),
    ("angry",   "You lied to us. Every single word was a lie."),
]

for emotion, text in lines:
    result = engine.speak(
        text,
        voice="af_bella",
        emotion=emotion,
        save_as=f"demo_character_{emotion}",
    )
    print(f"[{emotion:>7}] {result.audio_path} ({result.duration_seconds:.1f}s)")

print(f"\nGenerated {len(lines)} character lines.")
