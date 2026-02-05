#!/usr/bin/env python3
"""
Demo 1: Narrator — Long-form, controlled, expressive speech.

Shows: preset selection, emotion, pacing control.
No experimental features. Deterministic output.

    python demos/narrator.py
"""

from voice_soundboard import VoiceEngine

engine = VoiceEngine()

# A calm documentary narrator reading a passage.
result = engine.speak(
    "In the summer of nineteen sixty-nine, "
    "three astronauts left Earth and walked on the Moon. "
    "It was a moment that changed how humanity saw itself — "
    "not as passengers on a planet, but as explorers of a universe.",
    preset="narrator",
    emotion="calm",
    save_as="demo_narrator",
)

print(f"Saved: {result.audio_path}")
print(f"Voice: {result.voice_used}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Generated in: {result.generation_time:.2f}s")
print(f"Realtime factor: {result.realtime_factor:.1f}x")
