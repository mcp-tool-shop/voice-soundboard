#!/usr/bin/env python3
"""
Start here. Everything else is optional.

Generates a single audio file using the default voice and engine.
No configuration required -- just run it.

    python examples/hello_world.py

Output:
    output/hello_world.wav
"""

from voice_soundboard import VoiceEngine

engine = VoiceEngine()
result = engine.speak("Hello world!", save_as="hello_world")

print(f"Audio saved to: {result.audio_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
