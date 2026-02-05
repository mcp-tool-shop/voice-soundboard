#!/usr/bin/env python3
"""
Demo 3: Streaming Assistant — Low-latency speech for interactive use.

Shows: streaming generation, realtime playback, latency measurement.
No experimental features. Deterministic output.

    python demos/assistant.py
"""

import asyncio
import time
from voice_soundboard.streaming import StreamingEngine

engine = StreamingEngine()


async def main():
    text = (
        "Sure, I can help with that. "
        "The meeting is scheduled for three PM tomorrow. "
        "I've also added a reminder thirty minutes before."
    )

    print("Streaming assistant response...")
    t0 = time.perf_counter()

    result = await engine.stream_to_file(
        text,
        output_path="demo_assistant.wav",
        voice="af_bella",
    )

    wall_time = time.perf_counter() - t0

    print(f"Saved: {result.audio_path}")
    print(f"Duration: {result.total_duration:.1f}s of audio")
    print(f"Chunks: {result.total_chunks}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Realtime factor: {result.realtime_factor:.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
