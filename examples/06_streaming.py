#!/usr/bin/env python3
"""
Real-Time Streaming Example

Demonstrates low-latency audio generation:
- Real-time streaming with playback
- Chunk-based generation
- Progress callbacks
"""

import asyncio
from voice_soundboard import stream_realtime, StreamingEngine


async def main():
    print("Voice Soundboard - Streaming Demo")
    print("=" * 38)

    # 1. Real-time streaming with automatic playback
    print("\n1. Real-time streaming (plays as it generates):")
    text = """
    Voice Soundboard supports real-time streaming for low-latency applications.
    This is especially useful for interactive voice assistants, chatbots, and
    any application where responsiveness matters. The audio starts playing
    before the entire text has been processed, reducing perceived latency.
    """

    result = await stream_realtime(
        text.strip(),
        voice="af_bella",
        speed=1.0
    )

    print(f"   Total chunks: {result.total_chunks}")
    print(f"   Generation time: {result.generation_time:.2f}s")
    print(f"   Audio duration: {result.duration_seconds:.2f}s")
    print(f"   Realtime factor: {result.realtime_factor:.1f}x")

    # 2. Streaming with callbacks
    print("\n2. Streaming with progress callbacks:")

    chunks_received = 0

    def on_chunk(chunk_data, chunk_index):
        nonlocal chunks_received
        chunks_received += 1
        print(f"   Chunk {chunk_index}: {len(chunk_data)} samples")

    def on_progress(progress, total):
        percent = (progress / total) * 100 if total > 0 else 0
        print(f"   Progress: {percent:.0f}%")

    engine = StreamingEngine()
    result = await engine.stream_to_file(
        "This is a test of the streaming engine with callbacks.",
        output_path="streaming_output.wav",
        on_chunk=on_chunk,
        on_progress=on_progress,
        voice="bm_george"
    )

    print(f"   Saved to: {result.audio_path}")
    print(f"   Chunks received: {chunks_received}")

    # 3. Streaming long text
    print("\n3. Streaming long text (efficient for articles/documents):")
    long_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every
    letter of the English alphabet, making it useful for typography testing.
    In the world of text-to-speech, streaming is essential for handling long
    documents without blocking the user interface. Each sentence can be
    processed and played independently, creating a seamless audio experience.
    """

    result = await stream_realtime(
        long_text.strip(),
        voice="af_bella",
        speed=1.1  # Slightly faster for long content
    )

    print(f"   Processed {len(long_text)} characters")
    print(f"   Realtime factor: {result.realtime_factor:.1f}x")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
