#!/usr/bin/env python3
"""
Basic benchmark harness for Voice Soundboard.

Measures what matters:
- Engine startup time (model load)
- Single utterance generation time
- Throughput (characters per second)
- Realtime factor (audio duration / generation time)

Usage:
    python benchmarks/bench_basic.py

Environment:
    Set VOICE_SOUNDBOARD_MODELS to your models directory if needed.
    Results vary by hardware. GPU (CUDA) is significantly faster than CPU.
"""

import time
import statistics
import sys

# Short, medium, long test sentences.
TEXTS = {
    "short": "Hello world.",
    "medium": (
        "Voice Soundboard converts text to speech locally. "
        "It supports multiple voices, emotions, and presets."
    ),
    "long": (
        "In the summer of nineteen sixty-nine, three astronauts left Earth "
        "and walked on the Moon. It was a moment that changed how humanity "
        "saw itself. Not as passengers on a planet, but as explorers of a "
        "universe. The mission lasted eight days. The walk lasted two hours. "
        "But the impact lasted forever."
    ),
}

WARMUP_RUNS = 1
BENCH_RUNS = 5


def measure_startup():
    """Measure cold-start model loading time."""
    from voice_soundboard import VoiceEngine

    t0 = time.perf_counter()
    engine = VoiceEngine()
    construct_time = time.perf_counter() - t0

    # Force model load with a minimal call
    t1 = time.perf_counter()
    engine.speak("test", save_as="_bench_warmup")
    first_call_time = time.perf_counter() - t1

    return {
        "construct_ms": construct_time * 1000,
        "first_call_ms": first_call_time * 1000,
        "total_startup_ms": (construct_time + first_call_time) * 1000,
    }


def measure_generation(engine, text, label, runs=BENCH_RUNS):
    """Measure generation time over multiple runs."""
    times = []
    durations = []
    rtfs = []

    for i in range(runs):
        result = engine.speak(text, save_as=f"_bench_{label}_{i}")
        times.append(result.generation_time)
        durations.append(result.duration_seconds)
        rtfs.append(result.realtime_factor)

    chars = len(text)
    avg_time = statistics.mean(times)

    return {
        "label": label,
        "chars": chars,
        "runs": runs,
        "avg_generation_ms": avg_time * 1000,
        "min_generation_ms": min(times) * 1000,
        "max_generation_ms": max(times) * 1000,
        "avg_duration_s": statistics.mean(durations),
        "avg_rtf": statistics.mean(rtfs),
        "chars_per_second": chars / avg_time if avg_time > 0 else 0,
    }


def main():
    print("Voice Soundboard Benchmark")
    print("=" * 50)

    # Step 1: Startup
    print("\n1. Startup (cold start)")
    startup = measure_startup()
    print(f"   Constructor:    {startup['construct_ms']:>8.1f} ms")
    print(f"   First call:     {startup['first_call_ms']:>8.1f} ms")
    print(f"   Total startup:  {startup['total_startup_ms']:>8.1f} ms")

    # Step 2: Import an already-warm engine
    from voice_soundboard import VoiceEngine
    engine = VoiceEngine()
    engine.speak("warmup", save_as="_bench_warmup2")  # ensure model loaded

    # Step 3: Generation benchmarks
    print(f"\n2. Generation ({BENCH_RUNS} runs each, warm engine)")
    print(f"   {'Label':<8} {'Chars':>5} {'Avg ms':>8} {'Min ms':>8} "
          f"{'RTF':>6} {'Chars/s':>8}")
    print(f"   {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    for label, text in TEXTS.items():
        result = measure_generation(engine, text, label)
        print(f"   {result['label']:<8} {result['chars']:>5} "
              f"{result['avg_generation_ms']:>8.1f} "
              f"{result['min_generation_ms']:>8.1f} "
              f"{result['avg_rtf']:>6.1f}x "
              f"{result['chars_per_second']:>8.0f}")

    # Step 4: Device info
    from voice_soundboard import Config
    from unittest.mock import patch
    with patch('pathlib.Path.mkdir'):
        config = Config()
    print(f"\n3. Environment")
    print(f"   Device: {config.device}")
    print(f"   Python: {sys.version.split()[0]}")

    print(f"\nDone. Results are hardware-dependent.")
    print(f"Clean up: delete output/_bench_* files.")


if __name__ == "__main__":
    main()
