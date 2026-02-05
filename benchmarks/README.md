# Benchmarks

Measure real performance on your hardware.

## Run

```bash
python benchmarks/bench_basic.py
```

## What It Measures

| Metric | What It Means |
|--------|--------------|
| **Startup (cold)** | Time to construct engine + load model on first call |
| **Generation time** | Wall-clock time to produce audio from text |
| **Realtime factor (RTF)** | Audio duration / generation time. RTF > 1.0 means faster than real-time. |
| **Chars/second** | Text throughput |

## Typical Results

Results depend on hardware. These are rough expectations:

| Hardware | Short (12 chars) | Medium (100 chars) | Long (300 chars) | RTF |
|----------|------------------|--------------------|--------------------|------|
| RTX 3080 | ~100ms | ~200ms | ~500ms | ~6x |
| CPU (modern) | ~300ms | ~800ms | ~2000ms | ~2x |

Kokoro ONNX is the primary engine. It runs at approximately 2-6x real-time
depending on text length and hardware.

## No Comparison Marketing

These benchmarks measure *this system's* performance for self-knowledge.
We do not compare against other TTS systems. If you need comparisons,
run the benchmarks yourself against your alternatives.

## Environment Assumptions

- Models downloaded to the configured model directory
- Warm engine (model already loaded) for generation benchmarks
- Single-threaded generation (no batching)
- WAV output (no post-processing)
