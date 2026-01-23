# TTS Best Practices 2026

State-of-the-art techniques and recommendations for high-quality speech synthesis.

## Top Open-Source Models (2026)

### Tier 1: Production Ready

| Model | Strength | Latency | Voice Cloning | Languages |
|-------|----------|---------|---------------|-----------|
| **Fish Speech V1.5** | Quality + DualAR | Real-time | 10s reference | EN, ZH, JP, +10 |
| **CosyVoice2** | Streaming (150ms) | 150ms | Zero-shot | Multilingual |
| **IndexTTS2** | Emotion control | Real-time | 3s reference | EN, ZH |

### Tier 2: Specialized Use Cases

| Model | Best For | Notes |
|-------|----------|-------|
| **Chatterbox** | Emotion + paralinguistics | [laugh], [cough] tags |
| **XTTS-v2** | Multilingual cloning | 17+ languages |
| **Kokoro** | Lightweight (82M params) | Edge deployment |
| **F5-TTS** | Voice cloning quality | Flow matching |

### Tier 3: Research/Experimental

| Model | Innovation |
|-------|------------|
| **VoxCPM** | Tokenizer-free continuous space |
| **NeuTTS Air** | 3-second on-device cloning |
| **VibeVoice** | Low-frame-rate tokenization |

## Quality Checklist

### Audio Quality

- [ ] **Sample rate:** 22050-48000 Hz (24000 Hz common)
- [ ] **Bit depth:** 16-bit minimum, 32-bit float internal
- [ ] **Noise floor:** <-60 dB SNR in training data
- [ ] **Clipping:** None (normalize to -3 dB peak)

### Model Selection

- [ ] Match model to use case (streaming vs batch)
- [ ] Consider VRAM requirements
- [ ] Verify language support
- [ ] Check voice cloning quality needs

### Evaluation

- [ ] MOS testing with diverse listeners
- [ ] WER/CER for intelligibility
- [ ] Speaker similarity for cloning
- [ ] Real-world A/B testing

## Voice Quality Parameters

### Recommended Defaults

```python
DEFAULT_VOICE_PARAMS = {
    # Naturalness
    "jitter_percent": 0.5,      # Slight natural variation
    "shimmer_percent": 2.0,     # Normal amplitude variation
    "hnr_db": 20,               # Clear but not synthetic

    # Prosody
    "pitch_range_semitones": 6, # Natural pitch variation
    "speaking_rate_wpm": 150,   # Comfortable listening
    "pause_ratio": 0.15,        # Natural breathing

    # Emotion
    "emotion_intensity": 0.5,   # Balanced expression
    "exaggeration": 0.5,        # Chatterbox default
}
```

### Voice Character Presets

```python
VOICE_PRESETS = {
    "narrator": {
        "pace": 0.9,
        "pitch_variation": 0.3,
        "warmth": 0.7,
        "clarity": 0.9
    },
    "conversational": {
        "pace": 1.0,
        "pitch_variation": 0.5,
        "warmth": 0.6,
        "naturalness": 0.9
    },
    "announcer": {
        "pace": 0.95,
        "pitch_variation": 0.4,
        "projection": 0.8,
        "clarity": 0.95
    },
    "intimate": {
        "pace": 0.85,
        "breathiness": 0.4,
        "warmth": 0.8,
        "volume": 0.7
    }
}
```

## Prosody Control

### Best Practices

1. **Don't over-control:** Let the model's natural prosody shine
2. **Use SSML sparingly:** For specific effects only
3. **Reference audio:** Better than explicit parameters for style
4. **Test edge cases:** Long sentences, unusual punctuation

### Pitch Contour Guidelines

```python
# Good: Subtle variations
pitch_range = (0.95, 1.05)  # ±5%

# Bad: Extreme variations (sounds robotic)
pitch_range = (0.7, 1.3)    # ±30%
```

### Duration Guidelines

```python
# Natural pause scaling
pause_after_comma = 150  # ms
pause_after_period = 400  # ms
pause_after_paragraph = 800  # ms

# Emphasis (subtle)
emphasis_duration_scale = 1.15  # 15% longer
emphasis_pitch_scale = 1.08     # 8% higher
```

## Voice Cloning Best Practices

### Reference Audio Quality

| Requirement | Recommendation |
|-------------|----------------|
| **Duration** | 5-15 seconds optimal |
| **Content** | Natural speech, not reading |
| **Noise** | Clean, <-40 dB noise floor |
| **Emotion** | Neutral or target emotion |
| **Recording** | Close-mic, minimal reverb |

### Cloning Pipeline

```python
# 1. Validate reference audio
if not validate_reference(audio):
    raise ValueError("Reference audio quality insufficient")

# 2. Extract speaker embedding
speaker_emb = extract_speaker_embedding(audio)

# 3. Verify embedding quality
similarity = verify_speaker_embedding(speaker_emb, audio)
if similarity < 0.7:
    warn("Low speaker embedding confidence")

# 4. Generate with appropriate parameters
output = generate(
    text=text,
    speaker_embedding=speaker_emb,
    cfg_weight=0.5,  # Balance quality vs similarity
)
```

### Cross-Lingual Cloning

```python
# When source and target languages differ
CROSS_LINGUAL_SETTINGS = {
    "same_family": {  # e.g., EN→FR
        "cfg_weight": 0.5,
        "speaker_weight": 0.8
    },
    "different_family": {  # e.g., EN→ZH
        "cfg_weight": 0.3,  # Lower for more adaptation
        "speaker_weight": 0.6
    }
}
```

## Emotion & Expressiveness

### Chatterbox Parameters

```python
# Emotion exaggeration scale
# 0.0 = monotone/flat
# 0.5 = natural (default)
# 1.0 = theatrical/dramatic

EMOTION_PRESETS = {
    "neutral": {"exaggeration": 0.3, "cfg_weight": 0.5},
    "conversational": {"exaggeration": 0.5, "cfg_weight": 0.5},
    "expressive": {"exaggeration": 0.7, "cfg_weight": 0.4},
    "dramatic": {"exaggeration": 0.9, "cfg_weight": 0.3},
}
```

### Paralinguistic Tags

```python
# Chatterbox-Turbo native tags
SUPPORTED_TAGS = [
    "[laugh]",
    "[chuckle]",
    "[cough]",
    "[sigh]",
    "[gasp]",
    "[breath]",
]

# Usage
text = "That's hilarious [laugh] I can't believe it!"
```

## Latency Optimization

### Streaming TTS

```python
# Chunked generation for low latency
async def stream_tts(text, websocket):
    chunks = split_into_chunks(text, max_words=10)

    for chunk in chunks:
        audio = await generate_chunk(chunk)
        await websocket.send(audio)
        # First audio plays while rest generates
```

### Caching Strategies

```python
# Cache frequently used phrases
PHRASE_CACHE = LRUCache(maxsize=1000)

def generate_with_cache(text, voice):
    cache_key = hash((text, voice))
    if cache_key in PHRASE_CACHE:
        return PHRASE_CACHE[cache_key]

    audio = generate(text, voice)
    PHRASE_CACHE[cache_key] = audio
    return audio
```

### Model Optimization

1. **Quantization:** INT8 for 2-4x speedup
2. **Distillation:** Smaller student models
3. **Batching:** Process multiple requests together
4. **GPU selection:** Match model to hardware

## Evaluation Metrics

### Objective Metrics

| Metric | Target | Measures |
|--------|--------|----------|
| **MOS** | >4.0 | Overall naturalness |
| **WER** | <5% | Intelligibility |
| **Speaker Similarity** | >0.80 | Voice cloning accuracy |
| **F0 RMSE** | <20 Hz | Prosody accuracy |
| **Real-time Factor** | >10x | Speed |

### Subjective Testing

```python
# A/B testing framework
def run_ab_test(system_a, system_b, test_sentences, n_listeners=20):
    results = []
    for sentence in test_sentences:
        audio_a = system_a.generate(sentence)
        audio_b = system_b.generate(sentence)

        for listener in sample_listeners(n_listeners):
            preference = listener.compare(audio_a, audio_b)
            results.append({
                "sentence": sentence,
                "preference": preference,
                "listener": listener.id
            })

    return analyze_preferences(results)
```

## Common Pitfalls

### Avoid These Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Over-controlling prosody | Sounds robotic | Use style references |
| Ignoring text normalization | "Dr." → "Doctor" | Proper preprocessing |
| Mismatched sample rates | Artifacts | Consistent pipeline |
| Training on noisy data | Poor quality | Clean data or enhancement |
| Extreme voice cloning | Uncanny valley | Moderate parameters |

### Quality Degradation Signs

- Metallic/robotic quality → Check vocoder
- Mispronunciations → Improve G2P or use phoneme input
- Unnatural pauses → Adjust duration model
- Monotone output → Increase prosody variation
- Speaker drift → Strengthen speaker conditioning

## Deployment Checklist

### Pre-Launch

- [ ] Load test with expected traffic
- [ ] Monitor VRAM usage
- [ ] Set up error handling for edge cases
- [ ] Implement request queuing
- [ ] Configure timeouts appropriately

### Production

- [ ] Monitor latency percentiles (p50, p95, p99)
- [ ] Track generation failures
- [ ] Log audio quality metrics
- [ ] Set up alerts for degradation
- [ ] Plan for model updates

## References

- Best Open-Source TTS Models 2026: https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models
- SiliconFlow TTS Guide: https://www.siliconflow.com/articles/en/best-open-source-text-to-speech-models
- Chatterbox Documentation: https://github.com/resemble-ai/chatterbox
- Fish Speech: https://github.com/fishaudio/fish-speech
- IndexTTS2: https://github.com/index-tts/index-tts
