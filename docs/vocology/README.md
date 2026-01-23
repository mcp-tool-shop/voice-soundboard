# Vocology & AI Voice Science Reference

> A comprehensive guide to voice science, speech synthesis, and 2026 best practices for the Voice Soundboard project.

## What is Vocology?

**Vocology** is the science and practice of voice habilitation — not just repairing voices, but building and strengthening them to meet specific needs. Coined by **Ingo Titze** at the National Center for Voice and Speech (NCVS), it combines:

- Speech-language pathology
- Otolaryngology (laryngology)
- Vocal music pedagogy
- Theatre voice training

> "To habilitate means to enable, equip for, capacitate"

## Documentation Index

### Core Voice Science

| Document | Description |
|----------|-------------|
| [Voice Quality Parameters](./voice-quality-parameters.md) | F0, jitter, shimmer, HNR, spectral tilt, CPP |
| [Phonation Types](./phonation-types.md) | Modal, breathy, creaky, harsh, falsetto |
| [Voice Characteristics](./voice-characteristics.md) | Timbre, resonance, texture descriptors |
| [Vocal Biomarkers](./vocal-biomarkers.md) | Health detection via voice analysis |
| [Voice Metrics Reference](./voice-metrics-reference.md) | **NEW** Normative values by gender, age, language |

### Prosody & Rhythm

| Document | Description |
|----------|-------------|
| [Prosody Deep Dive](./prosody-deep-dive.md) | **NEW** Comprehensive guide to speech rhythm, intonation, stress |
| [Prosody Control](./prosody-control.md) | Pitch contours, duration, SSML for TTS |

**Prosody Deep Dive** covers:
- Intonation, stress, rhythm fundamentals
- Speech rhythm typology (stress-timed, syllable-timed, mora-timed)
- Rhythm Zone Theory (RZT) - signal-based rhythm detection
- Cognitive neuroscience of speech rhythm (neural oscillations)
- ToBI prosodic annotation system
- Emotional prosody and affective computing

### TTS & Implementation

| Document | Description |
|----------|-------------|
| [TTS Architecture 2026](./tts-architecture-2026.md) | Neural vocoders, mel-spectrograms, pipelines |
| [Disentanglement](./disentanglement.md) | Timbre-emotion separation, speaker embeddings |
| [Humanizing AI Vocals](./humanizing-ai-vocals.md) | **NEW** Breath sounds, timing, pitch variations |
| [Best Practices 2026](./best-practices-2026.md) | State-of-the-art models and techniques |

**Humanizing AI Vocals** covers:
- Why AI vocals sound artificial (missing breaths, perfect timing, stable pitch)
- Adding realistic breath sounds (timing, types, volume levels)
- Pitch humanization (drift, scooping, micro-variations)
- Timing adjustments (nudging syllables off-grid)
- Formant and vibrato modulation
- Processing chain and tools

## Quick Reference

### Core Voice Parameters

| Parameter | What It Measures | Healthy Range | TTS Use |
|-----------|------------------|---------------|---------|
| **F0** | Pitch (fundamental frequency) | M: 85-180Hz, F: 165-255Hz | Primary pitch |
| **Jitter** | Pitch variation cycle-to-cycle | <1% | Naturalness |
| **Shimmer** | Amplitude variation | <3-5% | Breathiness |
| **HNR** | Signal clarity | >15-20 dB | Hoarseness |
| **Spectral Tilt** | High-freq energy rolloff | Varies | Age, breathiness |
| **CPP** | Overall periodicity | Higher = clearer | Best quality metric |

### Phonation Spectrum

```
BREATHY ←——————— MODAL ———————→ CREAKY/PRESSED
(open glottis)              (constricted glottis)
   ↓                              ↓
Airy, soft               Vocal fry, tense
intimate                 authoritative
```

### Voice Texture Descriptors

- **Warm** - Inviting, comforting, full resonance
- **Rich** - Full-bodied, deep, luxurious
- **Smooth/Silky** - Flowing, no roughness
- **Raspy/Gravelly** - Friction, rough edges
- **Husky** - Deep + slightly rough
- **Breathy** - Airy, intimate, soft

## Sources & References

### Academic - Voice Quality
- National Center for Voice and Speech (NCVS): https://ncvs.org/
- Voice Quality Types in North American English: https://pmc.ncbi.nlm.nih.gov/articles/PMC11288166/
- Phonation Types Overview: https://idiom.ucsd.edu/~mgarellek/files/Garellek_Phonetics_of_Voice_Handbook_final.pdf

### Academic - Prosody & Rhythm
- Prosody (Wikipedia): https://en.wikipedia.org/wiki/Prosody_(linguistics)
- Rhythm Zone Theory (Gibbon & Lin): https://arxiv.org/abs/1902.01267
- Neural Oscillations and Speech (Frontiers): https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00320/full
- ToBI Guidelines (Ohio State): https://www.ling.ohio-state.edu/research/phonetics/E_ToBI/
- Emotional Prosody Research: https://journals.sagepub.com/doi/10.1177/17456916231217722
- Speech Rhythm Typology (Isochrony): https://en.wikipedia.org/wiki/Isochrony

### Prosody Tools
- Prosogram: https://sites.google.com/site/prosogram/home
- ProsodyPro: https://www.homepages.ucl.ac.uk/~uclyyix/ProsodyPro/
- SProSIG (Speech Prosody Special Interest Group): https://sprosig.org/

### Industry (2026)
- Best Open-Source TTS Models 2026: https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models
- Vocal Biomarker Trends 2026: https://canaryspeech.com/blog/5-trends-in-2026/
- Marco-Voice Technical Report: https://arxiv.org/html/2508.02038v2
- Voice Cloning Survey 2025: https://arxiv.org/html/2505.00579v1
- TTS Prosody Modeling: https://apxml.com/courses/speech-recognition-synthesis-asr-tts/chapter-4-advanced-text-to-speech-synthesis/prosody-modeling-control-tts

### Models
- IndexTTS2: https://github.com/index-tts/index-tts
- Chatterbox: https://github.com/resemble-ai/chatterbox
- Fish Speech: https://github.com/fishaudio/fish-speech
- CosyVoice2: https://github.com/FunAudioLLM/CosyVoice
