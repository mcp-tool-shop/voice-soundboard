# Voice Preset Library Roadmap

## Vision
A curated library of voice presets that combine humanization, formant shifting, and (eventually) phonation effects to create distinct, production-ready voice characters.

---

## Phase 1: Foundation (v1.3.0) ✅ CURRENT

### What's Working
- [x] VoiceHumanizer (breaths + pitch variation) - **Score: 9/10**
- [x] FormantShifter (deeper/brighter) - **Score: 9/10**
- [x] RhythmAnalyzer (nPVI, RZT metrics)
- [x] Emotional states (EXCITED, CALM, TIRED, ANXIOUS, CONFIDENT, INTIMATE)

### What Needs Work
- [ ] Phonation effects (BREATHY, CREAKY) - currently garbled at high intensity
- [ ] Phonation smoothing and crossfade improvements

---

## Phase 2: Basic Preset Library (v1.4.0)

### Character Presets
Each preset combines multiple effects into a single, easy-to-use configuration.

```python
# Target API
from voice_soundboard.vocology import VoicePreset, apply_preset

# Simple usage
audio, sr = apply_preset(audio, VoicePreset.WARM_NARRATOR, sample_rate=sr)
```

### Initial Preset Categories

#### Narrator Voices
| Preset | Formant | Humanize | Description |
|--------|---------|----------|-------------|
| `WARM_NARRATOR` | -5% | calm, subtle breaths | Audiobook, documentary |
| `ENERGETIC_HOST` | +3% | excited, more breaths | Podcast, YouTube |
| `INTIMATE_STORYTELLER` | -3% | intimate, soft breaths | ASMR, meditation |

#### Character Voices
| Preset | Formant | Humanize | Description |
|--------|---------|----------|-------------|
| `DEEP_AUTHORITY` | -12% | confident | News anchor, commander |
| `YOUNG_BRIGHT` | +10% | excited | Youthful, energetic |
| `ELDERLY_WISE` | -8% | tired, slower | Aged character |
| `CHILD_LIKE` | +15% | excited, faster | Young character |

#### Emotional Overlays
| Preset | Settings | Use Case |
|--------|----------|----------|
| `NERVOUS` | high jitter, faster breaths | Anxious character |
| `SLEEPY` | low energy, slower, subtle | Tired character |
| `EXCITED` | more variation, faster | Happy, energetic |
| `SAD` | lower pitch drift, slower | Melancholy |

---

## Phase 3: Advanced Presets (v1.5.0)

### Fix Phonation Effects
- [ ] Lower default intensity (0.2-0.3 instead of 0.5-0.6)
- [ ] Add crossfade/smoothing between original and effect
- [ ] Test with real TTS output, not just reference audio

### Phonation-Enhanced Presets
| Preset | Formant | Phonation | Humanize |
|--------|---------|-----------|----------|
| `BREATHY_INTIMATE` | -3% | breathy 20% | intimate |
| `GRAVELLY_TOUGH` | -10% | creaky 15% | confident |
| `WHISPER_SECRET` | 0% | whisper 40% | intimate |
| `HUSKY_SEDUCTIVE` | -5% | breathy 25% | calm |

---

## Phase 4: Preset Chains & Layering (v1.6.0)

### Preset Chains
Allow combining presets in sequence:

```python
chain = PresetChain([
    (VoicePreset.DEEP_AUTHORITY, 1.0),
    (VoicePreset.TIRED, 0.5),  # 50% tired overlay
])
audio, sr = chain.apply(audio, sample_rate=sr)
```

### Context-Aware Presets
Automatically adjust based on content:
- Question detection → rising intonation
- Exclamation → more energy
- Long pause → deeper breath
- Emphasis words → pitch accent

---

## Phase 5: User-Created Presets (v1.7.0)

### Preset Editor
```python
# Create custom preset
my_preset = VoicePreset.custom(
    name="my_narrator",
    formant_ratio=0.95,
    humanize=HumanizeConfig.for_emotion(EmotionalState.CALM),
    description="My custom warm narrator voice"
)

# Save to library
my_preset.save("~/.voice-soundboard/presets/my_narrator.json")
```

### Preset Sharing
- JSON export/import
- Community preset repository
- Preset thumbnails (spectrogram previews)

---

## Implementation Priority

### v1.3.0 (NOW)
1. ✅ Commit current working features
2. ✅ Document what's working vs experimental

### v1.4.0 (Next)
1. Create `VoicePreset` enum with 10-12 basic presets
2. Create `apply_preset()` convenience function
3. Add preset tests
4. Write preset documentation with audio examples

### v1.5.0 (After phonation fix)
1. Fix phonation intensity/smoothing
2. Add phonation-based presets
3. A/B testing framework for preset tuning

---

## File Structure

```
voice_soundboard/vocology/
├── __init__.py
├── presets/
│   ├── __init__.py
│   ├── library.py      # VoicePreset enum, apply_preset()
│   ├── narrators.py    # Narrator preset configs
│   ├── characters.py   # Character preset configs
│   └── emotions.py     # Emotional overlay configs
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Preset quality score | >8/10 average |
| Preset variety | 20+ distinct presets |
| API simplicity | 1-line preset application |
| Documentation | Audio examples for each preset |

---

## Notes

- Always test presets with real TTS output, not just sine waves
- Keep intensity subtle - natural > dramatic
- Each preset should have a clear use case
- Presets should be composable (can layer/combine)

---

## Lessons Learned (v1.2.0 Testing Session)

### What Worked Well
- **Humanization pipeline** scored 9/10 - breaths + pitch variation make a real difference
- **Formant shifting** scored 9/10 - deeper/brighter voices work great
- **DEEP_AUTHORITY preset** - gravelly texture via jitter was effective

### Current Limitations
1. **Presets sound too similar** - formant shifts alone aren't enough differentiation
2. **Phonation effects unusable** - BREATHY/CREAKY are garbled at any useful intensity
3. **Testing with wrong voice gender** - HUSKY_INTIMATE on male voice = "dude with husky voice"

### Ideas for Better Differentiation

#### Short-term (v1.3.x)
- [ ] **Larger formant shifts** - current 2-8% too subtle, try 15-25% for character voices
- [ ] **Pitch shifting** - add actual F0 shift, not just formant (requires pitch-shift algo)
- [ ] **Speed/tempo variation** - time-stretch for fast/slow speakers
- [ ] **EQ profiles** - boost/cut frequency bands for "radio voice", "telephone", etc.

#### Medium-term (v1.4.x)
- [ ] **Fix phonation synthesis** - lower intensity defaults, better crossfade blending
- [ ] **Combine multiple effects** - formant + phonation + EQ as layered chain
- [ ] **Voice-specific presets** - separate male/female/child preset variants
- [ ] **A/B testing framework** - systematic comparison of preset variations

#### Long-term (v1.5.x)
- [ ] **Neural voice conversion** - use ML models for more dramatic transformations
- [ ] **Reference audio matching** - "make it sound like this sample"
- [ ] **Real-time preset preview** - instant feedback while tweaking parameters

### Research Directions
- Study vocoder-based voice modification (WORLD, STRAIGHT)
- Look into pitch-synchronous overlap-add (PSOLA) for cleaner pitch shifting
- Investigate spectral envelope manipulation beyond simple formant ratios
- Consider LPC-based voice transformation for more dramatic effects

### Testing Protocol for Future Sessions
1. Always use **real TTS audio** (not synthetic tones)
2. Test each preset with **male AND female voices**
3. Compare **before/after** side-by-side
4. Score on **distinctiveness** not just quality
5. Document which parameters made the biggest audible difference
