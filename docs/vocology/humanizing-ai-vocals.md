# Humanizing AI Vocals: Breath Sounds & Realism

> Adding the subtle imperfections that make synthesized speech sound natural and alive.

---

## Why AI Vocals Sound Artificial

AI-generated vocals often lack authenticity due to several factors:

| Issue | Description | Human Reality |
|-------|-------------|---------------|
| **Missing respiratory patterns** | Continuous sound without breathing | Humans breathe every 3-5 seconds |
| **Perfect precision** | Metronomic timing | Humans drift ±20-50ms from beat |
| **Pitch stability** | Unnaturally consistent | Natural voices waver ±5-10 cents |
| **Absent formant shifts** | Static vocal tract | Constantly adjusting mouth shape |
| **Uniform vibrato** | Consistent modulation | Varies with emotion and phrase |
| **No mouth sounds** | Clean signal only | Lip smacks, tongue clicks, saliva |

---

## Adding Realistic Breath Sounds

### Timing and Placement

```
Breath timing guidelines:
- Position breaths 50-200ms BEFORE vocal phrases
- Quick inhales (100-200ms) for energetic sections
- Longer breaths (300-500ms) for ballads/slow speech
- Consider phrase length when determining breath preparation
- Longer phrases need deeper breaths
```

**Breath placement patterns:**

| Context | Breath Type | Duration | Position |
|---------|-------------|----------|----------|
| **Sentence start** | Deep inhale | 300-400ms | 150-200ms before |
| **Mid-phrase** | Quick catch breath | 100-150ms | 50-100ms before |
| **Emotional moment** | Audible gasp | 200-300ms | Immediate |
| **List items** | Tiny sips | 50-100ms | Between items |
| **After exertion** | Heavy panting | 400-600ms | After phrase |

### Types of Breath Sounds

Layer multiple breath types for realism:

1. **Nasal breathing** - Softer, more subtle
2. **Mouth breaths** - More audible, emotional
3. **Vocal fry onset** - Creaky start to phrases
4. **Lip smacks** - Before/after certain sounds
5. **Tongue clicks** - Natural speech preparation
6. **Saliva sounds** - Wet mouth noises (subtle!)

### Breath Sample Recording Tips

```
Recording diverse breaths:
- Different emotional states (calm, excited, tired, anxious)
- Various intensities (whisper to forceful)
- Multiple breath types (nose, mouth, mixed)
- Clean recordings (no room noise)
- Match microphone/room to target vocal
```

### Breath Volume Guidelines

| Vocal Style | Breath Volume | Notes |
|-------------|---------------|-------|
| **Intimate/ASMR** | -12 to -6 dB below vocal | Breaths should be audible |
| **Pop/Podcast** | -18 to -12 dB below vocal | Present but not distracting |
| **Rock/Energetic** | -12 to -8 dB below vocal | Match energy level |
| **Classical/Narration** | -24 to -18 dB below vocal | Barely perceptible |

---

## Humanization Processing Techniques

### 1. Pitch Variations

Add micro-pitch imperfections:

```
Pitch humanization targets:
- Gentle pitch drift at phrase endings (±5-15 cents)
- Subtle scooping into notes (start 20-50 cents flat)
- Overshoot on high notes (briefly sharp, then settle)
- Phrase-final lowering (declarative sentences drop)
- Random micro-variations (±3-5 cents jitter)
```

**Pitch drift automation:**
```
Start of phrase: On pitch
Mid-phrase: Slight drift (±5 cents)
End of phrase: Drift down 10-20 cents (statements)
                Drift up 10-30 cents (questions)
```

### 2. Formant Shifting

Simulate changing vocal tract shapes:

```
Formant automation ideas:
- Slight formant rise on stressed syllables
- Formant lowering on relaxed phrases
- Micro-shifts during vowel transitions
- Match speaker's head movement (if video)
```

| Emotional State | Formant Shift |
|-----------------|---------------|
| **Excited** | +2-5% (brighter) |
| **Tired/Sad** | -2-5% (darker) |
| **Intimate** | -3-8% (warmer) |
| **Authoritative** | Stable to -2% |

### 3. Dynamic Range Manipulation

Natural vocals have dynamic variation:

```
Dynamic processing:
- Variable compression (not constant ratio)
- Saturation on louder sections
- Allow transients on consonants
- Softer dynamics on breath phrases
- Louder attacks on stressed words
```

### 4. Timing Adjustments

Perfect timing sounds robotic:

```
Timing humanization:
- Nudge syllables ±10-30ms from grid
- Slightly early for anticipation/excitement
- Slightly late for relaxed/tired delivery
- Vary gap between phrases (not metronomic)
- Extend emotional words slightly
```

**Timing patterns by emotion:**

| Emotion | Timing Tendency |
|---------|-----------------|
| **Excited** | Slightly ahead of beat |
| **Relaxed** | Behind the beat |
| **Confident** | On or slightly ahead |
| **Hesitant** | Variable, behind |
| **Angry** | Hard on beat, clipped |

### 5. Vibrato Inconsistencies

Natural vibrato varies:

```
Vibrato parameters to modulate:
- Rate: 5-7 Hz (varies with emotion)
- Depth: ±30-100 cents (varies with intensity)
- Onset delay: 150-400ms into sustained notes
- Fade: Often increases, then decreases at phrase end
```

| Emotional State | Vibrato Rate | Vibrato Depth |
|-----------------|--------------|---------------|
| **Calm** | 5-5.5 Hz | ±30-50 cents |
| **Emotional** | 5.5-6.5 Hz | ±50-80 cents |
| **Tense** | 6-7 Hz | ±40-60 cents |
| **Weak/Tired** | 4.5-5.5 Hz | ±20-40 cents |

---

## Layered Approach

**Key principle**: Layer multiple subtle approaches rather than relying on a single solution.

### Recommended Processing Chain

```
1. Breath insertion (sample-based)
2. Timing adjustments (±20ms nudges)
3. Pitch humanization (micro-variations)
4. Formant automation (emotion matching)
5. Dynamic variation (compression + saturation)
6. Vibrato modulation (rate/depth variation)
7. Room/space processing (subtle reverb)
8. Final EQ (match reference vocal)
```

### Intensity by Application

| Application | Humanization Level | Focus Areas |
|-------------|-------------------|-------------|
| **Audiobook** | Medium | Breaths, timing, pitch drift |
| **Podcast** | Light-Medium | Breaths, natural pauses |
| **Music Vocal** | Heavy | All parameters |
| **Voice Assistant** | Light | Breaths, slight timing |
| **Character Voice** | Heavy | Formants, dynamics, emotion |

---

## Tools and Plugins

### Pitch & Formant

| Tool | Best For |
|------|----------|
| **Melodyne** | Detailed pitch curve editing |
| **Auto-Tune** | Humanize function for variations |
| **Little AlterBoy** | Real-time formant shifting |
| **VocalSynth 2** | Harmonic character adjustment |
| **SoundID VoiceAI** | 50 presets with naturalisation |

### Breath Libraries

- Native Instruments breath collections
- Custom recordings (recommended for matching)
- Splice breath sample packs
- Film/game vocal FX libraries

### Processing

| Tool | Use |
|------|-----|
| **Waves Vocal Rider** | Natural level automation |
| **FabFilter Pro-Q** | Surgical EQ matching |
| **iZotope VocalSynth** | Character and modulation |
| **Soundtoys Decapitator** | Subtle saturation |

---

## Implementation in Voice Soundboard

### Breath Insertion Algorithm

```python
def insert_breaths(audio, breath_samples, phrase_boundaries):
    """
    Insert breath sounds before phrases.

    Parameters:
        audio: Main vocal audio
        breath_samples: List of breath audio samples
        phrase_boundaries: List of (start_time, end_time) tuples
    """
    for i, (start, end) in enumerate(phrase_boundaries):
        # Determine breath type based on context
        phrase_duration = end - start

        if phrase_duration > 3.0:  # Long phrase needs deep breath
            breath = select_breath(breath_samples, type='deep')
            offset = 0.3  # 300ms before phrase
        elif i == 0:  # First phrase
            breath = select_breath(breath_samples, type='medium')
            offset = 0.2
        else:  # Mid-phrase breath
            breath = select_breath(breath_samples, type='quick')
            offset = 0.1

        # Insert breath
        breath_position = start - offset
        audio = mix_at_position(audio, breath, breath_position)

    return audio
```

### Pitch Humanization

```python
def humanize_pitch(f0_contour, sample_rate):
    """
    Add natural pitch variations to F0 contour.
    """
    # Add micro-jitter (±5 cents)
    jitter = np.random.normal(0, 5, len(f0_contour))

    # Add slow drift (phrase-level)
    drift = generate_drift_contour(len(f0_contour), max_cents=15)

    # Add phrase-final lowering
    final_drop = generate_final_drop(len(f0_contour), drop_cents=20)

    return f0_contour + jitter + drift + final_drop
```

---

## Quality Checklist

Before finalizing humanized vocals:

- [ ] Breaths sound natural and well-timed
- [ ] No metronomic timing patterns
- [ ] Pitch has subtle variations (not robotic stability)
- [ ] Stressed syllables have appropriate emphasis
- [ ] Phrase endings have natural pitch movement
- [ ] Vibrato varies with emotional content
- [ ] Dynamics match speaking/singing style
- [ ] No obvious processing artifacts
- [ ] Matches reference human vocal quality

---

## References

- [How to Add Breath Sounds and Realism to AI Vocals](https://www.sonarworks.com/blog/learn/how-to-add-breath-sounds-and-realism-to-ai-vocals) - Sonarworks
- Melodyne Documentation
- iZotope Vocal Production Guide

---

*Last updated: 2026-01-23*
