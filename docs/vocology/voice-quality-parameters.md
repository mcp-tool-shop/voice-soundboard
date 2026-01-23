# Voice Quality Parameters

Scientific measurements used to quantify voice characteristics for analysis and synthesis.

## Fundamental Frequency (F0)

**What it is:** The lowest frequency component of the voice signal, corresponding to perceived pitch.

**Measurement:** Hertz (Hz)

**Typical Ranges:**
- Adult males: 85-180 Hz (average ~120 Hz)
- Adult females: 165-255 Hz (average ~210 Hz)
- Children: 250-400 Hz

**In TTS:** Primary control for pitch. Most systems expose this directly or via percentage modifiers.

```python
# Example: Pitch control
engine.speak("Hello", pitch_shift=1.2)  # 20% higher pitch
```

## Jitter

**What it is:** Cycle-to-cycle variation in fundamental frequency. Measures pitch instability.

**Types:**
- **Local Jitter (Jitter%)**: Percent variation between adjacent periods
- **RAP**: Relative Average Perturbation (3-point average)
- **PPQ5**: 5-point Period Perturbation Quotient

**Healthy Range:** <1% for sustained vowels

**Voice Quality Impact:**
- Low jitter (~0.5%): Smooth, controlled voice
- Normal jitter (0.5-1%): Natural human variation
- High jitter (>1%): Rough, unstable, potentially pathological

**In TTS:** Adding controlled jitter increases naturalness. Too much sounds broken.

```python
# Synthesis parameter
jitter_percent = 0.8  # Slightly rough, natural
```

## Shimmer

**What it is:** Cycle-to-cycle variation in amplitude. Measures loudness instability.

**Types:**
- **Local Shimmer (Shimmer%)**: Percent variation between adjacent amplitudes
- **APQ3**: 3-point Amplitude Perturbation Quotient
- **APQ5**: 5-point Amplitude Perturbation Quotient
- **APQ11**: 11-point smoothed quotient

**Healthy Range:** <3-5% for sustained vowels

**Voice Quality Impact:**
- Low shimmer (<2%): Clear, steady voice
- Normal shimmer (2-4%): Natural variation
- High shimmer (>5%): Breathy, hoarse quality
- Very high shimmer (>10%): Pathological or intentionally breathy (Klatt 1990)

**In TTS:** Higher shimmer creates breathiness. Useful for intimate/ASMR voices.

```python
# Breathy voice synthesis
shimmer_percent = 8.0  # Breathy, airy quality
```

## Harmonics-to-Noise Ratio (HNR)

**What it is:** Ratio of periodic (harmonic) energy to aperiodic (noise) energy in the voice.

**Measurement:** Decibels (dB)

**Typical Ranges:**
- Healthy voice: 15-25 dB
- Slightly hoarse: 10-15 dB
- Hoarse/breathy: 5-10 dB
- Severely disordered: <5 dB

**Inverse: NHR (Noise-to-Harmonics Ratio)**
- Higher NHR = more noise = hoarser voice

**Voice Quality Impact:**
- High HNR (>20 dB): Clear, ringing, "clean" voice
- Medium HNR (12-18 dB): Normal conversational voice
- Low HNR (<10 dB): Hoarse, rough, breathy

**In TTS:** Control turbulent noise injection for breathiness/hoarseness.

```python
# Target HNR for synthesis
hnr_target_db = 18.0  # Clear but natural
```

## Spectral Tilt

**What it is:** The rate at which spectral energy decreases with frequency. Measured as slope of the spectral envelope.

**Measurement:** dB/octave

**Characteristics:**
- **Steep tilt** (more negative): Breathy, soft, muted highs
- **Shallow tilt** (less negative): Pressed, bright, strong highs

**Voice Quality Impact:**
- Breathy voice: Steeper tilt (weak harmonics, more noise)
- Pressed voice: Shallower tilt (strong harmonics throughout)
- Modal voice: Moderate tilt

**In TTS:** Adjusting spectral tilt changes perceived voice "brightness" and breathiness.

```python
# Spectral tilt control
spectral_tilt_db_octave = -12  # Slightly breathy
```

## Cepstral Peak Prominence (CPP)

**What it is:** Measures the prominence of the cepstral peak, indicating overall voice periodicity and clarity.

**Why it matters:** Unlike jitter/shimmer, CPP works for both sustained vowels AND connected speech. It's the **gold standard** for dysphonia assessment.

**Measurement:** Decibels (dB)

**Typical Ranges:**
- Clear voice: >8 dB
- Mildly dysphonic: 5-8 dB
- Moderately dysphonic: 3-5 dB
- Severely dysphonic: <3 dB

**Advantages over Jitter/Shimmer:**
1. Valid for sentences, not just vowels
2. Doesn't require F0 tracking
3. Robust to recording conditions
4. Single metric captures overall quality

**In TTS Evaluation:** Primary metric for synthetic voice quality assessment.

## Formants (F1-F4)

**What they are:** Resonant frequencies of the vocal tract that shape vowel identity and voice "color."

**Key Formants:**
| Formant | Frequency Range | Controls |
|---------|-----------------|----------|
| F1 | 250-900 Hz | Tongue height (open/close) |
| F2 | 700-2500 Hz | Tongue frontness/backness |
| F3 | 1800-3500 Hz | Lip rounding, voice color |
| F4 | 2500-4500 Hz | Individual voice quality |
| Singer's Formant | ~2800-3400 Hz | Projection, "ring" |

**Voice Character Impact:**
- Lower formants → Deeper, more resonant voice
- Higher formants → Brighter, more forward voice
- Clustered F3-F5 → "Singer's formant" for projection

**In TTS:** Formant shifting changes perceived speaker size/gender without changing pitch.

```python
# Formant shift for voice modification
formant_shift_ratio = 0.95  # Slightly larger vocal tract (deeper)
```

## Summary Table

| Parameter | Measures | Range | Low Value | High Value |
|-----------|----------|-------|-----------|------------|
| F0 | Pitch | 80-400 Hz | Deep voice | High voice |
| Jitter | Pitch stability | 0-2% | Smooth | Rough |
| Shimmer | Amplitude stability | 0-10% | Steady | Breathy |
| HNR | Clarity | 0-30 dB | Hoarse | Clear |
| Spectral Tilt | Brightness | -6 to -18 dB/oct | Bright/pressed | Soft/breathy |
| CPP | Overall quality | 0-15 dB | Dysphonic | Clear |
| F1-F4 | Resonance | Varies | Dark timbre | Bright timbre |

## References

- Titze, I.R. (1994). Principles of Voice Production
- Klatt, D.H. & Klatt, L.C. (1990). Analysis, synthesis, and perception of voice quality variations
- Phonalyze Voice Quality Metrics: https://blog.phonalyze.com/voice-quality-metrics-and-their-clinical-interpretation/
- NCVS Tutorials: https://ncvs.org/tutorials/
