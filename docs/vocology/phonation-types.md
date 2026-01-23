# Phonation Types

Phonation refers to how the vocal folds vibrate to produce voice. Different phonation types create distinct voice qualities.

## The Phonation Continuum

Voice quality exists on a continuum based on glottal aperture (how open/closed the vocal folds are):

```
VOICELESS → BREATHY → MODAL → CREAKY → GLOTTAL STOP
   ↓           ↓         ↓        ↓          ↓
 (open)    (partly    (normal)  (tight)   (closed)
            open)
```

## Primary Phonation Types

### 1. Modal Voice

**Description:** The default, "normal" phonation with regular vocal fold vibration.

**Characteristics:**
- Vocal folds fully adducted (closed during vibration)
- Regular, periodic vibration
- Clear harmonic structure
- Balanced airflow

**Acoustic Markers:**
- Low jitter (<1%)
- Low shimmer (<3%)
- High HNR (>20 dB)
- Strong harmonics throughout spectrum

**Perceptual Quality:** Clear, natural, neutral

**TTS Use:** Default voice quality for most synthesis.

---

### 2. Breathy Voice

**Description:** Produced with incomplete vocal fold closure, allowing air to escape.

**Characteristics:**
- Glottis remains partially open
- Air turbulence creates noise
- Reduced vocal fold contact
- Lower subglottal pressure

**Acoustic Markers:**
- High shimmer (5-10%)
- Low HNR (10-15 dB)
- Steep spectral tilt
- Weak high-frequency harmonics
- Visible aspiration noise

**Perceptual Quality:** Soft, intimate, airy, sexy, vulnerable

**Associated Terms:** Lax, slack, murmured, aspirated

**TTS Use:**
- ASMR content
- Intimate narration
- Romantic characters
- Whispering effects

```python
# Breathy voice parameters
breathiness = {
    "shimmer": 0.08,      # 8%
    "hnr": 12,            # dB
    "spectral_tilt": -15, # dB/octave
    "aspiration_noise": 0.3
}
```

---

### 3. Creaky Voice (Vocal Fry)

**Description:** Very low frequency, irregular vibration with tightly adducted vocal folds.

**Characteristics:**
- Strongly adducted vocal folds
- Low longitudinal tension
- Very low F0 (often 20-70 Hz)
- Irregular pulse timing
- Thick vocal folds

**Acoustic Markers:**
- High jitter (often >2%)
- Irregular period lengths
- Subharmonics or period doubling
- Low F0, often below normal range

**Perceptual Quality:** Gravelly, popping, rattling, authoritative, casual

**Associated Terms:** Vocal fry, laryngealization, glottalization, stiff voice

**TTS Use:**
- Casual speech patterns
- Authority/gravitas
- Certain accents (American English end-of-sentence)
- Character voices

```python
# Creaky voice parameters
creakiness = {
    "jitter": 0.03,       # 3%
    "f0_floor": 50,       # Hz
    "irregularity": 0.4,
    "subharmonics": True
}
```

---

### 4. Harsh/Pressed Voice

**Description:** Produced with very high tension in the vocal folds and larynx.

**Characteristics:**
- Extremely strong medial compression
- Excessive adductive tension
- Ventricular fold involvement (sometimes)
- High subglottal pressure

**Acoustic Markers:**
- Irregular pitch and amplitude
- Noise throughout spectrum
- Reduced harmonic clarity
- High spectral energy
- Shallow spectral tilt

**Perceptual Quality:** Strained, tense, effortful, aggressive

**Associated Terms:** Tense voice, pressed phonation, tight voice

**TTS Use:**
- Anger expression
- Shouting/yelling
- Intense emotional states
- Stressed characters

```python
# Harsh voice parameters
harshness = {
    "tension": 0.8,
    "spectral_tilt": -6,  # Shallow
    "noise_injection": 0.2,
    "intensity_boost": 1.3
}
```

---

### 5. Falsetto

**Description:** High-pitched phonation with stretched, thin vocal folds.

**Characteristics:**
- Vocal folds stretched longitudinally
- Thin, stiff edges
- Only ligamental portion vibrates
- Limited dynamic range

**Acoustic Markers:**
- High F0 (often >300 Hz for males)
- Relatively weak harmonics
- Breathy quality possible
- Limited amplitude variation

**Perceptual Quality:** Light, airy, high, ethereal

**TTS Use:**
- Surprise/shock expressions
- Character voices
- Singing synthesis
- Comedy effects

---

### 6. Whisper

**Description:** Voiceless speech produced with turbulent airflow through a narrowed glottis.

**Characteristics:**
- Vocal folds do not vibrate
- Turbulent noise at glottis
- Articulation preserved
- No fundamental frequency

**Acoustic Markers:**
- No harmonic structure
- Broadband noise
- Formant patterns preserved in noise
- No measurable F0

**Perceptual Quality:** Secretive, intimate, quiet

**TTS Use:**
- ASMR
- Dramatic effect
- Secretive dialogue
- Stealth/quiet scenes

---

## Mixed Phonation Types

### Whispery Voice (Murmur)
Combination of whisper + modal: Some vocal fold vibration with added turbulence.

### Creaky Falsetto
High pitch with irregular vibration.

### Breathy Creaky
Low, airy voice with irregular pulses.

## Phonation Type Summary

| Type | Glottis | F0 | Jitter | Shimmer | HNR | Perceptual |
|------|---------|-----|--------|---------|-----|------------|
| Modal | Normal | Normal | Low | Low | High | Clear, neutral |
| Breathy | Open | Normal | Low | High | Low | Soft, airy |
| Creaky | Tight | Very low | High | Variable | Variable | Gravelly |
| Harsh | Very tight | Variable | High | High | Low | Tense, strained |
| Falsetto | Stretched | High | Low | Low | Medium | Light, airy |
| Whisper | Narrow, no vibration | None | N/A | N/A | Very low | Quiet, secretive |

## Implementation in Voice Soundboard

```python
from voice_soundboard import VoiceEngine

engine = VoiceEngine()

# Different phonation types via parameters
engine.speak("Normal voice", phonation="modal")
engine.speak("Soft and intimate", phonation="breathy", breathiness=0.6)
engine.speak("Casual and relaxed", phonation="creaky", creakiness=0.4)
engine.speak("Angry and intense", phonation="harsh", tension=0.8)
engine.speak("Quiet secret", phonation="whisper")
```

## References

- Laver, J. (1980). The Phonetic Description of Voice Quality
- Gordon, M. & Ladefoged, P. (2001). Phonation types: a cross-linguistic overview
- Garellek, M. (2019). The phonetics of voice
- Voice Quality Symbols (VoQS): https://en.wikipedia.org/wiki/Voice_Quality_Symbols
