# Voice Characteristics & Descriptors

A comprehensive taxonomy of voice qualities for character design, casting, and synthesis.

## The Voice Equation

```
Voice Quality = Vocal Tract Shape + Laryngeal Anatomy + Learned Behavior
                      ↓                    ↓                  ↓
               (resonance)           (phonation)         (style)
```

## Primary Voice Dimensions

### 1. Pitch (Frequency)

| Descriptor | F0 Range | Character Association |
|------------|----------|----------------------|
| Very deep | <100 Hz | Powerful, ominous, mature |
| Deep | 100-130 Hz | Authoritative, calm, masculine |
| Medium-low | 130-165 Hz | Trustworthy, warm |
| Medium | 165-200 Hz | Neutral, versatile |
| Medium-high | 200-250 Hz | Friendly, approachable |
| High | 250-350 Hz | Youthful, energetic |
| Very high | >350 Hz | Childlike, excited |

### 2. Resonance (Where Sound Vibrates)

| Type | Location | Sound Quality | Character Type |
|------|----------|---------------|----------------|
| **Chest voice** | Lower resonators | Deep, powerful, rich | Leaders, villains |
| **Head voice** | Upper resonators | Light, bright, clear | Innocent, ethereal |
| **Nasal** | Nasal cavity | Twangy, pointed | Nerdy, whiny |
| **Throaty** | Pharynx | Guttural, raw | Tough, rugged |
| **Forward/Bright** | Front of face | Projecting, clear | Announcers, teachers |
| **Back/Dark** | Back of throat | Mellow, subdued | Jazz singers, philosophers |

### 3. Timbre (Tonal Color)

The unique "fingerprint" that distinguishes one voice from another.

#### Timbre Categories

**Bright Timbres:**
- Clear, cutting, forward
- Strong high-frequency harmonics
- Associated with: youth, energy, alertness

**Dark Timbres:**
- Rich, mellow, subdued
- Weak high-frequency harmonics
- Associated with: maturity, calm, mystery

**Warm Timbres:**
- Full, inviting, comforting
- Balanced harmonic spectrum
- Associated with: trustworthiness, friendliness

**Cold Timbres:**
- Thin, distant, clinical
- Sparse harmonic content
- Associated with: detachment, technology, villains

## Texture Descriptors

### Smooth Textures
| Term | Description | Use Case |
|------|-------------|----------|
| **Smooth** | Even, flowing, no roughness | Narration, luxury brands |
| **Silky** | Luxurious, gentle, flowing | Romance, ASMR |
| **Velvety** | Rich, soft, enveloping | Jazz, late night |
| **Honeyed** | Sweet, pleasant, melodic | Children's content |
| **Dulcet** | Sweet, soothing, musical | Meditation, relaxation |

### Rough Textures
| Term | Description | Use Case |
|------|-------------|----------|
| **Raspy** | Friction, slight roughness | Rock singers, tough characters |
| **Gravelly** | Like gravel, coarse | Anti-heroes, aged characters |
| **Gritty** | Raw, textured, edgy | Action, drama |
| **Husky** | Deep + rough, often sensual | Film noir, seduction |
| **Hoarse** | Strained, damaged quality | Illness, exhaustion |
| **Scratchy** | Dry, irritated sound | Elderly, sick characters |

### Airy Textures
| Term | Description | Use Case |
|------|-------------|----------|
| **Breathy** | Air escaping, soft | Intimacy, ASMR |
| **Airy** | Light, ethereal | Fantasy, dreams |
| **Wispy** | Thin, delicate | Ghosts, whispers |
| **Feathery** | Extremely light | Fairy characters |

### Strong Textures
| Term | Description | Use Case |
|------|-------------|----------|
| **Booming** | Loud, resonant, powerful | Gods, announcements |
| **Thunderous** | Deep, commanding | Villains, authority |
| **Stentorian** | Extremely loud, clear | Military, formal |
| **Resonant** | Ringing, full | Leaders, orators |

## Emotional/Character Qualities

### Positive Qualities
| Quality | Voice Features | Use |
|---------|---------------|-----|
| **Warm** | Low-mid pitch, full resonance | Friendly characters |
| **Friendly** | Varied pitch, moderate pace | Approachable |
| **Enthusiastic** | Higher pitch, fast, varied | Excitement |
| **Confident** | Steady, moderate pace, full | Leadership |
| **Soothing** | Slow, low, steady | Meditation, comfort |
| **Cheerful** | Higher pitch, upward inflections | Happy content |

### Negative Qualities
| Quality | Voice Features | Use |
|---------|---------------|-----|
| **Cold** | Flat, monotone, clipped | Villains, robots |
| **Menacing** | Low, slow, deliberate | Threats |
| **Nervous** | Fast, high, unsteady | Anxious characters |
| **Sarcastic** | Exaggerated inflection | Comedy |
| **Whiny** | High, nasal, complaining | Annoying characters |
| **Gruff** | Low, harsh, abrupt | Tough characters |

## Age Characteristics

### Young Voices (Child - Teen)
- Higher F0 (250-400+ Hz)
- Higher formants
- Less controlled breath
- Thinner timbre
- More pitch variation

### Adult Voices (20-50)
- Stable F0
- Full harmonic development
- Controlled breath support
- Maximum dynamic range

### Aged Voices (60+)
- F0 may rise (males) or lower (females)
- Increased jitter/shimmer
- Reduced breath capacity
- Thinner timbre
- Potential tremor
- Raspier quality

## Gender Characteristics

### Typical Male Features
- Lower F0 (85-180 Hz)
- Lower formants
- Larger vocal tract resonance
- Can access very low chest voice

### Typical Female Features
- Higher F0 (165-255 Hz)
- Higher formants
- Brighter timbre tendency
- More pitch variation (often)

### Androgynous Features
- Mid-range F0 (150-200 Hz)
- Ambiguous formant placement
- Neutral resonance balance

## Accent & Regional Qualities

Accents affect:
- Vowel formant patterns
- Consonant articulation
- Prosodic patterns (rhythm, intonation)
- Speaking rate

Not covered in detail here - see dedicated accent documentation.

## Voice Soundboard Descriptors Mapping

```python
# Map descriptive terms to synthesis parameters
VOICE_DESCRIPTORS = {
    # Texture
    "smooth": {"jitter": 0.003, "shimmer": 0.02, "hnr": 22},
    "raspy": {"jitter": 0.015, "shimmer": 0.06, "noise": 0.1},
    "breathy": {"shimmer": 0.08, "hnr": 12, "aspiration": 0.3},
    "gravelly": {"jitter": 0.02, "creaky": 0.5, "f0_variation": 0.15},

    # Resonance
    "deep": {"formant_shift": 0.9, "f0_shift": 0.85},
    "bright": {"formant_shift": 1.1, "spectral_tilt": -8},
    "warm": {"formant_shift": 0.95, "hnr": 20, "shimmer": 0.03},
    "nasal": {"nasal_coupling": 0.6},

    # Character
    "authoritative": {"f0_shift": 0.9, "pace": 0.9, "steadiness": 0.9},
    "friendly": {"f0_shift": 1.05, "variation": 0.3, "warmth": 0.7},
    "menacing": {"f0_shift": 0.8, "pace": 0.7, "creaky": 0.3},
}
```

## Creating Character Voices

### Example: Wise Old Mentor
```python
mentor_voice = {
    "f0_shift": 0.85,        # Lower pitch
    "jitter": 0.012,         # Slight age tremor
    "shimmer": 0.04,         # Some breathiness
    "pace": 0.85,            # Slower, deliberate
    "formant_shift": 0.95,   # Slightly larger resonance
    "warmth": 0.8            # Warm, kind quality
}
```

### Example: Energetic Young Hero
```python
hero_voice = {
    "f0_shift": 1.1,         # Slightly higher
    "jitter": 0.004,         # Very smooth
    "pace": 1.15,            # Faster, energetic
    "variation": 0.4,        # Expressive pitch range
    "brightness": 0.7        # Forward, projecting
}
```

### Example: Mysterious Villain
```python
villain_voice = {
    "f0_shift": 0.8,         # Deep
    "creaky": 0.3,           # Slight vocal fry
    "pace": 0.8,             # Slow, deliberate
    "variation": 0.15,       # Controlled, measured
    "darkness": 0.8          # Back resonance
}
```

## References

- Master List of Words to Describe Voices: https://www.bryndonovan.com/2015/12/14/master-list-of-words-to-describe-voices/
- 250+ Ways to Describe Voices: https://kathysteinemann.com/Musings/voices/
- Voquent Vocal Traits Guide: https://www.voquent.com/blog/vocal-characteristics-how-should-your-character-sound/
- Voice Types (Kennedy Center): https://www.kennedy-center.org/education/resources-for-educators/classroom-resources/media-and-interactives/media/opera/understanding-different-voice-types/
