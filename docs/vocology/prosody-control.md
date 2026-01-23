# Prosody Control in TTS

Controlling the rhythm, intonation, and expressiveness of synthesized speech.

## What is Prosody?

**Prosody** encompasses the suprasegmental features of speech that convey meaning beyond individual phonemes:

- **Pitch (F0):** Perceived highness/lowness
- **Duration:** Timing of segments and pauses
- **Intensity:** Loudness/energy patterns
- **Rhythm:** Patterns of stressed/unstressed syllables
- **Intonation:** Pitch contours across phrases

## Prosodic Functions

### Linguistic Functions
| Function | Example | Prosodic Cue |
|----------|---------|--------------|
| **Question vs Statement** | "You're coming?" vs "You're coming." | Rising vs falling pitch |
| **Emphasis** | "I said BLUE, not green" | Higher pitch, longer duration |
| **Phrasing** | "Let's eat, grandma" vs "Let's eat grandma" | Pause placement |
| **New vs Given** | Focus on new information | Pitch accent |

### Paralinguistic Functions
| Function | Prosodic Cue |
|----------|--------------|
| **Emotion** | Pitch range, rate, intensity patterns |
| **Attitude** | Sarcasm via exaggerated contours |
| **Speaker State** | Tiredness = slow, low; excitement = fast, high |

## Prosody Control Methods

### 1. SSML (Speech Synthesis Markup Language)

```xml
<speak>
  <!-- Pitch control -->
  <prosody pitch="+20%">Higher pitch</prosody>
  <prosody pitch="-10%">Lower pitch</prosody>
  <prosody pitch="high">Relatively high</prosody>

  <!-- Rate control -->
  <prosody rate="slow">Speaking slowly</prosody>
  <prosody rate="150%">Speaking faster</prosody>

  <!-- Volume control -->
  <prosody volume="loud">Speaking loudly</prosody>
  <prosody volume="-6dB">Quieter</prosody>

  <!-- Combined -->
  <prosody pitch="+10%" rate="slow" volume="soft">
    Gentle, high, slow speech
  </prosody>

  <!-- Pauses -->
  <break time="500ms"/>
  <break strength="strong"/>

  <!-- Emphasis -->
  <emphasis level="strong">Important</emphasis>

  <!-- Say-as for special content -->
  <say-as interpret-as="date">2026-01-23</say-as>
  <say-as interpret-as="telephone">+1-555-123-4567</say-as>
</speak>
```

### 2. Explicit Duration/Pitch Control

FastSpeech-style explicit prediction:

```python
# Duration control
phoneme_durations = duration_predictor(text_encoding)
# [10, 15, 8, 12, ...] frames per phoneme

# Pitch control
pitch_contour = pitch_predictor(text_encoding)
# [120, 125, 130, 128, ...] Hz per frame

# Modify for expressiveness
pitch_contour = pitch_contour * pitch_scale + pitch_shift
phoneme_durations = phoneme_durations * duration_scale
```

### 3. Reference Encoder (Style Tokens)

Learn prosody from reference audio:

```python
# Extract prosody from reference
prosody_embedding = reference_encoder(reference_mel)

# Apply to new synthesis
output = decoder(text_encoding, speaker_embedding, prosody_embedding)
```

### 4. VAE-Based Control

Latent space for prosody variation:

```python
# Sample from prosody latent space
z_prosody = torch.randn(batch_size, prosody_dim)

# Decode with sampled prosody
output = decoder(text_encoding, speaker_embedding, z_prosody)

# Interpolate between prosody styles
z_interp = alpha * z_style_a + (1 - alpha) * z_style_b
```

### 5. Natural Language Prompts

Modern systems support text descriptions:

```python
# Describe desired prosody in natural language
engine.speak(
    "Hello everyone!",
    style="excitedly, with rising intonation and fast pace"
)

# Or use emotion labels
engine.speak(
    "I can't believe it",
    emotion="surprised",
    intensity=0.8
)
```

## Pitch Contour Modeling

### ToBI (Tones and Break Indices)

Standard annotation system for English intonation:

| Tone | Symbol | Description |
|------|--------|-------------|
| High | H* | Peak pitch accent |
| Low | L* | Low pitch accent |
| Rising | L+H* | Rise to peak |
| Falling | H+L* | Fall from peak |
| Boundary rise | H% | Question ending |
| Boundary fall | L% | Statement ending |

### F0 Prediction Approaches

1. **Frame-level regression:** Predict F0 for each acoustic frame
2. **Phoneme-level:** Average F0 per phoneme, interpolate
3. **Quantized:** Discretize F0 into bins, treat as classification
4. **Continuous normalizing flows:** Model full F0 distribution

```python
# Frame-level F0 prediction
class PitchPredictor(nn.Module):
    def forward(self, text_encoding, speaker_embedding):
        # Predict log-F0 and voicing flag
        log_f0 = self.f0_predictor(text_encoding)
        voicing = self.voicing_predictor(text_encoding)
        return log_f0, voicing
```

## Duration Modeling

### Duration Prediction

```python
class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim):
        self.conv_layers = nn.Sequential(
            Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            LayerNorm(hidden_dim),
            ReLU(),
            Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            LayerNorm(hidden_dim),
            ReLU(),
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, text_encoding):
        # Predict duration in frames per phoneme
        x = self.conv_layers(text_encoding)
        durations = self.linear(x).squeeze(-1)
        durations = F.softplus(durations)  # Ensure positive
        return durations
```

### Length Regulator

Expand phoneme sequence to frame sequence:

```python
def length_regulate(phoneme_encoding, durations):
    """Expand phonemes to match acoustic frames."""
    # durations: [batch, phonemes] integer frame counts
    # Returns: [batch, frames, hidden]

    output = []
    for i, dur in enumerate(durations):
        output.append(phoneme_encoding[i].repeat(dur, 1))
    return torch.cat(output, dim=0)
```

## Pauses and Phrasing

### Automatic Pause Prediction

```python
# Predict pause durations at word boundaries
pause_predictor = PausePredictor()

# Input: text with word boundaries marked
text = "Hello, | world. | How are you?"

# Output: pause durations in ms
pauses = pause_predictor(text)
# [0, 200, 500, 0, 0, 0]  # Pauses after punctuation
```

### Phrase Break Types

| Break | Typical Duration | Context |
|-------|------------------|---------|
| None | 0 ms | Within word |
| Minor | 50-150 ms | Between words |
| Medium | 150-400 ms | Comma, clause boundary |
| Major | 400-800 ms | Period, paragraph |
| Long | >800 ms | Section break, dramatic pause |

## Energy/Intensity Control

```python
# Energy prediction
class EnergyPredictor(nn.Module):
    def forward(self, text_encoding):
        # Predict log energy per frame
        log_energy = self.predictor(text_encoding)
        return log_energy

# Apply energy contour
mel_spectrogram = mel_spectrogram + energy_contour.unsqueeze(-1)
```

## Word-Level Markup

Recent research on word-specific control:

```python
# Word-level prosody tags
text = """
The <emphasis>IMPORTANT</emphasis> thing is
to <slow>speak clearly</slow> and
<high_pitch>expressively</high_pitch>.
"""

# Parsed into per-word modifications
word_mods = [
    {"word": "IMPORTANT", "emphasis": 1.5, "pitch": 1.2},
    {"word": "speak", "rate": 0.7},
    {"word": "clearly", "rate": 0.7},
    {"word": "expressively", "pitch": 1.3},
]
```

## Voice Soundboard Prosody API

### Current Implementation
```python
from voice_soundboard import VoiceEngine

engine = VoiceEngine()

# Speed control
result = engine.speak("Hello world", speed=0.8)  # Slower

# Style hints (natural language)
result = engine.speak("Hello world", style="excitedly")
```

### Enhanced Prosody Control (v1.2.0+)
```python
from voice_soundboard import VoiceEngine
from voice_soundboard.prosody import ProsodyContour

engine = VoiceEngine()

# SSML support
ssml = """
<speak>
  <prosody pitch="+10%" rate="slow">
    Welcome to Voice Soundboard.
  </prosody>
  <break time="500ms"/>
  <emphasis level="strong">Amazing</emphasis> quality!
</speak>
"""
result = engine.speak_ssml(ssml)

# Explicit contour control
contour = ProsodyContour()
contour.set_pitch_range(low=100, high=200)
contour.add_emphasis(word="Amazing", pitch_boost=1.3, duration_boost=1.2)

result = engine.speak("Amazing quality!", prosody=contour)

# Reference-based prosody
result = engine.speak(
    "Hello world",
    prosody_reference="expressive_sample.wav"
)
```

## Evaluation Metrics

| Metric | Measures | Calculation |
|--------|----------|-------------|
| **F0 RMSE** | Pitch accuracy | √(Σ(pred_f0 - true_f0)²/N) |
| **F0 Correlation** | Contour shape | Pearson correlation |
| **Duration RMSE** | Timing accuracy | √(Σ(pred_dur - true_dur)²/N) |
| **Rhythm Similarity** | Overall rhythm | DTW of duration sequences |
| **MOS-Prosody** | Subjective naturalness | Human ratings |

## References

- TTS Prosody Modeling and Control: https://apxml.com/courses/speech-recognition-synthesis-asr-tts/chapter-4-advanced-text-to-speech-synthesis/prosody-modeling-control-tts
- SSML Specification: https://www.w3.org/TR/speech-synthesis/
- FastSpeech 2: https://arxiv.org/abs/2006.04558
- Word-level Prosody Control: https://www.isca-archive.org/interspeech_2024/korotkova24_interspeech.pdf
- Prosody-TTS: https://arxiv.org/abs/2110.02854
