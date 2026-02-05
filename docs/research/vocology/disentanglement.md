# Voice Disentanglement

Separating independent voice components for fine-grained control in speech synthesis.

## What is Disentanglement?

**Disentanglement** is the process of separating a complex voice signal into independent, controllable components. This allows mixing and matching different aspects of voice.

```
┌─────────────────────────────────────────────────────────────┐
│                    VOICE DISENTANGLEMENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Speech Signal                                              │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────┐                                                │
│   │ Encoder │                                                │
│   └────┬────┘                                                │
│        │                                                     │
│        ├────────┬────────┬────────┬────────┐                │
│        ▼        ▼        ▼        ▼        ▼                │
│   ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐        │
│   │Content ││ Timbre ││Prosody ││Emotion ││ Phase  │        │
│   │  (what)││ (who)  ││ (how)  ││(feeling││(timing)│        │
│   └────────┘└────────┘└────────┘└────────┘└────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Content (Linguistic Information)
**What is being said**

- Phoneme sequence
- Word identity
- Pronunciation

**Representation:** Text, phonemes, semantic tokens

**Independence:** Should be speaker-agnostic

### 2. Timbre (Speaker Identity)
**Who is speaking**

- Vocal tract shape
- Fundamental voice quality
- Consistent across utterances

**Representation:** Speaker embedding (256-512 dim vector)

**Independence:** Should be content-agnostic

### 3. Prosody (Speaking Style)
**How it's being said**

- Pitch contour (F0)
- Duration/rhythm
- Energy patterns
- Phrasing

**Representation:** Prosody embedding or explicit F0/duration

**Independence:** Partially correlated with content and emotion

### 4. Emotion (Affective State)
**The feeling behind speech**

- Emotional category (happy, sad, angry, etc.)
- Arousal level
- Valence

**Representation:** Emotion embedding or categorical label

**Independence:** Affects prosody but separable from timbre

### 5. Acoustic Details
**Fine-grained audio characteristics**

- Phase information
- Micro-variations
- Recording conditions

**Representation:** Low-level acoustic features

## Disentanglement Techniques

### 1. Adversarial Training

Use gradient reversal to prevent information leakage.

```python
# Prevent speaker info in content encoder
content_encoder = ContentEncoder()
speaker_classifier = SpeakerClassifier()

# Forward pass
content_features = content_encoder(mel_spectrogram)
speaker_pred = speaker_classifier(GradientReversal(content_features))

# Loss encourages content features to be speaker-independent
loss = content_loss + adversarial_weight * speaker_classification_loss
```

### 2. Information Bottleneck

Limit capacity to force compression.

```python
# Force content encoder through narrow bottleneck
class ContentEncoder(nn.Module):
    def __init__(self):
        self.encoder = Encoder(output_dim=64)  # Small dimension
        # Can't fit speaker info in 64 dims if content needs it all
```

### 3. Separate Encoders

Dedicated encoder for each component.

```python
# SpeechSplit-style architecture
content_embedding = content_encoder(speech)
rhythm_embedding = rhythm_encoder(speech)
pitch_embedding = pitch_encoder(speech)
timbre_embedding = timbre_encoder(speech)

# Reconstruct with any combination
reconstructed = decoder(content_embedding,
                       rhythm_embedding,
                       pitch_embedding,
                       new_timbre_embedding)  # Voice conversion!
```

### 4. Contrastive Learning

Learn embeddings where similar items cluster together.

```python
# Contrastive loss for speaker embedding
# Same speaker utterances should be close
# Different speaker utterances should be far

positive_pairs = [(spk_a_utt1, spk_a_utt2), ...]
negative_pairs = [(spk_a_utt1, spk_b_utt1), ...]

loss = contrastive_loss(speaker_encoder, positive_pairs, negative_pairs)
```

### 5. Flow-Based Separation

Normalizing flows for invertible transformations.

```python
# VITS-style flow
z = flow.forward(mel_spectrogram, speaker_embedding)
# z is content, disentangled from speaker

# Reconstruct with different speaker
mel_new_speaker = flow.reverse(z, new_speaker_embedding)
```

## State-of-the-Art Systems (2026)

### Marco-Voice
**Speaker-Emotion Disentanglement**

```
Speech → Encoders → [Speaker Emb] + [Emotion Emb] + [Content]
                           ↓              ↓
                    Contrastive      Adaptive
                      Learning       Attention
```

- Achieves 0.8275 speaker similarity (highest)
- Independent control over emotion and identity
- Adaptive cross-attention for style blending

### IndexTTS2
**Timbre-Emotion Separation**

```
Text + Timbre Prompt + Style Prompt → Decoder → Speech
              ↓              ↓
    (voice identity)  (emotion/prosody)
```

- Zero-shot timbre cloning
- Explicit emotion prompt
- Frame-accurate duration control

### Mega-TTS
**Four-Component Disentanglement**

```
Speech = Content + Timbre + Prosody + Phase
```

- Uses mel-spectrogram intermediate
- Each component separately controllable
- Large-scale training (>100K hours)

### DIS-Vector
**Fine-Grained Control**

```
Speech → [Content] + [Pitch] + [Rhythm] + [Timbre]
              ↓          ↓         ↓          ↓
         Phonemes    F0 curve   Duration   Identity
```

- Separate embedding space per component
- Zero-shot cross-lingual voice cloning
- Precise attribute manipulation

## Practical Applications

### Voice Conversion
```python
# Take content from speaker A, timbre from speaker B
content = content_encoder(speaker_a_audio)
timbre = timbre_encoder(speaker_b_audio)
converted = decoder(content, timbre)
```

### Emotion Transfer
```python
# Same speaker, different emotion
content = content_encoder(neutral_audio)
timbre = timbre_encoder(neutral_audio)  # Same speaker
emotion = emotion_encoder(angry_reference)  # Different emotion
result = decoder(content, timbre, emotion)
```

### Prosody Cloning
```python
# Clone speaking style from reference
content = text_to_content("Hello world")
timbre = my_voice_embedding
prosody = prosody_encoder(expressive_reference)
result = decoder(content, timbre, prosody)
```

### Cross-Lingual Voice Cloning
```python
# Speak French with English speaker's voice
french_content = french_text_encoder("Bonjour le monde")
english_timbre = timbre_encoder(english_speaker_audio)
result = decoder(french_content, english_timbre)
```

## Implementation in Voice Soundboard

### Current Capabilities (v1.1.0)
- Speaker selection (Kokoro voices)
- Style hints (natural language)
- Emotion exaggeration (Chatterbox)
- Voice cloning (F5-TTS)

### Enhanced Disentanglement (v1.2.0+)
```python
from voice_soundboard import VoiceEngine
from voice_soundboard.disentangle import (
    extract_timbre,
    extract_prosody,
    extract_emotion
)

engine = VoiceEngine()

# Extract components from reference
timbre = extract_timbre("reference_speaker.wav")
prosody = extract_prosody("expressive_reading.wav")
emotion = extract_emotion("happy_sample.wav")

# Synthesize with mixed components
result = engine.speak(
    "Hello world!",
    timbre=timbre,
    prosody=prosody,
    emotion=emotion
)
```

## Evaluation Metrics

| Metric | Measures | Target |
|--------|----------|--------|
| **Speaker Similarity** | Timbre preservation | >0.80 |
| **Content WER** | Linguistic accuracy | <5% |
| **Emotion Accuracy** | Emotion classification | >85% |
| **Prosody Correlation** | F0/duration match | >0.90 |
| **MOS** | Overall naturalness | >4.0 |

## Challenges

1. **Entanglement Leakage:** Components may still correlate
2. **Training Data:** Need diverse, labeled data
3. **Quality vs Control Tradeoff:** More disentanglement can reduce quality
4. **Cross-Domain Generalization:** May not work on unseen speakers/emotions

## References

- Marco-Voice Technical Report: https://arxiv.org/html/2508.02038v2
- Voice Cloning Survey 2025: https://arxiv.org/html/2505.00579v1
- DIS-Vector Project: https://github.com/NN-Project-1/dis-Vector-Embedding
- IndexTTS2: https://github.com/index-tts/index-tts
- SpeechSplit: https://arxiv.org/abs/2004.11284
