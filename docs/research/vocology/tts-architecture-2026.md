# TTS Architecture 2026

State-of-the-art text-to-speech system architectures and components.

## The Modern TTS Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TEXT-TO-SPEECH PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │   Text   │ → │ Linguis- │ → │ Acoustic │ → │ Neural Vocoder   │  │
│  │  Input   │   │tic Proc. │   │  Model   │   │ (Waveform Gen)   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘  │
│       ↓              ↓              ↓                ↓               │
│  "Hello!"      Phonemes      Mel-Spectrogram     Audio Waveform     │
│               /h ə l oʊ/     [80×T matrix]       [samples]          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Text/Linguistic Frontend

**Purpose:** Convert text to linguistic features.

**Stages:**
1. **Text Normalization:** Numbers, abbreviations, symbols → words
2. **Tokenization:** Split into words/subwords
3. **G2P (Grapheme-to-Phoneme):** Text → phoneme sequences
4. **Prosody Prediction:** Stress, phrasing, emphasis

**Modern Approaches (2026):**
- Transformer-based G2P models
- End-to-end character-level models (skip explicit G2P)
- Large language model integration for context understanding

### 2. Acoustic Model

**Purpose:** Predict acoustic features from linguistic input.

**Output Types:**
- **Mel-Spectrogram:** 80-dimensional log-mel filterbank features
- **Linear Spectrogram:** Full frequency resolution
- **Audio Codes/Tokens:** Discrete neural codec tokens (emerging)

**Major Architectures (2026):**

| Model | Architecture | Key Innovation |
|-------|--------------|----------------|
| **Tacotron 2** | Seq2seq + Attention | Attention-based alignment |
| **FastSpeech 2** | Non-autoregressive | Parallel generation, explicit duration |
| **VITS** | VAE + Flow + GAN | End-to-end, high quality |
| **Grad-TTS** | Diffusion | Score-based generation |
| **NaturalSpeech 3** | DiT + FACodec | Factorized codec, disentanglement |

### 3. Neural Vocoder

**Purpose:** Convert acoustic features → audio waveform.

**Types:**

#### Autoregressive (Legacy)
- **WaveNet:** Original neural vocoder, slow
- **WaveRNN:** Faster single-RNN approach

#### GAN-Based (Current Standard)
- **HiFi-GAN:** Fast, high quality, most popular
- **BigVGAN:** Universal, handles diverse audio
- **UnivNet:** Multi-resolution spectrogram discriminator

#### Diffusion-Based (Emerging)
- **DiffWave:** Diffusion for waveform
- **WaveGrad:** Gradient-based diffusion

#### Flow-Based
- **WaveGlow:** Flow-based, parallel generation

**2026 Vocoder Comparison:**

| Vocoder | Speed (RTF) | Quality (MOS) | VRAM |
|---------|-------------|---------------|------|
| HiFi-GAN | 50-100x RT | 4.2-4.5 | ~1GB |
| BigVGAN | 30-50x RT | 4.3-4.6 | ~2GB |
| Vocos | 100x+ RT | 4.0-4.3 | <1GB |

## Mel-Spectrogram Details

```python
# Typical mel-spectrogram parameters
MEL_CONFIG = {
    "sample_rate": 24000,      # Audio sample rate
    "n_fft": 1024,             # FFT window size
    "hop_length": 256,         # Samples between frames
    "win_length": 1024,        # Window length
    "n_mels": 80,              # Number of mel bands
    "fmin": 0,                 # Minimum frequency
    "fmax": 12000,             # Maximum frequency (Nyquist/2)
}

# Results in:
# - ~93 frames per second of audio
# - 80-dimensional feature vector per frame
# - Compact representation of spectral content
```

## End-to-End Architectures

### VITS (2021, Still Relevant)
```
Text → Text Encoder → Flow → Decoder → Waveform
              ↓
    Stochastic Duration Predictor
```
- Variational inference
- Normalizing flows for expressiveness
- End-to-end training (no separate vocoder)

### Tortoise-TTS
```
Text → Autoregressive Transformer → Diffusion Decoder → Waveform
                    ↓
         Voice Conditioning (CLVP)
```
- Autoregressive for naturalness
- Diffusion for quality
- Voice cloning via conditioning

### Fish Speech V1.5 (2026)
```
Text → DualAR Transformer → Semantic Tokens → Acoustic Decoder → Waveform
              ↓                    ↓
    Speaker Embedding      Style Embedding
```
- Dual autoregressive for content + style
- 300K+ hours training data
- State-of-the-art quality

### IndexTTS2 (2026)
```
Text → GPT-style AR → Speech Tokens → Flow Decoder → Waveform
              ↓              ↓
    Timbre Prompt    Emotion Prompt
```
- Emotion-timbre disentanglement
- Duration control
- Frame-accurate timing

## Speaker Embeddings

**Purpose:** Encode speaker identity for multi-speaker or voice cloning.

**Types:**

### Pre-computed Embeddings
- **x-vector:** DNN-based, ~512 dimensions
- **d-vector:** GE2E trained, ~256 dimensions
- **ECAPA-TDNN:** State-of-the-art speaker verification

### Learned Embeddings
- **Speaker lookup table:** Fixed embedding per known speaker
- **Reference encoder:** Compute embedding from audio sample

### Zero-Shot Voice Cloning
```python
# Extract speaker embedding from reference audio
speaker_embedding = encoder.embed_speaker(reference_audio)

# Generate speech with cloned voice
audio = tts.generate(
    text="Hello world",
    speaker_embedding=speaker_embedding
)
```

## Neural Audio Codecs

**Emerging paradigm:** Replace mel-spectrograms with discrete tokens.

### Key Codecs (2026)

| Codec | Tokens/sec | Bitrate | Use Case |
|-------|------------|---------|----------|
| **EnCodec** | 75 | 1.5-24 kbps | Compression + TTS |
| **DAC** | 86 | 8 kbps | High quality |
| **Mimi** | 12.5 | Ultra-low | LLM integration |
| **SpeechTokenizer** | 50 | Semantic | Content extraction |
| **FACodec** | Variable | Factorized | Disentanglement |

### Advantages of Codec-Based TTS
1. LLM-compatible (discrete tokens)
2. Lower compute for acoustic model
3. Natural disentanglement
4. Streaming-friendly

## Latency Considerations

### Streaming TTS Pipeline
```
Text Chunk → Acoustic Model → Vocoder → Audio Chunk
    ↓              ↓             ↓
  ~10ms         ~50ms        ~20ms

Total first-chunk latency: ~80-150ms
```

### Latency Optimization Techniques
1. **Chunked processing:** Generate audio in segments
2. **Causal models:** No future context dependency
3. **Speculative decoding:** Parallel candidate generation
4. **Distillation:** Smaller, faster student models

## 2026 State-of-the-Art Models

| Model | Company/Lab | Strength | Open Source |
|-------|-------------|----------|-------------|
| **Fish Speech V1.5** | FishAudio | Quality + Speed | Yes |
| **CosyVoice2** | Alibaba | 150ms streaming | Yes |
| **IndexTTS2** | Index-TTS | Emotion control | Yes |
| **Chatterbox** | Resemble AI | Emotion + cloning | Yes |
| **XTTS-v2** | Coqui | Multilingual | Yes |
| **Eleven Labs** | ElevenLabs | Production quality | No |
| **Azure Neural TTS** | Microsoft | Enterprise scale | No |

## References

- Deep Learning Speech Synthesis (Wikipedia): https://en.wikipedia.org/wiki/Deep_learning_speech_synthesis
- Best Open-Source TTS Models 2026: https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models
- HiFi-GAN Paper: https://arxiv.org/abs/2010.05646
- VITS Paper: https://arxiv.org/abs/2106.06103
- Fish Speech: https://github.com/fishaudio/fish-speech
