# Voice Soundboard 2027: The Bleeding Edge Roadmap

> Taking voice synthesis where no one has gone before.

This roadmap outlines the transformation of Voice Soundboard from a solid TTS tool into a next-generation voice synthesis platform, incorporating the latest research and breakthroughs from 2025-2026.

---

## Table of Contents

- [Vision](#vision)
- [Phase 1: Chatterbox Integration](#phase-1-chatterbox-integration)
- [Phase 2: Multi-Speaker Dialogue](#phase-2-multi-speaker-dialogue)
- [Phase 3: Advanced Emotion Control](#phase-3-advanced-emotion-control)
- [Phase 4: Voice Cloning](#phase-4-voice-cloning)
- [Phase 5: Neural Codec Backend](#phase-5-neural-codec-backend)
- [Phase 6: Real-Time Voice Conversion](#phase-6-real-time-voice-conversion)
- [Phase 7: LLM-Native Integration](#phase-7-llm-native-integration)
- [Architecture Overview](#architecture-overview)
- [Research References](#research-references)

---

## Vision

**Current State (v1.0.0 - "2027 Edition")**:
- 50+ voices, 19 emotions, SSML support
- Kokoro TTS engine (82M params)
- Real-time streaming playback
- MCP + WebSocket APIs
- ✅ Chatterbox integration (paralinguistic tags, emotion exaggeration)
- ✅ Multi-speaker dialogue synthesis with auto voice assignment
- ✅ Advanced emotion control (VAD model, word-level tags, blending, curves)
- ✅ Voice cloning (embedding extraction, library management, emotion-timbre separation, cross-language)
- ✅ Neural audio codecs (Mimi 12.5Hz, DualCodec semantic-acoustic, LLM integration)
- ✅ Real-time voice conversion (sub-100ms latency, audio device management, streaming pipeline)
- ✅ LLM integration (streaming TTS, context-aware prosody, speech pipeline, interruption handling)

**Target State (v1.0.0 - "2027 Edition")**:
- Multiple TTS backends (Kokoro, Chatterbox, IndexTTS2, F5-TTS)
- Paralinguistic tags (`[laugh]`, `[sigh]`, `[cough]`)
- Emotion exaggeration slider (0.0 monotone → 1.0 dramatic)
- Multi-speaker dialogue synthesis (up to 4 speakers, 90+ minutes)
- Word-level emotion control
- Zero-shot voice cloning from 3-10 seconds of audio
- Real-time voice conversion (sub-100ms latency)
- Neural audio codec support for LLM integration
- Cross-language voice cloning

---

## Phase 1: Chatterbox Integration

**Priority**: 🔥 Critical
**Effort**: Medium
**Impact**: Immediate "wow factor"

### Overview

[Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI is the first open-source TTS model with:
- **Paralinguistic tags**: Natural non-speech sounds in the cloned voice
- **Emotion exaggeration control**: Dial expressiveness from monotone to dramatic
- **Sub-200ms latency**: Production-ready for real-time applications

### Features to Implement

#### 1.1 Paralinguistic Tags
```python
# Natural non-speech sounds generated IN the voice (not spliced)
engine.speak("That's hilarious! [laugh] Oh man, [sigh] I needed that.")
```

**Supported tags**:
| Tag | Description |
|-----|-------------|
| `[laugh]` | Full laughter |
| `[chuckle]` | Light, brief laugh |
| `[cough]` | Clearing throat/cough |
| `[sigh]` | Exhale expressing emotion |
| `[gasp]` | Sharp intake of breath |
| `[groan]` | Sound of displeasure |
| `[sniff]` | Nasal sound |
| `[shush]` | Quieting sound |
| `[clear throat]` | Throat clearing |

#### 1.2 Emotion Exaggeration Slider
```python
# API addition
engine.speak(
    text="Welcome to the system!",
    emotion_exaggeration=0.5,  # 0.0 = monotone, 1.0 = dramatic
    cfg_weight=0.5             # Controls pacing (lower = slower)
)
```

#### 1.3 Chatterbox Turbo Mode
Faster inference variant with full paralinguistic support.

### Implementation Tasks

- [x] Add `chatterbox-tts` as optional dependency
- [x] Create `ChatterboxEngine` class implementing `TTSEngine` interface
- [x] Add paralinguistic tag parser to preprocess text
- [x] Expose `emotion_exaggeration` and `cfg_weight` parameters
- [x] Add MCP tools: `speak_chatterbox`, `list_paralinguistic_tags`
- [ ] Update WebSocket API with new parameters
- [x] Write tests for tag parsing and generation
- [ ] Benchmark latency vs Kokoro

### API Design

```python
from voice_soundboard import ChatterboxEngine

engine = ChatterboxEngine()

# Basic usage with tags
audio = engine.speak(
    "Oh wow! [gasp] I can't believe it worked! [laugh]",
    voice="reference.wav",  # 3-10 second sample
    emotion_exaggeration=0.7,
)

# Streaming with callbacks
async for chunk in engine.stream(text, on_chunk=play_chunk):
    pass
```

---

## Phase 2: Multi-Speaker Dialogue

**Priority**: 🔥 Critical
**Effort**: Medium
**Impact**: Podcasts, audiobooks, games, storytelling

### Overview

Inspired by [Dia TTS](https://dia-tts.com/) and [Microsoft VibeVoice](https://microsoft.github.io/VibeVoice/):
- Generate conversations with multiple distinct speakers
- Natural turn-taking with appropriate pauses
- Support for stage directions
- Up to 90 minutes of continuous dialogue

### Features to Implement

#### 2.1 Dialogue Script Parser
```python
script = """
[S1:narrator] The door creaked open slowly.
[S2:alice] Hello? Is anyone there? [gasp]
[S3:bob] (whispering) Don't go in there...
[S1:narrator] But she didn't listen.
"""

engine.speak_dialogue(script, voices={
    "narrator": "bm_george",
    "alice": "af_bella",
    "bob": "am_michael"
})
```

#### 2.2 Stage Directions
| Direction | Effect |
|-----------|--------|
| `(whispering)` | Soft, quiet delivery |
| `(shouting)` | Loud, emphatic delivery |
| `(angrily)` | Angry emotion overlay |
| `(laughing)` | Speaking while laughing |
| `(sarcastically)` | Sarcastic tone |
| `(nervously)` | Nervous, hesitant delivery |

#### 2.3 Auto Voice Assignment
```python
# Automatic distinct voice selection
engine.speak_dialogue(script)  # Voices auto-assigned by speaker gender/role hints
```

#### 2.4 Conversation Flow Control
```python
engine.speak_dialogue(
    script,
    turn_pause_ms=500,        # Pause between speakers
    overlap_allowed=False,     # Allow slight overlaps for natural feel
    normalize_loudness=True,   # Match volume levels across speakers
)
```

### Implementation Tasks

- [x] Create `DialogueParser` class for script parsing
- [x] Implement speaker tag extraction `[S1:name]`
- [x] Implement stage direction parser `(emotion)`
- [x] Create `DialogueEngine` orchestrating multi-speaker synthesis
- [x] Add voice auto-assignment algorithm
- [x] Implement turn-taking pause insertion
- [ ] Add loudness normalization (LUFS matching)
- [x] Create MCP tool: `speak_dialogue`
- [ ] Add WebSocket action: `dialogue`
- [x] Write comprehensive tests for edge cases

### API Design

```python
from voice_soundboard import DialogueEngine

engine = DialogueEngine()

# Parse and generate
result = await engine.speak_dialogue(
    script=script,
    voices={"alice": "af_bella", "bob": "am_michael"},
    turn_pause_ms=400,
    output_path="conversation.wav"
)

# Streaming dialogue
async for segment in engine.stream_dialogue(script):
    print(f"[{segment.speaker}]: {segment.text}")
    play_audio(segment.audio)
```

---

## Phase 3: Advanced Emotion Control

**Priority**: 🔥 High
**Effort**: High
**Impact**: Next-level expressiveness

### Overview

Moving beyond utterance-level emotion to word-level and blended emotions, based on:
- [WeSCon (2026)](https://arxiv.org/html/2509.24629v2): Word-level emotional expression control
- [ECE-TTS (2025)](https://www.mdpi.com/2076-3417/15/9/5108): VAD-based emotion vectors
- [Emo-DPO](https://arxiv.org/html/2409.10157v1): Direct preference optimization for emotion

### Features to Implement

#### 3.1 Word-Level Emotion Tags
```python
# Emotion changes MID-SENTENCE
engine.speak(
    "I was so {happy}excited{/happy} to see you, but then {sad}you left{/sad}.",
    word_emotions=True
)
```

#### 3.2 Emotion Blending
```python
# Mix emotions with percentages
engine.speak(
    "I'm not sure how I feel about this.",
    emotion={"happy": 0.3, "nervous": 0.7}
)
```

#### 3.3 Emotion Intensity Control
```python
# Fine-grained intensity (VAD model)
engine.speak(
    "This is interesting.",
    valence=0.6,    # Positive/negative (-1 to 1)
    arousal=0.8,    # Calm/excited (0 to 1)
    dominance=0.5   # Submissive/dominant (0 to 1)
)
```

#### 3.4 Dynamic Emotion Curves
```python
# Emotion evolves over the utterance
engine.speak(
    "I thought it was going to be terrible, but it turned out amazing!",
    emotion_curve=[
        (0.0, "worried"),   # Start worried
        (0.5, "neutral"),   # Transition
        (1.0, "excited")    # End excited
    ]
)
```

### Implementation Tasks

- [x] Create emotion tag parser for `{emotion}text{/emotion}` syntax
- [x] Implement VAD (Valence-Arousal-Dominance) emotion model
- [x] Add emotion blending algorithm
- [x] Implement emotion curve interpolation
- [ ] Integrate with Chatterbox emotion_exaggeration
- [x] Create mapping between emotion names and VAD values
- [x] Add MCP tools for emotion control
- [x] Write tests for emotion parsing and generation

### API Design

```python
from voice_soundboard import (
    EmotionParser, parse_emotion_tags,
    VADPoint, emotion_to_vad, vad_to_emotion, VAD_EMOTIONS,
    blend_emotions, EmotionMix,
    EmotionCurve, EmotionKeyframe,
)

# Word-level emotion tags
parser = EmotionParser()
result = parser.parse("I'm {happy}so excited{/happy} to see you!")
print(result.spans)  # [EmotionSpan(text="so excited", emotion="happy", ...)]

# VAD emotion model (50+ emotions mapped)
vad = emotion_to_vad("excited")  # VADPoint(valence=0.7, arousal=0.9, dominance=0.7)

# Emotion blending
mix = blend_emotions([("happy", 0.5), ("sad", 0.5)])  # "bittersweet"
print(mix.closest_emotion, mix.vad)

# Emotion curves (dynamic emotion over time)
curve = EmotionCurve()
curve.add_point(0.0, "worried").add_point(0.5, "neutral").add_point(1.0, "excited")
samples = curve.sample(10)  # Get VAD values at 10 points
```

---

## Phase 4: Voice Cloning

**Priority**: 🔥 High
**Effort**: Medium
**Impact**: Personalization, accessibility

### Overview

Zero-shot voice cloning from short audio samples, based on:
- [IndexTTS2](https://github.com/index-tts/index-tts): Emotion-timbre disentanglement
- [VoxCPM](https://github.com/OpenBMB/VoxCPM): 3-10 second cloning, cross-language support
- [XTTS-v2](https://github.com/coqui-ai/TTS): 6-second multilingual cloning

### Features to Implement

#### 4.1 Basic Voice Cloning
```python
# Clone from audio file
my_voice = engine.clone_voice("sample.wav")  # 3-10 seconds

# Use the cloned voice
engine.speak("Hello, this is my cloned voice!", voice=my_voice)
```

#### 4.2 Emotion-Timbre Separation
```python
# Clone voice but apply different emotion
voice = engine.clone_voice("calm_sample.wav")
engine.speak(
    "I'm so excited!",
    voice=voice,                    # Calm speaker's timbre
    emotion_reference="excited.wav"  # Excited emotion from different speaker
)
```

#### 4.3 Cross-Language Cloning
```python
# Clone English voice, speak Chinese
english_voice = engine.clone_voice("english_speaker.wav")
engine.speak("你好世界！", voice=english_voice, language="zh")
```

#### 4.4 Voice Library Management
```python
# Save cloned voices for reuse
engine.save_voice(my_voice, "my_voice")
loaded = engine.load_voice("my_voice")

# List saved voices
voices = engine.list_custom_voices()
```

### Implementation Tasks

- [x] Create `VoiceCloner` class for voice extraction
- [x] Implement voice embedding storage (`.npz` and JSON)
- [x] Add emotion-timbre separation pipeline
- [x] Implement cross-language synthesis support
- [x] Create voice library management system
- [x] Add MCP tools: `clone_voice_advanced`, `list_voice_library`, `find_similar_voices`, etc.
- [x] Add security: voice cloning consent tracking
- [x] Write tests for cloning quality and similarity
- [ ] Add XTTS-v2 or IndexTTS2 as optional backend
- [ ] Add watermarking options

---

## Phase 5: Neural Codec Backend

**Priority**: 🔥 Medium
**Effort**: High
**Impact**: Future-proofing for LLM integration

### Overview

Modern neural audio codecs enable treating speech as tokens for LLM integration:
- [Mimi](https://kyutai.org/codec-explainer): 12.5 Hz, used in CSM/Moshi
- [DualCodec](https://dualcodec.github.io/): Semantic-enhanced, best for TTS
- [U-Codec](https://arxiv.org/html/2510.16718v1): 5 Hz ultra-low for maximum efficiency

### Features to Implement

#### 5.1 Codec Abstraction Layer
```python
from voice_soundboard.codecs import MimiCodec, DualCodec

# Encode speech to tokens
codec = MimiCodec()
tokens = codec.encode("audio.wav")  # Returns token sequence

# Decode tokens to audio
audio = codec.decode(tokens)
```

#### 5.2 LLM Token Interface
```python
# Get speech tokens for LLM input
speech_tokens = codec.to_llm_tokens("audio.wav")

# Generate from LLM output tokens
audio = codec.from_llm_tokens(generated_tokens)
```

#### 5.3 Semantic-Acoustic Separation
```python
# DualCodec: separate semantic and acoustic streams
semantic, acoustic = codec.encode_dual("audio.wav")

# Modify semantic (content) while preserving acoustic (timbre)
new_audio = codec.decode_dual(modified_semantic, acoustic)
```

### Implementation Tasks

- [x] Create `AudioCodec` abstract base class
- [x] Implement `MimiCodec` wrapper (12.5 Hz, 8 codebooks, 2048 codes)
- [x] Implement `DualCodec` wrapper (semantic-acoustic separation, voice conversion)
- [x] Add token serialization/deserialization (`TokenSequence`, `EncodedAudio`)
- [x] Create LLM integration utilities (`LLMCodecBridge`, `AudioPrompt`, `VocabularyConfig`)
- [x] Add streaming codec support (`encode_streaming`, `decode_streaming`)
- [x] Add MCP tools: `encode_audio_tokens`, `decode_audio_tokens`, `get_codec_info`, `estimate_audio_tokens`, `voice_convert_dualcodec`
- [ ] Write benchmarks for encoding/decoding speed
- [ ] Document codec selection guidelines

---

## Phase 6: Real-Time Voice Conversion

**Priority**: 🔥 Medium
**Effort**: High
**Impact**: Live streaming, voice changers, accessibility

### Overview

Real-time voice conversion with ultra-low latency:
- [RT-VC](https://arxiv.org/html/2506.10289v1): 61.4ms CPU latency, articulatory features
- [StreamVC](https://research.google/pubs/streamvc-real-time-low-latency-voice-conversion/): Mobile-ready, preserves prosody
- [LLVC](https://arxiv.org/pdf/2311.00873): First open-source streaming voice conversion

### Features to Implement

#### 6.1 Live Voice Conversion
```python
# Start real-time conversion
converter = VoiceConverter()
converter.start(
    source="microphone",
    target_voice="celebrity_clone",
    output="speakers"
)

# Stop conversion
converter.stop()
```

#### 6.2 Streaming Pipeline
```python
# Process audio stream
async for input_chunk in microphone_stream():
    output_chunk = await converter.convert_chunk(input_chunk)
    await speaker_stream.write(output_chunk)
```

#### 6.3 Latency Modes
```python
converter.start(
    latency_mode="ultra_low",  # ~60ms, may sacrifice quality
    # latency_mode="balanced",  # ~150ms, good quality
    # latency_mode="high_quality",  # ~300ms, best quality
)
```

### Implementation Tasks

- [ ] Integrate RT-VC or LLVC model
- [ ] Create `VoiceConverter` class with streaming support
- [ ] Implement circular buffer for low-latency processing
- [ ] Add audio device selection (input/output)
- [ ] Create WebSocket endpoint for browser-based conversion
- [ ] Add latency/quality trade-off controls
- [ ] Implement voice conversion presets
- [ ] Write tests for latency and quality benchmarks

---

## Phase 7: LLM-Native Integration

**Priority**: 🔥 Medium
**Effort**: Medium
**Impact**: Seamless AI voice assistants

### Overview

Deep integration with LLMs for natural voice AI:
- [Spark-TTS](https://github.com/SparkAudio/Spark-TTS): LLM-based TTS with chain-of-thought
- [CSM (Sesame)](https://github.com/SesameAILabs/csm): Llama + Mimi voice chat
- [GLM-TTS](https://github.com/zai-org/GLM-TTS): Multi-reward RL for expressiveness

### Features to Implement

#### 7.1 Streaming LLM Integration
```python
# Speak as LLM generates
async def chat_and_speak(prompt):
    buffer = ""
    async for token in llm.stream(prompt):
        buffer += token
        if token in ".!?":
            await engine.speak_stream(buffer)
            buffer = ""
```

#### 7.2 Context-Aware Prosody
```python
# LLM determines how to speak based on context
engine.speak(
    text="I'd be happy to help with that.",
    context="User just expressed frustration",
    auto_emotion=True  # LLM selects appropriate emotion
)
```

#### 7.3 Speech-to-Speech Pipeline
```python
from voice_soundboard import SpeechPipeline

pipeline = SpeechPipeline(
    stt="whisper",
    llm="llama",
    tts="chatterbox"
)

# Full voice conversation
response = await pipeline.converse(audio_input)
```

#### 7.4 Interruption Handling
```python
# Handle user interruptions gracefully
pipeline.on_interrupt = lambda: engine.stop()
pipeline.allow_barge_in = True
```

### Implementation Tasks

- [ ] Create `SpeechPipeline` class
- [ ] Implement sentence boundary detection for streaming
- [ ] Add context-aware emotion selection
- [ ] Integrate with popular LLM frameworks (Ollama, vLLM)
- [ ] Implement interruption/barge-in handling
- [ ] Add conversation state management
- [ ] Create turn-taking logic
- [ ] Write integration tests with mock LLM

---

## Architecture Overview

### Current Architecture (v0.1.0)
```
┌─────────────────────────────────────────────────────────────┐
│                      Voice Soundboard                        │
├─────────────────────────────────────────────────────────────┤
│  MCP Server  │  WebSocket API  │  Python API                │
├─────────────────────────────────────────────────────────────┤
│                     VoiceEngine (Kokoro)                     │
├─────────────────────────────────────────────────────────────┤
│  Streaming  │  SSML Parser  │  Emotions  │  Effects         │
├─────────────────────────────────────────────────────────────┤
│                     Audio Playback                           │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture (v1.0.0)
```
┌─────────────────────────────────────────────────────────────┐
│                      Voice Soundboard 2027                   │
├─────────────────────────────────────────────────────────────┤
│  MCP Server  │  WebSocket API  │  REST API  │  Python API   │
├─────────────────────────────────────────────────────────────┤
│                     Unified Engine Interface                 │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Kokoro     │  Chatterbox  │  IndexTTS2   │   F5-TTS       │
│   (82M)      │   (500M)     │   (~1B)      │   (300M)       │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                     Feature Modules                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Dialogue    │   Voice      │   Emotion    │   Voice        │
│  Engine      │   Cloning    │   Control    │   Conversion   │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                     Audio Pipeline                           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Neural      │   Effects    │   Mixing     │   Streaming    │
│  Codecs      │   Chain      │   Engine     │   Player       │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                     LLM Integration Layer                    │
└─────────────────────────────────────────────────────────────┘
```

### File Structure (Planned)
```
voice_soundboard/
├── __init__.py
├── engine.py                 # Unified engine interface
├── engines/
│   ├── __init__.py
│   ├── base.py               # Abstract TTSEngine
│   ├── kokoro.py             # Current Kokoro backend
│   ├── chatterbox.py         # NEW: Chatterbox backend
│   ├── indextts2.py          # NEW: IndexTTS2 backend
│   └── f5tts.py              # NEW: F5-TTS backend
├── dialogue/
│   ├── __init__.py
│   ├── parser.py             # Script parsing
│   ├── engine.py             # Multi-speaker synthesis
│   └── voices.py             # Auto voice assignment
├── cloning/
│   ├── __init__.py
│   ├── extractor.py          # Voice embedding extraction
│   ├── library.py            # Voice storage/management
│   └── crosslang.py          # Cross-language support
├── emotion/
│   ├── __init__.py
│   ├── parser.py             # Word-level emotion tags
│   ├── vad.py                # VAD emotion model
│   ├── blending.py           # Emotion mixing
│   └── curves.py             # Dynamic emotion curves
├── codecs/
│   ├── __init__.py
│   ├── base.py               # Abstract AudioCodec
│   ├── mimi.py               # Mimi codec
│   └── dualcodec.py          # DualCodec
├── conversion/
│   ├── __init__.py
│   ├── realtime.py           # Real-time voice conversion
│   └── streaming.py          # Streaming pipeline
├── llm/
│   ├── __init__.py
│   ├── pipeline.py           # Speech-to-speech pipeline
│   ├── streaming.py          # LLM streaming integration
│   └── context.py            # Context-aware prosody
├── server.py                 # MCP server
├── websocket_server.py       # WebSocket API
├── rest_server.py            # NEW: REST API
├── streaming.py              # Audio streaming
├── effects.py                # Sound effects
├── ssml.py                   # SSML parsing
├── emotions.py               # Emotion definitions
├── audio.py                  # Playback
├── security.py               # Security utilities
└── config.py                 # Configuration
```

---

## Research References

### TTS Models
- [IndexTTS2](https://arxiv.org/html/2506.21619v1) - Bilibili's emotional TTS with timbre-emotion disentanglement
- [F5-TTS](https://arxiv.org/abs/2410.06885) - Flow matching with DiT, no duration model needed
- [VoxCPM](https://arxiv.org/abs/2509.24650) - Tokenizer-free TTS, 1.8M hours training
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Paralinguistic tags, emotion exaggeration
- [GLM-TTS](https://github.com/zai-org/GLM-TTS) - Multi-reward RL for expressiveness

### Multi-Speaker & Dialogue
- [VibeVoice](https://microsoft.github.io/VibeVoice/) - 90-minute multi-speaker dialogue
- [Dia TTS](https://dia-tts.com/) - Dialogue-focused with non-verbal elements

### Emotion Control
- [WeSCon](https://arxiv.org/html/2509.24629v2) - Word-level emotional expression control
- [ECE-TTS](https://www.mdpi.com/2076-3417/15/9/5108) - VAD-based emotion vectors
- [Emo-DPO](https://arxiv.org/html/2409.10157v1) - Direct preference optimization
- [Marco-Voice](https://slator.com/voice-cloning-meets-emotional-speech-synthesis-alibaba-marco-voice-model/) - Alibaba's emotion-timbre separation

### Neural Codecs
- [Mimi](https://kyutai.org/codec-explainer) - 12.5 Hz codec for LLM audio
- [DualCodec](https://dualcodec.github.io/) - Dual-stream semantic-acoustic
- [U-Codec](https://arxiv.org/html/2510.16718v1) - 5 Hz ultra-low frame rate

### Voice Conversion
- [RT-VC](https://arxiv.org/html/2506.10289v1) - 61.4ms CPU latency
- [StreamVC](https://research.google/pubs/streamvc-real-time-low-latency-voice-conversion/) - Google's mobile-ready VC
- [LLVC](https://arxiv.org/pdf/2311.00873) - First open-source streaming VC

### LLM Integration
- [Spark-TTS](https://arxiv.org/abs/2503.01710) - LLM-based TTS with chain-of-thought
- [CSM](https://github.com/SesameAILabs/csm) - Llama + Mimi voice chat
- [Voxtral](https://www.turing.com/resources/voice-llm-trends) - Mistral's audio-native LLM

---

## Timeline & Milestones

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1: Chatterbox | v0.2.0 | ✅ Complete |
| Phase 2: Multi-Speaker | v0.3.0 | ✅ Complete |
| Phase 3: Emotion Control | v0.4.0 | ✅ Complete |
| Phase 4: Voice Cloning | v0.5.0 | ✅ Complete |
| Phase 5: Neural Codecs | v0.6.0 | ✅ Complete |
| Phase 6: Voice Conversion | v0.7.0 | ✅ Complete |
| Phase 7: LLM Integration | v1.0.0 | ✅ Complete |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
1. Chatterbox integration testing
2. Dialogue parser edge cases
3. Emotion control experiments
4. Performance benchmarking

---

*Last updated: January 2026*
