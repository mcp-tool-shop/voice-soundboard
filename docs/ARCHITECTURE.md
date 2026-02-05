# Architecture & Core Invariants

This document defines what Voice Soundboard **is** at the engine level.
These abstractions are frozen. Changes require a major version bump.

---

## Core Lifecycle

Every speech generation follows this pipeline. Stages run in order.
Each stage has a single responsibility and a defined contract.

```
Text ──> Normalize ──> Interpret ──> Resolve ──> Synthesize ──> Output
  │          │             │            │            │            │
  │     normalizer.py  interpreter.py  config.py   engine.py   audio.py
  │                       ssml.py      emotions.py
  │
  └── or SSML input (parsed first, then enters at Normalize)
```

### Stage 1: Input

**Contract:** Accepts `str` (plain text or SSML).

- Plain text passes through directly.
- SSML is parsed via `parse_ssml()` into plain text + `SSMLParams`.
- SSML parsing uses `defusedxml` (XXE-safe). This is non-negotiable.

### Stage 2: Normalize

**Contract:** `str -> str`. Pure function. No side effects.

- Expands currency, abbreviations, emojis, math symbols.
- Controlled by `normalize=True` (default) on `speak()`.
- Skippable. Never changes meaning, only pronunciation.

### Stage 3: Interpret

**Contract:** `(style, emotion, preset, voice, speed) -> (voice, speed, preset)`.

Priority order (highest wins):
1. Explicit `voice=` / `speed=` arguments
2. `preset=` resolution (maps to voice + speed)
3. `style=` natural language interpretation
4. `emotion=` parameter defaults
5. `Config` defaults (`af_bella`, speed 1.0)

This priority order is an invariant. It must never change.

### Stage 4: Resolve

**Contract:** All parameters are concrete after this stage.

- `voice` is a valid Kokoro voice ID (validated against model)
- `speed` is clamped to `[0.5, 2.0]`
- No Optional values remain

### Stage 5: Synthesize

**Contract:** `(text, voice, speed) -> (samples: np.ndarray, sample_rate: int)`.

- Delegates to the loaded TTS engine (`kokoro_onnx` by default).
- Engine is lazy-loaded on first call.
- Must produce valid PCM float32 audio samples.
- Must set `sample_rate` (Kokoro: 24000 Hz).

### Stage 6: Output

**Contract:** `(samples, sample_rate) -> SpeechResult`.

- Audio is saved as WAV to `Config.output_dir`.
- Filename is either user-provided (sanitized) or hash-based.
- `SpeechResult` contains: `audio_path`, `duration_seconds`, `generation_time`, `voice_used`, `sample_rate`, `realtime_factor`.

---

## Frozen Interfaces

### VoiceEngine

```python
class VoiceEngine:
    def __init__(self, config: Config | None = None): ...
    def speak(self, text, *, voice=None, preset=None, speed=None,
              style=None, emotion=None, save_as=None, normalize=True) -> SpeechResult: ...
    def speak_raw(self, text, *, voice=None, speed=1.0, normalize=True) -> tuple[np.ndarray, int]: ...
    def list_voices(self) -> list[str]: ...
    def list_presets(self) -> dict: ...
    def get_voice_info(self, voice: str) -> dict: ...
```

**Invariants:**
- `speak()` always returns a `SpeechResult` with a valid file path.
- `speak_raw()` always returns `(samples, sample_rate)` with no file I/O.
- Engine is lazy-loaded. Construction never downloads or loads models.
- `speak()` never raises on valid input. Invalid input raises `ValueError`.
- Missing models raise `FileNotFoundError` with download instructions.

### TTSEngine (Abstract Base)

```python
class TTSEngine(ABC):
    @property
    def name(self) -> str: ...
    @property
    def capabilities(self) -> EngineCapabilities: ...
    def speak(self, text, voice=None, speed=1.0, **kwargs) -> EngineResult: ...
    def speak_raw(self, text, voice=None, speed=1.0, **kwargs) -> tuple[np.ndarray, int]: ...
    def list_voices(self) -> list[str]: ...
```

**Invariants:**
- Every engine must implement `speak()`, `speak_raw()`, `list_voices()`.
- `EngineCapabilities` declares what the engine supports. No lying.
- `speak_raw()` must not write files. It returns in-memory audio only.

### StreamingEngine

```python
class StreamingEngine:
    async def stream(self, text, ...) -> AsyncGenerator[StreamChunk, None]: ...
    async def stream_to_file(self, text, output_path, ...) -> StreamResult: ...
```

**Invariants:**
- `StreamChunk.is_final=True` is the last chunk. Always sent exactly once.
- Chunks are yielded in order. No reordering.
- `StreamResult.total_chunks` matches the number of yielded chunks.

### SpeechResult

```python
@dataclass
class SpeechResult:
    audio_path: Path           # Always exists on disk after speak()
    duration_seconds: float    # > 0 for non-empty input
    generation_time: float     # Wall clock seconds for synthesis
    voice_used: str            # The actual voice ID used
    sample_rate: int           # Always 24000 for Kokoro
    realtime_factor: float     # duration / generation_time
```

**Invariants:**
- `audio_path` points to a readable `.wav` file.
- `realtime_factor > 0` for any successful synthesis.
- `voice_used` is always the resolved voice, never None.

### Config

```python
@dataclass
class Config:
    output_dir: Path       # Resolved from VOICE_SOUNDBOARD_DIR or cwd
    cache_dir: Path        # Resolved from VOICE_SOUNDBOARD_DIR or cwd
    model_dir: Path        # Resolved from VOICE_SOUNDBOARD_MODELS or base/models
    device: str            # "cuda" or "cpu" (auto-detected)
    default_voice: str     # "af_bella"
    default_speed: float   # 1.0
    sample_rate: int       # 24000
    use_gpu: bool          # True
    cache_models: bool     # True
```

**Invariants:**
- `output_dir` is created on construction. Always writable.
- `model_dir` is not created on construction (models may not exist yet).
- `device` is auto-detected but can be overridden.
- `to_dict()` produces a JSON-serializable dict.

---

## What Cannot Change Without a Major Version

1. The lifecycle stage order (Normalize -> Interpret -> Resolve -> Synthesize -> Output)
2. The parameter priority order (explicit > preset > style > emotion > defaults)
3. `SpeechResult` field names and types
4. `VoiceEngine.speak()` return type
5. `VoiceEngine.speak_raw()` return type
6. Default voice (`af_bella`) and default speed (`1.0`)
7. WAV as the default output format
8. Lazy model loading (construction is always cheap)
