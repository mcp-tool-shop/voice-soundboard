# API Stability Guarantees

Every public symbol in Voice Soundboard has a stability level. This tells you
whether it's safe to depend on across releases.

The current API version is available as `voice_soundboard.API_VERSION`.

---

## Versioning Rules

Voice Soundboard follows [Semantic Versioning](https://semver.org/):

| Version | Meaning |
|---------|---------|
| **0.x.y** | Experimental. Anything can change. |
| **1.x.y** | Stable public API. See rules below. |

### What counts as a breaking change

Any of these on a **stable** symbol requires a major version bump:

- Removing a public function, class, or method
- Changing a return type
- Adding a required parameter (without a default)
- Renaming a parameter
- Changing the meaning of an existing parameter
- Removing a field from `SpeechResult` or `Config`

### What does NOT count as a breaking change

- Adding a new optional parameter (with a default)
- Adding a new function or class to `__all__`
- Fixing a bug (even if someone depended on the broken behavior)
- Changing internal/experimental APIs
- Adding new voices, presets, or emotions
- Performance improvements

### Deprecation Policy

Before removing a stable symbol:

1. Add a `DeprecationWarning` for at least one minor release
2. Document the deprecation in CHANGELOG.md
3. Provide a migration path in the warning message
4. Remove in the next major version

```python
import warnings
warnings.warn(
    "quick_speak() is deprecated, use VoiceEngine().speak() instead. "
    "Will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

## Stability Levels

### Stable

Will not break between minor versions (e.g. 1.1 -> 1.2). If a breaking change
is necessary, it will be announced in CHANGELOG.md with a deprecation period of
at least one minor release.

### Advanced

Stable in spirit but covers complex features. The interfaces are unlikely to
change, but edge-case behavior may be adjusted between minor versions. Breaking
changes will be documented.

### Experimental

May change or be removed in any release without notice. Do not depend on these
in production code unless you pin an exact version.

---

## Symbol Stability Map

### Stable -- `from voice_soundboard import ...`

| Symbol | Since | Description |
|--------|-------|-------------|
| `VoiceEngine` | 1.0.0 | Core TTS engine |
| `VoiceEngine.speak()` | 1.0.0 | Generate speech from text |
| `VoiceEngine.speak_raw()` | 1.0.0 | Generate raw audio samples |
| `VoiceEngine.list_voices()` | 1.0.0 | List available voices |
| `VoiceEngine.list_presets()` | 1.0.0 | List voice presets |
| `SpeechResult` | 1.0.0 | Speech generation result |
| `Config` | 1.0.0 | Engine configuration |
| `quick_speak()` | 1.0.0 | One-liner speech generation |
| `play_audio()` | 1.0.0 | Audio playback |
| `stop_playback()` | 1.0.0 | Stop playback |
| `VOICE_PRESETS` | 1.0.0 | Preset definitions |
| `KOKORO_VOICES` | 1.0.0 | Voice definitions |
| `get_emotion_params()` | 1.0.0 | Emotion parameter lookup |
| `list_emotions()` | 1.0.0 | Available emotions |
| `get_effect()` / `play_effect()` / `list_effects()` | 1.0.0 | Sound effects |
| `API_VERSION` | 1.1.0 | API version constant |

### Advanced -- import from subpackages

| Subpackage | Since | Description |
|------------|-------|-------------|
| `voice_soundboard.engines` | 1.0.0 | TTS engine backends (Kokoro, Chatterbox, F5-TTS) |
| `voice_soundboard.dialogue` | 1.0.0 | Multi-speaker dialogue synthesis |
| `voice_soundboard.emotion` | 1.0.0 | VAD model, emotion blending, curves |
| `voice_soundboard.cloning` | 1.0.0 | Voice cloning and library |
| `voice_soundboard.presets` | 1.1.0 | Voice preset catalog with search |

### Experimental -- may change without notice

| Subpackage | Since | Description |
|------------|-------|-------------|
| `voice_soundboard.codecs` | 1.1.0 | Neural audio codecs |
| `voice_soundboard.conversion` | 1.1.0 | Real-time voice conversion |
| `voice_soundboard.llm` | 1.1.0 | LLM conversation pipeline |
| `voice_soundboard.vocology` | 1.2.0 | Voice humanization, formant analysis |
| `voice_soundboard.studio` | 1.1.0 | Voice Studio sessions |

---

## How to Check Stability

If it's in `from voice_soundboard import X` -- it's **stable**.

If it's in `from voice_soundboard.engines import X` -- it's **advanced**.

If it's in `from voice_soundboard.codecs import X` -- it's **experimental**.

Programmatically: `voice_soundboard.API_VERSION` returns the API version.

When in doubt, check this document.
