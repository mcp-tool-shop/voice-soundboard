# v1.0 Contract

This document declares what Voice Soundboard v1.0 means.
It sets expectations so users know whether to depend on this platform.

---

## What v1.0 Means

Voice Soundboard v1.0 is a **stable, local text-to-speech library** with:

- A frozen Python API (`VoiceEngine.speak()`)
- A frozen CLI (`voice-soundboard speak "..."`)
- A frozen MCP tool surface (40+ tools)
- 54+ voices, 19 emotions, 5 presets
- Semantic versioning (breaking changes require a major version bump)

It does **not** mean "feature-complete for all use cases."
It means "safe to depend on for the use cases it covers."

---

## What Is Locked (Will Not Break)

These are the frozen interfaces from [ARCHITECTURE.md](ARCHITECTURE.md).
Changing any of these requires a major version bump (v2.0):

| Interface | Contract |
|-----------|----------|
| `VoiceEngine.speak()` | Returns `SpeechResult` with `.audio_path` to a valid WAV |
| `VoiceEngine.speak_raw()` | Returns `(np.ndarray, int)` with no file I/O |
| `SpeechResult` fields | `audio_path`, `duration_seconds`, `generation_time`, `voice_used`, `sample_rate`, `realtime_factor` |
| `Config` defaults | `af_bella` voice, `1.0` speed, `24000` sample rate |
| Parameter priority | explicit > preset > style > emotion > defaults |
| Output format | WAV (PCM float32) |
| Lazy loading | Construction never loads models |
| CLI subcommand | `voice-soundboard speak "text"` with `--voice`, `--preset`, `--emotion`, `--style`, `--speed`, `-o` |
| `API_VERSION` | Bumped on any breaking change |

## What May Change (Minor Versions)

These can change between minor releases (v1.x) without warning:

| Area | What Can Change |
|------|-----------------|
| **New voices** | Voices may be added. Existing voice IDs remain stable. |
| **New presets** | Presets may be added. Existing preset names remain stable. |
| **New emotions** | Emotions may be added. Existing emotion names remain stable. |
| **New `speak()` parameters** | Optional parameters may be added. Existing parameters remain stable. |
| **Experimental modules** | `codecs/`, `conversion/`, `llm/`, `vocology/`, `studio/` may change or be removed. |
| **MCP tools** | New tools may be added. Existing tool names and parameter schemas remain stable. |
| **Performance** | Generation speed may improve. No regression guarantees. |
| **Internal modules** | `normalizer.py`, `interpreter.py`, `security.py`, `errors.py` are internal. |

## What Is Intentionally Excluded

These are **not goals** for v1.0:

| Non-Goal | Reason |
|----------|--------|
| Real-time bidirectional voice chat | Not a conversation system. Use LLM module (experimental) if needed. |
| Cloud deployment | Designed for local machines. No HTTP auth, no multi-tenant. |
| Audio editing / mixing | Not an audio editor. Use ffmpeg or similar. |
| Voice recognition (STT) | One direction only: text to speech. |
| Guaranteed voice cloning quality | Cloning depends on input audio quality. Results vary. |
| Cross-platform GUI | No native GUI. Web demo is for evaluation only. |
| Batch processing API | Process one text at a time. Batch by calling `speak()` in a loop. |

---

## Post-v1.0 Extension Rules

Future development follows these rules:

1. **Additive only.** New features are added as new optional parameters,
   new subpackages, or new install extras. Nothing existing breaks.

2. **Deprecation before removal.** Stable symbols get a `DeprecationWarning`
   for at least one minor release before removal in a major version.

3. **Experimental stays experimental.** Modules marked experimental in
   [FEATURE_FLAGS.md](FEATURE_FLAGS.md) can change freely between minor
   releases. Moving a module from experimental to stable is a one-way door.

4. **Changelog required.** Every change gets a CHANGELOG entry classified
   as patch, minor, or breaking per [RELEASE.md](RELEASE.md).

5. **Tests guard contracts.** The smoke tests in `tests/smoke/test_api_contract.py`
   verify the frozen interfaces. If a change breaks a smoke test, it's breaking.

---

## Release Readiness

v1.0 is tagged when all of these are true:

- [ ] All smoke tests pass
- [ ] All core tests pass
- [ ] Three canonical demos run without errors
- [ ] Benchmark harness produces results
- [ ] `__version__` set to `"1.0.0"`
- [ ] `API_VERSION` set to `1`
- [ ] CHANGELOG `[Unreleased]` moved to `[1.0.0]`
- [ ] All Phase 1-5 docs present and linked from README
