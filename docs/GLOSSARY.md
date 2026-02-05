# Glossary

One word per concept. If two words mean the same thing, only one appears here.

---

| Term | Definition |
|------|-----------|
| **Engine** | A TTS backend that converts text to audio. Kokoro is the default engine. Chatterbox and F5-TTS are optional engines. |
| **Voice** | A specific speaker identity within an engine, identified by an ID like `af_bella`. Voices have a fixed gender, accent, and style. |
| **Preset** | A named combination of voice + speed + style (e.g. `assistant`, `narrator`). A shortcut so you don't have to pick settings manually. |
| **Emotion** | A named mood applied to speech (e.g. `happy`, `calm`). Affects speed, pitch mapping, and voice selection. |
| **Style** | A natural language description of how to speak (e.g. `"warmly and cheerfully"`). Interpreted into voice, speed, and preset parameters. |
| **Prosody** | The rhythm, stress, and intonation of speech. Controlled indirectly through speed, emotion, and style. Not a user-facing parameter. |
| **Streaming** | Generating and playing audio in chunks as it's produced, instead of waiting for the full file. Used for low-latency output. |
| **Cloning** | Creating a new voice from a short (3-10 second) audio sample of a real person's speech. Requires explicit consent. |
| **MCP** | Model Context Protocol. A standard for AI agents to call tools. Voice Soundboard exposes 40+ tools via MCP. |
| **Paralinguistic tag** | A non-speech sound inserted into text, like `[laugh]` or `[sigh]`. Requires the Chatterbox engine. |
| **SSML** | Speech Synthesis Markup Language. An XML format for fine-grained control over pauses, emphasis, and pronunciation. |
| **Effect** | A short procedurally-generated sound (e.g. `chime`, `success`, `error`). Not speech. |

## Module Naming Conventions

Some modules share similar names. This is intentional:

| Module | Scope | When to Use |
|--------|-------|-------------|
| `emotions.py` | Core. Maps emotion names to synthesis parameters. | Basic: `get_emotion_params("happy")` |
| `emotion/` | Optional. Advanced VAD model, blending, curves. | Advanced: `blend_emotions([("happy", 0.7), ("surprised", 0.3)])` |

Rule: if you need a named emotion for `speak(emotion="happy")`, that's `emotions.py`.
If you need to blend, parse inline tags, or create emotion curves, that's `emotion/`.

## API / CLI Parameter Mapping

| API Parameter | CLI Flag | Notes |
|---------------|----------|-------|
| `voice=` | `--voice` | |
| `preset=` | `--preset` | |
| `emotion=` | `--emotion` | |
| `style=` | `--style` | Natural language hint |
| `speed=` | `--speed` | |
| `save_as=` | `-o` / `--output` | Different name (CLI convention) |

## Deprecated / Avoided Terms

These terms are **not used** in public APIs or documentation:

| Avoid | Use Instead | Reason |
|-------|-------------|--------|
| "model" (for a voice) | **voice** | "Model" means the neural network. A voice is a speaker identity. |
| "tone" (as a parameter) | **emotion** or **style** | Ambiguous. Use emotion for a named mood, style for a natural language description. |
| "persona" | **preset** | Preset is the established term in this project. |
| "timbre" | **voice** | Timbre is a technical acoustic property. Users pick a voice. |
| "utterance" | **text** or **speech** | Unnecessary jargon. |
| "speaker" (in API) | **voice** | "Speaker" is acceptable in dialogue contexts only (e.g. `[S1:narrator]`). |
| "profile" | **voice** (or **preset**) | VoiceProfile in cloning is a legacy name; represents a cloned voice instance. |
