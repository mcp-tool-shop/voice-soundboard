#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Integration Examples

This file demonstrates how to use Voice Soundboard with AI agents via MCP.

To use with Claude Desktop, add this to your config:

{
  "mcpServers": {
    "voice-soundboard": {
      "command": "python",
      "args": ["-m", "voice_soundboard.server"]
    }
  }
}

The MCP server exposes 40+ tools for AI agents to use.
This file shows equivalent Python code for testing and understanding.
"""

from voice_soundboard import (
    VoiceEngine,
    play_audio,
    list_effects,
    play_effect,
    get_effect,
    list_emotions,
    blend_emotions,
    KOKORO_VOICES,
    VOICE_PRESETS,
    stream_realtime,
    parse_ssml,
    DialogueEngine,
    VoiceCloner,
    VoiceLibrary,
    emotion_to_vad,
)
import asyncio


def mcp_speak_example():
    """
    MCP Tool: speak

    Generate speech with natural language control.

    Parameters:
        text (str): Text to speak
        voice (str, optional): Voice ID
        speed (float, optional): Speaking speed (0.5-2.0)
        style (str, optional): Natural language style hint
        preset (str, optional): Voice preset name
        play (bool, optional): Play audio on server
    """
    print("\n" + "=" * 50)
    print("MCP Tool: speak")
    print("=" * 50)

    engine = VoiceEngine()

    # Basic usage
    result = engine.speak("Hello from the MCP server!")
    print(f"Generated: {result.audio_path}")

    # With all parameters
    result = engine.speak(
        text="This demonstrates all parameters.",
        voice="af_bella",
        speed=1.1,
        style="warmly and cheerfully",
        preset="assistant",
    )
    print(f"Full params: {result.audio_path}")
    play_audio(result.audio_path)


def mcp_list_voices_example():
    """
    MCP Tool: list_voices

    List all available voices with filtering options.

    Parameters:
        filter_gender (str, optional): 'male' or 'female'
        filter_accent (str, optional): 'american', 'british', etc.
    """
    print("\n" + "=" * 50)
    print("MCP Tool: list_voices")
    print("=" * 50)

    # All voices
    print(f"Total voices: {len(KOKORO_VOICES)}")

    # Filter by gender
    male_voices = {k: v for k, v in KOKORO_VOICES.items() if v.get('gender') == 'male'}
    female_voices = {k: v for k, v in KOKORO_VOICES.items() if v.get('gender') == 'female'}
    print(f"Male voices: {len(male_voices)}")
    print(f"Female voices: {len(female_voices)}")

    # Filter by accent
    american = {k: v for k, v in KOKORO_VOICES.items() if 'american' in v.get('name', '').lower()}
    british = {k: v for k, v in KOKORO_VOICES.items() if 'british' in v.get('name', '').lower()}
    print(f"American: {len(american)}, British: {len(british)}")


def mcp_list_presets_example():
    """
    MCP Tool: list_presets

    List available voice presets with descriptions.
    """
    print("\n" + "=" * 50)
    print("MCP Tool: list_presets")
    print("=" * 50)

    for name, config in VOICE_PRESETS.items():
        print(f"  {name}:")
        print(f"    Voice: {config.get('voice', 'default')}")
        print(f"    Speed: {config.get('speed', 1.0)}")
        print(f"    Description: {config.get('description', 'No description')}")


def mcp_list_emotions_example():
    """
    MCP Tool: list_emotions

    List available emotions with characteristics.
    """
    print("\n" + "=" * 50)
    print("MCP Tool: list_emotions")
    print("=" * 50)

    emotions = list_emotions()
    print(f"Available emotions ({len(emotions)}):")
    for emotion in emotions:
        vad = emotion_to_vad(emotion)
        print(f"  {emotion:12} V:{vad.valence:+.2f} A:{vad.arousal:.2f} D:{vad.dominance:.2f}")


def mcp_blend_emotions_example():
    """
    MCP Tool: blend_emotions

    Blend multiple emotions with weights.

    Parameters:
        emotions (list): List of [emotion, weight] pairs
    """
    print("\n" + "=" * 50)
    print("MCP Tool: blend_emotions")
    print("=" * 50)

    # Blend happy and surprised
    result = blend_emotions([("happy", 0.7), ("surprised", 0.3)])
    print(f"70% happy + 30% surprised = {result.closest_emotion}")
    print(f"VAD: V={result.vad.valence:.2f} A={result.vad.arousal:.2f} D={result.vad.dominance:.2f}")

    # Blend for bittersweet
    result = blend_emotions([("happy", 0.5), ("sad", 0.5)])
    print(f"50% happy + 50% sad = {result.closest_emotion}")


def mcp_sound_effect_example():
    """
    MCP Tool: sound_effect

    Play or save a sound effect.

    Parameters:
        effect (str): Effect name
        save_path (str, optional): Path to save the effect
    """
    print("\n" + "=" * 50)
    print("MCP Tool: sound_effect")
    print("=" * 50)

    effects = list_effects()
    print(f"Available effects: {effects}")

    # Play effect
    play_effect("chime")
    print("Played: chime")

    # Save effect
    effect = get_effect("success")
    effect.save("success_sound.wav")
    print("Saved: success_sound.wav")


def mcp_speak_ssml_example():
    """
    MCP Tool: speak_ssml

    Generate speech from SSML markup.

    Parameters:
        ssml (str): SSML-formatted text
        voice (str, optional): Voice ID
        play (bool, optional): Play audio on server
    """
    print("\n" + "=" * 50)
    print("MCP Tool: speak_ssml")
    print("=" * 50)

    ssml = '''
    <speak>
        Hello! <break time="500ms"/>
        <prosody rate="slow">This is slower speech.</prosody>
        <break time="300ms"/>
        <emphasis level="strong">This is emphasized!</emphasis>
    </speak>
    '''

    text, params = parse_ssml(ssml)
    print(f"Parsed: {text.strip()}")

    engine = VoiceEngine()
    result = engine.speak(text, speed=params.speed)
    play_audio(result.audio_path)


async def mcp_speak_dialogue_example():
    """
    MCP Tool: speak_dialogue

    Generate multi-speaker dialogue.

    Parameters:
        script (str): Dialogue script with speaker tags
        voices (dict, optional): Speaker to voice ID mapping
        turn_pause_ms (int, optional): Pause between speakers
        play (bool, optional): Play audio on server
    """
    print("\n" + "=" * 50)
    print("MCP Tool: speak_dialogue")
    print("=" * 50)

    script = """
[S1:host] Welcome to the podcast!
[S2:guest] Thanks for having me!
[S1:host] Let's talk about Voice Soundboard.
[S2:guest] (excited) It's an amazing tool for AI voice synthesis!
"""

    engine = DialogueEngine()
    result = await engine.speak_dialogue(
        script,
        voices={"host": "am_michael", "guest": "af_bella"},
        turn_pause_ms=400
    )

    print(f"Generated: {result.audio_path}")
    play_audio(result.audio_path)


def mcp_clone_voice_example():
    """
    MCP Tool: clone_voice

    Clone a voice from an audio sample.

    Parameters:
        audio_path (str): Path to audio sample (3-10 seconds)
        voice_id (str): Unique ID for the cloned voice
        consent_given (bool): Acknowledgment of consent

    Note: Requires an actual audio sample to demonstrate.
    """
    print("\n" + "=" * 50)
    print("MCP Tool: clone_voice")
    print("=" * 50)

    print("Voice cloning API:")
    print("""
    # Clone a voice
    cloner = VoiceCloner()
    result = cloner.clone(
        audio_path="sample.wav",
        voice_id="custom_voice",
        consent_given=True
    )

    # Use the cloned voice
    audio = cloner.speak("Hello!", voice=result.voice_id)

    # List cloned voices
    library = VoiceLibrary()
    voices = library.list_voices()
    """)


def mcp_list_voice_library_example():
    """
    MCP Tool: list_voice_library

    List all cloned voices in the library.

    Parameters:
        query (str, optional): Search query
        tags (list, optional): Filter by tags
        min_quality (float, optional): Minimum quality score
    """
    print("\n" + "=" * 50)
    print("MCP Tool: list_voice_library")
    print("=" * 50)

    library = VoiceLibrary()
    voices = library.list_voices()
    print(f"Cloned voices in library: {len(voices)}")

    for voice in voices[:5]:
        print(f"  - {voice.voice_id}: {voice.name or 'Unnamed'}")


async def mcp_speak_realtime_example():
    """
    MCP Tool: speak_realtime

    Stream speech with ultra-low latency.

    Parameters:
        text (str): Text to speak
        voice (str, optional): Voice ID
        emotion (str, optional): Emotion name
        speed (float, optional): Speaking speed
    """
    print("\n" + "=" * 50)
    print("MCP Tool: speak_realtime")
    print("=" * 50)

    result = await stream_realtime(
        "This is real-time streaming with low latency!",
        voice="af_bella",
        speed=1.0
    )

    print(f"Chunks: {result.total_chunks}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Realtime factor: {result.realtime_factor:.1f}x")


def mcp_get_emotion_vad_example():
    """
    MCP Tool: get_emotion_vad

    Get VAD (Valence-Arousal-Dominance) values for an emotion.

    Parameters:
        emotion (str): Emotion name
    """
    print("\n" + "=" * 50)
    print("MCP Tool: get_emotion_vad")
    print("=" * 50)

    emotions = ["happy", "sad", "angry", "calm", "excited"]
    for emotion in emotions:
        vad = emotion_to_vad(emotion)
        print(f"{emotion}:")
        print(f"  Valence: {vad.valence:+.2f} (negative to positive)")
        print(f"  Arousal: {vad.arousal:.2f} (calm to excited)")
        print(f"  Dominance: {vad.dominance:.2f} (submissive to dominant)")


def main():
    """Run all MCP examples."""
    print("=" * 60)
    print("Voice Soundboard - MCP Integration Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the 40+ tools available via MCP.")
    print("To use with Claude Desktop, add the server to your config.")

    # Run synchronous examples
    mcp_speak_example()
    mcp_list_voices_example()
    mcp_list_presets_example()
    mcp_list_emotions_example()
    mcp_blend_emotions_example()
    mcp_sound_effect_example()
    mcp_speak_ssml_example()
    mcp_clone_voice_example()
    mcp_list_voice_library_example()
    mcp_get_emotion_vad_example()

    # Run async examples
    asyncio.run(mcp_speak_dialogue_example())
    asyncio.run(mcp_speak_realtime_example())

    print("\n" + "=" * 60)
    print("MCP Examples Complete!")
    print("=" * 60)
    print("\nTo start the MCP server:")
    print("  python -m voice_soundboard.server")
    print("\nOr add to Claude Desktop config:")
    print("""
{
  "mcpServers": {
    "voice-soundboard": {
      "command": "python",
      "args": ["-m", "voice_soundboard.server"]
    }
  }
}
""")


if __name__ == "__main__":
    main()
