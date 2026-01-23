#!/usr/bin/env python3
"""
Basic Speech Generation Example

Demonstrates the core VoiceEngine functionality:
- Simple text-to-speech
- Voice selection
- Presets (assistant, narrator, announcer)
- Emotions
- Natural language styles
"""

from voice_soundboard import VoiceEngine, play_audio

def main():
    # Initialize the engine
    engine = VoiceEngine()
    print("Voice Soundboard - Basic Speech Demo")
    print("=" * 40)

    # 1. Simple speech
    print("\n1. Simple speech:")
    result = engine.speak("Hello! Welcome to Voice Soundboard.")
    print(f"   Generated: {result.audio_path}")
    print(f"   Duration: {result.duration_seconds:.2f}s")
    play_audio(result.audio_path)

    # 2. Different voices
    print("\n2. Different voices:")
    voices = [
        ("af_bella", "American Female - Bella"),
        ("bm_george", "British Male - George"),
        ("jf_alpha", "Japanese Female - Alpha"),
    ]
    for voice_id, description in voices:
        print(f"   {description}:")
        result = engine.speak("This is my voice.", voice=voice_id)
        play_audio(result.audio_path)

    # 3. Presets
    print("\n3. Voice presets:")
    presets = ["assistant", "narrator", "announcer", "storyteller", "whisper"]
    for preset in presets:
        print(f"   Preset: {preset}")
        result = engine.speak(f"Speaking as the {preset}.", preset=preset)
        play_audio(result.audio_path)

    # 4. Emotions
    print("\n4. Emotions:")
    emotions = ["happy", "sad", "excited", "calm", "angry"]
    for emotion in emotions:
        print(f"   Emotion: {emotion}")
        result = engine.speak(f"I'm feeling {emotion} right now.", emotion=emotion)
        play_audio(result.audio_path)

    # 5. Natural language styles
    print("\n5. Natural language styles:")
    styles = [
        "warmly and cheerfully",
        "slowly and mysteriously",
        "excitedly, like an announcer",
        "calmly, in a british accent",
    ]
    for style in styles:
        print(f"   Style: '{style}'")
        result = engine.speak("Let me tell you something interesting.", style=style)
        play_audio(result.audio_path)

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
