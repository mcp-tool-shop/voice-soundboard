#!/usr/bin/env python3
"""
Voice Cloning Example

Demonstrates voice cloning features:
- Clone from audio sample
- Use cloned voice for synthesis
- Voice library management
- Cross-language cloning
"""

from pathlib import Path
from voice_soundboard import VoiceCloner, VoiceLibrary, play_audio


def main():
    print("Voice Soundboard - Voice Cloning Demo")
    print("=" * 42)

    # Check for sample audio
    sample_path = Path("sample_voice.wav")
    if not sample_path.exists():
        print("\nTo run this demo, you need a voice sample.")
        print("Create a 3-10 second WAV file called 'sample_voice.wav'")
        print("with clear speech (no background noise).")

        # Create a demo without actual cloning
        print("\n--- Running in demo mode (no actual cloning) ---")
        demonstrate_api()
        return

    cloner = VoiceCloner()

    # 1. Clone a voice (requires consent)
    print("\n1. Cloning voice from sample:")
    print("   IMPORTANT: Voice cloning requires consent from the voice owner")

    result = cloner.clone(
        audio_path=sample_path,
        voice_id="my_voice",
        consent_given=True,
        consent_notes="Self-recording for personal use"
    )
    print(f"   Cloned voice ID: {result.voice_id}")
    print(f"   Quality score: {result.quality_score:.2f}")

    # 2. Use the cloned voice
    print("\n2. Speaking with cloned voice:")
    audio = cloner.speak(
        "Hello! This is my cloned voice speaking.",
        voice=result.voice_id
    )
    play_audio(audio)

    # 3. Cross-language (if supported)
    print("\n3. Cross-language cloning:")
    languages = ["en", "fr", "de", "es"]
    phrases = {
        "en": "Hello, how are you?",
        "fr": "Bonjour, comment allez-vous?",
        "de": "Hallo, wie geht es Ihnen?",
        "es": "Hola, como estas?",
    }
    for lang in languages:
        audio = cloner.speak(
            phrases[lang],
            voice=result.voice_id,
            language=lang
        )
        print(f"   {lang.upper()}: {phrases[lang]}")
        play_audio(audio)

    # 4. Voice library management
    print("\n4. Voice library:")
    library = VoiceLibrary()
    voices = library.list_voices()
    print(f"   Total voices: {len(voices)}")
    for voice in voices:
        print(f"   - {voice.voice_id}: {voice.name or 'Unnamed'}")

    print("\nDemo complete!")


def demonstrate_api():
    """Show the API without actual cloning."""
    print("\n--- Voice Cloning API Demo ---")

    print("""
    # Clone a voice
    cloner = VoiceCloner()
    result = cloner.clone(
        audio_path="sample.wav",       # 3-10 second sample
        voice_id="my_voice",           # Unique ID for this voice
        consent_given=True,            # Required for ethical use
        consent_notes="Owner consent"  # Documentation
    )

    # Use the cloned voice
    audio = cloner.speak("Hello!", voice=result.voice_id)

    # Cross-language cloning
    audio = cloner.speak("Bonjour!", voice=result.voice_id, language="fr")

    # Manage voice library
    library = VoiceLibrary()
    voices = library.list_voices()
    library.delete_voice("my_voice")
    """)


if __name__ == "__main__":
    main()
