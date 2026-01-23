#!/usr/bin/env python3
"""
Paralinguistic Tags Example (Chatterbox)

Demonstrates natural non-speech sounds:
- [laugh], [chuckle], [sigh]
- [gasp], [cough], [groan]
- Emotion exaggeration control
"""

try:
    from voice_soundboard.engines import ChatterboxEngine, CHATTERBOX_AVAILABLE
except ImportError:
    CHATTERBOX_AVAILABLE = False

from voice_soundboard import play_audio


def main():
    if not CHATTERBOX_AVAILABLE:
        print("Chatterbox is not installed.")
        print("Install with: pip install voice-soundboard[chatterbox]")
        return

    engine = ChatterboxEngine()
    print("Voice Soundboard - Paralinguistic Tags Demo")
    print("=" * 45)

    # 1. Basic tags
    print("\n1. Basic paralinguistic tags:")
    examples = [
        "That's hilarious! [laugh] Oh man, I needed that.",
        "Well... [sigh] I suppose you're right.",
        "Wait, what? [gasp] I can't believe it!",
        "Excuse me. [cough] Where were we?",
        "[chuckle] That's pretty funny.",
    ]
    for text in examples:
        print(f"   Text: {text}")
        result = engine.speak(text)
        play_audio(result.audio_path)

    # 2. Emotion exaggeration
    print("\n2. Emotion exaggeration levels:")
    text = "I'm so excited about this project!"
    levels = [0.0, 0.3, 0.5, 0.7, 1.0]
    for level in levels:
        print(f"   Exaggeration: {level} ({'monotone' if level == 0 else 'dramatic' if level == 1 else 'moderate'})")
        result = engine.speak(text, emotion_exaggeration=level)
        play_audio(result.audio_path)

    # 3. Combining tags with exaggeration
    print("\n3. Combined tags and emotion:")
    result = engine.speak(
        "Oh no! [gasp] This is terrible! [groan] What are we going to do?",
        emotion_exaggeration=0.8
    )
    play_audio(result.audio_path)

    print("\nDemo complete!")
    print("\nSupported tags: [laugh], [chuckle], [sigh], [gasp], [cough], [groan], [sniff], [shush], [clear throat]")


if __name__ == "__main__":
    main()
