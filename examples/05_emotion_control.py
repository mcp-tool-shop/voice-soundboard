#!/usr/bin/env python3
"""
Advanced Emotion Control Example

Demonstrates:
- Word-level emotion tags
- VAD emotion model (Valence-Arousal-Dominance)
- Emotion blending
- Dynamic emotion curves
"""

from voice_soundboard import (
    VoiceEngine,
    play_audio,
    EmotionParser,
    blend_emotions,
    EmotionCurve,
    emotion_to_vad,
    vad_to_emotion,
)


def main():
    engine = VoiceEngine()
    print("Voice Soundboard - Emotion Control Demo")
    print("=" * 45)

    # 1. Word-level emotion tags
    print("\n1. Word-level emotion tags:")
    text = "I was so {happy}excited{/happy} to see you, but then {sad}you had to leave{/sad}."
    print(f"   Text: {text}")

    parser = EmotionParser()
    result = parser.parse(text)
    print(f"   Emotion spans: {len(result.spans)}")
    for span in result.spans:
        print(f"   - '{span.text}' [{span.emotion}]")

    # Generate with emotion-aware synthesis
    audio = engine.speak(text)
    play_audio(audio.audio_path)

    # 2. VAD Emotion Model
    print("\n2. VAD Emotion Model (Valence-Arousal-Dominance):")
    emotions = ["happy", "sad", "angry", "calm", "excited", "fearful"]
    for emotion in emotions:
        vad = emotion_to_vad(emotion)
        print(f"   {emotion:10} -> V:{vad.valence:+.2f}  A:{vad.arousal:.2f}  D:{vad.dominance:.2f}")

    # Find emotion from VAD values
    print("\n   Reverse lookup:")
    test_vads = [
        (0.8, 0.8, 0.7),   # High positive, high energy
        (-0.6, 0.7, 0.3),  # Negative, high energy, low control
        (0.3, 0.2, 0.5),   # Slightly positive, calm
    ]
    for v, a, d in test_vads:
        emotion = vad_to_emotion(v, a, d)
        print(f"   V:{v:+.1f} A:{a:.1f} D:{d:.1f} -> {emotion}")

    # 3. Emotion Blending
    print("\n3. Emotion Blending:")
    blends = [
        [("happy", 0.5), ("sad", 0.5)],        # Bittersweet
        [("happy", 0.7), ("surprised", 0.3)],  # Pleasant surprise
        [("angry", 0.4), ("sad", 0.6)],        # Bitter disappointment
        [("calm", 0.8), ("happy", 0.2)],       # Content
    ]
    for blend in blends:
        result = blend_emotions(blend)
        weights = " + ".join(f"{int(w*100)}% {e}" for e, w in blend)
        print(f"   {weights}")
        print(f"   -> {result.closest_emotion} (V:{result.vad.valence:+.2f}, A:{result.vad.arousal:.2f})")

        audio = engine.speak(
            f"This is how {result.closest_emotion} sounds.",
            emotion=result.closest_emotion
        )
        play_audio(audio.audio_path)

    # 4. Emotion Curves
    print("\n4. Dynamic Emotion Curves:")
    curve = EmotionCurve()
    curve.add_point(0.0, "worried")
    curve.add_point(0.3, "anxious")
    curve.add_point(0.6, "hopeful")
    curve.add_point(1.0, "excited")

    print("   Keyframes:")
    for kf in curve.keyframes:
        print(f"   - t={kf.position:.1f}: {kf.emotion}")

    # Sample the curve
    print("\n   Sampled emotions:")
    samples = curve.sample(5)
    for i, sample in enumerate(samples):
        t = i / (len(samples) - 1)
        emotion = vad_to_emotion(sample.valence, sample.arousal, sample.dominance)
        print(f"   t={t:.2f}: {emotion}")

    # Use curve for a sentence
    text = "I thought it was going to be terrible, but it turned out amazing!"
    print(f"\n   Text with emotion curve: '{text}'")
    audio = engine.speak(text, emotion="excited")  # End emotion
    play_audio(audio.audio_path)

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
