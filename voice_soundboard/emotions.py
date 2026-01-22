"""
Emotion Control - Add emotional parameters to speech.

Maps emotions to voice characteristics, speed, and text modifications
that influence how Kokoro renders the speech.

Supported emotions:
- happy: Upbeat, faster, brighter voice
- sad: Slower, softer
- excited: Fast, energetic
- calm: Slow, measured
- angry: Emphatic, faster
- fearful: Faster, higher pitch feel
- surprised: Quick bursts
- neutral: Default

Example:
    params = get_emotion_params("excited")
    engine.speak(text, **params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionParams:
    """Parameters derived from an emotion."""
    speed: float = 1.0
    voice_preference: Optional[str] = None  # Suggested voice for this emotion
    text_prefix: str = ""  # Add to start of text (can influence prosody)
    text_suffix: str = ""  # Add to end
    punctuation_boost: bool = False  # Add emphasis punctuation
    pause_multiplier: float = 1.0  # Adjust pause lengths


# Emotion definitions
EMOTIONS: dict[str, EmotionParams] = {
    # Positive emotions
    "happy": EmotionParams(
        speed=1.1,
        voice_preference="af_bella",  # Warm, friendly
        punctuation_boost=True,
    ),
    "excited": EmotionParams(
        speed=1.25,
        voice_preference="af_nicole",  # Energetic
        punctuation_boost=True,
    ),
    "joyful": EmotionParams(
        speed=1.15,
        voice_preference="af_bella",
        punctuation_boost=True,
    ),

    # Calm emotions
    "calm": EmotionParams(
        speed=0.85,
        voice_preference="bm_george",  # Deep, measured
        pause_multiplier=1.3,
    ),
    "peaceful": EmotionParams(
        speed=0.8,
        voice_preference="bf_emma",
        pause_multiplier=1.4,
    ),
    "neutral": EmotionParams(
        speed=1.0,
    ),

    # Negative emotions
    "sad": EmotionParams(
        speed=0.85,
        voice_preference="bf_emma",  # Softer
        pause_multiplier=1.2,
    ),
    "melancholy": EmotionParams(
        speed=0.8,
        voice_preference="bm_george",
        pause_multiplier=1.3,
    ),
    "angry": EmotionParams(
        speed=1.15,
        voice_preference="bm_lewis",  # Stronger
        punctuation_boost=True,
    ),
    "frustrated": EmotionParams(
        speed=1.1,
        voice_preference="bm_lewis",
    ),

    # High-energy emotions
    "fearful": EmotionParams(
        speed=1.2,
        voice_preference="af_sarah",
        pause_multiplier=0.8,
    ),
    "surprised": EmotionParams(
        speed=1.2,
        voice_preference="af_nicole",
        punctuation_boost=True,
    ),
    "urgent": EmotionParams(
        speed=1.3,
        voice_preference="am_adam",
        pause_multiplier=0.7,
    ),

    # Professional emotions
    "confident": EmotionParams(
        speed=1.0,
        voice_preference="bm_george",
    ),
    "serious": EmotionParams(
        speed=0.95,
        voice_preference="bm_george",
        pause_multiplier=1.1,
    ),
    "professional": EmotionParams(
        speed=1.0,
        voice_preference="am_adam",
    ),

    # Storytelling emotions
    "mysterious": EmotionParams(
        speed=0.9,
        voice_preference="bm_george",
        pause_multiplier=1.3,
    ),
    "dramatic": EmotionParams(
        speed=0.95,
        voice_preference="bm_george",
        pause_multiplier=1.2,
        punctuation_boost=True,
    ),
    "whimsical": EmotionParams(
        speed=1.1,
        voice_preference="af_bella",
    ),
}


def get_emotion_params(emotion: str) -> EmotionParams:
    """
    Get parameters for an emotion.

    Args:
        emotion: Emotion name (case-insensitive)

    Returns:
        EmotionParams with speed, voice preferences, etc.

    Example:
        params = get_emotion_params("excited")
        print(f"Speed: {params.speed}, Voice: {params.voice_preference}")
    """
    emotion = emotion.lower().strip()

    if emotion in EMOTIONS:
        return EMOTIONS[emotion]

    # Try to find a close match
    for key in EMOTIONS:
        if key.startswith(emotion) or emotion.startswith(key):
            return EMOTIONS[key]

    # Default to neutral
    return EMOTIONS["neutral"]


def list_emotions() -> list[str]:
    """List all available emotions."""
    return sorted(EMOTIONS.keys())


def apply_emotion_to_text(text: str, emotion: str) -> str:
    """
    Modify text to enhance emotional expression.

    Adds punctuation emphasis for certain emotions.

    Args:
        text: Original text
        emotion: Emotion name

    Returns:
        Modified text
    """
    params = get_emotion_params(emotion)

    if params.punctuation_boost:
        # Add subtle emphasis through punctuation
        # Replace periods with exclamation for excited/happy
        if emotion in ("excited", "happy", "joyful", "surprised"):
            # Only boost some sentences, not all
            sentences = text.split(". ")
            boosted = []
            for i, s in enumerate(sentences):
                if i % 2 == 0 and not s.endswith("!") and not s.endswith("?"):
                    boosted.append(s.rstrip(".") + "!")
                else:
                    boosted.append(s)
            text = ". ".join(boosted)

    return text


def get_emotion_voice_params(
    emotion: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
) -> dict:
    """
    Get voice engine parameters for an emotion.

    Args:
        emotion: Emotion name
        voice: Override voice (uses emotion default if None)
        speed: Override speed (uses emotion default if None)

    Returns:
        Dict with 'voice' and 'speed' keys for engine.speak()
    """
    params = get_emotion_params(emotion)

    return {
        "voice": voice or params.voice_preference,
        "speed": speed if speed is not None else params.speed,
    }


# Emotion intensity modifiers
def intensify_emotion(emotion: str, intensity: float = 1.0) -> EmotionParams:
    """
    Get emotion parameters with intensity modifier.

    Args:
        emotion: Base emotion
        intensity: 0.5 (subtle) to 2.0 (intense)

    Returns:
        Modified EmotionParams
    """
    base = get_emotion_params(emotion)
    intensity = max(0.5, min(2.0, intensity))

    # Scale speed deviation from 1.0
    speed_delta = (base.speed - 1.0) * intensity
    new_speed = 1.0 + speed_delta

    # Scale pause multiplier
    pause_delta = (base.pause_multiplier - 1.0) * intensity
    new_pause = 1.0 + pause_delta

    return EmotionParams(
        speed=max(0.5, min(2.0, new_speed)),
        voice_preference=base.voice_preference,
        text_prefix=base.text_prefix,
        text_suffix=base.text_suffix,
        punctuation_boost=base.punctuation_boost and intensity > 0.8,
        pause_multiplier=new_pause,
    )


if __name__ == "__main__":
    print("Available Emotions:")
    print("=" * 40)

    for emotion in list_emotions():
        params = get_emotion_params(emotion)
        print(f"  {emotion:15} speed={params.speed:.2f} voice={params.voice_preference or 'default'}")

    print("\nExample usage:")
    params = get_emotion_voice_params("excited")
    print(f"  excited -> {params}")

    # Test text modification
    text = "This is a test. I am very happy. The weather is nice."
    modified = apply_emotion_to_text(text, "happy")
    print(f"\nOriginal: {text}")
    print(f"Happy:    {modified}")
