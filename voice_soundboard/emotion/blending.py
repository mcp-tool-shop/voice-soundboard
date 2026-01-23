"""
Emotion Blending System.

Blend multiple emotions with weights to create nuanced emotional expressions.

Examples:
    # Bittersweet = happy + sad
    blended = blend_emotions([("happy", 0.6), ("sad", 0.4)])

    # Nervous excitement = excited + anxious
    blended = blend_emotions([("excited", 0.7), ("anxious", 0.3)])

Supports:
- Weighted emotion mixing
- VAD space blending for perceptually correct results
- Intensity preservation
- Dominant emotion detection
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from voice_soundboard.emotion.vad import (
    VADPoint,
    VAD_EMOTIONS,
    emotion_to_vad,
    vad_to_emotion,
    interpolate_vad,
)


@dataclass
class EmotionMix:
    """Result of blending multiple emotions."""
    vad: VADPoint                           # Blended VAD values
    components: List[Tuple[str, float]]     # Original (emotion, weight) pairs
    dominant_emotion: str                   # Closest named emotion
    intensity: float                        # Overall intensity (0-1)
    secondary_emotions: List[str]           # Other close emotions

    def to_synthesis_params(self) -> dict:
        """
        Convert to TTS synthesis parameters.

        Returns dict with:
        - emotion: Dominant emotion name
        - intensity: 0-1 intensity
        - valence, arousal, dominance: VAD values
        """
        return {
            "emotion": self.dominant_emotion,
            "intensity": self.intensity,
            "valence": self.vad.valence,
            "arousal": self.vad.arousal,
            "dominance": self.vad.dominance,
        }

    def describe(self) -> str:
        """Get a human-readable description of the blend."""
        parts = [f"{e}:{w:.0%}" for e, w in self.components]
        return f"{self.dominant_emotion} ({' + '.join(parts)})"


def blend_vad(vad_weights: List[Tuple[VADPoint, float]]) -> VADPoint:
    """
    Blend multiple VAD points with weights.

    Args:
        vad_weights: List of (VADPoint, weight) tuples

    Returns:
        Blended VADPoint

    Raises:
        ValueError: If weights don't sum to ~1.0 or list is empty
    """
    if not vad_weights:
        raise ValueError("At least one VAD point required")

    # Normalize weights
    total_weight = sum(w for _, w in vad_weights)
    if total_weight == 0:
        raise ValueError("Weights cannot all be zero")

    normalized = [(vad, w / total_weight) for vad, w in vad_weights]

    # Weighted average in VAD space
    blended_v = sum(vad.valence * w for vad, w in normalized)
    blended_a = sum(vad.arousal * w for vad, w in normalized)
    blended_d = sum(vad.dominance * w for vad, w in normalized)

    return VADPoint(
        valence=blended_v,
        arousal=blended_a,
        dominance=blended_d,
    )


def blend_emotions(
    emotion_weights: List[Tuple[str, float]],
    normalize_weights: bool = True,
) -> EmotionMix:
    """
    Blend multiple named emotions with weights.

    Args:
        emotion_weights: List of (emotion_name, weight) tuples
        normalize_weights: Auto-normalize weights to sum to 1.0

    Returns:
        EmotionMix with blended result

    Examples:
        # Bittersweet
        result = blend_emotions([("happy", 0.6), ("sad", 0.4)])

        # Nervous excitement
        result = blend_emotions([("excited", 0.7), ("anxious", 0.3)])
    """
    if not emotion_weights:
        raise ValueError("At least one emotion required")

    # Convert emotions to VAD
    vad_weights = []
    for emotion, weight in emotion_weights:
        try:
            vad = emotion_to_vad(emotion)
            vad_weights.append((vad, weight))
        except ValueError:
            # Unknown emotion - skip with warning
            print(f"Warning: Unknown emotion '{emotion}', skipping")
            continue

    if not vad_weights:
        raise ValueError("No valid emotions to blend")

    # Blend in VAD space
    blended_vad = blend_vad(vad_weights)

    # Find closest named emotions
    closest = vad_to_emotion(blended_vad, top_n=3)
    dominant = closest[0][0] if closest else "neutral"
    secondary = [e for e, _ in closest[1:3]]

    # Calculate intensity based on distance from neutral and arousal
    neutral_vad = VAD_EMOTIONS["neutral"]
    distance = blended_vad.distance(neutral_vad)
    max_distance = math.sqrt(2**2 + 1**2 + 1**2)
    intensity = min(1.0, distance / max_distance + blended_vad.arousal * 0.3)

    return EmotionMix(
        vad=blended_vad,
        components=emotion_weights,
        dominant_emotion=dominant,
        intensity=intensity,
        secondary_emotions=secondary,
    )


def transition_emotion(
    from_emotion: str,
    to_emotion: str,
    progress: float,
) -> EmotionMix:
    """
    Create a transitional blend between two emotions.

    Args:
        from_emotion: Starting emotion
        to_emotion: Ending emotion
        progress: Transition progress (0.0 = start, 1.0 = end)

    Returns:
        EmotionMix at the transition point
    """
    progress = max(0.0, min(1.0, progress))

    from_weight = 1.0 - progress
    to_weight = progress

    return blend_emotions([
        (from_emotion, from_weight),
        (to_emotion, to_weight),
    ])


def create_emotion_gradient(
    emotions: List[str],
    steps: int = 10,
) -> List[EmotionMix]:
    """
    Create a gradient through multiple emotions.

    Args:
        emotions: List of emotion names to transition through
        steps: Number of steps in the gradient

    Returns:
        List of EmotionMix objects representing the gradient
    """
    if len(emotions) < 2:
        raise ValueError("At least two emotions required for gradient")

    gradient = []
    segments = len(emotions) - 1
    steps_per_segment = steps // segments

    for i in range(segments):
        from_emotion = emotions[i]
        to_emotion = emotions[i + 1]

        for step in range(steps_per_segment):
            progress = step / steps_per_segment
            blend = transition_emotion(from_emotion, to_emotion, progress)
            gradient.append(blend)

    # Add final emotion
    gradient.append(blend_emotions([(emotions[-1], 1.0)]))

    return gradient


def get_complementary_emotion(emotion: str) -> str:
    """
    Get an emotion that complements (contrasts with) the given emotion.

    Args:
        emotion: Source emotion

    Returns:
        Complementary emotion name
    """
    vad = emotion_to_vad(emotion)

    # Invert valence and adjust arousal
    complementary_vad = VADPoint(
        valence=-vad.valence,
        arousal=1.0 - vad.arousal if vad.arousal > 0.5 else vad.arousal,
        dominance=vad.dominance,
    )

    closest = vad_to_emotion(complementary_vad, top_n=1)
    return closest[0][0] if closest else "neutral"


def get_similar_emotions(emotion: str, count: int = 3) -> List[str]:
    """
    Get emotions similar to the given emotion.

    Args:
        emotion: Source emotion
        count: Number of similar emotions to return

    Returns:
        List of similar emotion names
    """
    vad = emotion_to_vad(emotion)
    closest = vad_to_emotion(vad, top_n=count + 1)

    # Skip the exact match
    return [e for e, _ in closest if e != emotion][:count]


def emotion_distance(emotion1: str, emotion2: str) -> float:
    """
    Calculate perceptual distance between two emotions.

    Args:
        emotion1: First emotion
        emotion2: Second emotion

    Returns:
        Distance (0.0 = identical, ~1.7 = maximally different)
    """
    vad1 = emotion_to_vad(emotion1)
    vad2 = emotion_to_vad(emotion2)
    return vad1.distance(vad2)


# Common emotion blends with names
NAMED_BLENDS = {
    "bittersweet": [("happy", 0.5), ("sad", 0.5)],
    "nervous_excitement": [("excited", 0.6), ("anxious", 0.4)],
    "melancholic_joy": [("joyful", 0.4), ("melancholy", 0.6)],
    "angry_sadness": [("angry", 0.5), ("sad", 0.5)],
    "hopeful_anxiety": [("hopeful", 0.6), ("anxious", 0.4)],
    "nostalgic_happiness": [("nostalgic", 0.5), ("happy", 0.5)],
    "tender_sadness": [("tender", 0.4), ("sad", 0.6)],
    "awe": [("amazed", 0.5), ("fearful", 0.3), ("happy", 0.2)],
    "contemptuous_amusement": [("contemptuous", 0.5), ("amused", 0.5)],
    "resigned": [("sad", 0.4), ("calm", 0.4), ("peaceful", 0.2)],
}


def get_named_blend(name: str) -> Optional[EmotionMix]:
    """
    Get a predefined emotion blend by name.

    Args:
        name: Blend name (e.g., "bittersweet", "nervous_excitement")

    Returns:
        EmotionMix if found, None otherwise
    """
    if name.lower() in NAMED_BLENDS:
        return blend_emotions(NAMED_BLENDS[name.lower()])
    return None


def list_named_blends() -> List[str]:
    """List all available named emotion blends."""
    return list(NAMED_BLENDS.keys())
