"""
VAD Emotion Model (Valence-Arousal-Dominance).

Maps emotions to a 3-dimensional space:
- Valence: Pleasant (-1) to Unpleasant (+1) → we use Pleasant (+1) to Unpleasant (-1)
- Arousal: Calm (0) to Excited (1)
- Dominance: Submissive (0) to Dominant (1)

This model allows for:
- Precise emotion interpolation
- Emotion blending with correct perceptual results
- Mapping between named emotions and continuous values

Based on Russell's Circumplex Model and PAD Emotional State Model.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math


@dataclass
class VADPoint:
    """A point in VAD emotion space."""
    valence: float      # -1 (negative) to +1 (positive)
    arousal: float      # 0 (calm) to 1 (excited)
    dominance: float    # 0 (submissive) to 1 (dominant)

    def __post_init__(self):
        """Clamp values to valid ranges."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

    def distance(self, other: "VADPoint") -> float:
        """Calculate Euclidean distance to another VAD point."""
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple (valence, arousal, dominance)."""
        return (self.valence, self.arousal, self.dominance)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "VADPoint":
        """Create from tuple."""
        return cls(valence=t[0], arousal=t[1], dominance=t[2])

    def __add__(self, other: "VADPoint") -> "VADPoint":
        """Add two VAD points."""
        return VADPoint(
            valence=self.valence + other.valence,
            arousal=self.arousal + other.arousal,
            dominance=self.dominance + other.dominance,
        )

    def __mul__(self, scalar: float) -> "VADPoint":
        """Multiply by scalar."""
        return VADPoint(
            valence=self.valence * scalar,
            arousal=self.arousal * scalar,
            dominance=self.dominance * scalar,
        )

    def __rmul__(self, scalar: float) -> "VADPoint":
        """Right multiply by scalar."""
        return self.__mul__(scalar)


# VAD values for named emotions
# Based on research: Bradley & Lang (1999), Warriner et al. (2013)
VAD_EMOTIONS: dict[str, VADPoint] = {
    # Positive emotions
    "happy": VADPoint(valence=0.8, arousal=0.6, dominance=0.7),
    "joyful": VADPoint(valence=0.9, arousal=0.7, dominance=0.7),
    "excited": VADPoint(valence=0.7, arousal=0.9, dominance=0.7),
    "enthusiastic": VADPoint(valence=0.7, arousal=0.8, dominance=0.7),
    "content": VADPoint(valence=0.7, arousal=0.3, dominance=0.6),
    "peaceful": VADPoint(valence=0.6, arousal=0.2, dominance=0.5),
    "calm": VADPoint(valence=0.5, arousal=0.2, dominance=0.5),
    "relaxed": VADPoint(valence=0.6, arousal=0.2, dominance=0.5),
    "proud": VADPoint(valence=0.7, arousal=0.5, dominance=0.8),
    "confident": VADPoint(valence=0.6, arousal=0.5, dominance=0.8),
    "hopeful": VADPoint(valence=0.6, arousal=0.4, dominance=0.5),
    "loving": VADPoint(valence=0.9, arousal=0.5, dominance=0.6),
    "tender": VADPoint(valence=0.7, arousal=0.3, dominance=0.4),
    "grateful": VADPoint(valence=0.8, arousal=0.4, dominance=0.5),
    "amused": VADPoint(valence=0.7, arousal=0.6, dominance=0.6),

    # Neutral emotions
    "neutral": VADPoint(valence=0.0, arousal=0.3, dominance=0.5),
    "serious": VADPoint(valence=0.0, arousal=0.4, dominance=0.6),
    "thoughtful": VADPoint(valence=0.2, arousal=0.3, dominance=0.5),
    "curious": VADPoint(valence=0.3, arousal=0.5, dominance=0.5),

    # Negative emotions
    "sad": VADPoint(valence=-0.7, arousal=0.3, dominance=0.3),
    "melancholy": VADPoint(valence=-0.5, arousal=0.2, dominance=0.3),
    "disappointed": VADPoint(valence=-0.5, arousal=0.3, dominance=0.3),
    "lonely": VADPoint(valence=-0.6, arousal=0.2, dominance=0.2),
    "nostalgic": VADPoint(valence=-0.2, arousal=0.3, dominance=0.4),
    "bored": VADPoint(valence=-0.3, arousal=0.1, dominance=0.4),

    # High arousal negative
    "angry": VADPoint(valence=-0.7, arousal=0.8, dominance=0.8),
    "furious": VADPoint(valence=-0.9, arousal=0.9, dominance=0.8),
    "frustrated": VADPoint(valence=-0.6, arousal=0.7, dominance=0.5),
    "annoyed": VADPoint(valence=-0.4, arousal=0.5, dominance=0.5),
    "irritated": VADPoint(valence=-0.5, arousal=0.6, dominance=0.5),

    # Fear/anxiety spectrum
    "fearful": VADPoint(valence=-0.7, arousal=0.8, dominance=0.2),
    "terrified": VADPoint(valence=-0.9, arousal=0.9, dominance=0.1),
    "anxious": VADPoint(valence=-0.5, arousal=0.7, dominance=0.3),
    "nervous": VADPoint(valence=-0.4, arousal=0.6, dominance=0.3),
    "worried": VADPoint(valence=-0.4, arousal=0.5, dominance=0.3),
    "tense": VADPoint(valence=-0.3, arousal=0.6, dominance=0.4),

    # Surprise spectrum
    "surprised": VADPoint(valence=0.2, arousal=0.8, dominance=0.4),
    "shocked": VADPoint(valence=-0.2, arousal=0.9, dominance=0.3),
    "amazed": VADPoint(valence=0.6, arousal=0.8, dominance=0.5),
    "astonished": VADPoint(valence=0.3, arousal=0.8, dominance=0.4),

    # Disgust/contempt
    "disgusted": VADPoint(valence=-0.7, arousal=0.5, dominance=0.6),
    "contemptuous": VADPoint(valence=-0.5, arousal=0.4, dominance=0.7),

    # Complex emotions
    "sarcastic": VADPoint(valence=-0.2, arousal=0.4, dominance=0.7),
    "bitter": VADPoint(valence=-0.6, arousal=0.4, dominance=0.5),
    "wistful": VADPoint(valence=-0.1, arousal=0.2, dominance=0.4),
    "dramatic": VADPoint(valence=0.0, arousal=0.8, dominance=0.7),
    "mysterious": VADPoint(valence=0.0, arousal=0.4, dominance=0.6),
    "whimsical": VADPoint(valence=0.5, arousal=0.5, dominance=0.5),
    "playful": VADPoint(valence=0.6, arousal=0.6, dominance=0.6),
    "mischievous": VADPoint(valence=0.4, arousal=0.6, dominance=0.6),

    # Professional/performance
    "professional": VADPoint(valence=0.2, arousal=0.4, dominance=0.7),
    "authoritative": VADPoint(valence=0.1, arousal=0.5, dominance=0.8),
    "urgent": VADPoint(valence=-0.1, arousal=0.8, dominance=0.7),
    "empathetic": VADPoint(valence=0.4, arousal=0.4, dominance=0.4),
    "reassuring": VADPoint(valence=0.5, arousal=0.3, dominance=0.6),

    # Special states
    "whisper": VADPoint(valence=0.0, arousal=0.1, dominance=0.3),
    "sleepy": VADPoint(valence=0.1, arousal=0.1, dominance=0.3),
    "dreamy": VADPoint(valence=0.3, arousal=0.2, dominance=0.4),
}


def emotion_to_vad(emotion: str) -> VADPoint:
    """
    Convert an emotion name to VAD values.

    Args:
        emotion: Emotion name (e.g., "happy", "sad", "angry")

    Returns:
        VADPoint with valence, arousal, dominance values

    Raises:
        ValueError: If emotion is not recognized
    """
    emotion_lower = emotion.lower().strip()

    if emotion_lower in VAD_EMOTIONS:
        return VAD_EMOTIONS[emotion_lower]

    # Try to find partial match
    for name, vad in VAD_EMOTIONS.items():
        if emotion_lower in name or name in emotion_lower:
            return vad

    raise ValueError(f"Unknown emotion: {emotion}. Available: {list(VAD_EMOTIONS.keys())}")


def vad_to_emotion(vad: VADPoint, top_n: int = 1) -> List[Tuple[str, float]]:
    """
    Find the closest named emotion(s) to a VAD point.

    Args:
        vad: VAD point to match
        top_n: Number of closest emotions to return

    Returns:
        List of (emotion_name, distance) tuples, sorted by distance
    """
    distances = []
    for name, emotion_vad in VAD_EMOTIONS.items():
        dist = vad.distance(emotion_vad)
        distances.append((name, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_n]


def interpolate_vad(vad1: VADPoint, vad2: VADPoint, t: float) -> VADPoint:
    """
    Linearly interpolate between two VAD points.

    Args:
        vad1: Starting VAD point (t=0)
        vad2: Ending VAD point (t=1)
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated VAD point
    """
    t = max(0.0, min(1.0, t))
    return VADPoint(
        valence=vad1.valence + (vad2.valence - vad1.valence) * t,
        arousal=vad1.arousal + (vad2.arousal - vad1.arousal) * t,
        dominance=vad1.dominance + (vad2.dominance - vad1.dominance) * t,
    )


def get_emotion_intensity(vad: VADPoint) -> float:
    """
    Calculate the intensity/magnitude of an emotion.

    Higher arousal and distance from neutral = higher intensity.

    Args:
        vad: VAD point to measure

    Returns:
        Intensity value (0.0 to 1.0)
    """
    neutral = VAD_EMOTIONS["neutral"]
    distance = vad.distance(neutral)

    # Normalize by max possible distance (~1.7)
    max_distance = math.sqrt(2**2 + 1**2 + 1**2)
    normalized = distance / max_distance

    # Weight arousal more heavily
    arousal_weight = 0.4
    intensity = normalized * (1 - arousal_weight) + vad.arousal * arousal_weight

    return min(1.0, intensity)


def classify_emotion_category(vad: VADPoint) -> str:
    """
    Classify a VAD point into a broad emotion category.

    Categories:
    - positive_high: Happy, excited, joyful
    - positive_low: Calm, content, peaceful
    - negative_high: Angry, fearful, anxious
    - negative_low: Sad, melancholy, bored
    - neutral: Neutral, serious, thoughtful

    Args:
        vad: VAD point to classify

    Returns:
        Category name
    """
    if abs(vad.valence) < 0.2 and vad.arousal < 0.5:
        return "neutral"
    elif vad.valence > 0:
        if vad.arousal > 0.5:
            return "positive_high"
        else:
            return "positive_low"
    else:
        if vad.arousal > 0.5:
            return "negative_high"
        else:
            return "negative_low"


def list_emotions_by_category(category: Optional[str] = None) -> dict[str, List[str]]:
    """
    List emotions grouped by category.

    Args:
        category: Optional specific category to list

    Returns:
        Dict of category -> list of emotion names
    """
    categories: dict[str, List[str]] = {
        "positive_high": [],
        "positive_low": [],
        "negative_high": [],
        "negative_low": [],
        "neutral": [],
    }

    for name, vad in VAD_EMOTIONS.items():
        cat = classify_emotion_category(vad)
        categories[cat].append(name)

    if category:
        return {category: categories.get(category, [])}

    return categories
