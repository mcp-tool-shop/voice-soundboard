"""
Dynamic Emotion Curves.

Define emotion trajectories over the duration of an utterance.

Examples:
    # Start neutral, build to excited
    curve = EmotionCurve()
    curve.add_point(0.0, "neutral")
    curve.add_point(0.7, "excited")
    curve.add_point(1.0, "happy")

    # Get emotion at any position
    emotion = curve.get_emotion_at(0.5)  # Somewhere between neutral and excited

Supports:
- Linear interpolation between keyframes
- Smooth easing functions
- Pre-built curve patterns (build-up, arc, fade)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import math

from voice_soundboard.emotion.vad import (
    VADPoint,
    emotion_to_vad,
    vad_to_emotion,
    interpolate_vad,
    VAD_EMOTIONS,
)
from voice_soundboard.emotion.blending import EmotionMix, blend_vad


@dataclass
class EmotionKeyframe:
    """A single point on an emotion curve."""
    position: float         # 0.0 to 1.0 (normalized time/position)
    emotion: str            # Emotion name
    vad: VADPoint = None    # Computed VAD values
    intensity: float = 1.0  # Intensity at this point
    easing: str = "linear"  # Easing to next keyframe

    def __post_init__(self):
        """Compute VAD if not provided."""
        if self.vad is None:
            try:
                self.vad = emotion_to_vad(self.emotion)
            except ValueError:
                self.vad = VAD_EMOTIONS["neutral"]

        # Clamp position
        self.position = max(0.0, min(1.0, self.position))


class EmotionCurve:
    """
    Defines an emotion trajectory over normalized time/position.

    Example:
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(0.3, "curious")
        curve.add_point(0.7, "excited")
        curve.add_point(1.0, "happy")

        # Sample at any position
        vad = curve.get_vad_at(0.5)
        emotion = curve.get_emotion_at(0.5)
    """

    # Easing functions
    EASINGS: dict[str, Callable[[float], float]] = {
        "linear": lambda t: t,
        "ease_in": lambda t: t * t,
        "ease_out": lambda t: t * (2 - t),
        "ease_in_out": lambda t: t * t * (3 - 2 * t),
        "ease_in_cubic": lambda t: t * t * t,
        "ease_out_cubic": lambda t: 1 - (1 - t) ** 3,
        "ease_in_out_cubic": lambda t: 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2,
        "step": lambda t: 0.0 if t < 0.5 else 1.0,
        "hold": lambda t: 0.0,  # Stay at previous value
    }

    def __init__(self, default_easing: str = "linear"):
        """
        Initialize an empty emotion curve.

        Args:
            default_easing: Default easing between keyframes
        """
        self.keyframes: List[EmotionKeyframe] = []
        self.default_easing = default_easing

    def add_point(
        self,
        position: float,
        emotion: str,
        intensity: float = 1.0,
        easing: Optional[str] = None,
    ) -> "EmotionCurve":
        """
        Add a keyframe to the curve.

        Args:
            position: Normalized position (0.0 to 1.0)
            emotion: Emotion name
            intensity: Emotion intensity
            easing: Easing to next keyframe (or default)

        Returns:
            Self for chaining
        """
        keyframe = EmotionKeyframe(
            position=position,
            emotion=emotion,
            intensity=intensity,
            easing=easing or self.default_easing,
        )
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.position)
        return self

    def remove_point(self, position: float, tolerance: float = 0.01) -> bool:
        """
        Remove a keyframe at the given position.

        Args:
            position: Position to remove
            tolerance: Position matching tolerance

        Returns:
            True if removed, False if not found
        """
        for i, kf in enumerate(self.keyframes):
            if abs(kf.position - position) <= tolerance:
                self.keyframes.pop(i)
                return True
        return False

    def clear(self) -> "EmotionCurve":
        """Clear all keyframes."""
        self.keyframes = []
        return self

    def get_vad_at(self, position: float) -> VADPoint:
        """
        Get the VAD values at a position on the curve.

        Args:
            position: Normalized position (0.0 to 1.0)

        Returns:
            Interpolated VAD point
        """
        position = max(0.0, min(1.0, position))

        if not self.keyframes:
            return VAD_EMOTIONS["neutral"]

        if len(self.keyframes) == 1:
            return self.keyframes[0].vad

        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes):
            if kf.position >= position:
                next_kf = kf
                if i > 0:
                    prev_kf = self.keyframes[i - 1]
                break
            prev_kf = kf

        # Handle edge cases
        if position <= prev_kf.position:
            return prev_kf.vad
        if position >= next_kf.position:
            return next_kf.vad

        # Calculate local t (0-1 between keyframes)
        segment_length = next_kf.position - prev_kf.position
        if segment_length == 0:
            return prev_kf.vad

        local_t = (position - prev_kf.position) / segment_length

        # Apply easing
        easing_fn = self.EASINGS.get(prev_kf.easing, self.EASINGS["linear"])
        eased_t = easing_fn(local_t)

        # Interpolate VAD
        return interpolate_vad(prev_kf.vad, next_kf.vad, eased_t)

    def get_emotion_at(self, position: float) -> str:
        """
        Get the closest named emotion at a position.

        Args:
            position: Normalized position

        Returns:
            Emotion name
        """
        vad = self.get_vad_at(position)
        closest = vad_to_emotion(vad, top_n=1)
        return closest[0][0] if closest else "neutral"

    def get_intensity_at(self, position: float) -> float:
        """
        Get the interpolated intensity at a position.

        Args:
            position: Normalized position

        Returns:
            Intensity value (0-1)
        """
        position = max(0.0, min(1.0, position))

        if not self.keyframes:
            return 1.0

        if len(self.keyframes) == 1:
            return self.keyframes[0].intensity

        # Find surrounding keyframes and interpolate
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes):
            if kf.position >= position:
                next_kf = kf
                if i > 0:
                    prev_kf = self.keyframes[i - 1]
                break
            prev_kf = kf

        if position <= prev_kf.position:
            return prev_kf.intensity
        if position >= next_kf.position:
            return next_kf.intensity

        segment_length = next_kf.position - prev_kf.position
        if segment_length == 0:
            return prev_kf.intensity

        local_t = (position - prev_kf.position) / segment_length
        return prev_kf.intensity + (next_kf.intensity - prev_kf.intensity) * local_t

    def sample(self, num_samples: int = 10) -> List[Tuple[float, VADPoint, str]]:
        """
        Sample the curve at regular intervals.

        Args:
            num_samples: Number of samples

        Returns:
            List of (position, vad, emotion) tuples
        """
        samples = []
        for i in range(num_samples):
            pos = i / (num_samples - 1) if num_samples > 1 else 0.0
            vad = self.get_vad_at(pos)
            emotion = self.get_emotion_at(pos)
            samples.append((pos, vad, emotion))
        return samples

    def to_keyframes_dict(self) -> List[dict]:
        """Convert curve to serializable format."""
        return [
            {
                "position": kf.position,
                "emotion": kf.emotion,
                "intensity": kf.intensity,
                "easing": kf.easing,
            }
            for kf in self.keyframes
        ]

    @classmethod
    def from_keyframes_dict(cls, data: List[dict]) -> "EmotionCurve":
        """Create curve from serialized format."""
        curve = cls()
        for kf_data in data:
            curve.add_point(
                position=kf_data["position"],
                emotion=kf_data["emotion"],
                intensity=kf_data.get("intensity", 1.0),
                easing=kf_data.get("easing"),
            )
        return curve

    def __len__(self) -> int:
        """Get number of keyframes."""
        return len(self.keyframes)


def create_linear_curve(
    start_emotion: str,
    end_emotion: str,
) -> EmotionCurve:
    """
    Create a simple linear transition between two emotions.

    Args:
        start_emotion: Starting emotion
        end_emotion: Ending emotion

    Returns:
        EmotionCurve with two keyframes
    """
    curve = EmotionCurve()
    curve.add_point(0.0, start_emotion)
    curve.add_point(1.0, end_emotion)
    return curve


def create_arc_curve(
    start_emotion: str,
    peak_emotion: str,
    end_emotion: str,
    peak_position: float = 0.5,
) -> EmotionCurve:
    """
    Create an arc curve that rises to a peak and falls.

    Args:
        start_emotion: Starting emotion
        peak_emotion: Peak emotion
        end_emotion: Ending emotion
        peak_position: Position of peak (0-1)

    Returns:
        EmotionCurve with arc shape
    """
    curve = EmotionCurve(default_easing="ease_in_out")
    curve.add_point(0.0, start_emotion)
    curve.add_point(peak_position, peak_emotion)
    curve.add_point(1.0, end_emotion)
    return curve


def create_buildup_curve(
    start_emotion: str,
    end_emotion: str,
    buildup_speed: float = 0.7,
) -> EmotionCurve:
    """
    Create a curve that slowly builds to final emotion.

    Args:
        start_emotion: Starting emotion
        end_emotion: Final emotion
        buildup_speed: How late the transition happens (0.5-0.9)

    Returns:
        EmotionCurve with slow build
    """
    curve = EmotionCurve()
    curve.add_point(0.0, start_emotion, easing="ease_in_cubic")
    curve.add_point(buildup_speed, start_emotion, easing="ease_out_cubic")
    curve.add_point(1.0, end_emotion)
    return curve


def create_fade_curve(
    start_emotion: str,
    end_emotion: str,
    fade_start: float = 0.3,
) -> EmotionCurve:
    """
    Create a curve that quickly fades to final emotion.

    Args:
        start_emotion: Starting emotion
        end_emotion: Final emotion
        fade_start: When fade begins (0.1-0.5)

    Returns:
        EmotionCurve with fade pattern
    """
    curve = EmotionCurve()
    curve.add_point(0.0, start_emotion, easing="ease_out_cubic")
    curve.add_point(fade_start, end_emotion, easing="linear")
    curve.add_point(1.0, end_emotion)
    return curve


def create_wave_curve(
    base_emotion: str,
    peak_emotion: str,
    num_waves: int = 2,
) -> EmotionCurve:
    """
    Create a wave pattern oscillating between emotions.

    Args:
        base_emotion: Base/trough emotion
        peak_emotion: Peak emotion
        num_waves: Number of oscillations

    Returns:
        EmotionCurve with wave pattern
    """
    curve = EmotionCurve(default_easing="ease_in_out")

    for i in range(num_waves * 2 + 1):
        pos = i / (num_waves * 2)
        is_peak = i % 2 == 1
        emotion = peak_emotion if is_peak else base_emotion
        curve.add_point(pos, emotion)

    return curve


# Pre-built emotion curves for common narrative patterns
NARRATIVE_CURVES = {
    "tension_build": create_buildup_curve("calm", "anxious"),
    "joy_arc": create_arc_curve("neutral", "excited", "happy"),
    "sadness_fade": create_fade_curve("happy", "sad"),
    "suspense": create_wave_curve("tense", "fearful", num_waves=3),
    "revelation": create_arc_curve("curious", "surprised", "amazed"),
    "comfort": create_linear_curve("anxious", "calm"),
    "anger_buildup": create_buildup_curve("annoyed", "angry", buildup_speed=0.8),
    "resolution": create_arc_curve("tense", "relieved", "peaceful"),
}


def get_narrative_curve(name: str) -> Optional[EmotionCurve]:
    """
    Get a pre-built narrative emotion curve.

    Args:
        name: Curve name

    Returns:
        EmotionCurve if found, None otherwise
    """
    return NARRATIVE_CURVES.get(name.lower())


def list_narrative_curves() -> List[str]:
    """List available narrative curves."""
    return list(NARRATIVE_CURVES.keys())
