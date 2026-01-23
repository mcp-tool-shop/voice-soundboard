"""
Advanced Emotion Control Module.

Provides fine-grained emotion control for TTS synthesis:
- Word-level emotion tags: {happy}Hello{/happy} world!
- VAD emotion model (Valence-Arousal-Dominance)
- Emotion blending: Mix multiple emotions with weights
- Dynamic emotion curves: Interpolate emotions over utterance

Example:
    from voice_soundboard.emotion import EmotionParser, EmotionCurve, blend_emotions

    # Parse inline emotion tags
    parser = EmotionParser()
    result = parser.parse("I'm {excited}so happy{/excited} to see you!")

    # Blend emotions
    blended = blend_emotions([
        ("happy", 0.7),
        ("surprised", 0.3)
    ])

    # Create emotion curve
    curve = EmotionCurve()
    curve.add_point(0.0, "neutral")
    curve.add_point(0.5, "excited")
    curve.add_point(1.0, "happy")
"""

from voice_soundboard.emotion.parser import (
    EmotionParser,
    EmotionSpan,
    ParsedEmotionText,
    parse_emotion_tags,
)
from voice_soundboard.emotion.vad import (
    VADPoint,
    emotion_to_vad,
    vad_to_emotion,
    VAD_EMOTIONS,
    interpolate_vad,
)
from voice_soundboard.emotion.blending import (
    blend_emotions,
    blend_vad,
    EmotionMix,
    list_named_blends,
    get_named_blend,
)
from voice_soundboard.emotion.curves import (
    EmotionCurve,
    EmotionKeyframe,
    create_linear_curve,
    create_arc_curve,
    list_narrative_curves,
    get_narrative_curve,
)

__all__ = [
    # Parser
    "EmotionParser",
    "EmotionSpan",
    "ParsedEmotionText",
    "parse_emotion_tags",
    # VAD Model
    "VADPoint",
    "emotion_to_vad",
    "vad_to_emotion",
    "VAD_EMOTIONS",
    "interpolate_vad",
    # Blending
    "blend_emotions",
    "blend_vad",
    "EmotionMix",
    "list_named_blends",
    "get_named_blend",
    # Curves
    "EmotionCurve",
    "EmotionKeyframe",
    "create_linear_curve",
    "create_arc_curve",
    "list_narrative_curves",
    "get_narrative_curve",
]
