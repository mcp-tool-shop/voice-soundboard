"""
Tests for Advanced Emotion Control Module.

Tests cover:
- VAD emotion model
- Emotion tag parser
- Emotion blending
- Emotion curves
"""

import pytest
import math

from voice_soundboard.emotion.vad import (
    VADPoint,
    emotion_to_vad,
    vad_to_emotion,
    VAD_EMOTIONS,
    interpolate_vad,
    get_emotion_intensity,
    classify_emotion_category,
    list_emotions_by_category,
)
from voice_soundboard.emotion.parser import (
    EmotionParser,
    EmotionSpan,
    ParsedEmotionText,
    parse_emotion_tags,
    create_tagged_text,
    merge_adjacent_spans,
)
from voice_soundboard.emotion.blending import (
    blend_emotions,
    blend_vad,
    EmotionMix,
    transition_emotion,
    create_emotion_gradient,
    get_complementary_emotion,
    get_similar_emotions,
    emotion_distance,
    get_named_blend,
    list_named_blends,
)
from voice_soundboard.emotion.curves import (
    EmotionCurve,
    EmotionKeyframe,
    create_linear_curve,
    create_arc_curve,
    create_buildup_curve,
    create_fade_curve,
    create_wave_curve,
    get_narrative_curve,
    list_narrative_curves,
)


class TestVADPoint:
    """Tests for VADPoint dataclass."""

    def test_creation(self):
        """Test VADPoint creation."""
        vad = VADPoint(valence=0.5, arousal=0.7, dominance=0.3)
        assert vad.valence == 0.5
        assert vad.arousal == 0.7
        assert vad.dominance == 0.3

    def test_clamping(self):
        """Test values are clamped to valid ranges."""
        vad = VADPoint(valence=1.5, arousal=-0.5, dominance=2.0)
        assert vad.valence == 1.0
        assert vad.arousal == 0.0
        assert vad.dominance == 1.0

    def test_distance(self):
        """Test Euclidean distance calculation."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)

        distance = vad1.distance(vad2)
        expected = math.sqrt(1 + 1 + 1)
        assert abs(distance - expected) < 0.001

    def test_to_tuple(self):
        """Test conversion to tuple."""
        vad = VADPoint(valence=0.5, arousal=0.7, dominance=0.3)
        t = vad.to_tuple()
        assert t == (0.5, 0.7, 0.3)

    def test_from_tuple(self):
        """Test creation from tuple."""
        vad = VADPoint.from_tuple((0.5, 0.7, 0.3))
        assert vad.valence == 0.5
        assert vad.arousal == 0.7
        assert vad.dominance == 0.3

    def test_addition(self):
        """Test VADPoint addition."""
        vad1 = VADPoint(valence=0.3, arousal=0.4, dominance=0.5)
        vad2 = VADPoint(valence=0.2, arousal=0.1, dominance=0.2)
        result = vad1 + vad2
        assert abs(result.valence - 0.5) < 0.001
        assert abs(result.arousal - 0.5) < 0.001
        assert abs(result.dominance - 0.7) < 0.001

    def test_multiplication(self):
        """Test scalar multiplication."""
        vad = VADPoint(valence=0.5, arousal=0.4, dominance=0.3)
        result = vad * 2.0
        assert abs(result.valence - 1.0) < 0.001
        assert abs(result.arousal - 0.8) < 0.001
        assert abs(result.dominance - 0.6) < 0.001


class TestEmotionToVAD:
    """Tests for emotion_to_vad function."""

    def test_known_emotions(self):
        """Test conversion of known emotions."""
        emotions = ["happy", "sad", "angry", "fearful", "surprised"]
        for emotion in emotions:
            vad = emotion_to_vad(emotion)
            assert isinstance(vad, VADPoint)

    def test_happy_is_positive_valence(self):
        """Test happy has positive valence."""
        vad = emotion_to_vad("happy")
        assert vad.valence > 0

    def test_sad_is_negative_valence(self):
        """Test sad has negative valence."""
        vad = emotion_to_vad("sad")
        assert vad.valence < 0

    def test_excited_is_high_arousal(self):
        """Test excited has high arousal."""
        vad = emotion_to_vad("excited")
        assert vad.arousal > 0.7

    def test_calm_is_low_arousal(self):
        """Test calm has low arousal."""
        vad = emotion_to_vad("calm")
        assert vad.arousal < 0.4

    def test_case_insensitive(self):
        """Test emotion lookup is case-insensitive."""
        vad1 = emotion_to_vad("HAPPY")
        vad2 = emotion_to_vad("happy")
        assert vad1.valence == vad2.valence

    def test_unknown_emotion_raises(self):
        """Test unknown emotion raises ValueError."""
        with pytest.raises(ValueError):
            emotion_to_vad("nonexistent_emotion_xyz")


class TestVADToEmotion:
    """Tests for vad_to_emotion function."""

    def test_returns_closest_emotion(self):
        """Test returns closest named emotion."""
        vad = VAD_EMOTIONS["happy"]
        closest = vad_to_emotion(vad, top_n=1)
        assert len(closest) == 1
        assert closest[0][0] == "happy"

    def test_returns_multiple(self):
        """Test returning multiple closest emotions."""
        vad = VADPoint(valence=0.7, arousal=0.6, dominance=0.6)
        closest = vad_to_emotion(vad, top_n=3)
        assert len(closest) == 3
        # All should be reasonably close
        assert all(dist < 1.0 for _, dist in closest)

    def test_sorted_by_distance(self):
        """Test results are sorted by distance."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        closest = vad_to_emotion(vad, top_n=5)
        distances = [d for _, d in closest]
        assert distances == sorted(distances)


class TestInterpolateVAD:
    """Tests for interpolate_vad function."""

    def test_t_zero_returns_first(self):
        """Test t=0 returns first VAD point."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 0.0)
        assert result.valence == 0.0
        assert result.arousal == 0.0

    def test_t_one_returns_second(self):
        """Test t=1 returns second VAD point."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 1.0)
        assert result.valence == 1.0
        assert result.arousal == 1.0

    def test_t_half_returns_midpoint(self):
        """Test t=0.5 returns midpoint."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 0.5)
        assert abs(result.valence - 0.5) < 0.001
        assert abs(result.arousal - 0.5) < 0.001

    def test_t_clamped(self):
        """Test t is clamped to 0-1."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)

        result_low = interpolate_vad(vad1, vad2, -0.5)
        assert result_low.valence == 0.0

        result_high = interpolate_vad(vad1, vad2, 1.5)
        assert result_high.valence == 1.0


class TestEmotionIntensity:
    """Tests for get_emotion_intensity function."""

    def test_neutral_is_low_intensity(self):
        """Test neutral has low intensity."""
        vad = VAD_EMOTIONS["neutral"]
        intensity = get_emotion_intensity(vad)
        assert intensity < 0.4

    def test_excited_is_high_intensity(self):
        """Test excited has high intensity."""
        vad = VAD_EMOTIONS["excited"]
        intensity = get_emotion_intensity(vad)
        assert intensity > 0.5

    def test_intensity_in_range(self):
        """Test intensity is always 0-1."""
        for vad in VAD_EMOTIONS.values():
            intensity = get_emotion_intensity(vad)
            assert 0 <= intensity <= 1


class TestClassifyEmotionCategory:
    """Tests for classify_emotion_category function."""

    def test_happy_is_positive_high(self):
        """Test happy is positive_high category."""
        vad = VAD_EMOTIONS["happy"]
        category = classify_emotion_category(vad)
        assert category == "positive_high"

    def test_calm_is_positive_low(self):
        """Test calm is positive_low category."""
        vad = VAD_EMOTIONS["calm"]
        category = classify_emotion_category(vad)
        assert category == "positive_low"

    def test_angry_is_negative_high(self):
        """Test angry is negative_high category."""
        vad = VAD_EMOTIONS["angry"]
        category = classify_emotion_category(vad)
        assert category == "negative_high"

    def test_sad_is_negative_low(self):
        """Test sad is negative_low category."""
        vad = VAD_EMOTIONS["sad"]
        category = classify_emotion_category(vad)
        assert category == "negative_low"


class TestEmotionParser:
    """Tests for EmotionParser class."""

    def test_parse_single_tag(self):
        """Test parsing single emotion tag."""
        parser = EmotionParser()
        result = parser.parse("I'm {happy}so glad{/happy} to see you!")

        assert result.plain_text == "I'm so glad to see you!"
        assert len(result.spans) == 1
        assert result.spans[0].emotion == "happy"
        assert result.spans[0].text == "so glad"

    def test_parse_multiple_tags(self):
        """Test parsing multiple emotion tags."""
        parser = EmotionParser()
        result = parser.parse("{sad}Goodbye{/sad} and {happy}hello{/happy}!")

        assert len(result.spans) == 2
        emotions = {s.emotion for s in result.spans}
        assert "sad" in emotions
        assert "happy" in emotions

    def test_parse_with_intensity(self):
        """Test parsing tags with intensity."""
        parser = EmotionParser()
        result = parser.parse("{excited:0.8}Wow{/excited}!")

        assert len(result.spans) == 1
        assert result.spans[0].emotion == "excited"
        assert result.spans[0].intensity == 0.8

    def test_parse_no_tags(self):
        """Test parsing text without tags."""
        parser = EmotionParser()
        result = parser.parse("Just regular text here.")

        assert result.plain_text == "Just regular text here."
        assert len(result.spans) == 0
        assert not result.has_emotion_tags()

    def test_has_tags(self):
        """Test has_tags detection."""
        parser = EmotionParser()
        assert parser.has_tags("{happy}text{/happy}")
        assert not parser.has_tags("plain text")

    def test_remove_tags(self):
        """Test tag removal."""
        parser = EmotionParser()
        text = "I'm {happy}so glad{/happy}!"
        plain = parser.remove_tags(text)
        assert plain == "I'm so glad!"

    def test_extract_emotions(self):
        """Test emotion extraction."""
        parser = EmotionParser()
        text = "{happy}text{/happy} {sad}more{/sad} {happy}again{/happy}"
        emotions = parser.extract_emotions(text)
        assert set(emotions) == {"happy", "sad"}

    def test_case_insensitive_tags(self):
        """Test case-insensitive tag parsing."""
        parser = EmotionParser()
        result = parser.parse("{HAPPY}text{/HAPPY}")
        assert result.spans[0].emotion == "happy"

    def test_unclosed_tag_handled(self):
        """Test unclosed tags are handled gracefully."""
        parser = EmotionParser()
        result = parser.parse("{happy}unclosed text")
        assert "unclosed text" in result.plain_text


class TestParseEmotionTags:
    """Tests for parse_emotion_tags convenience function."""

    def test_basic_parsing(self):
        """Test basic parsing works."""
        result = parse_emotion_tags("{happy}hello{/happy}")
        assert result.has_emotion_tags()
        assert result.spans[0].emotion == "happy"


class TestCreateTaggedText:
    """Tests for create_tagged_text function."""

    def test_basic_tagging(self):
        """Test basic text tagging."""
        tagged = create_tagged_text("hello", "happy")
        assert tagged == "{happy}hello{/happy}"

    def test_with_intensity(self):
        """Test tagging with intensity."""
        tagged = create_tagged_text("hello", "excited", 0.7)
        assert tagged == "{excited:0.7}hello{/excited}"


class TestBlendEmotions:
    """Tests for blend_emotions function."""

    def test_single_emotion(self):
        """Test blending single emotion returns itself."""
        result = blend_emotions([("happy", 1.0)])
        assert result.dominant_emotion == "happy"

    def test_two_emotions(self):
        """Test blending two emotions."""
        result = blend_emotions([("happy", 0.5), ("sad", 0.5)])
        assert isinstance(result, EmotionMix)
        assert result.vad is not None

    def test_weights_normalize(self):
        """Test weights are auto-normalized."""
        result = blend_emotions([("happy", 2.0), ("sad", 2.0)])
        # Should work and treat as 50/50
        assert result.vad is not None

    def test_returns_closest_emotion(self):
        """Test dominant emotion is closest named emotion."""
        result = blend_emotions([("happy", 1.0)])
        assert result.dominant_emotion == "happy"

    def test_bittersweet_blend(self):
        """Test classic bittersweet blend."""
        result = blend_emotions([("happy", 0.5), ("sad", 0.5)])
        # Should result in moderate valence
        assert -0.5 < result.vad.valence < 0.5


class TestBlendVAD:
    """Tests for blend_vad function."""

    def test_single_vad(self):
        """Test blending single VAD."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = blend_vad([(vad, 1.0)])
        assert result.valence == vad.valence

    def test_two_vads(self):
        """Test blending two VADs."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = blend_vad([(vad1, 0.5), (vad2, 0.5)])
        assert abs(result.valence - 0.5) < 0.001


class TestTransitionEmotion:
    """Tests for transition_emotion function."""

    def test_progress_zero(self):
        """Test progress=0 returns start emotion."""
        result = transition_emotion("happy", "sad", 0.0)
        assert result.dominant_emotion == "happy"

    def test_progress_one(self):
        """Test progress=1 returns end emotion."""
        result = transition_emotion("happy", "sad", 1.0)
        assert result.dominant_emotion == "sad"

    def test_progress_half(self):
        """Test progress=0.5 returns blend."""
        result = transition_emotion("happy", "sad", 0.5)
        # Should be somewhere in between
        assert result.vad is not None


class TestEmotionCurve:
    """Tests for EmotionCurve class."""

    def test_empty_curve(self):
        """Test empty curve returns neutral."""
        curve = EmotionCurve()
        vad = curve.get_vad_at(0.5)
        assert vad == VAD_EMOTIONS["neutral"]

    def test_single_keyframe(self):
        """Test single keyframe returns that emotion."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        emotion = curve.get_emotion_at(0.0)
        assert emotion == "happy"

    def test_two_keyframes(self):
        """Test interpolation between two keyframes."""
        curve = EmotionCurve()
        curve.add_point(0.0, "sad")
        curve.add_point(1.0, "happy")

        # At start
        assert curve.get_emotion_at(0.0) == "sad"
        # At end
        assert curve.get_emotion_at(1.0) == "happy"

    def test_sample_curve(self):
        """Test sampling curve at multiple points."""
        curve = EmotionCurve()
        curve.add_point(0.0, "neutral")
        curve.add_point(1.0, "excited")

        samples = curve.sample(5)
        assert len(samples) == 5
        # First should be close to neutral
        assert samples[0][2] in ["neutral", "serious", "thoughtful", "curious"]

    def test_chained_add_point(self):
        """Test add_point can be chained."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm").add_point(1.0, "excited")
        assert len(curve) == 2

    def test_remove_point(self):
        """Test removing keyframes."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(0.5, "sad")

        assert len(curve) == 2
        removed = curve.remove_point(0.5)
        assert removed
        assert len(curve) == 1

    def test_clear(self):
        """Test clearing curve."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")
        curve.clear()
        assert len(curve) == 0


class TestPrebuiltCurves:
    """Tests for pre-built curve functions."""

    def test_linear_curve(self):
        """Test linear curve creation."""
        curve = create_linear_curve("happy", "sad")
        assert len(curve) == 2
        assert curve.get_emotion_at(0.0) == "happy"
        assert curve.get_emotion_at(1.0) == "sad"

    def test_arc_curve(self):
        """Test arc curve creation."""
        curve = create_arc_curve("calm", "excited", "happy")
        assert len(curve) == 3

    def test_buildup_curve(self):
        """Test buildup curve creation."""
        curve = create_buildup_curve("calm", "excited")
        assert len(curve) >= 2

    def test_fade_curve(self):
        """Test fade curve creation."""
        curve = create_fade_curve("excited", "calm")
        assert len(curve) >= 2

    def test_wave_curve(self):
        """Test wave curve creation."""
        curve = create_wave_curve("calm", "excited", num_waves=2)
        assert len(curve) >= 3


class TestNarrativeCurves:
    """Tests for narrative curve presets."""

    def test_list_narrative_curves(self):
        """Test listing available curves."""
        curves = list_narrative_curves()
        assert len(curves) > 0
        assert "tension_build" in curves
        assert "joy_arc" in curves

    def test_get_narrative_curve(self):
        """Test getting narrative curve by name."""
        curve = get_narrative_curve("tension_build")
        assert curve is not None
        assert len(curve) >= 2

    def test_unknown_curve_returns_none(self):
        """Test unknown curve returns None."""
        curve = get_narrative_curve("nonexistent_curve")
        assert curve is None


class TestNamedBlends:
    """Tests for named emotion blends."""

    def test_list_named_blends(self):
        """Test listing available blends."""
        blends = list_named_blends()
        assert len(blends) > 0
        assert "bittersweet" in blends

    def test_get_named_blend(self):
        """Test getting blend by name."""
        blend = get_named_blend("bittersweet")
        assert blend is not None
        assert isinstance(blend, EmotionMix)

    def test_unknown_blend_returns_none(self):
        """Test unknown blend returns None."""
        blend = get_named_blend("nonexistent_blend")
        assert blend is None


class TestEmotionDistance:
    """Tests for emotion_distance function."""

    def test_same_emotion_zero_distance(self):
        """Test same emotion has zero distance."""
        dist = emotion_distance("happy", "happy")
        assert dist == 0.0

    def test_opposite_emotions_large_distance(self):
        """Test opposite emotions have large distance."""
        dist = emotion_distance("happy", "sad")
        assert dist > 1.0

    def test_similar_emotions_small_distance(self):
        """Test similar emotions have small distance."""
        dist = emotion_distance("happy", "joyful")
        assert dist < 0.5


class TestSimilarEmotions:
    """Tests for get_similar_emotions function."""

    def test_returns_similar(self):
        """Test returns similar emotions."""
        similar = get_similar_emotions("happy", count=3)
        assert len(similar) == 3
        assert "happy" not in similar  # Should not include self

    def test_similar_are_close(self):
        """Test returned emotions are close."""
        similar = get_similar_emotions("excited")
        for emotion in similar:
            dist = emotion_distance("excited", emotion)
            assert dist < 1.0


class TestComplementaryEmotion:
    """Tests for get_complementary_emotion function."""

    def test_returns_different(self):
        """Test returns different emotion."""
        comp = get_complementary_emotion("happy")
        assert comp != "happy"

    def test_opposite_valence(self):
        """Test complementary has opposite valence."""
        comp = get_complementary_emotion("happy")
        vad_happy = emotion_to_vad("happy")
        vad_comp = emotion_to_vad(comp)
        # Should have opposite valence sign
        assert vad_happy.valence * vad_comp.valence <= 0


class TestEmotionKeyframe:
    """Tests for EmotionKeyframe dataclass."""

    def test_auto_vad_computation(self):
        """Test VAD is auto-computed from emotion."""
        kf = EmotionKeyframe(position=0.5, emotion="happy")
        assert kf.vad is not None
        assert kf.vad.valence > 0

    def test_position_clamping(self):
        """Test position is clamped to 0-1."""
        kf = EmotionKeyframe(position=1.5, emotion="happy")
        assert kf.position == 1.0


class TestParsedEmotionText:
    """Tests for ParsedEmotionText methods."""

    def test_get_emotion_at_position(self):
        """Test getting emotion at character position."""
        result = parse_emotion_tags("Hi {happy}there{/happy}!")

        # Before span
        assert result.get_emotion_at_position(0) == "neutral"
        # In span (starts at 3)
        assert result.get_emotion_at_position(4) == "happy"

    def test_get_emotions_used(self):
        """Test getting list of emotions used."""
        result = parse_emotion_tags("{happy}a{/happy} {sad}b{/sad} {happy}c{/happy}")
        emotions = result.get_emotions_used()
        assert set(emotions) == {"happy", "sad"}

    def test_get_emotion_timeline(self):
        """Test getting emotion timeline."""
        result = parse_emotion_tags("{happy}hello{/happy}")
        timeline = result.get_emotion_timeline()
        assert len(timeline) >= 1
