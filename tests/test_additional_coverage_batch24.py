"""
Additional test coverage batch 24: emotion/curves.py and emotion/parser.py.

Tests for dynamic emotion curves and word-level emotion tag parsing.
"""

import pytest
import math

from voice_soundboard.emotion.curves import (
    EmotionKeyframe,
    EmotionCurve,
    create_linear_curve,
    create_arc_curve,
    create_buildup_curve,
    create_fade_curve,
    create_wave_curve,
    NARRATIVE_CURVES,
    get_narrative_curve,
    list_narrative_curves,
)
from voice_soundboard.emotion.parser import (
    EmotionTagType,
    EmotionSpan,
    ParsedEmotionText,
    EmotionParser,
    parse_emotion_tags,
    create_tagged_text,
    merge_adjacent_spans,
)
from voice_soundboard.emotion.vad import VADPoint, VAD_EMOTIONS


# ============================================================================
# EmotionKeyframe Tests
# ============================================================================

class TestEmotionKeyframe:
    """Tests for EmotionKeyframe dataclass."""

    def test_keyframe_creation_basic(self):
        """Test basic keyframe creation."""
        kf = EmotionKeyframe(position=0.5, emotion="happy")
        assert kf.position == 0.5
        assert kf.emotion == "happy"
        assert kf.intensity == 1.0
        assert kf.easing == "linear"

    def test_keyframe_computes_vad(self):
        """Test VAD is computed from emotion name."""
        kf = EmotionKeyframe(position=0.5, emotion="happy")
        assert kf.vad is not None
        assert kf.vad == VAD_EMOTIONS["happy"]

    def test_keyframe_unknown_emotion_defaults_to_neutral(self):
        """Test unknown emotion defaults to neutral VAD."""
        kf = EmotionKeyframe(position=0.5, emotion="nonexistent_xyz")
        assert kf.vad == VAD_EMOTIONS["neutral"]

    def test_keyframe_clamps_position_low(self):
        """Test position is clamped to minimum 0.0."""
        kf = EmotionKeyframe(position=-0.5, emotion="happy")
        assert kf.position == 0.0

    def test_keyframe_clamps_position_high(self):
        """Test position is clamped to maximum 1.0."""
        kf = EmotionKeyframe(position=1.5, emotion="happy")
        assert kf.position == 1.0

    def test_keyframe_custom_intensity(self):
        """Test custom intensity value."""
        kf = EmotionKeyframe(position=0.5, emotion="happy", intensity=0.7)
        assert kf.intensity == 0.7

    def test_keyframe_custom_easing(self):
        """Test custom easing value."""
        kf = EmotionKeyframe(position=0.5, emotion="happy", easing="ease_in")
        assert kf.easing == "ease_in"

    def test_keyframe_preserves_explicit_vad(self):
        """Test explicit VAD is preserved."""
        custom_vad = VADPoint(valence=0.9, arousal=0.9, dominance=0.9)
        kf = EmotionKeyframe(position=0.5, emotion="happy", vad=custom_vad)
        assert kf.vad == custom_vad


# ============================================================================
# EmotionCurve Tests
# ============================================================================

class TestEmotionCurve:
    """Tests for EmotionCurve class."""

    def test_curve_creation_empty(self):
        """Test creating empty curve."""
        curve = EmotionCurve()
        assert len(curve) == 0

    def test_curve_add_point(self):
        """Test adding a point to curve."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        assert len(curve) == 1
        assert curve.keyframes[0].emotion == "happy"

    def test_curve_add_multiple_points(self):
        """Test adding multiple points."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(0.5, "excited")
        curve.add_point(1.0, "happy")
        assert len(curve) == 3

    def test_curve_points_are_sorted(self):
        """Test points are automatically sorted by position."""
        curve = EmotionCurve()
        curve.add_point(1.0, "happy")
        curve.add_point(0.0, "calm")
        curve.add_point(0.5, "excited")
        positions = [kf.position for kf in curve.keyframes]
        assert positions == [0.0, 0.5, 1.0]

    def test_curve_chaining(self):
        """Test add_point returns self for chaining."""
        curve = EmotionCurve()
        result = curve.add_point(0.0, "calm").add_point(1.0, "happy")
        assert result is curve
        assert len(curve) == 2

    def test_curve_remove_point(self):
        """Test removing a point from curve."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(0.5, "excited")
        curve.add_point(1.0, "happy")

        result = curve.remove_point(0.5)
        assert result is True
        assert len(curve) == 2

    def test_curve_remove_point_not_found(self):
        """Test removing non-existent point returns False."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")

        result = curve.remove_point(0.5)
        assert result is False

    def test_curve_remove_point_with_tolerance(self):
        """Test removing point with tolerance."""
        curve = EmotionCurve()
        curve.add_point(0.5, "excited")

        result = curve.remove_point(0.51, tolerance=0.02)
        assert result is True
        assert len(curve) == 0

    def test_curve_clear(self):
        """Test clearing all keyframes."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(1.0, "happy")
        curve.clear()
        assert len(curve) == 0

    def test_curve_clear_chaining(self):
        """Test clear returns self for chaining."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        result = curve.clear()
        assert result is curve

    def test_curve_get_vad_empty(self):
        """Test getting VAD from empty curve returns neutral."""
        curve = EmotionCurve()
        vad = curve.get_vad_at(0.5)
        assert vad == VAD_EMOTIONS["neutral"]

    def test_curve_get_vad_single_point(self):
        """Test getting VAD from single-point curve."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        vad = curve.get_vad_at(0.0)
        assert vad == VAD_EMOTIONS["happy"]

    def test_curve_get_vad_at_keyframe(self):
        """Test getting VAD at exact keyframe position."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(1.0, "excited")

        vad = curve.get_vad_at(0.0)
        assert vad.valence == pytest.approx(VAD_EMOTIONS["calm"].valence)

    def test_curve_get_vad_interpolated(self):
        """Test getting interpolated VAD between keyframes."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(1.0, "excited")

        vad = curve.get_vad_at(0.5)
        calm_vad = VAD_EMOTIONS["calm"]
        excited_vad = VAD_EMOTIONS["excited"]

        expected_valence = (calm_vad.valence + excited_vad.valence) / 2
        assert vad.valence == pytest.approx(expected_valence)

    def test_curve_get_vad_clamps_position(self):
        """Test position is clamped for get_vad_at."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(1.0, "excited")

        vad_low = curve.get_vad_at(-0.5)
        vad_high = curve.get_vad_at(1.5)

        assert vad_low.valence == pytest.approx(VAD_EMOTIONS["calm"].valence)
        assert vad_high.valence == pytest.approx(VAD_EMOTIONS["excited"].valence)

    def test_curve_get_emotion_at(self):
        """Test getting emotion name at position."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")

        emotion = curve.get_emotion_at(0.0)
        assert emotion == "happy"

    def test_curve_get_intensity_at_empty(self):
        """Test getting intensity from empty curve."""
        curve = EmotionCurve()
        intensity = curve.get_intensity_at(0.5)
        assert intensity == 1.0

    def test_curve_get_intensity_at_single(self):
        """Test getting intensity from single-point curve."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy", intensity=0.7)
        intensity = curve.get_intensity_at(0.0)
        assert intensity == 0.7

    def test_curve_get_intensity_interpolated(self):
        """Test getting interpolated intensity."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm", intensity=0.5)
        curve.add_point(1.0, "excited", intensity=1.0)

        intensity = curve.get_intensity_at(0.5)
        assert intensity == pytest.approx(0.75)

    def test_curve_sample(self):
        """Test sampling curve at regular intervals."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm")
        curve.add_point(1.0, "excited")

        samples = curve.sample(num_samples=5)
        assert len(samples) == 5
        assert samples[0][0] == 0.0
        assert samples[-1][0] == 1.0

    def test_curve_sample_returns_tuples(self):
        """Test sample returns (position, vad, emotion) tuples."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")

        samples = curve.sample(num_samples=3)
        pos, vad, emotion = samples[0]
        assert isinstance(pos, float)
        assert isinstance(vad, VADPoint)
        assert isinstance(emotion, str)

    def test_curve_to_keyframes_dict(self):
        """Test serializing curve to dict."""
        curve = EmotionCurve()
        curve.add_point(0.0, "calm", intensity=0.8, easing="ease_in")
        curve.add_point(1.0, "excited")

        data = curve.to_keyframes_dict()
        assert len(data) == 2
        assert data[0]["position"] == 0.0
        assert data[0]["emotion"] == "calm"
        assert data[0]["intensity"] == 0.8
        assert data[0]["easing"] == "ease_in"

    def test_curve_from_keyframes_dict(self):
        """Test deserializing curve from dict."""
        data = [
            {"position": 0.0, "emotion": "calm", "intensity": 0.8},
            {"position": 1.0, "emotion": "excited"},
        ]
        curve = EmotionCurve.from_keyframes_dict(data)
        assert len(curve) == 2
        assert curve.keyframes[0].emotion == "calm"
        assert curve.keyframes[0].intensity == 0.8

    def test_curve_len(self):
        """Test __len__ returns keyframe count."""
        curve = EmotionCurve()
        assert len(curve) == 0
        curve.add_point(0.5, "happy")
        assert len(curve) == 1


class TestEmotionCurveEasings:
    """Tests for emotion curve easing functions."""

    def test_easing_linear(self):
        """Test linear easing."""
        easing = EmotionCurve.EASINGS["linear"]
        assert easing(0.0) == 0.0
        assert easing(0.5) == 0.5
        assert easing(1.0) == 1.0

    def test_easing_ease_in(self):
        """Test ease_in easing (starts slow)."""
        easing = EmotionCurve.EASINGS["ease_in"]
        assert easing(0.0) == 0.0
        assert easing(0.5) < 0.5  # Slower at start
        assert easing(1.0) == 1.0

    def test_easing_ease_out(self):
        """Test ease_out easing (ends slow)."""
        easing = EmotionCurve.EASINGS["ease_out"]
        assert easing(0.0) == 0.0
        assert easing(0.5) > 0.5  # Faster at start
        assert easing(1.0) == 1.0

    def test_easing_step(self):
        """Test step easing."""
        easing = EmotionCurve.EASINGS["step"]
        assert easing(0.0) == 0.0
        assert easing(0.4) == 0.0
        assert easing(0.5) == 1.0
        assert easing(1.0) == 1.0

    def test_easing_hold(self):
        """Test hold easing (stays at 0)."""
        easing = EmotionCurve.EASINGS["hold"]
        assert easing(0.0) == 0.0
        assert easing(0.5) == 0.0
        assert easing(1.0) == 0.0

    def test_curve_uses_easing(self):
        """Test curve applies easing during interpolation."""
        curve_linear = EmotionCurve()
        curve_linear.add_point(0.0, "calm", easing="linear")
        curve_linear.add_point(1.0, "excited")

        curve_ease_in = EmotionCurve()
        curve_ease_in.add_point(0.0, "calm", easing="ease_in")
        curve_ease_in.add_point(1.0, "excited")

        vad_linear = curve_linear.get_vad_at(0.5)
        vad_ease_in = curve_ease_in.get_vad_at(0.5)

        # ease_in should be closer to start at midpoint
        assert vad_ease_in.arousal < vad_linear.arousal


class TestCurveCreationFunctions:
    """Tests for curve creation helper functions."""

    def test_create_linear_curve(self):
        """Test creating linear transition curve."""
        curve = create_linear_curve("happy", "sad")
        assert len(curve) == 2
        assert curve.keyframes[0].emotion == "happy"
        assert curve.keyframes[1].emotion == "sad"

    def test_create_arc_curve(self):
        """Test creating arc curve."""
        curve = create_arc_curve("calm", "excited", "happy")
        assert len(curve) == 3
        assert curve.keyframes[0].emotion == "calm"
        assert curve.keyframes[1].emotion == "excited"
        assert curve.keyframes[2].emotion == "happy"

    def test_create_arc_curve_custom_peak(self):
        """Test arc curve with custom peak position."""
        curve = create_arc_curve("calm", "excited", "happy", peak_position=0.7)
        assert curve.keyframes[1].position == 0.7

    def test_create_buildup_curve(self):
        """Test creating buildup curve."""
        curve = create_buildup_curve("calm", "anxious")
        assert len(curve) == 3
        assert curve.keyframes[0].emotion == "calm"
        assert curve.keyframes[-1].emotion == "anxious"

    def test_create_buildup_curve_custom_speed(self):
        """Test buildup curve with custom speed."""
        curve = create_buildup_curve("calm", "anxious", buildup_speed=0.8)
        assert curve.keyframes[1].position == 0.8

    def test_create_fade_curve(self):
        """Test creating fade curve."""
        curve = create_fade_curve("happy", "sad")
        assert len(curve) == 3
        assert curve.keyframes[0].emotion == "happy"
        assert curve.keyframes[-1].emotion == "sad"

    def test_create_fade_curve_custom_start(self):
        """Test fade curve with custom fade start."""
        curve = create_fade_curve("happy", "sad", fade_start=0.4)
        assert curve.keyframes[1].position == 0.4

    def test_create_wave_curve(self):
        """Test creating wave curve."""
        curve = create_wave_curve("calm", "excited", num_waves=2)
        assert len(curve) == 5  # 2 waves * 2 + 1

    def test_create_wave_curve_alternates(self):
        """Test wave curve alternates between emotions."""
        curve = create_wave_curve("calm", "excited", num_waves=2)
        emotions = [kf.emotion for kf in curve.keyframes]
        assert emotions[0] == "calm"
        assert emotions[1] == "excited"
        assert emotions[2] == "calm"


class TestNarrativeCurves:
    """Tests for pre-built narrative curves."""

    def test_narrative_curves_exist(self):
        """Test narrative curves dictionary exists."""
        assert len(NARRATIVE_CURVES) > 0

    def test_tension_build_exists(self):
        """Test tension_build curve exists."""
        assert "tension_build" in NARRATIVE_CURVES

    def test_joy_arc_exists(self):
        """Test joy_arc curve exists."""
        assert "joy_arc" in NARRATIVE_CURVES

    def test_get_narrative_curve_found(self):
        """Test getting existing narrative curve."""
        curve = get_narrative_curve("tension_build")
        assert curve is not None
        assert isinstance(curve, EmotionCurve)

    def test_get_narrative_curve_case_insensitive(self):
        """Test narrative curve lookup is case insensitive."""
        curve = get_narrative_curve("TENSION_BUILD")
        assert curve is not None

    def test_get_narrative_curve_not_found(self):
        """Test getting non-existent curve returns None."""
        curve = get_narrative_curve("nonexistent_xyz")
        assert curve is None

    def test_list_narrative_curves(self):
        """Test listing narrative curves."""
        curves = list_narrative_curves()
        assert isinstance(curves, list)
        assert "tension_build" in curves
        assert len(curves) == len(NARRATIVE_CURVES)


# ============================================================================
# EmotionParser Tests
# ============================================================================

class TestEmotionSpan:
    """Tests for EmotionSpan dataclass."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = EmotionSpan(
            text="hello world",
            emotion="happy",
            intensity=0.8,
            start_char=0,
            end_char=11,
        )
        assert span.text == "hello world"
        assert span.emotion == "happy"
        assert span.intensity == 0.8

    def test_span_word_count(self):
        """Test word count calculation."""
        span = EmotionSpan(text="hello world today", emotion="happy")
        assert span.word_count() == 3

    def test_span_word_count_empty(self):
        """Test word count for empty text."""
        span = EmotionSpan(text="", emotion="happy")
        assert span.word_count() == 0


class TestParsedEmotionText:
    """Tests for ParsedEmotionText dataclass."""

    def test_get_emotion_at_position_in_span(self):
        """Test getting emotion at position within a span."""
        span = EmotionSpan(
            text="happy text",
            emotion="happy",
            start_char=5,
            end_char=15,
        )
        parsed = ParsedEmotionText(
            original_text="test {happy}happy text{/happy} end",
            plain_text="test happy text end",
            spans=[span],
        )
        assert parsed.get_emotion_at_position(7) == "happy"

    def test_get_emotion_at_position_outside_span(self):
        """Test getting emotion at position outside spans."""
        span = EmotionSpan(
            text="happy text",
            emotion="happy",
            start_char=5,
            end_char=15,
        )
        parsed = ParsedEmotionText(
            original_text="test {happy}happy text{/happy} end",
            plain_text="test happy text end",
            spans=[span],
        )
        assert parsed.get_emotion_at_position(0) == "neutral"

    def test_get_emotion_at_word(self):
        """Test getting emotion at word index."""
        span = EmotionSpan(
            text="happy",
            emotion="happy",
            start_word=1,
            end_word=2,
        )
        parsed = ParsedEmotionText(
            original_text="test {happy}happy{/happy} end",
            plain_text="test happy end",
            spans=[span],
        )
        assert parsed.get_emotion_at_word(1) == "happy"
        assert parsed.get_emotion_at_word(0) == "neutral"

    def test_get_emotion_timeline(self):
        """Test getting emotion timeline."""
        span = EmotionSpan(
            text="happy",
            emotion="happy",
            start_char=5,
            end_char=10,
        )
        parsed = ParsedEmotionText(
            original_text="test {happy}happy{/happy}",
            plain_text="test happy",
            spans=[span],
        )
        timeline = parsed.get_emotion_timeline()
        assert isinstance(timeline, list)
        assert len(timeline) >= 1

    def test_get_emotion_timeline_empty(self):
        """Test getting timeline from empty text."""
        parsed = ParsedEmotionText(
            original_text="",
            plain_text="",
            spans=[],
        )
        timeline = parsed.get_emotion_timeline()
        assert timeline == [(0.0, "neutral")]

    def test_has_emotion_tags(self):
        """Test checking for emotion tags."""
        parsed_with = ParsedEmotionText(
            original_text="{happy}text{/happy}",
            plain_text="text",
            spans=[EmotionSpan(text="text", emotion="happy")],
        )
        parsed_without = ParsedEmotionText(
            original_text="text",
            plain_text="text",
            spans=[],
        )
        assert parsed_with.has_emotion_tags() is True
        assert parsed_without.has_emotion_tags() is False

    def test_get_emotions_used(self):
        """Test getting list of used emotions."""
        spans = [
            EmotionSpan(text="a", emotion="happy"),
            EmotionSpan(text="b", emotion="sad"),
            EmotionSpan(text="c", emotion="happy"),  # Duplicate
        ]
        parsed = ParsedEmotionText(
            original_text="",
            plain_text="a b c",
            spans=spans,
        )
        emotions = parsed.get_emotions_used()
        assert set(emotions) == {"happy", "sad"}


class TestEmotionParser:
    """Tests for EmotionParser class."""

    def test_parser_creation(self):
        """Test creating parser."""
        parser = EmotionParser()
        assert parser.default_emotion == "neutral"
        assert parser.default_intensity == 1.0

    def test_parser_custom_defaults(self):
        """Test parser with custom defaults."""
        parser = EmotionParser(default_emotion="calm", default_intensity=0.5)
        assert parser.default_emotion == "calm"
        assert parser.default_intensity == 0.5

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        parser = EmotionParser()
        result = parser.parse("")
        assert result.plain_text == ""
        assert len(result.spans) == 0

    def test_parse_no_tags(self):
        """Test parsing text without tags."""
        parser = EmotionParser()
        result = parser.parse("Hello world")
        assert result.plain_text == "Hello world"
        assert len(result.spans) == 0

    def test_parse_single_tag(self):
        """Test parsing single emotion tag."""
        parser = EmotionParser()
        result = parser.parse("I'm {happy}so excited{/happy}!")
        assert result.plain_text == "I'm so excited!"
        assert len(result.spans) == 1
        assert result.spans[0].emotion == "happy"
        assert result.spans[0].text == "so excited"

    def test_parse_multiple_tags(self):
        """Test parsing multiple emotion tags."""
        parser = EmotionParser()
        result = parser.parse("{happy}Hello{/happy} and {sad}goodbye{/sad}")
        assert len(result.spans) == 2

    def test_parse_with_intensity(self):
        """Test parsing tag with intensity modifier."""
        parser = EmotionParser()
        result = parser.parse("{happy:0.7}somewhat happy{/happy}")
        assert len(result.spans) == 1
        assert result.spans[0].intensity == 0.7

    def test_parse_intensity_clamped(self):
        """Test intensity is clamped to valid range."""
        parser = EmotionParser()
        result = parser.parse("{happy:1.5}very happy{/happy}")
        assert result.spans[0].intensity == 1.0

    def test_parse_nested_tags(self):
        """Test parsing nested emotion tags."""
        parser = EmotionParser(allow_nesting=True)
        result = parser.parse("{happy}Great {excited}amazing{/excited} day!{/happy}")
        assert len(result.spans) >= 1
        # Should have both happy and excited spans

    def test_parse_unclosed_tag(self):
        """Test parsing unclosed tag includes remaining text."""
        parser = EmotionParser()
        result = parser.parse("{happy}Hello world")
        assert len(result.spans) == 1
        assert "Hello world" in result.spans[0].text

    def test_parse_unmatched_closing_tag(self):
        """Test unmatched closing tag is treated as literal."""
        parser = EmotionParser()
        result = parser.parse("Hello {/happy} world")
        assert "{/happy}" in result.plain_text

    def test_remove_tags(self):
        """Test removing all tags from text."""
        parser = EmotionParser()
        result = parser.remove_tags("{happy}Hello{/happy} {sad}world{/sad}")
        assert result == "Hello world"

    def test_has_tags_true(self):
        """Test detecting presence of tags."""
        parser = EmotionParser()
        assert parser.has_tags("{happy}test{/happy}") is True

    def test_has_tags_false(self):
        """Test detecting absence of tags."""
        parser = EmotionParser()
        assert parser.has_tags("plain text") is False

    def test_extract_emotions(self):
        """Test extracting list of emotions from text."""
        parser = EmotionParser()
        emotions = parser.extract_emotions("{happy}a{/happy} {sad}b{/sad} {happy}c{/happy}")
        assert set(emotions) == {"happy", "sad"}


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_parse_emotion_tags(self):
        """Test parse_emotion_tags convenience function."""
        result = parse_emotion_tags("{happy}Hello{/happy}")
        assert isinstance(result, ParsedEmotionText)
        assert len(result.spans) == 1

    def test_create_tagged_text_simple(self):
        """Test creating tagged text."""
        result = create_tagged_text("Hello", "happy")
        assert result == "{happy}Hello{/happy}"

    def test_create_tagged_text_with_intensity(self):
        """Test creating tagged text with intensity."""
        result = create_tagged_text("Hello", "happy", intensity=0.7)
        assert result == "{happy:0.7}Hello{/happy}"

    def test_create_tagged_text_full_intensity(self):
        """Test creating tagged text with full intensity omits modifier."""
        result = create_tagged_text("Hello", "happy", intensity=1.0)
        assert result == "{happy}Hello{/happy}"


class TestMergeAdjacentSpans:
    """Tests for merge_adjacent_spans function."""

    def test_merge_empty_list(self):
        """Test merging empty list."""
        result = merge_adjacent_spans([])
        assert result == []

    def test_merge_single_span(self):
        """Test merging single span."""
        spans = [EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5)]
        result = merge_adjacent_spans(spans)
        assert len(result) == 1

    def test_merge_adjacent_same_emotion(self):
        """Test merging adjacent spans with same emotion."""
        spans = [
            EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5, intensity=1.0),
            EmotionSpan(text=" world", emotion="happy", start_char=5, end_char=11, intensity=1.0),
        ]
        result = merge_adjacent_spans(spans)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_no_merge_different_emotions(self):
        """Test no merging for different emotions."""
        spans = [
            EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5),
            EmotionSpan(text=" world", emotion="sad", start_char=5, end_char=11),
        ]
        result = merge_adjacent_spans(spans)
        assert len(result) == 2

    def test_no_merge_non_adjacent(self):
        """Test no merging for non-adjacent spans."""
        spans = [
            EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5),
            EmotionSpan(text="world", emotion="happy", start_char=10, end_char=15),
        ]
        result = merge_adjacent_spans(spans)
        assert len(result) == 2

    def test_no_merge_different_intensity(self):
        """Test no merging when intensity differs."""
        spans = [
            EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5, intensity=0.5),
            EmotionSpan(text=" world", emotion="happy", start_char=5, end_char=11, intensity=1.0),
        ]
        result = merge_adjacent_spans(spans)
        assert len(result) == 2

    def test_merge_preserves_order(self):
        """Test merge sorts spans by position."""
        spans = [
            EmotionSpan(text=" world", emotion="happy", start_char=5, end_char=11, intensity=1.0),
            EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5, intensity=1.0),
        ]
        result = merge_adjacent_spans(spans)
        assert len(result) == 1
        assert result[0].start_char == 0
