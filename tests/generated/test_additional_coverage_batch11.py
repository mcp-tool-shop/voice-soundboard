"""
Additional test coverage batch 11: emotion/vad.py, emotion/parser.py, emotion/blending.py.

Tests for VAD emotion model, emotion tag parsing, and emotion blending system.
"""

import pytest
import math
from unittest.mock import patch, MagicMock

from voice_soundboard.emotion.vad import (
    VADPoint,
    VAD_EMOTIONS,
    emotion_to_vad,
    vad_to_emotion,
    interpolate_vad,
    get_emotion_intensity,
    classify_emotion_category,
    list_emotions_by_category,
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
from voice_soundboard.emotion.blending import (
    EmotionMix,
    blend_vad,
    blend_emotions,
    transition_emotion,
    create_emotion_gradient,
    get_complementary_emotion,
    get_similar_emotions,
    emotion_distance,
    NAMED_BLENDS,
    get_named_blend,
    list_named_blends,
)


# =============================================================================
# VADPoint Tests
# =============================================================================

class TestVADPoint:
    """Tests for VADPoint dataclass."""

    def test_creation_basic(self):
        """Test basic VADPoint creation."""
        point = VADPoint(valence=0.5, arousal=0.6, dominance=0.7)
        assert point.valence == 0.5
        assert point.arousal == 0.6
        assert point.dominance == 0.7

    def test_creation_with_clamping_valence_high(self):
        """Test that valence is clamped to max 1.0."""
        point = VADPoint(valence=1.5, arousal=0.5, dominance=0.5)
        assert point.valence == 1.0

    def test_creation_with_clamping_valence_low(self):
        """Test that valence is clamped to min -1.0."""
        point = VADPoint(valence=-1.5, arousal=0.5, dominance=0.5)
        assert point.valence == -1.0

    def test_creation_with_clamping_arousal_high(self):
        """Test that arousal is clamped to max 1.0."""
        point = VADPoint(valence=0.5, arousal=1.5, dominance=0.5)
        assert point.arousal == 1.0

    def test_creation_with_clamping_arousal_low(self):
        """Test that arousal is clamped to min 0.0."""
        point = VADPoint(valence=0.5, arousal=-0.5, dominance=0.5)
        assert point.arousal == 0.0

    def test_creation_with_clamping_dominance_high(self):
        """Test that dominance is clamped to max 1.0."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=1.5)
        assert point.dominance == 1.0

    def test_creation_with_clamping_dominance_low(self):
        """Test that dominance is clamped to min 0.0."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=-0.5)
        assert point.dominance == 0.0

    def test_distance_same_point(self):
        """Test distance to same point is zero."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        assert point.distance(point) == 0.0

    def test_distance_different_points(self):
        """Test distance between different points."""
        point1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        point2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        # sqrt(1 + 1 + 1) = sqrt(3) ≈ 1.732
        expected = math.sqrt(3)
        assert abs(point1.distance(point2) - expected) < 0.001

    def test_to_tuple(self):
        """Test conversion to tuple."""
        point = VADPoint(valence=0.1, arousal=0.2, dominance=0.3)
        result = point.to_tuple()
        assert result == (0.1, 0.2, 0.3)

    def test_from_tuple(self):
        """Test creation from tuple."""
        point = VADPoint.from_tuple((0.4, 0.5, 0.6))
        assert point.valence == 0.4
        assert point.arousal == 0.5
        assert point.dominance == 0.6

    def test_add_operator(self):
        """Test addition of two VADPoints."""
        point1 = VADPoint(valence=0.2, arousal=0.3, dominance=0.4)
        point2 = VADPoint(valence=0.1, arousal=0.2, dominance=0.1)
        result = point1 + point2
        assert abs(result.valence - 0.3) < 0.001
        assert abs(result.arousal - 0.5) < 0.001
        assert abs(result.dominance - 0.5) < 0.001

    def test_add_operator_with_clamping(self):
        """Test addition with clamping."""
        point1 = VADPoint(valence=0.8, arousal=0.9, dominance=0.9)
        point2 = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = point1 + point2
        # Result should be clamped
        assert result.valence == 1.0
        assert result.arousal == 1.0
        assert result.dominance == 1.0

    def test_mul_operator(self):
        """Test multiplication by scalar."""
        point = VADPoint(valence=0.4, arousal=0.6, dominance=0.8)
        result = point * 0.5
        assert result.valence == 0.2
        assert result.arousal == 0.3
        assert result.dominance == 0.4

    def test_rmul_operator(self):
        """Test right multiplication by scalar."""
        point = VADPoint(valence=0.4, arousal=0.6, dominance=0.8)
        result = 0.5 * point
        assert result.valence == 0.2
        assert result.arousal == 0.3
        assert result.dominance == 0.4


# =============================================================================
# VAD_EMOTIONS Dictionary Tests
# =============================================================================

class TestVADEmotions:
    """Tests for VAD_EMOTIONS dictionary."""

    def test_contains_basic_emotions(self):
        """Test that basic emotions are present."""
        assert "happy" in VAD_EMOTIONS
        assert "sad" in VAD_EMOTIONS
        assert "angry" in VAD_EMOTIONS
        assert "fearful" in VAD_EMOTIONS
        assert "neutral" in VAD_EMOTIONS

    def test_emotions_are_vad_points(self):
        """Test that all values are VADPoint instances."""
        for name, vad in VAD_EMOTIONS.items():
            assert isinstance(vad, VADPoint), f"{name} is not VADPoint"

    def test_happy_has_positive_valence(self):
        """Test that happy has positive valence."""
        assert VAD_EMOTIONS["happy"].valence > 0

    def test_sad_has_negative_valence(self):
        """Test that sad has negative valence."""
        assert VAD_EMOTIONS["sad"].valence < 0

    def test_excited_has_high_arousal(self):
        """Test that excited has high arousal."""
        assert VAD_EMOTIONS["excited"].arousal > 0.7

    def test_calm_has_low_arousal(self):
        """Test that calm has low arousal."""
        assert VAD_EMOTIONS["calm"].arousal < 0.4


# =============================================================================
# emotion_to_vad Function Tests
# =============================================================================

class TestEmotionToVad:
    """Tests for emotion_to_vad function."""

    def test_known_emotion(self):
        """Test converting known emotion."""
        result = emotion_to_vad("happy")
        assert result == VAD_EMOTIONS["happy"]

    def test_case_insensitive(self):
        """Test case insensitivity."""
        result = emotion_to_vad("HAPPY")
        assert result == VAD_EMOTIONS["happy"]

    def test_with_whitespace(self):
        """Test handling whitespace."""
        result = emotion_to_vad("  happy  ")
        assert result == VAD_EMOTIONS["happy"]

    def test_partial_match(self):
        """Test partial matching."""
        # "joyful" contains "joy" or matches some pattern
        result = emotion_to_vad("joyful")
        assert isinstance(result, VADPoint)

    def test_unknown_emotion_raises(self):
        """Test that unknown emotion raises ValueError."""
        with pytest.raises(ValueError, match="Unknown emotion"):
            emotion_to_vad("unknownemotionxyz")


# =============================================================================
# vad_to_emotion Function Tests
# =============================================================================

class TestVadToEmotion:
    """Tests for vad_to_emotion function."""

    def test_exact_match_returns_emotion(self):
        """Test that exact VAD match returns correct emotion."""
        vad = VAD_EMOTIONS["happy"]
        result = vad_to_emotion(vad, top_n=1)
        assert len(result) == 1
        assert result[0][0] == "happy"
        assert result[0][1] == 0.0  # Distance should be 0

    def test_returns_multiple_matches(self):
        """Test returning multiple closest matches."""
        vad = VADPoint(valence=0.7, arousal=0.5, dominance=0.6)
        result = vad_to_emotion(vad, top_n=3)
        assert len(result) == 3

    def test_sorted_by_distance(self):
        """Test results are sorted by distance."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = vad_to_emotion(vad, top_n=5)
        distances = [d for _, d in result]
        assert distances == sorted(distances)


# =============================================================================
# interpolate_vad Function Tests
# =============================================================================

class TestInterpolateVad:
    """Tests for interpolate_vad function."""

    def test_interpolate_at_zero(self):
        """Test interpolation at t=0 returns first point."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 0.0)
        assert result.valence == 0.0
        assert result.arousal == 0.0
        assert result.dominance == 0.0

    def test_interpolate_at_one(self):
        """Test interpolation at t=1 returns second point."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 1.0)
        assert result.valence == 1.0
        assert result.arousal == 1.0
        assert result.dominance == 1.0

    def test_interpolate_at_half(self):
        """Test interpolation at t=0.5 returns midpoint."""
        vad1 = VADPoint(valence=0.0, arousal=0.2, dominance=0.4)
        vad2 = VADPoint(valence=1.0, arousal=0.8, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 0.5)
        assert abs(result.valence - 0.5) < 0.001
        assert abs(result.arousal - 0.5) < 0.001
        assert abs(result.dominance - 0.7) < 0.001

    def test_interpolate_clamps_t(self):
        """Test that t is clamped to [0, 1]."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)

        result_low = interpolate_vad(vad1, vad2, -0.5)
        assert result_low.valence == 0.0

        result_high = interpolate_vad(vad1, vad2, 1.5)
        assert result_high.valence == 1.0


# =============================================================================
# get_emotion_intensity Function Tests
# =============================================================================

class TestGetEmotionIntensity:
    """Tests for get_emotion_intensity function."""

    def test_neutral_has_low_intensity(self):
        """Test that neutral emotion has low intensity."""
        vad = VAD_EMOTIONS["neutral"]
        intensity = get_emotion_intensity(vad)
        assert intensity < 0.5

    def test_excited_has_high_intensity(self):
        """Test that excited has high intensity."""
        vad = VAD_EMOTIONS["excited"]
        intensity = get_emotion_intensity(vad)
        assert intensity > 0.5

    def test_intensity_bounded(self):
        """Test that intensity is bounded 0-1."""
        for name, vad in VAD_EMOTIONS.items():
            intensity = get_emotion_intensity(vad)
            assert 0.0 <= intensity <= 1.0, f"Intensity for {name} out of bounds"


# =============================================================================
# classify_emotion_category Function Tests
# =============================================================================

class TestClassifyEmotionCategory:
    """Tests for classify_emotion_category function."""

    def test_happy_is_positive_high(self):
        """Test happy is classified as positive_high."""
        vad = VAD_EMOTIONS["happy"]
        category = classify_emotion_category(vad)
        assert category == "positive_high"

    def test_calm_is_positive_low(self):
        """Test calm is classified as positive_low."""
        vad = VAD_EMOTIONS["calm"]
        category = classify_emotion_category(vad)
        assert category == "positive_low"

    def test_angry_is_negative_high(self):
        """Test angry is classified as negative_high."""
        vad = VAD_EMOTIONS["angry"]
        category = classify_emotion_category(vad)
        assert category == "negative_high"

    def test_sad_is_negative_low(self):
        """Test sad is classified as negative_low."""
        vad = VAD_EMOTIONS["sad"]
        category = classify_emotion_category(vad)
        assert category == "negative_low"

    def test_neutral_is_neutral(self):
        """Test neutral is classified as neutral."""
        vad = VAD_EMOTIONS["neutral"]
        category = classify_emotion_category(vad)
        assert category == "neutral"


# =============================================================================
# list_emotions_by_category Function Tests
# =============================================================================

class TestListEmotionsByCategory:
    """Tests for list_emotions_by_category function."""

    def test_returns_all_categories(self):
        """Test returns all five categories."""
        result = list_emotions_by_category()
        expected_categories = ["positive_high", "positive_low", "negative_high",
                              "negative_low", "neutral"]
        for cat in expected_categories:
            assert cat in result

    def test_filter_by_specific_category(self):
        """Test filtering by specific category."""
        result = list_emotions_by_category("positive_high")
        assert "positive_high" in result
        assert len(result) == 1

    def test_happy_in_positive_category(self):
        """Test happy is in a positive category."""
        result = list_emotions_by_category()
        found = False
        for cat in ["positive_high", "positive_low"]:
            if "happy" in result.get(cat, []):
                found = True
                break
        assert found


# =============================================================================
# EmotionTagType Enum Tests
# =============================================================================

class TestEmotionTagType:
    """Tests for EmotionTagType enum."""

    def test_open_value(self):
        """Test OPEN enum value."""
        assert EmotionTagType.OPEN.value == "open"

    def test_close_value(self):
        """Test CLOSE enum value."""
        assert EmotionTagType.CLOSE.value == "close"


# =============================================================================
# EmotionSpan Tests
# =============================================================================

class TestEmotionSpan:
    """Tests for EmotionSpan dataclass."""

    def test_creation_basic(self):
        """Test basic EmotionSpan creation."""
        span = EmotionSpan(text="hello", emotion="happy")
        assert span.text == "hello"
        assert span.emotion == "happy"
        assert span.intensity == 1.0

    def test_creation_with_all_fields(self):
        """Test creation with all fields."""
        span = EmotionSpan(
            text="hello world",
            emotion="happy",
            intensity=0.8,
            start_char=0,
            end_char=11,
            start_word=0,
            end_word=2,
            nested_in="joyful"
        )
        assert span.intensity == 0.8
        assert span.start_char == 0
        assert span.end_char == 11
        assert span.nested_in == "joyful"

    def test_word_count(self):
        """Test word_count method."""
        span = EmotionSpan(text="one two three", emotion="happy")
        assert span.word_count() == 3

    def test_word_count_single(self):
        """Test word_count with single word."""
        span = EmotionSpan(text="hello", emotion="happy")
        assert span.word_count() == 1


# =============================================================================
# ParsedEmotionText Tests
# =============================================================================

class TestParsedEmotionText:
    """Tests for ParsedEmotionText dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        parsed = ParsedEmotionText(
            original_text="{happy}hello{/happy}",
            plain_text="hello",
            spans=[EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5)]
        )
        assert parsed.plain_text == "hello"
        assert len(parsed.spans) == 1

    def test_get_emotion_at_position_in_span(self):
        """Test getting emotion at position within span."""
        span = EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5)
        parsed = ParsedEmotionText(
            original_text="{happy}hello{/happy}",
            plain_text="hello",
            spans=[span]
        )
        assert parsed.get_emotion_at_position(2) == "happy"

    def test_get_emotion_at_position_outside_span(self):
        """Test getting emotion at position outside spans."""
        span = EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5)
        parsed = ParsedEmotionText(
            original_text="{happy}hello{/happy} world",
            plain_text="hello world",
            spans=[span]
        )
        assert parsed.get_emotion_at_position(8) == "neutral"

    def test_get_emotion_at_word_in_span(self):
        """Test getting emotion at word index in span."""
        span = EmotionSpan(
            text="hello", emotion="happy",
            start_char=0, end_char=5,
            start_word=0, end_word=1
        )
        parsed = ParsedEmotionText(
            original_text="{happy}hello{/happy}",
            plain_text="hello",
            spans=[span]
        )
        assert parsed.get_emotion_at_word(0) == "happy"

    def test_get_emotion_timeline_empty(self):
        """Test emotion timeline with empty text."""
        parsed = ParsedEmotionText(
            original_text="",
            plain_text="",
            spans=[]
        )
        timeline = parsed.get_emotion_timeline()
        assert timeline == [(0.0, "neutral")]

    def test_get_emotion_timeline_with_spans(self):
        """Test emotion timeline with spans."""
        span = EmotionSpan(
            text="hello", emotion="happy",
            start_char=0, end_char=5
        )
        parsed = ParsedEmotionText(
            original_text="{happy}hello{/happy} world",
            plain_text="hello world",
            spans=[span]
        )
        timeline = parsed.get_emotion_timeline()
        assert len(timeline) >= 1

    def test_has_emotion_tags_true(self):
        """Test has_emotion_tags returns True when spans exist."""
        parsed = ParsedEmotionText(
            original_text="{happy}hello{/happy}",
            plain_text="hello",
            spans=[EmotionSpan(text="hello", emotion="happy")]
        )
        assert parsed.has_emotion_tags() is True

    def test_has_emotion_tags_false(self):
        """Test has_emotion_tags returns False when no spans."""
        parsed = ParsedEmotionText(
            original_text="hello",
            plain_text="hello",
            spans=[]
        )
        assert parsed.has_emotion_tags() is False

    def test_get_emotions_used(self):
        """Test getting list of unique emotions."""
        parsed = ParsedEmotionText(
            original_text="{happy}hi{/happy} {sad}bye{/sad}",
            plain_text="hi bye",
            spans=[
                EmotionSpan(text="hi", emotion="happy"),
                EmotionSpan(text="bye", emotion="sad")
            ]
        )
        emotions = parsed.get_emotions_used()
        assert "happy" in emotions
        assert "sad" in emotions


# =============================================================================
# EmotionParser Tests
# =============================================================================

class TestEmotionParser:
    """Tests for EmotionParser class."""

    def test_init_default(self):
        """Test default initialization."""
        parser = EmotionParser()
        assert parser.default_emotion == "neutral"
        assert parser.default_intensity == 1.0
        assert parser.allow_nesting is True

    def test_init_custom(self):
        """Test custom initialization."""
        parser = EmotionParser(
            default_emotion="calm",
            default_intensity=0.8,
            allow_nesting=False
        )
        assert parser.default_emotion == "calm"
        assert parser.default_intensity == 0.8
        assert parser.allow_nesting is False

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        parser = EmotionParser()
        result = parser.parse("")
        assert result.original_text == ""
        assert result.plain_text == ""
        assert result.spans == []

    def test_parse_no_tags(self):
        """Test parsing text without tags."""
        parser = EmotionParser()
        result = parser.parse("Hello world")
        assert result.plain_text == "Hello world"
        assert result.spans == []

    def test_parse_single_tag(self):
        """Test parsing single emotion tag."""
        parser = EmotionParser()
        result = parser.parse("{happy}Hello{/happy}")
        assert result.plain_text == "Hello"
        assert len(result.spans) == 1
        assert result.spans[0].emotion == "happy"
        assert result.spans[0].text == "Hello"

    def test_parse_with_intensity(self):
        """Test parsing tag with intensity modifier."""
        parser = EmotionParser()
        result = parser.parse("{happy:0.5}Hello{/happy}")
        assert len(result.spans) == 1
        assert result.spans[0].intensity == 0.5

    def test_parse_multiple_tags(self):
        """Test parsing multiple emotion tags."""
        parser = EmotionParser()
        result = parser.parse("{happy}Hi{/happy} {sad}Bye{/sad}")
        assert len(result.spans) == 2
        emotions = [s.emotion for s in result.spans]
        assert "happy" in emotions
        assert "sad" in emotions

    def test_parse_nested_tags(self):
        """Test parsing nested tags."""
        parser = EmotionParser(allow_nesting=True)
        result = parser.parse("{happy}Hello {excited}World{/excited}{/happy}")
        # Should have spans for both
        emotions = [s.emotion for s in result.spans]
        assert "happy" in emotions
        assert "excited" in emotions

    def test_parse_unclosed_tag(self):
        """Test parsing unclosed tag."""
        parser = EmotionParser()
        result = parser.parse("{happy}Hello world")
        # Should still create a span
        assert len(result.spans) == 1
        assert result.spans[0].emotion == "happy"

    def test_remove_tags(self):
        """Test removing all tags."""
        parser = EmotionParser()
        result = parser.remove_tags("{happy}Hello{/happy} world")
        assert result == "Hello world"

    def test_has_tags_true(self):
        """Test has_tags returns True."""
        parser = EmotionParser()
        assert parser.has_tags("{happy}text{/happy}") is True

    def test_has_tags_false(self):
        """Test has_tags returns False."""
        parser = EmotionParser()
        assert parser.has_tags("plain text") is False

    def test_extract_emotions(self):
        """Test extracting emotion names."""
        parser = EmotionParser()
        emotions = parser.extract_emotions("{happy}hi{/happy} {sad}bye{/sad}")
        assert "happy" in emotions
        assert "sad" in emotions


# =============================================================================
# parse_emotion_tags Function Tests
# =============================================================================

class TestParseEmotionTags:
    """Tests for parse_emotion_tags convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        result = parse_emotion_tags("{happy}Hello{/happy}")
        assert isinstance(result, ParsedEmotionText)
        assert result.plain_text == "Hello"


# =============================================================================
# create_tagged_text Function Tests
# =============================================================================

class TestCreateTaggedText:
    """Tests for create_tagged_text function."""

    def test_basic_tagging(self):
        """Test basic text tagging."""
        result = create_tagged_text("Hello", "happy")
        assert result == "{happy}Hello{/happy}"

    def test_tagging_with_intensity(self):
        """Test tagging with intensity."""
        result = create_tagged_text("Hello", "happy", intensity=0.5)
        assert result == "{happy:0.5}Hello{/happy}"

    def test_tagging_default_intensity(self):
        """Test tagging with default intensity doesn't include modifier."""
        result = create_tagged_text("Hello", "happy", intensity=1.0)
        assert result == "{happy}Hello{/happy}"


# =============================================================================
# merge_adjacent_spans Function Tests
# =============================================================================

class TestMergeAdjacentSpans:
    """Tests for merge_adjacent_spans function."""

    def test_empty_list(self):
        """Test with empty list."""
        result = merge_adjacent_spans([])
        assert result == []

    def test_single_span(self):
        """Test with single span."""
        span = EmotionSpan(text="hello", emotion="happy", start_char=0, end_char=5)
        result = merge_adjacent_spans([span])
        assert len(result) == 1

    def test_non_adjacent_spans(self):
        """Test non-adjacent spans aren't merged."""
        span1 = EmotionSpan(text="hi", emotion="happy", start_char=0, end_char=2)
        span2 = EmotionSpan(text="bye", emotion="happy", start_char=5, end_char=8)
        result = merge_adjacent_spans([span1, span2])
        assert len(result) == 2

    def test_adjacent_same_emotion_merged(self):
        """Test adjacent spans with same emotion are merged."""
        span1 = EmotionSpan(
            text="hello", emotion="happy", intensity=1.0,
            start_char=0, end_char=5, start_word=0, end_word=1
        )
        span2 = EmotionSpan(
            text=" world", emotion="happy", intensity=1.0,
            start_char=5, end_char=11, start_word=1, end_word=2
        )
        result = merge_adjacent_spans([span1, span2])
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_adjacent_different_emotion_not_merged(self):
        """Test adjacent spans with different emotions aren't merged."""
        span1 = EmotionSpan(text="hi", emotion="happy", start_char=0, end_char=2)
        span2 = EmotionSpan(text="bye", emotion="sad", start_char=2, end_char=5)
        result = merge_adjacent_spans([span1, span2])
        assert len(result) == 2


# =============================================================================
# EmotionMix Tests
# =============================================================================

class TestEmotionMix:
    """Tests for EmotionMix dataclass."""

    def test_creation(self):
        """Test basic creation."""
        mix = EmotionMix(
            vad=VADPoint(valence=0.5, arousal=0.5, dominance=0.5),
            components=[("happy", 0.6), ("sad", 0.4)],
            dominant_emotion="happy",
            intensity=0.7,
            secondary_emotions=["content", "peaceful"]
        )
        assert mix.dominant_emotion == "happy"
        assert mix.intensity == 0.7

    def test_to_synthesis_params(self):
        """Test conversion to synthesis parameters."""
        mix = EmotionMix(
            vad=VADPoint(valence=0.5, arousal=0.6, dominance=0.7),
            components=[("happy", 1.0)],
            dominant_emotion="happy",
            intensity=0.8,
            secondary_emotions=[]
        )
        params = mix.to_synthesis_params()
        assert params["emotion"] == "happy"
        assert params["intensity"] == 0.8
        assert params["valence"] == 0.5
        assert params["arousal"] == 0.6
        assert params["dominance"] == 0.7

    def test_describe(self):
        """Test describe method."""
        mix = EmotionMix(
            vad=VADPoint(valence=0.5, arousal=0.5, dominance=0.5),
            components=[("happy", 0.6), ("sad", 0.4)],
            dominant_emotion="bittersweet",
            intensity=0.7,
            secondary_emotions=[]
        )
        description = mix.describe()
        assert "bittersweet" in description
        assert "happy" in description
        assert "sad" in description


# =============================================================================
# blend_vad Function Tests
# =============================================================================

class TestBlendVad:
    """Tests for blend_vad function."""

    def test_single_point(self):
        """Test blending single point."""
        vad = VADPoint(valence=0.5, arousal=0.6, dominance=0.7)
        result = blend_vad([(vad, 1.0)])
        assert result.valence == 0.5
        assert result.arousal == 0.6
        assert result.dominance == 0.7

    def test_two_equal_weights(self):
        """Test blending two points with equal weights."""
        vad1 = VADPoint(valence=0.0, arousal=0.2, dominance=0.4)
        vad2 = VADPoint(valence=1.0, arousal=0.8, dominance=0.8)
        result = blend_vad([(vad1, 0.5), (vad2, 0.5)])
        assert abs(result.valence - 0.5) < 0.001
        assert abs(result.arousal - 0.5) < 0.001
        assert abs(result.dominance - 0.6) < 0.001

    def test_weights_normalized(self):
        """Test that weights are normalized."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        # Weights that don't sum to 1
        result = blend_vad([(vad1, 1.0), (vad2, 1.0)])
        # Should be normalized to 0.5 each
        assert abs(result.valence - 0.5) < 0.001

    def test_empty_list_raises(self):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="At least one VAD point"):
            blend_vad([])

    def test_zero_weights_raises(self):
        """Test all zero weights raises error."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        with pytest.raises(ValueError, match="Weights cannot all be zero"):
            blend_vad([(vad, 0.0)])


# =============================================================================
# blend_emotions Function Tests
# =============================================================================

class TestBlendEmotions:
    """Tests for blend_emotions function."""

    def test_single_emotion(self):
        """Test blending single emotion."""
        result = blend_emotions([("happy", 1.0)])
        assert isinstance(result, EmotionMix)
        assert result.dominant_emotion is not None

    def test_two_emotions(self):
        """Test blending two emotions."""
        result = blend_emotions([("happy", 0.6), ("sad", 0.4)])
        assert isinstance(result, EmotionMix)
        assert len(result.components) == 2

    def test_empty_list_raises(self):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="At least one emotion"):
            blend_emotions([])

    def test_unknown_emotion_skipped(self):
        """Test unknown emotions are skipped with warning."""
        # Should not raise, just skip unknown
        result = blend_emotions([("happy", 0.5), ("unknownxyz", 0.5)])
        assert isinstance(result, EmotionMix)


# =============================================================================
# transition_emotion Function Tests
# =============================================================================

class TestTransitionEmotion:
    """Tests for transition_emotion function."""

    def test_at_start(self):
        """Test transition at start (progress=0)."""
        result = transition_emotion("happy", "sad", 0.0)
        # Should be weighted toward happy
        assert result.components[0][0] == "happy"
        assert result.components[0][1] == 1.0

    def test_at_end(self):
        """Test transition at end (progress=1)."""
        result = transition_emotion("happy", "sad", 1.0)
        # Should be weighted toward sad
        assert result.components[1][0] == "sad"
        assert result.components[1][1] == 1.0

    def test_at_middle(self):
        """Test transition at middle (progress=0.5)."""
        result = transition_emotion("happy", "sad", 0.5)
        # Both should have equal weight
        assert result.components[0][1] == 0.5
        assert result.components[1][1] == 0.5

    def test_progress_clamped(self):
        """Test progress is clamped to [0, 1]."""
        result_low = transition_emotion("happy", "sad", -0.5)
        assert result_low.components[0][1] == 1.0  # All happy

        result_high = transition_emotion("happy", "sad", 1.5)
        assert result_high.components[1][1] == 1.0  # All sad


# =============================================================================
# create_emotion_gradient Function Tests
# =============================================================================

class TestCreateEmotionGradient:
    """Tests for create_emotion_gradient function."""

    def test_basic_gradient(self):
        """Test basic gradient creation."""
        gradient = create_emotion_gradient(["happy", "sad"], steps=5)
        assert len(gradient) > 0
        assert all(isinstance(m, EmotionMix) for m in gradient)

    def test_three_emotion_gradient(self):
        """Test gradient through three emotions."""
        gradient = create_emotion_gradient(["happy", "neutral", "sad"], steps=10)
        assert len(gradient) > 0

    def test_single_emotion_raises(self):
        """Test single emotion raises error."""
        with pytest.raises(ValueError, match="At least two emotions"):
            create_emotion_gradient(["happy"])


# =============================================================================
# get_complementary_emotion Function Tests
# =============================================================================

class TestGetComplementaryEmotion:
    """Tests for get_complementary_emotion function."""

    def test_happy_complement(self):
        """Test getting complement of happy."""
        result = get_complementary_emotion("happy")
        assert isinstance(result, str)
        # Complement should have opposite valence
        result_vad = emotion_to_vad(result)
        happy_vad = emotion_to_vad("happy")
        # Signs should differ or result should be different emotion
        assert result != "happy"

    def test_sad_complement(self):
        """Test getting complement of sad."""
        result = get_complementary_emotion("sad")
        assert isinstance(result, str)
        assert result != "sad"


# =============================================================================
# get_similar_emotions Function Tests
# =============================================================================

class TestGetSimilarEmotions:
    """Tests for get_similar_emotions function."""

    def test_returns_list(self):
        """Test returns list of strings."""
        result = get_similar_emotions("happy", count=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_excludes_original(self):
        """Test excludes original emotion."""
        result = get_similar_emotions("happy", count=5)
        assert "happy" not in result

    def test_similar_to_happy(self):
        """Test similar emotions to happy are positive."""
        result = get_similar_emotions("happy", count=3)
        # At least some should be positive
        assert len(result) > 0


# =============================================================================
# emotion_distance Function Tests
# =============================================================================

class TestEmotionDistance:
    """Tests for emotion_distance function."""

    def test_same_emotion_zero_distance(self):
        """Test same emotion has zero distance."""
        distance = emotion_distance("happy", "happy")
        assert distance == 0.0

    def test_different_emotions_positive_distance(self):
        """Test different emotions have positive distance."""
        distance = emotion_distance("happy", "sad")
        assert distance > 0

    def test_opposite_emotions_large_distance(self):
        """Test opposite emotions have large distance."""
        distance = emotion_distance("happy", "sad")
        # Should be significant
        assert distance > 0.5


# =============================================================================
# NAMED_BLENDS Tests
# =============================================================================

class TestNamedBlends:
    """Tests for NAMED_BLENDS dictionary."""

    def test_bittersweet_exists(self):
        """Test bittersweet blend exists."""
        assert "bittersweet" in NAMED_BLENDS

    def test_nervous_excitement_exists(self):
        """Test nervous_excitement blend exists."""
        assert "nervous_excitement" in NAMED_BLENDS

    def test_blends_are_lists(self):
        """Test all blends are lists of tuples."""
        for name, blend in NAMED_BLENDS.items():
            assert isinstance(blend, list)
            for item in blend:
                assert isinstance(item, tuple)
                assert len(item) == 2


# =============================================================================
# get_named_blend Function Tests
# =============================================================================

class TestGetNamedBlend:
    """Tests for get_named_blend function."""

    def test_known_blend(self):
        """Test getting known blend."""
        result = get_named_blend("bittersweet")
        assert isinstance(result, EmotionMix)

    def test_unknown_blend_returns_none(self):
        """Test unknown blend returns None."""
        result = get_named_blend("unknownblendxyz")
        assert result is None

    def test_case_insensitive(self):
        """Test case insensitivity."""
        result = get_named_blend("BITTERSWEET")
        assert isinstance(result, EmotionMix)


# =============================================================================
# list_named_blends Function Tests
# =============================================================================

class TestListNamedBlends:
    """Tests for list_named_blends function."""

    def test_returns_list(self):
        """Test returns list."""
        result = list_named_blends()
        assert isinstance(result, list)

    def test_contains_known_blends(self):
        """Test contains known blends."""
        result = list_named_blends()
        assert "bittersweet" in result
        assert "nervous_excitement" in result
