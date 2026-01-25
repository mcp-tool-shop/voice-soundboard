"""
Additional test coverage batch 23: emotion/vad.py and emotion/blending.py.

Tests for VAD emotion model and emotion blending system.
"""

import pytest
import math
from unittest.mock import patch

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


# ============================================================================
# VADPoint Tests
# ============================================================================

class TestVADPoint:
    """Tests for VADPoint dataclass."""

    def test_vadpoint_creation_basic(self):
        """Test basic VADPoint creation."""
        point = VADPoint(valence=0.5, arousal=0.6, dominance=0.7)
        assert point.valence == 0.5
        assert point.arousal == 0.6
        assert point.dominance == 0.7

    def test_vadpoint_clamping_valence_high(self):
        """Test valence is clamped to max 1.0."""
        point = VADPoint(valence=1.5, arousal=0.5, dominance=0.5)
        assert point.valence == 1.0

    def test_vadpoint_clamping_valence_low(self):
        """Test valence is clamped to min -1.0."""
        point = VADPoint(valence=-1.5, arousal=0.5, dominance=0.5)
        assert point.valence == -1.0

    def test_vadpoint_clamping_arousal_high(self):
        """Test arousal is clamped to max 1.0."""
        point = VADPoint(valence=0.5, arousal=1.5, dominance=0.5)
        assert point.arousal == 1.0

    def test_vadpoint_clamping_arousal_low(self):
        """Test arousal is clamped to min 0.0."""
        point = VADPoint(valence=0.5, arousal=-0.5, dominance=0.5)
        assert point.arousal == 0.0

    def test_vadpoint_clamping_dominance_high(self):
        """Test dominance is clamped to max 1.0."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=1.5)
        assert point.dominance == 1.0

    def test_vadpoint_clamping_dominance_low(self):
        """Test dominance is clamped to min 0.0."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=-0.5)
        assert point.dominance == 0.0

    def test_vadpoint_distance_same_point(self):
        """Test distance to same point is zero."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        assert point.distance(point) == 0.0

    def test_vadpoint_distance_different_points(self):
        """Test distance between different points."""
        point1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        point2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        expected = math.sqrt(1**2 + 1**2 + 1**2)
        assert point1.distance(point2) == pytest.approx(expected)

    def test_vadpoint_to_tuple(self):
        """Test conversion to tuple."""
        point = VADPoint(valence=0.3, arousal=0.5, dominance=0.7)
        assert point.to_tuple() == (0.3, 0.5, 0.7)

    def test_vadpoint_from_tuple(self):
        """Test creation from tuple."""
        point = VADPoint.from_tuple((0.3, 0.5, 0.7))
        assert point.valence == 0.3
        assert point.arousal == 0.5
        assert point.dominance == 0.7

    def test_vadpoint_add(self):
        """Test adding two VADPoints."""
        p1 = VADPoint(valence=0.2, arousal=0.3, dominance=0.4)
        p2 = VADPoint(valence=0.3, arousal=0.2, dominance=0.1)
        result = p1 + p2
        assert result.valence == pytest.approx(0.5)
        assert result.arousal == pytest.approx(0.5)
        assert result.dominance == pytest.approx(0.5)

    def test_vadpoint_add_with_clamping(self):
        """Test adding VADPoints with result clamping."""
        p1 = VADPoint(valence=0.8, arousal=0.8, dominance=0.8)
        p2 = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = p1 + p2
        assert result.valence == 1.0  # Clamped
        assert result.arousal == 1.0  # Clamped
        assert result.dominance == 1.0  # Clamped

    def test_vadpoint_multiply_scalar(self):
        """Test multiplying VADPoint by scalar."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = point * 0.5
        assert result.valence == pytest.approx(0.25)
        assert result.arousal == pytest.approx(0.25)
        assert result.dominance == pytest.approx(0.25)

    def test_vadpoint_rmul_scalar(self):
        """Test right multiply by scalar."""
        point = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = 0.5 * point
        assert result.valence == pytest.approx(0.25)
        assert result.arousal == pytest.approx(0.25)
        assert result.dominance == pytest.approx(0.25)


class TestVADEmotions:
    """Tests for VAD_EMOTIONS dictionary."""

    def test_vad_emotions_contains_basic_emotions(self):
        """Test VAD_EMOTIONS contains basic emotions."""
        basic_emotions = ["happy", "sad", "angry", "fearful", "surprised", "neutral"]
        for emotion in basic_emotions:
            assert emotion in VAD_EMOTIONS

    def test_vad_emotions_all_valid_vadpoints(self):
        """Test all emotions have valid VADPoint values."""
        for name, vad in VAD_EMOTIONS.items():
            assert isinstance(vad, VADPoint)
            assert -1.0 <= vad.valence <= 1.0
            assert 0.0 <= vad.arousal <= 1.0
            assert 0.0 <= vad.dominance <= 1.0

    def test_happy_emotion_is_positive(self):
        """Test happy has positive valence."""
        assert VAD_EMOTIONS["happy"].valence > 0

    def test_sad_emotion_is_negative(self):
        """Test sad has negative valence."""
        assert VAD_EMOTIONS["sad"].valence < 0

    def test_excited_has_high_arousal(self):
        """Test excited has high arousal."""
        assert VAD_EMOTIONS["excited"].arousal > 0.7

    def test_calm_has_low_arousal(self):
        """Test calm has low arousal."""
        assert VAD_EMOTIONS["calm"].arousal < 0.3


class TestEmotionToVAD:
    """Tests for emotion_to_vad function."""

    def test_emotion_to_vad_known_emotion(self):
        """Test converting known emotion."""
        vad = emotion_to_vad("happy")
        assert vad == VAD_EMOTIONS["happy"]

    def test_emotion_to_vad_case_insensitive(self):
        """Test case insensitivity."""
        vad = emotion_to_vad("HAPPY")
        assert vad == VAD_EMOTIONS["happy"]

    def test_emotion_to_vad_with_whitespace(self):
        """Test handling whitespace."""
        vad = emotion_to_vad("  happy  ")
        assert vad == VAD_EMOTIONS["happy"]

    def test_emotion_to_vad_partial_match(self):
        """Test partial match finding."""
        # "joyful" should be found if we search for something containing it
        vad = emotion_to_vad("joyful")
        assert vad == VAD_EMOTIONS["joyful"]

    def test_emotion_to_vad_unknown_raises(self):
        """Test unknown emotion raises ValueError."""
        with pytest.raises(ValueError, match="Unknown emotion"):
            emotion_to_vad("nonexistent_emotion_xyz")


class TestVADToEmotion:
    """Tests for vad_to_emotion function."""

    def test_vad_to_emotion_exact_match(self):
        """Test finding exact emotion match."""
        happy_vad = VAD_EMOTIONS["happy"]
        result = vad_to_emotion(happy_vad, top_n=1)
        assert len(result) == 1
        assert result[0][0] == "happy"
        assert result[0][1] == 0.0  # Zero distance for exact match

    def test_vad_to_emotion_top_n(self):
        """Test returning top N matches."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = vad_to_emotion(vad, top_n=3)
        assert len(result) == 3

    def test_vad_to_emotion_sorted_by_distance(self):
        """Test results are sorted by distance."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = vad_to_emotion(vad, top_n=5)
        distances = [d for _, d in result]
        assert distances == sorted(distances)


class TestInterpolateVAD:
    """Tests for interpolate_vad function."""

    def test_interpolate_at_zero(self):
        """Test interpolation at t=0 returns start."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 0.0)
        assert result.valence == pytest.approx(0.0)
        assert result.arousal == pytest.approx(0.0)
        assert result.dominance == pytest.approx(0.0)

    def test_interpolate_at_one(self):
        """Test interpolation at t=1 returns end."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 1.0)
        assert result.valence == pytest.approx(1.0)
        assert result.arousal == pytest.approx(1.0)
        assert result.dominance == pytest.approx(1.0)

    def test_interpolate_at_midpoint(self):
        """Test interpolation at midpoint."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 0.5)
        assert result.valence == pytest.approx(0.5)
        assert result.arousal == pytest.approx(0.5)
        assert result.dominance == pytest.approx(0.5)

    def test_interpolate_clamps_t_low(self):
        """Test t is clamped to 0.0 minimum."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, -0.5)
        assert result.valence == pytest.approx(0.0)

    def test_interpolate_clamps_t_high(self):
        """Test t is clamped to 1.0 maximum."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = interpolate_vad(vad1, vad2, 1.5)
        assert result.valence == pytest.approx(1.0)


class TestGetEmotionIntensity:
    """Tests for get_emotion_intensity function."""

    def test_neutral_has_low_intensity(self):
        """Test neutral emotion has low intensity."""
        neutral_vad = VAD_EMOTIONS["neutral"]
        intensity = get_emotion_intensity(neutral_vad)
        assert intensity < 0.3

    def test_excited_has_high_intensity(self):
        """Test excited emotion has high intensity."""
        excited_vad = VAD_EMOTIONS["excited"]
        intensity = get_emotion_intensity(excited_vad)
        assert intensity > 0.5

    def test_intensity_in_valid_range(self):
        """Test intensity is always 0-1."""
        for vad in VAD_EMOTIONS.values():
            intensity = get_emotion_intensity(vad)
            assert 0.0 <= intensity <= 1.0


class TestClassifyEmotionCategory:
    """Tests for classify_emotion_category function."""

    def test_classify_positive_high(self):
        """Test classifying positive high arousal emotion."""
        vad = VADPoint(valence=0.7, arousal=0.8, dominance=0.5)
        assert classify_emotion_category(vad) == "positive_high"

    def test_classify_positive_low(self):
        """Test classifying positive low arousal emotion."""
        vad = VADPoint(valence=0.7, arousal=0.3, dominance=0.5)
        assert classify_emotion_category(vad) == "positive_low"

    def test_classify_negative_high(self):
        """Test classifying negative high arousal emotion."""
        vad = VADPoint(valence=-0.7, arousal=0.8, dominance=0.5)
        assert classify_emotion_category(vad) == "negative_high"

    def test_classify_negative_low(self):
        """Test classifying negative low arousal emotion."""
        vad = VADPoint(valence=-0.7, arousal=0.3, dominance=0.5)
        assert classify_emotion_category(vad) == "negative_low"

    def test_classify_neutral(self):
        """Test classifying neutral emotion."""
        vad = VADPoint(valence=0.0, arousal=0.3, dominance=0.5)
        assert classify_emotion_category(vad) == "neutral"


class TestListEmotionsByCategory:
    """Tests for list_emotions_by_category function."""

    def test_list_all_categories(self):
        """Test listing all categories."""
        result = list_emotions_by_category()
        assert "positive_high" in result
        assert "positive_low" in result
        assert "negative_high" in result
        assert "negative_low" in result
        assert "neutral" in result

    def test_list_specific_category(self):
        """Test listing specific category."""
        result = list_emotions_by_category("positive_high")
        assert "positive_high" in result
        assert len(result) == 1

    def test_list_unknown_category_returns_empty(self):
        """Test unknown category returns empty list."""
        result = list_emotions_by_category("unknown_category")
        assert result == {"unknown_category": []}

    def test_all_emotions_categorized(self):
        """Test all emotions are categorized."""
        categories = list_emotions_by_category()
        all_categorized = []
        for emotions in categories.values():
            all_categorized.extend(emotions)
        assert len(all_categorized) == len(VAD_EMOTIONS)


# ============================================================================
# Emotion Blending Tests
# ============================================================================

class TestEmotionMix:
    """Tests for EmotionMix dataclass."""

    def test_emotion_mix_creation(self):
        """Test basic EmotionMix creation."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        mix = EmotionMix(
            vad=vad,
            components=[("happy", 0.5), ("sad", 0.5)],
            dominant_emotion="content",
            intensity=0.5,
            secondary_emotions=["peaceful", "calm"],
        )
        assert mix.vad == vad
        assert mix.dominant_emotion == "content"
        assert len(mix.components) == 2

    def test_to_synthesis_params(self):
        """Test conversion to synthesis parameters."""
        vad = VADPoint(valence=0.5, arousal=0.6, dominance=0.7)
        mix = EmotionMix(
            vad=vad,
            components=[("happy", 1.0)],
            dominant_emotion="happy",
            intensity=0.8,
            secondary_emotions=[],
        )
        params = mix.to_synthesis_params()
        assert params["emotion"] == "happy"
        assert params["intensity"] == 0.8
        assert params["valence"] == 0.5
        assert params["arousal"] == 0.6
        assert params["dominance"] == 0.7

    def test_describe(self):
        """Test human-readable description."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        mix = EmotionMix(
            vad=vad,
            components=[("happy", 0.6), ("sad", 0.4)],
            dominant_emotion="content",
            intensity=0.5,
            secondary_emotions=[],
        )
        description = mix.describe()
        assert "content" in description
        assert "happy" in description
        assert "sad" in description


class TestBlendVAD:
    """Tests for blend_vad function."""

    def test_blend_single_vad(self):
        """Test blending single VAD point."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        result = blend_vad([(vad, 1.0)])
        assert result.valence == pytest.approx(0.5)
        assert result.arousal == pytest.approx(0.5)
        assert result.dominance == pytest.approx(0.5)

    def test_blend_equal_weights(self):
        """Test blending with equal weights."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = blend_vad([(vad1, 0.5), (vad2, 0.5)])
        assert result.valence == pytest.approx(0.5)
        assert result.arousal == pytest.approx(0.5)
        assert result.dominance == pytest.approx(0.5)

    def test_blend_unequal_weights(self):
        """Test blending with unequal weights."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = blend_vad([(vad1, 0.25), (vad2, 0.75)])
        assert result.valence == pytest.approx(0.75)
        assert result.arousal == pytest.approx(0.75)
        assert result.dominance == pytest.approx(0.75)

    def test_blend_normalizes_weights(self):
        """Test weights are normalized automatically."""
        vad1 = VADPoint(valence=0.0, arousal=0.0, dominance=0.0)
        vad2 = VADPoint(valence=1.0, arousal=1.0, dominance=1.0)
        result = blend_vad([(vad1, 1.0), (vad2, 3.0)])
        assert result.valence == pytest.approx(0.75)

    def test_blend_empty_raises(self):
        """Test blending empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one VAD point"):
            blend_vad([])

    def test_blend_zero_weights_raises(self):
        """Test all-zero weights raises ValueError."""
        vad = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        with pytest.raises(ValueError, match="Weights cannot all be zero"):
            blend_vad([(vad, 0.0)])


class TestBlendEmotions:
    """Tests for blend_emotions function."""

    def test_blend_single_emotion(self):
        """Test blending single emotion."""
        result = blend_emotions([("happy", 1.0)])
        assert result.dominant_emotion == "happy"

    def test_blend_two_emotions(self):
        """Test blending two emotions."""
        result = blend_emotions([("happy", 0.5), ("sad", 0.5)])
        assert isinstance(result, EmotionMix)
        assert len(result.components) == 2

    def test_blend_bittersweet(self):
        """Test creating bittersweet blend."""
        result = blend_emotions([("happy", 0.5), ("sad", 0.5)])
        # Should have characteristics of both
        assert result.vad.valence > VAD_EMOTIONS["sad"].valence
        assert result.vad.valence < VAD_EMOTIONS["happy"].valence

    def test_blend_empty_raises(self):
        """Test blending empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one emotion"):
            blend_emotions([])

    def test_blend_unknown_emotion_skipped(self):
        """Test unknown emotions are skipped with warning."""
        result = blend_emotions([("happy", 0.5), ("unknown_xyz", 0.5)])
        # Should still work, using only valid emotion
        assert result.dominant_emotion is not None

    def test_blend_all_unknown_raises(self):
        """Test all unknown emotions raises ValueError."""
        with pytest.raises(ValueError, match="No valid emotions"):
            blend_emotions([("unknown_xyz", 1.0)])

    def test_blend_has_secondary_emotions(self):
        """Test blend includes secondary emotions."""
        result = blend_emotions([("happy", 0.6), ("excited", 0.4)])
        # Should have at least one secondary emotion
        assert isinstance(result.secondary_emotions, list)


class TestTransitionEmotion:
    """Tests for transition_emotion function."""

    def test_transition_at_start(self):
        """Test transition at start (progress=0)."""
        result = transition_emotion("happy", "sad", 0.0)
        assert result.dominant_emotion == "happy"

    def test_transition_at_end(self):
        """Test transition at end (progress=1)."""
        result = transition_emotion("happy", "sad", 1.0)
        assert result.dominant_emotion == "sad"

    def test_transition_at_midpoint(self):
        """Test transition at midpoint."""
        result = transition_emotion("happy", "sad", 0.5)
        assert len(result.components) == 2

    def test_transition_clamps_progress(self):
        """Test progress is clamped to valid range."""
        result1 = transition_emotion("happy", "sad", -0.5)
        result2 = transition_emotion("happy", "sad", 0.0)
        # Both should be same (at start)
        assert result1.vad.valence == pytest.approx(result2.vad.valence)


class TestCreateEmotionGradient:
    """Tests for create_emotion_gradient function."""

    def test_gradient_two_emotions(self):
        """Test gradient between two emotions."""
        gradient = create_emotion_gradient(["happy", "sad"], steps=10)
        assert len(gradient) > 0

    def test_gradient_three_emotions(self):
        """Test gradient through three emotions."""
        gradient = create_emotion_gradient(["happy", "neutral", "sad"], steps=12)
        assert len(gradient) > 0

    def test_gradient_single_emotion_raises(self):
        """Test single emotion raises ValueError."""
        with pytest.raises(ValueError, match="At least two emotions"):
            create_emotion_gradient(["happy"], steps=10)

    def test_gradient_starts_with_first_emotion(self):
        """Test gradient starts near first emotion."""
        gradient = create_emotion_gradient(["happy", "sad"], steps=10)
        first = gradient[0]
        assert "happy" in first.describe() or first.vad.valence > 0

    def test_gradient_ends_with_last_emotion(self):
        """Test gradient ends with last emotion."""
        gradient = create_emotion_gradient(["happy", "sad"], steps=10)
        last = gradient[-1]
        assert "sad" in last.describe() or last.vad.valence < 0


class TestGetComplementaryEmotion:
    """Tests for get_complementary_emotion function."""

    def test_complementary_of_happy(self):
        """Test complementary of happy is negative."""
        result = get_complementary_emotion("happy")
        complementary_vad = emotion_to_vad(result)
        # Complementary should have opposite valence
        assert complementary_vad.valence < VAD_EMOTIONS["happy"].valence

    def test_complementary_of_sad(self):
        """Test complementary of sad is positive."""
        result = get_complementary_emotion("sad")
        complementary_vad = emotion_to_vad(result)
        assert complementary_vad.valence > VAD_EMOTIONS["sad"].valence

    def test_complementary_returns_valid_emotion(self):
        """Test complementary returns a valid emotion name."""
        result = get_complementary_emotion("neutral")
        assert result in VAD_EMOTIONS


class TestGetSimilarEmotions:
    """Tests for get_similar_emotions function."""

    def test_similar_to_happy(self):
        """Test getting similar emotions to happy."""
        result = get_similar_emotions("happy", count=3)
        assert len(result) <= 3
        assert "happy" not in result  # Shouldn't include itself

    def test_similar_emotions_are_positive_for_happy(self):
        """Test similar emotions to happy are positive."""
        result = get_similar_emotions("happy", count=3)
        for emotion in result:
            vad = emotion_to_vad(emotion)
            assert vad.valence > 0  # Should be positive

    def test_similar_returns_list(self):
        """Test similar returns a list."""
        result = get_similar_emotions("neutral", count=2)
        assert isinstance(result, list)


class TestEmotionDistance:
    """Tests for emotion_distance function."""

    def test_distance_same_emotion(self):
        """Test distance to same emotion is zero."""
        assert emotion_distance("happy", "happy") == 0.0

    def test_distance_different_emotions(self):
        """Test distance between different emotions."""
        distance = emotion_distance("happy", "sad")
        assert distance > 0

    def test_distance_is_symmetric(self):
        """Test distance is symmetric."""
        d1 = emotion_distance("happy", "sad")
        d2 = emotion_distance("sad", "happy")
        assert d1 == pytest.approx(d2)

    def test_distance_happy_sad_is_large(self):
        """Test happy and sad are far apart."""
        distance = emotion_distance("happy", "sad")
        # Should be a significant distance due to opposite valence
        assert distance > 1.0


class TestNamedBlends:
    """Tests for NAMED_BLENDS and related functions."""

    def test_named_blends_exist(self):
        """Test named blends dictionary exists and has items."""
        assert len(NAMED_BLENDS) > 0

    def test_bittersweet_blend_exists(self):
        """Test bittersweet blend exists."""
        assert "bittersweet" in NAMED_BLENDS

    def test_nervous_excitement_blend_exists(self):
        """Test nervous_excitement blend exists."""
        assert "nervous_excitement" in NAMED_BLENDS

    def test_get_named_blend_bittersweet(self):
        """Test getting bittersweet blend."""
        result = get_named_blend("bittersweet")
        assert result is not None
        assert isinstance(result, EmotionMix)

    def test_get_named_blend_case_insensitive(self):
        """Test named blend lookup is case insensitive."""
        result = get_named_blend("BITTERSWEET")
        assert result is not None

    def test_get_named_blend_unknown(self):
        """Test getting unknown blend returns None."""
        result = get_named_blend("unknown_blend_xyz")
        assert result is None

    def test_list_named_blends(self):
        """Test listing named blends."""
        result = list_named_blends()
        assert isinstance(result, list)
        assert "bittersweet" in result
        assert len(result) == len(NAMED_BLENDS)

    def test_all_named_blends_are_valid(self):
        """Test all named blends can be created."""
        for name in list_named_blends():
            result = get_named_blend(name)
            assert result is not None
            assert isinstance(result, EmotionMix)
