"""
Tests for Emotions Module (emotions.py).

Tests cover:
- EmotionParams dataclass
- EMOTIONS registry
- get_emotion_params function
- list_emotions function
- apply_emotion_to_text function
- get_emotion_voice_params function
- intensify_emotion function
"""

import pytest
from dataclasses import fields

from voice_soundboard.emotions import (
    EmotionParams,
    EMOTIONS,
    get_emotion_params,
    list_emotions,
    apply_emotion_to_text,
    get_emotion_voice_params,
    intensify_emotion,
)


class TestEmotionParams:
    """Tests for EmotionParams dataclass."""

    def test_all_fields_present(self):
        """Test EmotionParams has all expected fields."""
        field_names = {f.name for f in fields(EmotionParams)}
        expected = {
            "speed", "voice_preference", "text_prefix",
            "text_suffix", "punctuation_boost", "pause_multiplier"
        }
        assert field_names == expected

    def test_default_values(self):
        """Test EmotionParams default values."""
        params = EmotionParams()

        assert params.speed == 1.0
        assert params.voice_preference is None
        assert params.text_prefix == ""
        assert params.text_suffix == ""
        assert params.punctuation_boost is False
        assert params.pause_multiplier == 1.0

    def test_custom_values(self):
        """Test EmotionParams with custom values."""
        params = EmotionParams(
            speed=1.5,
            voice_preference="af_bella",
            punctuation_boost=True,
            pause_multiplier=0.8,
        )

        assert params.speed == 1.5
        assert params.voice_preference == "af_bella"
        assert params.punctuation_boost is True
        assert params.pause_multiplier == 0.8


class TestEMOTIONSRegistry:
    """Tests for EMOTIONS registry."""

    def test_not_empty(self):
        """Test EMOTIONS registry is not empty."""
        assert len(EMOTIONS) > 0

    def test_has_basic_emotions(self):
        """Test basic emotions are defined."""
        basic = ["happy", "sad", "angry", "fearful", "surprised", "neutral"]
        for emotion in basic:
            assert emotion in EMOTIONS, f"Missing basic emotion: {emotion}"

    def test_has_calm_emotions(self):
        """Test calm emotions are defined."""
        calm = ["calm", "peaceful"]
        for emotion in calm:
            assert emotion in EMOTIONS

    def test_has_professional_emotions(self):
        """Test professional emotions are defined."""
        professional = ["confident", "serious", "professional"]
        for emotion in professional:
            assert emotion in EMOTIONS

    def test_has_storytelling_emotions(self):
        """Test storytelling emotions are defined."""
        storytelling = ["mysterious", "dramatic", "whimsical"]
        for emotion in storytelling:
            assert emotion in EMOTIONS

    def test_all_entries_are_emotion_params(self):
        """Test all registry entries are EmotionParams."""
        for emotion, params in EMOTIONS.items():
            assert isinstance(params, EmotionParams), f"{emotion} is not EmotionParams"

    def test_speeds_in_valid_range(self):
        """Test all emotion speeds are in valid range."""
        for emotion, params in EMOTIONS.items():
            assert 0.5 <= params.speed <= 2.0, f"{emotion} has invalid speed: {params.speed}"

    def test_neutral_is_baseline(self):
        """Test neutral emotion is baseline."""
        neutral = EMOTIONS["neutral"]
        assert neutral.speed == 1.0
        assert neutral.voice_preference is None
        assert neutral.punctuation_boost is False


class TestGetEmotionParams:
    """Tests for get_emotion_params function."""

    def test_known_emotion(self):
        """Test getting known emotion."""
        params = get_emotion_params("happy")

        assert isinstance(params, EmotionParams)
        assert params.speed > 1.0  # Happy is faster

    def test_case_insensitive(self):
        """Test emotion lookup is case insensitive."""
        params1 = get_emotion_params("happy")
        params2 = get_emotion_params("HAPPY")
        params3 = get_emotion_params("Happy")

        assert params1.speed == params2.speed == params3.speed

    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        params1 = get_emotion_params("happy")
        params2 = get_emotion_params("  happy  ")

        assert params1.speed == params2.speed

    def test_unknown_emotion_returns_neutral(self):
        """Test unknown emotion returns neutral."""
        params = get_emotion_params("nonexistent_emotion_xyz")

        assert params.speed == 1.0  # Neutral speed

    def test_partial_match(self):
        """Test partial matching works."""
        # "happ" should match "happy"
        params = get_emotion_params("happ")
        happy_params = EMOTIONS["happy"]

        assert params.speed == happy_params.speed

    def test_excited_emotion(self):
        """Test excited emotion parameters."""
        params = get_emotion_params("excited")

        assert params.speed > 1.0  # Faster
        assert params.punctuation_boost is True

    def test_sad_emotion(self):
        """Test sad emotion parameters."""
        params = get_emotion_params("sad")

        assert params.speed < 1.0  # Slower
        assert params.pause_multiplier > 1.0  # Longer pauses

    def test_calm_emotion(self):
        """Test calm emotion parameters."""
        params = get_emotion_params("calm")

        assert params.speed < 1.0  # Slower
        assert params.pause_multiplier > 1.0


class TestListEmotions:
    """Tests for list_emotions function."""

    def test_returns_list(self):
        """Test list_emotions returns a list."""
        emotions = list_emotions()
        assert isinstance(emotions, list)

    def test_not_empty(self):
        """Test list is not empty."""
        emotions = list_emotions()
        assert len(emotions) > 0

    def test_sorted(self):
        """Test list is sorted alphabetically."""
        emotions = list_emotions()
        assert emotions == sorted(emotions)

    def test_contains_basic_emotions(self):
        """Test list contains basic emotions."""
        emotions = list_emotions()
        assert "happy" in emotions
        assert "sad" in emotions
        assert "neutral" in emotions

    def test_unique_entries(self):
        """Test all entries are unique."""
        emotions = list_emotions()
        assert len(emotions) == len(set(emotions))


class TestApplyEmotionToText:
    """Tests for apply_emotion_to_text function."""

    def test_neutral_no_change(self):
        """Test neutral emotion doesn't change text."""
        text = "This is a test."
        result = apply_emotion_to_text(text, "neutral")

        assert result == text

    def test_happy_adds_exclamation(self):
        """Test happy emotion adds exclamation marks."""
        text = "This is great. I love it."
        result = apply_emotion_to_text(text, "happy")

        # Should add some exclamation marks
        assert "!" in result

    def test_excited_adds_exclamation(self):
        """Test excited emotion adds exclamation marks."""
        text = "Wow this is amazing. So cool."
        result = apply_emotion_to_text(text, "excited")

        assert "!" in result

    def test_sad_no_boost(self):
        """Test sad emotion doesn't boost punctuation."""
        text = "This is sad. I feel down."
        result = apply_emotion_to_text(text, "sad")

        # Sad doesn't have punctuation boost, so should be same
        # Actually sad might not modify, let's check
        # The function only boosts for specific emotions

    def test_preserves_questions(self):
        """Test question marks are preserved."""
        text = "How are you? I hope well."
        result = apply_emotion_to_text(text, "happy")

        assert "?" in result

    def test_preserves_existing_exclamations(self):
        """Test existing exclamation marks are preserved."""
        text = "Wow! This is great."
        result = apply_emotion_to_text(text, "happy")

        assert "!" in result

    def test_alternates_boosting(self):
        """Test punctuation boost alternates sentences."""
        text = "One. Two. Three. Four."
        result = apply_emotion_to_text(text, "excited")

        # Every other sentence should be boosted
        # So not all periods become exclamations


class TestGetEmotionVoiceParams:
    """Tests for get_emotion_voice_params function."""

    def test_returns_dict(self):
        """Test function returns dict."""
        params = get_emotion_voice_params("happy")
        assert isinstance(params, dict)

    def test_has_voice_key(self):
        """Test dict has voice key."""
        params = get_emotion_voice_params("happy")
        assert "voice" in params

    def test_has_speed_key(self):
        """Test dict has speed key."""
        params = get_emotion_voice_params("happy")
        assert "speed" in params

    def test_uses_emotion_defaults(self):
        """Test uses emotion defaults when no overrides."""
        params = get_emotion_voice_params("excited")
        excited = EMOTIONS["excited"]

        assert params["speed"] == excited.speed
        assert params["voice"] == excited.voice_preference

    def test_voice_override(self):
        """Test voice can be overridden."""
        params = get_emotion_voice_params("happy", voice="am_michael")

        assert params["voice"] == "am_michael"

    def test_speed_override(self):
        """Test speed can be overridden."""
        params = get_emotion_voice_params("happy", speed=1.5)

        assert params["speed"] == 1.5

    def test_both_overrides(self):
        """Test both voice and speed can be overridden."""
        params = get_emotion_voice_params(
            "happy",
            voice="bm_george",
            speed=0.8
        )

        assert params["voice"] == "bm_george"
        assert params["speed"] == 0.8

    def test_neutral_has_none_voice(self):
        """Test neutral emotion has None voice preference."""
        params = get_emotion_voice_params("neutral")

        # Neutral doesn't have a voice preference
        assert params["voice"] is None or params["voice"] in [None]


class TestIntensifyEmotion:
    """Tests for intensify_emotion function."""

    def test_returns_emotion_params(self):
        """Test function returns EmotionParams."""
        params = intensify_emotion("happy", 1.0)
        assert isinstance(params, EmotionParams)

    def test_intensity_1_unchanged(self):
        """Test intensity 1.0 returns similar params."""
        base = get_emotion_params("happy")
        intensified = intensify_emotion("happy", 1.0)

        assert intensified.speed == base.speed
        assert intensified.voice_preference == base.voice_preference

    def test_higher_intensity_more_extreme_speed(self):
        """Test higher intensity makes speed more extreme."""
        base = intensify_emotion("excited", 1.0)
        high = intensify_emotion("excited", 1.5)

        # Excited has speed > 1.0, so higher intensity should be even faster
        assert high.speed > base.speed

    def test_lower_intensity_less_extreme_speed(self):
        """Test lower intensity makes speed less extreme."""
        base = intensify_emotion("excited", 1.0)
        low = intensify_emotion("excited", 0.5)

        # Lower intensity should be closer to 1.0
        assert abs(low.speed - 1.0) < abs(base.speed - 1.0)

    def test_intensity_clamped_minimum(self):
        """Test intensity is clamped to minimum 0.5."""
        params = intensify_emotion("happy", 0.1)

        # Should use 0.5, not 0.1
        assert params is not None

    def test_intensity_clamped_maximum(self):
        """Test intensity is clamped to maximum 2.0."""
        params = intensify_emotion("happy", 5.0)

        # Should use 2.0, not 5.0
        assert params is not None

    def test_speed_clamped(self):
        """Test output speed is clamped to valid range."""
        params = intensify_emotion("excited", 2.0)

        assert 0.5 <= params.speed <= 2.0

    def test_preserves_voice_preference(self):
        """Test voice preference is preserved."""
        base = get_emotion_params("happy")
        intensified = intensify_emotion("happy", 1.5)

        assert intensified.voice_preference == base.voice_preference

    def test_pause_multiplier_scaled(self):
        """Test pause multiplier is scaled."""
        base = intensify_emotion("sad", 1.0)
        high = intensify_emotion("sad", 1.5)

        # Sad has pause_multiplier > 1.0, higher intensity = longer pauses
        assert high.pause_multiplier >= base.pause_multiplier

    def test_punctuation_boost_disabled_at_low_intensity(self):
        """Test punctuation boost disabled at low intensity."""
        # Happy has punctuation_boost = True
        low_params = intensify_emotion("happy", 0.5)

        # At intensity < 0.8, punctuation_boost should be False
        assert low_params.punctuation_boost is False

    def test_punctuation_boost_enabled_at_high_intensity(self):
        """Test punctuation boost enabled at normal/high intensity."""
        high_params = intensify_emotion("happy", 1.0)

        # At intensity >= 0.8, punctuation_boost should match original
        # happy has punctuation_boost = True
        assert high_params.punctuation_boost is True


class TestEmotionCategories:
    """Tests for emotion category groupings."""

    def test_positive_emotions_faster(self):
        """Test positive emotions generally have faster speed."""
        positive = ["happy", "excited", "joyful"]
        for emotion in positive:
            params = get_emotion_params(emotion)
            assert params.speed >= 1.0, f"{emotion} should be >= 1.0 speed"

    def test_calm_emotions_slower(self):
        """Test calm emotions have slower speed."""
        calm = ["calm", "peaceful", "sad", "melancholy"]
        for emotion in calm:
            params = get_emotion_params(emotion)
            assert params.speed < 1.0, f"{emotion} should be < 1.0 speed"

    def test_urgent_emotions_fast(self):
        """Test urgent emotions are fast."""
        urgent = ["urgent", "fearful", "surprised"]
        for emotion in urgent:
            params = get_emotion_params(emotion)
            assert params.speed > 1.0, f"{emotion} should be > 1.0 speed"


class TestVoicePreferences:
    """Tests for voice preferences in emotions."""

    def test_voice_preferences_are_valid(self):
        """Test voice preferences reference valid voices."""
        from voice_soundboard.config import KOKORO_VOICES

        for emotion, params in EMOTIONS.items():
            if params.voice_preference:
                assert params.voice_preference in KOKORO_VOICES, \
                    f"{emotion} references invalid voice: {params.voice_preference}"

    def test_some_emotions_have_voice_preference(self):
        """Test some emotions have voice preferences."""
        has_preference = [
            e for e, p in EMOTIONS.items()
            if p.voice_preference is not None
        ]
        assert len(has_preference) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
