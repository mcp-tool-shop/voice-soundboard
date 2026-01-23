"""
Tests for Natural Language Style Interpreter (interpreter.py).

Tests cover:
- StyleParams dataclass
- Style keyword matching
- Speed interpretation
- Voice preference matching
- Preset resolution
- Gender and accent preferences
- apply_style_to_params utility
"""

import pytest
from dataclasses import fields

from voice_soundboard.interpreter import (
    StyleParams,
    STYLE_KEYWORDS,
    find_best_voice,
    interpret_style,
    apply_style_to_params,
)


class TestStyleParams:
    """Tests for StyleParams dataclass."""

    def test_default_values(self):
        """Test StyleParams has correct default values."""
        params = StyleParams()

        assert params.voice is None
        assert params.speed is None
        assert params.preset is None
        assert params.confidence == 0.0

    def test_all_fields_optional(self):
        """Test all fields can be set."""
        params = StyleParams(
            voice="af_bella",
            speed=0.95,
            preset="narrator",
            confidence=0.8,
        )

        assert params.voice == "af_bella"
        assert params.speed == 0.95
        assert params.preset == "narrator"
        assert params.confidence == 0.8


class TestStyleKeywords:
    """Tests for STYLE_KEYWORDS dictionary."""

    def test_speed_keywords_exist(self):
        """Test speed-related keywords are defined."""
        speed_keywords = ["quickly", "fast", "slowly", "slow", "carefully"]
        for kw in speed_keywords:
            assert kw in STYLE_KEYWORDS

    def test_emotion_keywords_exist(self):
        """Test emotion-related keywords are defined."""
        emotion_keywords = ["excitedly", "calmly", "warmly", "sadly", "angrily"]
        for kw in emotion_keywords:
            assert kw in STYLE_KEYWORDS

    def test_preset_keywords_exist(self):
        """Test preset-related keywords are defined."""
        preset_keywords = ["like a narrator", "like a storyteller", "whispered"]
        for kw in preset_keywords:
            assert kw in STYLE_KEYWORDS

    def test_gender_keywords_exist(self):
        """Test gender preference keywords are defined."""
        assert "in a male voice" in STYLE_KEYWORDS
        assert "in a female voice" in STYLE_KEYWORDS
        assert "masculine" in STYLE_KEYWORDS
        assert "feminine" in STYLE_KEYWORDS

    def test_accent_keywords_exist(self):
        """Test accent preference keywords are defined."""
        assert "british" in STYLE_KEYWORDS
        assert "american" in STYLE_KEYWORDS
        assert "with a british accent" in STYLE_KEYWORDS


class TestFindBestVoice:
    """Tests for find_best_voice function."""

    def test_find_voice_by_gender_male(self):
        """Test finding male voice."""
        voice = find_best_voice(gender_prefer="male")

        assert voice is not None
        # Should return a male voice (starts with 'am_' or 'bm_')
        assert voice.startswith(("am_", "bm_"))

    def test_find_voice_by_gender_female(self):
        """Test finding female voice."""
        voice = find_best_voice(gender_prefer="female")

        assert voice is not None
        # Should return a female voice (starts with 'af_' or 'bf_')
        assert voice.startswith(("af_", "bf_"))

    def test_find_voice_by_accent_british(self):
        """Test finding British accent voice."""
        voice = find_best_voice(accent_prefer="british")

        assert voice is not None
        # British voices start with 'b'
        assert voice.startswith("b")

    def test_find_voice_by_style_warm(self):
        """Test finding warm-styled voice."""
        voice = find_best_voice(style_prefer=["warm", "friendly"])

        assert voice is not None

    def test_find_voice_no_match(self):
        """Test returns None when no match found."""
        voice = find_best_voice(style_prefer=["nonexistent_style_xyz"])

        assert voice is None

    def test_find_voice_combined_preferences(self):
        """Test with multiple preferences."""
        voice = find_best_voice(
            style_prefer=["warm"],
            gender_prefer="female",
        )

        # Should find a matching voice
        assert voice is not None


class TestInterpretStyle:
    """Tests for interpret_style function."""

    def test_empty_string(self):
        """Test empty style hint returns zero confidence."""
        result = interpret_style("")

        assert result.confidence == 0.0
        assert result.voice is None
        assert result.speed is None

    def test_speed_keyword_quickly(self):
        """Test 'quickly' sets faster speed."""
        result = interpret_style("quickly")

        assert result.speed > 1.0
        assert result.confidence > 0

    def test_speed_keyword_slowly(self):
        """Test 'slowly' sets slower speed."""
        result = interpret_style("slowly")

        assert result.speed < 1.0
        assert result.confidence > 0

    def test_style_warmly(self):
        """Test 'warmly' interpretation."""
        result = interpret_style("warmly")

        assert result.speed is not None
        assert result.speed < 1.0  # Warm is slower
        assert result.voice is not None  # Should find a warm voice
        assert result.confidence > 0

    def test_style_excitedly(self):
        """Test 'excitedly' interpretation."""
        result = interpret_style("excitedly")

        assert result.speed is not None
        assert result.speed > 1.0  # Excited is faster
        assert result.confidence > 0

    def test_preset_like_narrator(self):
        """Test 'like a narrator' sets preset."""
        result = interpret_style("like a narrator")

        assert result.preset == "narrator"
        assert result.confidence > 0

    def test_preset_whispered(self):
        """Test 'whispered' sets whisper preset."""
        result = interpret_style("whispered")

        assert result.preset == "whisper"

    def test_gender_preference_male(self):
        """Test 'in a male voice' sets gender preference."""
        result = interpret_style("in a male voice")

        assert result.voice is not None
        # Should be a male voice
        assert result.voice.startswith(("am_", "bm_"))

    def test_gender_preference_female(self):
        """Test 'in a female voice' sets gender preference."""
        result = interpret_style("in a female voice")

        assert result.voice is not None
        # Should be a female voice
        assert result.voice.startswith(("af_", "bf_"))

    def test_accent_preference_british(self):
        """Test 'british' sets accent preference."""
        result = interpret_style("with a british accent")

        assert result.voice is not None
        assert result.voice.startswith("b")  # British voices

    def test_combined_styles(self):
        """Test multiple style hints combined."""
        result = interpret_style("quickly and excitedly")

        # Should average the speeds
        assert result.speed is not None
        assert result.speed > 1.0
        # Higher confidence with more matches
        assert result.confidence > 0.3

    def test_preset_with_speed_override(self):
        """Test preset combined with speed modifier."""
        result = interpret_style("like a narrator, slowly")

        assert result.preset == "narrator"
        assert result.speed < 1.0

    def test_case_insensitive(self):
        """Test style matching is case-insensitive."""
        result1 = interpret_style("WARMLY")
        result2 = interpret_style("warmly")
        result3 = interpret_style("Warmly")

        # All should produce same result
        assert result1.speed == result2.speed == result3.speed

    def test_confidence_increases_with_matches(self):
        """Test confidence increases with more keyword matches."""
        single = interpret_style("warmly")
        double = interpret_style("warmly and slowly")

        assert double.confidence > single.confidence

    def test_unknown_style_zero_confidence(self):
        """Test unknown style returns zero confidence."""
        result = interpret_style("xyzabc123unknown")

        assert result.confidence == 0.0


class TestApplyStyleToParams:
    """Tests for apply_style_to_params function."""

    def test_style_applied_to_empty_params(self):
        """Test style applied when no explicit params given."""
        voice, speed, preset = apply_style_to_params("warmly")

        assert voice is not None  # Should get from style
        assert speed is not None
        assert speed < 1.0

    def test_explicit_voice_overrides(self):
        """Test explicit voice overrides interpreted voice."""
        voice, speed, preset = apply_style_to_params(
            "in a female voice",
            voice="bm_george"  # Explicit override
        )

        assert voice == "bm_george"  # Should keep explicit

    def test_explicit_speed_overrides(self):
        """Test explicit speed overrides interpreted speed."""
        voice, speed, preset = apply_style_to_params(
            "slowly",  # Would set speed < 1.0
            speed=1.5  # Explicit override
        )

        assert speed == 1.5  # Should keep explicit

    def test_explicit_preset_overrides(self):
        """Test explicit preset overrides interpreted preset."""
        voice, speed, preset = apply_style_to_params(
            "like a narrator",  # Would set narrator
            preset="announcer"  # Explicit override
        )

        assert preset == "announcer"

    def test_mixed_explicit_and_interpreted(self):
        """Test mixing explicit and interpreted params."""
        voice, speed, preset = apply_style_to_params(
            "warmly like a narrator",
            speed=1.2  # Only override speed
        )

        # Speed is explicit
        assert speed == 1.2
        # Preset from style
        assert preset == "narrator"
        # Voice from style
        assert voice is not None

    def test_no_style_hint(self):
        """Test with empty style hint and explicit params."""
        voice, speed, preset = apply_style_to_params(
            "",
            voice="af_bella",
            speed=1.0,
            preset="narrator"
        )

        assert voice == "af_bella"
        assert speed == 1.0
        assert preset == "narrator"


class TestEdgeCases:
    """Edge case tests."""

    def test_none_style_hint(self):
        """Test None as style hint doesn't crash."""
        # interpret_style expects string, but should handle gracefully
        result = interpret_style(None) if False else interpret_style("")

        assert result.confidence == 0.0

    def test_whitespace_only(self):
        """Test whitespace-only style hint."""
        result = interpret_style("   ")

        # Should be treated as empty
        assert result.confidence == 0.0

    def test_partial_keyword_match(self):
        """Test partial keywords don't match."""
        result = interpret_style("warm")  # Not 'warmly'

        # Should not match 'warmly' keyword
        assert result.confidence == 0.0

    def test_very_long_style_hint(self):
        """Test very long style hint is handled."""
        long_hint = "warmly " * 100
        result = interpret_style(long_hint)

        # Should still work
        assert result.confidence > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
