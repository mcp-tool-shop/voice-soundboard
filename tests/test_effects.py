"""
Tests for Sound Effects Library (effects.py).

Tests cover:
- SoundEffect dataclass
- Effect generation functions
- EFFECTS registry
- get_effect, play_effect, list_effects utilities
- Envelope and tone generation helpers
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from voice_soundboard.effects import (
    SoundEffect,
    SAMPLE_RATE,
    EFFECTS,
    get_effect,
    list_effects,
    play_effect,
    _envelope,
    _generate_tone,
    chime_notification,
    chime_success,
    chime_error,
    chime_attention,
    click,
    pop,
    whoosh,
    alert_warning,
    alert_critical,
    alert_info,
    ambient_rain,
    ambient_white_noise,
    ambient_drone,
)


class TestSoundEffect:
    """Tests for SoundEffect dataclass."""

    def test_sound_effect_structure(self):
        """Test SoundEffect has correct fields."""
        samples = np.zeros(1000, dtype=np.float32)
        effect = SoundEffect(
            name="test",
            samples=samples,
            sample_rate=44100,
            duration=0.5,
        )

        assert effect.name == "test"
        assert len(effect.samples) == 1000
        assert effect.sample_rate == 44100
        assert effect.duration == 0.5

    def test_save_creates_file(self, tmp_path):
        """Test save method creates WAV file."""
        samples = np.zeros(1000, dtype=np.float32)
        effect = SoundEffect("test", samples, 44100, 0.5)

        path = tmp_path / "test.wav"
        result = effect.save(path)

        assert result == path
        assert path.exists()

    @patch('voice_soundboard.effects.play_audio')
    def test_play_calls_play_audio(self, mock_play):
        """Test play method calls play_audio."""
        samples = np.zeros(1000, dtype=np.float32)
        effect = SoundEffect("test", samples, 44100, 0.5)

        effect.play()

        mock_play.assert_called_once()


class TestSampleRate:
    """Tests for sample rate constant."""

    def test_sample_rate_is_44100(self):
        """Test standard sample rate is 44100."""
        assert SAMPLE_RATE == 44100


class TestEnvelope:
    """Tests for _envelope helper function."""

    def test_envelope_applies_attack(self):
        """Test attack ramp is applied."""
        samples = np.ones(1000)
        result = _envelope(samples, attack=0.01, decay=0.0)

        # First sample should be near 0
        assert result[0] < 0.1

    def test_envelope_applies_decay(self):
        """Test decay ramp is applied."""
        samples = np.ones(1000)
        result = _envelope(samples, attack=0.0, decay=0.01)

        # Last sample should be near 0
        assert result[-1] < 0.1

    def test_envelope_preserves_middle(self):
        """Test middle samples are unaffected."""
        samples = np.ones(10000)
        result = _envelope(samples, attack=0.01, decay=0.01)

        # Middle should be close to 1
        mid = len(result) // 2
        assert abs(result[mid] - 1.0) < 0.01


class TestGenerateTone:
    """Tests for _generate_tone helper function."""

    def test_generate_sine_wave(self):
        """Test generating sine wave."""
        tone = _generate_tone(440, 1.0, wave="sine")

        # Should be 44100 samples for 1 second
        assert len(tone) == 44100

        # Sine wave has zero mean
        assert abs(np.mean(tone)) < 0.01

    def test_generate_square_wave(self):
        """Test generating square wave."""
        tone = _generate_tone(440, 0.1, wave="square")

        # Square wave values are -1 or 1
        unique = np.unique(np.sign(tone))
        assert -1 in unique
        assert 1 in unique

    def test_generate_triangle_wave(self):
        """Test generating triangle wave."""
        tone = _generate_tone(440, 0.1, wave="triangle")

        # Triangle wave is bounded by -1 to 1
        assert np.max(tone) <= 1.01
        assert np.min(tone) >= -1.01

    def test_generate_sawtooth_wave(self):
        """Test generating sawtooth wave."""
        tone = _generate_tone(440, 0.1, wave="sawtooth")

        # Sawtooth is bounded by -1 to 1
        assert np.max(tone) <= 1.01
        assert np.min(tone) >= -1.01

    def test_unknown_wave_defaults_to_sine(self):
        """Test unknown wave type defaults to sine."""
        tone_unknown = _generate_tone(440, 0.1, wave="unknown")
        tone_sine = _generate_tone(440, 0.1, wave="sine")

        np.testing.assert_array_equal(tone_unknown, tone_sine)


class TestChimeEffects:
    """Tests for chime effect generators."""

    def test_chime_notification(self):
        """Test chime_notification generates valid effect."""
        effect = chime_notification()

        assert effect.name == "chime_notification"
        assert len(effect.samples) > 0
        assert effect.sample_rate == SAMPLE_RATE
        assert effect.duration > 0

    def test_chime_success(self):
        """Test chime_success generates valid effect."""
        effect = chime_success()

        assert effect.name == "chime_success"
        assert len(effect.samples) > 0
        assert effect.duration > 0

    def test_chime_error(self):
        """Test chime_error generates valid effect."""
        effect = chime_error()

        assert effect.name == "chime_error"
        assert len(effect.samples) > 0

    def test_chime_attention(self):
        """Test chime_attention generates valid effect."""
        effect = chime_attention()

        assert effect.name == "chime_attention"
        assert len(effect.samples) > 0


class TestUIEffects:
    """Tests for UI sound effect generators."""

    def test_click_effect(self):
        """Test click effect generates short sound."""
        effect = click()

        assert effect.name == "click"
        assert effect.duration < 0.1  # Very short

    def test_pop_effect(self):
        """Test pop effect generates short sound."""
        effect = pop()

        assert effect.name == "pop"
        assert effect.duration < 0.1

    def test_whoosh_effect(self):
        """Test whoosh effect generates sound."""
        effect = whoosh()

        assert effect.name == "whoosh"
        assert effect.duration > 0


class TestAlertEffects:
    """Tests for alert effect generators."""

    def test_alert_warning(self):
        """Test alert_warning generates valid effect."""
        effect = alert_warning()

        assert effect.name == "alert_warning"
        assert len(effect.samples) > 0

    def test_alert_critical(self):
        """Test alert_critical generates repeated beeps."""
        effect = alert_critical()

        assert effect.name == "alert_critical"
        # Should have multiple beeps
        assert effect.duration > 0.2

    def test_alert_info(self):
        """Test alert_info generates gentle tone."""
        effect = alert_info()

        assert effect.name == "alert_info"
        assert len(effect.samples) > 0


class TestAmbientEffects:
    """Tests for ambient sound effect generators."""

    def test_ambient_rain(self):
        """Test ambient_rain generates sound."""
        effect = ambient_rain(duration=1.0)

        assert effect.name == "ambient_rain"
        # Should be approximately 1 second
        expected_samples = int(SAMPLE_RATE * 1.0)
        assert abs(len(effect.samples) - expected_samples) < 100

    def test_ambient_white_noise(self):
        """Test ambient_white_noise generates noise."""
        effect = ambient_white_noise(duration=1.0)

        assert effect.name == "ambient_white_noise"
        # White noise has non-zero variance
        assert np.std(effect.samples) > 0

    def test_ambient_drone(self):
        """Test ambient_drone generates tonal sound."""
        effect = ambient_drone(duration=1.0, base_freq=110)

        assert effect.name == "ambient_drone"
        assert len(effect.samples) > 0


class TestEffectsRegistry:
    """Tests for EFFECTS registry."""

    def test_effects_not_empty(self):
        """Test EFFECTS registry has entries."""
        assert len(EFFECTS) > 0

    def test_chime_aliases(self):
        """Test chime has aliases."""
        assert "chime" in EFFECTS
        assert "chime_notification" in EFFECTS
        # Both should be the same function
        assert EFFECTS["chime"] == EFFECTS["chime_notification"]

    def test_success_alias(self):
        """Test success is alias for chime_success."""
        assert "success" in EFFECTS
        assert EFFECTS["success"] == EFFECTS["chime_success"]

    def test_error_alias(self):
        """Test error is alias for chime_error."""
        assert "error" in EFFECTS
        assert EFFECTS["error"] == EFFECTS["chime_error"]

    def test_warning_alias(self):
        """Test warning is alias for alert_warning."""
        assert "warning" in EFFECTS
        assert EFFECTS["warning"] == EFFECTS["alert_warning"]


class TestGetEffect:
    """Tests for get_effect function."""

    def test_get_effect_valid_name(self):
        """Test get_effect returns effect for valid name."""
        effect = get_effect("chime")

        assert isinstance(effect, SoundEffect)
        assert effect.name.startswith("chime")

    def test_get_effect_invalid_name(self):
        """Test get_effect raises for invalid name."""
        with pytest.raises(ValueError) as exc_info:
            get_effect("nonexistent_effect_xyz")

        assert "Unknown effect" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_get_effect_case_sensitive(self):
        """Test get_effect is case-sensitive."""
        # Should work with correct case
        effect = get_effect("chime")
        assert effect is not None

        # Should fail with wrong case
        with pytest.raises(ValueError):
            get_effect("CHIME")


class TestListEffects:
    """Tests for list_effects function."""

    def test_list_effects_returns_list(self):
        """Test list_effects returns a list."""
        effects = list_effects()

        assert isinstance(effects, list)
        assert len(effects) > 0

    def test_list_effects_unique(self):
        """Test list_effects returns unique names."""
        effects = list_effects()

        # No duplicates
        assert len(effects) == len(set(effects))

    def test_list_effects_sorted(self):
        """Test list_effects returns sorted list."""
        effects = list_effects()

        assert effects == sorted(effects)

    def test_list_effects_has_expected_effects(self):
        """Test expected effect names are in list."""
        effects = list_effects()

        # At least these should be present (one per function)
        expected_present = ["chime", "click", "pop", "warning"]
        for name in expected_present:
            # Either exact name or alias should be present
            assert any(name in e for e in effects) or name in effects


class TestPlayEffect:
    """Tests for play_effect function."""

    @patch.object(SoundEffect, 'play')
    def test_play_effect_calls_play(self, mock_play):
        """Test play_effect calls the effect's play method."""
        play_effect("chime")

        mock_play.assert_called_once()

    def test_play_effect_invalid_name(self):
        """Test play_effect raises for invalid name."""
        with pytest.raises(ValueError):
            play_effect("nonexistent_effect")


class TestEffectProperties:
    """Tests for effect audio properties."""

    def test_all_effects_have_valid_samples(self):
        """Test all effects generate valid audio samples."""
        for name in list_effects():
            effect = get_effect(name)

            # Should have samples
            assert len(effect.samples) > 0

            # Samples should be numpy array
            assert isinstance(effect.samples, np.ndarray)

            # Samples should be float32
            assert effect.samples.dtype == np.float32

    def test_effects_are_normalized(self):
        """Test effects don't significantly clip (values mostly in -1 to 1 range)."""
        for name in list_effects():
            if "ambient" not in name and "whoosh" not in name:  # Noise-based effects may slightly exceed
                effect = get_effect(name)

                # Check range - allow small overshoot from noise
                max_val = np.max(np.abs(effect.samples))
                assert max_val <= 1.5, f"Effect {name} significantly clips"

    def test_effects_have_audio_content(self):
        """Test effects aren't silent."""
        for name in list_effects():
            effect = get_effect(name)

            # Should have some non-zero content
            max_val = np.max(np.abs(effect.samples))
            assert max_val > 0.01, f"Effect {name} is silent"


class TestEffectDurations:
    """Tests for effect durations."""

    def test_ui_effects_are_short(self):
        """Test UI effects are short for responsiveness."""
        ui_effects = ["click", "pop"]
        for name in ui_effects:
            if name in EFFECTS:
                effect = get_effect(name)
                assert effect.duration < 0.5, f"{name} should be short"

    def test_chimes_are_medium_length(self):
        """Test chimes are medium duration."""
        chime_effects = ["chime", "success", "error"]
        for name in chime_effects:
            if name in EFFECTS:
                effect = get_effect(name)
                assert 0.1 < effect.duration < 1.0, f"{name} duration unexpected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
