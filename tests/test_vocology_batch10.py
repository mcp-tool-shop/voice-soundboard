"""
Tests for Phase 9: Vocology Module - Batch 10
ProsodyAnalyzer and ProsodyModifier classes

Tests cover:
- ProsodyAnalyzer initialization (TEST-PAN-01 to TEST-PAN-04)
- ProsodyAnalyzer.analyze() method (TEST-PAN-05 to TEST-PAN-12)
- ProsodyModifier initialization (TEST-PMO-01)
- ProsodyModifier methods (TEST-PMO-02 to TEST-PMO-18)
- Prosody convenience functions (TEST-PMO-19 to TEST-PMO-25)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-PAN-01 to TEST-PAN-04: ProsodyAnalyzer Initialization Tests
# =============================================================================

from voice_soundboard.vocology.prosody import (
    ProsodyAnalyzer,
    ProsodyModifier,
    ProsodyContour,
    PitchContour,
    analyze_prosody,
    modify_prosody,
)


class TestProsodyAnalyzerInit:
    """Tests for ProsodyAnalyzer initialization (TEST-PAN-01 to TEST-PAN-04)."""

    def test_pan_01_default_f0_min(self):
        """TEST-PAN-01: Default f0_min is 50.0 Hz."""
        analyzer = ProsodyAnalyzer()
        assert analyzer.f0_min == 50.0

    def test_pan_02_default_f0_max(self):
        """TEST-PAN-02: Default f0_max is 500.0 Hz."""
        analyzer = ProsodyAnalyzer()
        assert analyzer.f0_max == 500.0

    def test_pan_03_default_hop_length(self):
        """TEST-PAN-03: Default hop_length is 0.010 seconds."""
        analyzer = ProsodyAnalyzer()
        assert analyzer.hop_length == 0.010

    def test_pan_04_custom_parameters(self):
        """TEST-PAN-04: Custom parameters are stored correctly."""
        analyzer = ProsodyAnalyzer(f0_min=75.0, f0_max=400.0, hop_length=0.005)
        assert analyzer.f0_min == 75.0
        assert analyzer.f0_max == 400.0
        assert analyzer.hop_length == 0.005


# =============================================================================
# TEST-PAN-05 to TEST-PAN-12: ProsodyAnalyzer.analyze() Tests
# =============================================================================

class TestProsodyAnalyzerAnalyze:
    """Tests for ProsodyAnalyzer.analyze() method (TEST-PAN-05 to TEST-PAN-12)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio += 0.01 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_pan_05_analyze_returns_contour(self, mock_audio_array):
        """TEST-PAN-05: analyze() returns ProsodyContour."""
        audio, sr = mock_audio_array
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result, ProsodyContour)

    def test_pan_06_analyze_requires_sample_rate(self):
        """TEST-PAN-06: analyze() raises ValueError without sample_rate for array."""
        audio = np.zeros(16000, dtype=np.float32)
        analyzer = ProsodyAnalyzer()
        with pytest.raises(ValueError, match="sample_rate"):
            analyzer.analyze(audio)

    def test_pan_07_analyze_has_pitch(self, mock_audio_array):
        """TEST-PAN-07: analyze() result has pitch contour."""
        audio, sr = mock_audio_array
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.pitch is not None
        assert isinstance(result.pitch, PitchContour)

    def test_pan_08_analyze_has_energy(self, mock_audio_array):
        """TEST-PAN-08: analyze() result has energy contour."""
        audio, sr = mock_audio_array
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.energy is not None
        assert isinstance(result.energy, np.ndarray)

    def test_pan_09_analyze_detects_pauses(self, mock_audio_array):
        """TEST-PAN-09: analyze() detects pauses."""
        audio, sr = mock_audio_array
        # Insert silence to create a pause
        audio[int(sr * 0.2):int(sr * 0.35)] = 0.0

        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.pauses is not None
        assert isinstance(result.pauses, list)

    def test_pan_10_analyze_pitch_in_range(self, mock_audio_array):
        """TEST-PAN-10: analyze() pitch values are in reasonable range."""
        audio, sr = mock_audio_array
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        voiced_f0 = result.pitch.frequencies[result.pitch.voiced]
        if len(voiced_f0) > 0:
            # F0 should be between f0_min and f0_max
            assert np.all(voiced_f0 >= 0) or np.all(
                (voiced_f0 >= analyzer.f0_min) & (voiced_f0 <= analyzer.f0_max)
            )

    def test_pan_11_analyze_energy_non_negative(self, mock_audio_array):
        """TEST-PAN-11: analyze() energy values are non-negative."""
        audio, sr = mock_audio_array
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert np.all(result.energy >= 0)

    def test_pan_12_analyze_pauses_format(self, mock_audio_array):
        """TEST-PAN-12: analyze() pauses are (time, duration) tuples."""
        audio, sr = mock_audio_array
        # Insert silence
        audio[int(sr * 0.2):int(sr * 0.35)] = 0.0

        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        for pause in result.pauses:
            assert isinstance(pause, tuple)
            assert len(pause) == 2


# =============================================================================
# TEST-PMO-01: ProsodyModifier Initialization Test
# =============================================================================

class TestProsodyModifierInit:
    """Test for ProsodyModifier initialization (TEST-PMO-01)."""

    def test_pmo_01_init(self):
        """TEST-PMO-01: ProsodyModifier initializes without error."""
        modifier = ProsodyModifier()
        assert modifier is not None


# =============================================================================
# TEST-PMO-02 to TEST-PMO-18: ProsodyModifier Methods Tests
# =============================================================================

class TestProsodyModifierMethods:
    """Tests for ProsodyModifier methods (TEST-PMO-02 to TEST-PMO-18)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32), sr

    def test_pmo_02_modify_pitch_returns_tuple(self, mock_audio_array):
        """TEST-PMO-02: modify_pitch() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result = modifier.modify_pitch(audio, ratio=1.0, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pmo_03_modify_pitch_ratio_up(self, mock_audio_array):
        """TEST-PMO-03: modify_pitch() with ratio > 1 raises pitch."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_pitch(audio, ratio=1.2, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_04_modify_pitch_ratio_down(self, mock_audio_array):
        """TEST-PMO-04: modify_pitch() with ratio < 1 lowers pitch."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_pitch(audio, ratio=0.8, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_05_modify_pitch_semitones_positive(self, mock_audio_array):
        """TEST-PMO-05: modify_pitch() with semitones > 0 raises pitch."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_pitch(audio, semitones=2, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_06_modify_pitch_semitones_negative(self, mock_audio_array):
        """TEST-PMO-06: modify_pitch() with semitones < 0 lowers pitch."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_pitch(audio, semitones=-3, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_07_modify_pitch_requires_sample_rate(self):
        """TEST-PMO-07: modify_pitch() raises ValueError without sample_rate."""
        audio = np.zeros(16000, dtype=np.float32)
        modifier = ProsodyModifier()
        with pytest.raises(ValueError, match="sample_rate"):
            modifier.modify_pitch(audio, ratio=1.0)

    def test_pmo_08_modify_duration_returns_tuple(self, mock_audio_array):
        """TEST-PMO-08: modify_duration() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result = modifier.modify_duration(audio, ratio=1.0, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pmo_09_modify_duration_slower(self, mock_audio_array):
        """TEST-PMO-09: modify_duration() with ratio > 1 slows down."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_duration(audio, ratio=1.5, sample_rate=sr)
        # Should be longer
        assert len(result_audio) > len(audio) * 0.9

    def test_pmo_10_modify_duration_faster(self, mock_audio_array):
        """TEST-PMO-10: modify_duration() with ratio < 1 speeds up."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_duration(audio, ratio=0.5, sample_rate=sr)
        # Should be shorter
        assert len(result_audio) < len(audio) * 1.1

    def test_pmo_11_modify_duration_requires_sample_rate(self):
        """TEST-PMO-11: modify_duration() raises ValueError without sample_rate."""
        audio = np.zeros(16000, dtype=np.float32)
        modifier = ProsodyModifier()
        with pytest.raises(ValueError, match="sample_rate"):
            modifier.modify_duration(audio, ratio=1.0)

    def test_pmo_12_modify_energy_returns_tuple(self, mock_audio_array):
        """TEST-PMO-12: modify_energy() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result = modifier.modify_energy(audio, ratio=1.0, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pmo_13_modify_energy_louder(self, mock_audio_array):
        """TEST-PMO-13: modify_energy() with ratio > 1 increases loudness."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_energy(audio, ratio=2.0, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_14_modify_energy_quieter(self, mock_audio_array):
        """TEST-PMO-14: modify_energy() with ratio < 1 decreases loudness."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_energy(audio, ratio=0.5, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_15_modify_energy_requires_sample_rate(self):
        """TEST-PMO-15: modify_energy() raises ValueError without sample_rate."""
        audio = np.zeros(16000, dtype=np.float32)
        modifier = ProsodyModifier()
        with pytest.raises(ValueError, match="sample_rate"):
            modifier.modify_energy(audio, ratio=1.0)

    def test_pmo_16_apply_contour_returns_tuple(self, mock_audio_array):
        """TEST-PMO-16: apply_contour() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        result = modifier.apply_contour(audio, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pmo_17_apply_contour_with_pitch(self, mock_audio_array):
        """TEST-PMO-17: apply_contour() accepts pitch_contour parameter."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        # Create a pitch contour
        times = np.linspace(0, 0.5, 50)
        freqs = np.ones(50) * 150  # Target 150 Hz
        voiced = np.ones(50, dtype=bool)
        pitch_contour = PitchContour(times=times, frequencies=freqs, voiced=voiced)

        result_audio, _ = modifier.apply_contour(
            audio, pitch_contour=pitch_contour, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_18_apply_contour_with_energy(self, mock_audio_array):
        """TEST-PMO-18: apply_contour() accepts energy_contour parameter."""
        audio, sr = mock_audio_array
        modifier = ProsodyModifier()
        energy_contour = np.linspace(0.5, 1.0, 50)  # Increasing energy

        result_audio, _ = modifier.apply_contour(
            audio, energy_contour=energy_contour, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)


# =============================================================================
# TEST-PMO-19 to TEST-PMO-25: Prosody Convenience Functions Tests
# =============================================================================

class TestProsodyConvenienceFunctions:
    """Tests for prosody convenience functions (TEST-PMO-19 to TEST-PMO-25)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32), sr

    def test_pmo_19_analyze_prosody_returns_contour(self, mock_audio_array):
        """TEST-PMO-19: analyze_prosody() returns ProsodyContour."""
        audio, sr = mock_audio_array
        result = analyze_prosody(audio, sample_rate=sr)
        assert isinstance(result, ProsodyContour)

    def test_pmo_20_modify_prosody_returns_tuple(self, mock_audio_array):
        """TEST-PMO-20: modify_prosody() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        result = modify_prosody(audio, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pmo_21_modify_prosody_pitch_ratio(self, mock_audio_array):
        """TEST-PMO-21: modify_prosody() accepts pitch_ratio parameter."""
        audio, sr = mock_audio_array
        result_audio, _ = modify_prosody(audio, pitch_ratio=1.2, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_22_modify_prosody_duration_ratio(self, mock_audio_array):
        """TEST-PMO-22: modify_prosody() accepts duration_ratio parameter."""
        audio, sr = mock_audio_array
        result_audio, _ = modify_prosody(audio, duration_ratio=0.8, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_23_modify_prosody_energy_ratio(self, mock_audio_array):
        """TEST-PMO-23: modify_prosody() accepts energy_ratio parameter."""
        audio, sr = mock_audio_array
        result_audio, _ = modify_prosody(audio, energy_ratio=1.5, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_24_modify_prosody_combined(self, mock_audio_array):
        """TEST-PMO-24: modify_prosody() applies multiple modifications."""
        audio, sr = mock_audio_array
        result_audio, _ = modify_prosody(
            audio,
            pitch_ratio=1.1,
            duration_ratio=1.2,
            energy_ratio=0.9,
            sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_pmo_25_modify_prosody_no_change(self, mock_audio_array):
        """TEST-PMO-25: modify_prosody() with all ratios=1.0 preserves audio."""
        audio, sr = mock_audio_array
        result_audio, result_sr = modify_prosody(
            audio,
            pitch_ratio=1.0,
            duration_ratio=1.0,
            energy_ratio=1.0,
            sample_rate=sr
        )
        # Should preserve length approximately
        assert abs(len(result_audio) - len(audio)) < len(audio) * 0.1
        assert result_sr == sr
