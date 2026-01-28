"""
Test Additional Coverage Batch 50: Vocology Rhythm Tests

Tests for:
- RhythmClass enum
- RhythmBand enum
- RhythmMetrics dataclass
- RhythmZone dataclass
- RZTAnalysis dataclass
- RhythmAnalyzer class
- RhythmModifier class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== RhythmClass Enum Tests ==============

class TestRhythmClassEnum:
    """Tests for RhythmClass enum."""

    def test_rhythm_class_stress_timed(self):
        """Test RhythmClass.STRESS_TIMED value."""
        from voice_soundboard.vocology.rhythm import RhythmClass
        assert RhythmClass.STRESS_TIMED.value == "stress_timed"

    def test_rhythm_class_syllable_timed(self):
        """Test RhythmClass.SYLLABLE_TIMED value."""
        from voice_soundboard.vocology.rhythm import RhythmClass
        assert RhythmClass.SYLLABLE_TIMED.value == "syllable_timed"

    def test_rhythm_class_mora_timed(self):
        """Test RhythmClass.MORA_TIMED value."""
        from voice_soundboard.vocology.rhythm import RhythmClass
        assert RhythmClass.MORA_TIMED.value == "mora_timed"

    def test_rhythm_class_mixed(self):
        """Test RhythmClass.MIXED value."""
        from voice_soundboard.vocology.rhythm import RhythmClass
        assert RhythmClass.MIXED.value == "mixed"


# ============== RhythmBand Enum Tests ==============

class TestRhythmBandEnum:
    """Tests for RhythmBand enum."""

    def test_rhythm_band_delta(self):
        """Test RhythmBand.DELTA for phrase rhythm."""
        from voice_soundboard.vocology.rhythm import RhythmBand
        assert RhythmBand.DELTA.value == "delta"

    def test_rhythm_band_theta(self):
        """Test RhythmBand.THETA for syllable rate."""
        from voice_soundboard.vocology.rhythm import RhythmBand
        assert RhythmBand.THETA.value == "theta"

    def test_rhythm_band_alpha(self):
        """Test RhythmBand.ALPHA for phoneme rate."""
        from voice_soundboard.vocology.rhythm import RhythmBand
        assert RhythmBand.ALPHA.value == "alpha"

    def test_rhythm_band_beta(self):
        """Test RhythmBand.BETA for fast articulation."""
        from voice_soundboard.vocology.rhythm import RhythmBand
        assert RhythmBand.BETA.value == "beta"


# ============== RhythmMetrics Tests ==============

class TestRhythmMetrics:
    """Tests for RhythmMetrics dataclass."""

    def test_rhythm_metrics_creation(self):
        """Test RhythmMetrics basic creation."""
        from voice_soundboard.vocology.rhythm import RhythmMetrics
        metrics = RhythmMetrics(
            percent_v=45.0,
            delta_v=0.05,
            delta_c=0.04,
            npvi_v=50.0,
            rpvi_c=0.03,
            varco_v=40.0,
            varco_c=35.0,
            speech_rate=5.0,
            articulation_rate=6.0
        )
        assert metrics.percent_v == 45.0
        assert metrics.speech_rate == 5.0

    def test_rhythm_metrics_stress_timed_classification(self):
        """Test RhythmMetrics classifies as stress-timed."""
        from voice_soundboard.vocology.rhythm import RhythmMetrics, RhythmClass
        metrics = RhythmMetrics(
            percent_v=40.0,  # < 45
            delta_v=0.08,
            delta_c=0.06,
            npvi_v=60.0,  # > 55
            rpvi_c=0.05,
            varco_v=50.0,
            varco_c=45.0,
            speech_rate=4.5,
            articulation_rate=5.5
        )
        assert metrics.rhythm_class == RhythmClass.STRESS_TIMED

    def test_rhythm_metrics_syllable_timed_classification(self):
        """Test RhythmMetrics classifies as syllable-timed."""
        from voice_soundboard.vocology.rhythm import RhythmMetrics, RhythmClass
        metrics = RhythmMetrics(
            percent_v=50.0,  # > 45
            delta_v=0.03,
            delta_c=0.03,
            npvi_v=40.0,  # < 45
            rpvi_c=0.02,
            varco_v=30.0,
            varco_c=25.0,
            speech_rate=5.5,
            articulation_rate=6.5
        )
        assert metrics.rhythm_class == RhythmClass.SYLLABLE_TIMED

    def test_rhythm_metrics_mora_timed_classification(self):
        """Test RhythmMetrics classifies as mora-timed."""
        from voice_soundboard.vocology.rhythm import RhythmMetrics, RhythmClass
        metrics = RhythmMetrics(
            percent_v=48.0,
            delta_v=0.02,
            delta_c=0.02,
            npvi_v=30.0,  # < 35
            rpvi_c=0.015,
            varco_v=20.0,
            varco_c=18.0,
            speech_rate=6.0,
            articulation_rate=7.0
        )
        assert metrics.rhythm_class == RhythmClass.MORA_TIMED

    def test_rhythm_metrics_to_dict(self):
        """Test RhythmMetrics.to_dict method."""
        from voice_soundboard.vocology.rhythm import RhythmMetrics
        metrics = RhythmMetrics(
            percent_v=45.0,
            delta_v=0.05,
            delta_c=0.04,
            npvi_v=50.0,
            rpvi_c=0.03,
            varco_v=40.0,
            varco_c=35.0,
            speech_rate=5.0,
            articulation_rate=6.0
        )
        d = metrics.to_dict()
        assert "percent_v" in d
        assert "rhythm_class" in d
        assert d["speech_rate"] == 5.0


# ============== RhythmZone Tests ==============

class TestRhythmZone:
    """Tests for RhythmZone dataclass."""

    def test_rhythm_zone_creation(self):
        """Test RhythmZone basic creation."""
        from voice_soundboard.vocology.rhythm import RhythmZone, RhythmBand
        zone = RhythmZone(
            start_time=0.0,
            end_time=0.5,
            dominant_frequency=5.0,
            energy=0.8,
            band=RhythmBand.THETA
        )
        assert zone.start_time == 0.0
        assert zone.end_time == 0.5
        assert zone.dominant_frequency == 5.0

    def test_rhythm_zone_duration_property(self):
        """Test RhythmZone.duration property."""
        from voice_soundboard.vocology.rhythm import RhythmZone, RhythmBand
        zone = RhythmZone(
            start_time=1.0,
            end_time=2.5,
            dominant_frequency=6.0,
            energy=0.7,
            band=RhythmBand.THETA
        )
        assert zone.duration == 1.5


# ============== RZTAnalysis Tests ==============

class TestRZTAnalysis:
    """Tests for RZTAnalysis dataclass."""

    def test_rzt_analysis_creation(self):
        """Test RZTAnalysis basic creation."""
        from voice_soundboard.vocology.rhythm import RZTAnalysis, RhythmZone, RhythmBand
        zones = [
            RhythmZone(0.0, 0.5, 5.0, 0.8, RhythmBand.THETA),
            RhythmZone(0.5, 1.0, 6.0, 0.7, RhythmBand.THETA),
        ]
        analysis = RZTAnalysis(
            zones=zones,
            envelope_spectrum=np.array([0.1, 0.5, 0.3]),
            rhythm_frequencies=[1.5, 5.5],
            phrase_rhythm=1.5,
            syllable_rhythm=5.5,
            sample_rate=24000
        )
        assert len(analysis.zones) == 2
        assert analysis.phrase_rhythm == 1.5

    def test_rzt_analysis_estimated_syllable_rate(self):
        """Test RZTAnalysis.estimated_syllable_rate property."""
        from voice_soundboard.vocology.rhythm import RZTAnalysis
        analysis = RZTAnalysis(
            zones=[],
            envelope_spectrum=np.array([]),
            rhythm_frequencies=[],
            phrase_rhythm=1.5,
            syllable_rhythm=5.5,
            sample_rate=24000
        )
        assert analysis.estimated_syllable_rate == 5.5

    def test_rzt_analysis_phrase_duration(self):
        """Test RZTAnalysis.phrase_duration property."""
        from voice_soundboard.vocology.rhythm import RZTAnalysis
        analysis = RZTAnalysis(
            zones=[],
            envelope_spectrum=np.array([]),
            rhythm_frequencies=[],
            phrase_rhythm=2.0,  # 2 Hz = 0.5 sec per phrase
            syllable_rhythm=5.0,
            sample_rate=24000
        )
        assert analysis.phrase_duration == 0.5


# ============== RhythmAnalyzer Tests ==============

class TestRhythmAnalyzer:
    """Tests for RhythmAnalyzer class."""

    def test_rhythm_analyzer_init(self):
        """Test RhythmAnalyzer initialization."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer
        analyzer = RhythmAnalyzer(sample_rate=24000)
        assert analyzer.sample_rate == 24000

    def test_rhythm_analyzer_calculate_percent_v(self):
        """Test RhythmAnalyzer._calculate_percent_v method."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer
        analyzer = RhythmAnalyzer()
        v_intervals = [0.1, 0.15, 0.12]  # Total: 0.37
        c_intervals = [0.05, 0.08, 0.1]  # Total: 0.23
        # Percent V = 100 * 0.37 / 0.60 = 61.67%
        percent_v = analyzer._calculate_percent_v(v_intervals, c_intervals)
        assert abs(percent_v - 61.67) < 1.0

    def test_rhythm_analyzer_calculate_npvi(self):
        """Test RhythmAnalyzer._calculate_npvi method."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer
        analyzer = RhythmAnalyzer()
        # Equal intervals should have low nPVI
        equal_intervals = [0.1, 0.1, 0.1, 0.1]
        npvi = analyzer._calculate_npvi(equal_intervals)
        assert npvi == 0.0

    def test_rhythm_analyzer_calculate_npvi_varied(self):
        """Test RhythmAnalyzer._calculate_npvi with varied intervals."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer
        analyzer = RhythmAnalyzer()
        # Varied intervals should have higher nPVI
        varied_intervals = [0.1, 0.2, 0.1, 0.2]
        npvi = analyzer._calculate_npvi(varied_intervals)
        assert npvi > 0.0

    def test_rhythm_analyzer_calculate_rpvi(self):
        """Test RhythmAnalyzer._calculate_rpvi method."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer
        analyzer = RhythmAnalyzer()
        intervals = [0.1, 0.2, 0.15]
        rpvi = analyzer._calculate_rpvi(intervals)
        # Expected: (|0.1-0.2| + |0.2-0.15|) / 2 = (0.1 + 0.05) / 2 = 0.075
        assert abs(rpvi - 0.075) < 0.01

    def test_rhythm_analyzer_calculate_varco(self):
        """Test RhythmAnalyzer._calculate_varco method."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer
        analyzer = RhythmAnalyzer()
        intervals = [0.1, 0.1, 0.1]  # No variation
        varco = analyzer._calculate_varco(intervals)
        assert varco == 0.0

    @patch('voice_soundboard.vocology.rhythm.RhythmAnalyzer._load_audio')
    @patch('scipy.signal.hilbert')
    @patch('scipy.signal.butter')
    @patch('scipy.signal.filtfilt')
    def test_rhythm_analyzer_extract_envelope(self, mock_filtfilt, mock_butter, mock_hilbert, mock_load):
        """Test RhythmAnalyzer._extract_envelope method."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer

        analyzer = RhythmAnalyzer(sample_rate=24000)
        audio = np.random.randn(24000).astype(np.float32)

        # Mock Hilbert transform
        mock_hilbert.return_value = audio + 1j * np.random.randn(24000)
        mock_butter.return_value = (np.array([1.0]), np.array([1.0]))
        mock_filtfilt.return_value = np.abs(audio)

        envelope = analyzer._extract_envelope(audio, 24000)
        assert len(envelope) == len(audio)

    @patch('voice_soundboard.vocology.rhythm.RhythmAnalyzer._load_audio')
    def test_rhythm_analyzer_analyze_band_energy(self, mock_load):
        """Test RhythmAnalyzer.analyze_band_energy method."""
        from voice_soundboard.vocology.rhythm import RhythmAnalyzer, RhythmBand

        analyzer = RhythmAnalyzer(sample_rate=24000)
        audio = np.random.randn(24000).astype(np.float32)

        with patch.object(analyzer, '_extract_envelope', return_value=np.abs(audio)):
            with patch.object(analyzer, '_compute_envelope_spectrum') as mock_spectrum:
                mock_spectrum.return_value = (np.random.rand(100), np.linspace(0, 50, 100))

                band_energy = analyzer.analyze_band_energy(audio, 24000)

                assert RhythmBand.DELTA in band_energy
                assert RhythmBand.THETA in band_energy


# ============== RhythmModifier Tests ==============

class TestRhythmModifier:
    """Tests for RhythmModifier class."""

    def test_rhythm_modifier_init(self):
        """Test RhythmModifier initialization."""
        from voice_soundboard.vocology.rhythm import RhythmModifier
        modifier = RhythmModifier(sample_rate=24000)
        assert modifier.sample_rate == 24000
        assert modifier.analyzer is not None

    def test_rhythm_modifier_init_creates_analyzer(self):
        """Test RhythmModifier creates internal analyzer."""
        from voice_soundboard.vocology.rhythm import RhythmModifier, RhythmAnalyzer
        modifier = RhythmModifier(sample_rate=22050)
        assert isinstance(modifier.analyzer, RhythmAnalyzer)
        assert modifier.analyzer.sample_rate == 22050

    @patch('voice_soundboard.vocology.rhythm.RhythmModifier._load_audio')
    def test_rhythm_modifier_add_variability(self, mock_load):
        """Test RhythmModifier.add_variability method."""
        from voice_soundboard.vocology.rhythm import RhythmModifier, RZTAnalysis, RhythmZone, RhythmBand

        modifier = RhythmModifier(sample_rate=24000)
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        mock_rzt = RZTAnalysis(
            zones=[RhythmZone(0.0, 0.5, 5.0, 0.8, RhythmBand.THETA)],
            envelope_spectrum=np.array([]),
            rhythm_frequencies=[],
            phrase_rhythm=1.5,
            syllable_rhythm=5.0,
            sample_rate=24000
        )

        with patch.object(modifier.analyzer, 'analyze_rzt', return_value=mock_rzt):
            with patch.object(modifier, '_create_variability_map', return_value=np.ones(len(audio))):
                with patch.object(modifier, '_apply_stretch_map', return_value=audio):
                    output, sr = modifier.add_variability(audio, amount=0.1, sample_rate=24000)
                    assert sr == 24000

    @patch('voice_soundboard.vocology.rhythm.RhythmModifier._load_audio')
    def test_rhythm_modifier_adjust_rhythm_class_same(self, mock_load):
        """Test RhythmModifier.adjust_rhythm_class when already at target."""
        from voice_soundboard.vocology.rhythm import RhythmModifier, RhythmClass, RhythmMetrics

        modifier = RhythmModifier(sample_rate=24000)
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        mock_metrics = RhythmMetrics(
            percent_v=50.0,
            delta_v=0.03,
            delta_c=0.03,
            npvi_v=40.0,
            rpvi_c=0.02,
            varco_v=30.0,
            varco_c=25.0,
            speech_rate=5.5,
            articulation_rate=6.5
        )

        with patch.object(modifier.analyzer, 'analyze_metrics', return_value=mock_metrics):
            output, sr = modifier.adjust_rhythm_class(audio, RhythmClass.SYLLABLE_TIMED, sample_rate=24000)
            # Should return original audio since already at target
            np.testing.assert_array_equal(output, audio)
