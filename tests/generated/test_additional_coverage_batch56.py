"""
Test Additional Coverage Batch 56: Vocology Formants Tests

Tests for:
- FormantFrequencies dataclass
- FormantAnalysis dataclass
- FormantAnalyzer class
- FormantShifter class
- Convenience functions (analyze_formants, shift_formants)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== FormantFrequencies Tests ==============

class TestFormantFrequencies:
    """Tests for FormantFrequencies dataclass."""

    def test_formant_frequencies_creation(self):
        """Test FormantFrequencies basic creation."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(
            f1=500.0,
            f2=1500.0,
            f3=2500.0,
            f4=3500.0
        )
        assert formants.f1 == 500.0
        assert formants.f2 == 1500.0
        assert formants.f3 == 2500.0
        assert formants.f4 == 3500.0

    def test_formant_frequencies_with_f5(self):
        """Test FormantFrequencies with f5."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0
        )
        assert formants.f5 == 4500.0

    def test_formant_frequencies_f5_default_none(self):
        """Test FormantFrequencies f5 defaults to None."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0)
        assert formants.f5 is None

    def test_formant_frequencies_as_list_four(self):
        """Test FormantFrequencies.as_list with 4 formants."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0)
        assert formants.as_list == [500.0, 1500.0, 2500.0, 3500.0]

    def test_formant_frequencies_as_list_five(self):
        """Test FormantFrequencies.as_list with 5 formants."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0
        )
        assert formants.as_list == [500.0, 1500.0, 2500.0, 3500.0, 4500.0]

    def test_formant_frequencies_singer_formant_present_true(self):
        """Test singer_formant_present returns True for clustered formants."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        # Singer's formant: F3, F4, F5 cluster around 3000 Hz
        formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2900.0, f4=3000.0, f5=3100.0
        )
        assert formants.singer_formant_present() is True

    def test_formant_frequencies_singer_formant_present_false_no_f5(self):
        """Test singer_formant_present returns False without f5."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0)
        assert formants.singer_formant_present() is False

    def test_formant_frequencies_singer_formant_present_false_spread(self):
        """Test singer_formant_present returns False for spread formants."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        # Formants not clustered enough
        formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0
        )
        assert formants.singer_formant_present() is False

    def test_formant_frequencies_with_bandwidths(self):
        """Test FormantFrequencies with bandwidths."""
        from voice_soundboard.vocology.formants import FormantFrequencies
        formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0,
            bandwidths=[100.0, 150.0, 200.0, 250.0]
        )
        assert len(formants.bandwidths) == 4


# ============== FormantAnalysis Tests ==============

class TestFormantAnalysis:
    """Tests for FormantAnalysis dataclass."""

    def test_formant_analysis_creation(self):
        """Test FormantAnalysis basic creation."""
        from voice_soundboard.vocology.formants import FormantAnalysis, FormantFrequencies
        formants_array = np.array([
            [500.0, 1500.0, 2500.0, 3500.0],
            [510.0, 1520.0, 2480.0, 3520.0]
        ])
        mean_formants = FormantFrequencies(f1=505.0, f2=1510.0, f3=2490.0, f4=3510.0)
        analysis = FormantAnalysis(
            formants=formants_array,
            mean_formants=mean_formants,
            std_formants=[5.0, 10.0, 10.0, 10.0],
            sample_rate=24000,
            hop_length=240
        )
        assert analysis.sample_rate == 24000
        assert analysis.hop_length == 240

    def test_formant_analysis_n_frames(self):
        """Test FormantAnalysis.n_frames property."""
        from voice_soundboard.vocology.formants import FormantAnalysis, FormantFrequencies
        formants_array = np.zeros((10, 4))
        mean_formants = FormantFrequencies(f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0)
        analysis = FormantAnalysis(
            formants=formants_array, mean_formants=mean_formants,
            std_formants=[0.0]*4, sample_rate=24000, hop_length=240
        )
        assert analysis.n_frames == 10

    def test_formant_analysis_n_formants(self):
        """Test FormantAnalysis.n_formants property."""
        from voice_soundboard.vocology.formants import FormantAnalysis, FormantFrequencies
        formants_array = np.zeros((10, 5))
        mean_formants = FormantFrequencies(f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0)
        analysis = FormantAnalysis(
            formants=formants_array, mean_formants=mean_formants,
            std_formants=[0.0]*5, sample_rate=24000, hop_length=240
        )
        assert analysis.n_formants == 5

    def test_formant_analysis_get_frame(self):
        """Test FormantAnalysis.get_frame method."""
        from voice_soundboard.vocology.formants import FormantAnalysis, FormantFrequencies
        formants_array = np.array([
            [500.0, 1500.0, 2500.0, 3500.0],
            [600.0, 1600.0, 2600.0, 3600.0]
        ])
        mean_formants = FormantFrequencies(f1=550.0, f2=1550.0, f3=2550.0, f4=3550.0)
        analysis = FormantAnalysis(
            formants=formants_array, mean_formants=mean_formants,
            std_formants=[50.0]*4, sample_rate=24000, hop_length=240
        )
        frame0 = analysis.get_frame(0)
        assert frame0.f1 == 500.0
        assert frame0.f4 == 3500.0

    def test_formant_analysis_get_frame_with_f5(self):
        """Test FormantAnalysis.get_frame with 5 formants."""
        from voice_soundboard.vocology.formants import FormantAnalysis, FormantFrequencies
        formants_array = np.array([
            [500.0, 1500.0, 2500.0, 3500.0, 4500.0]
        ])
        mean_formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0
        )
        analysis = FormantAnalysis(
            formants=formants_array, mean_formants=mean_formants,
            std_formants=[0.0]*5, sample_rate=24000, hop_length=240
        )
        frame = analysis.get_frame(0)
        assert frame.f5 == 4500.0


# ============== FormantAnalyzer Tests ==============

class TestFormantAnalyzer:
    """Tests for FormantAnalyzer class."""

    def test_formant_analyzer_init(self):
        """Test FormantAnalyzer initialization."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer(n_formants=5, lpc_order=12)
        assert analyzer.n_formants == 5
        assert analyzer.lpc_order == 12

    def test_formant_analyzer_default_lpc_order(self):
        """Test FormantAnalyzer default LPC order calculation."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer(n_formants=5)
        # Default: 2 * n_formants + 2 = 12
        assert analyzer.lpc_order == 12

    def test_formant_analyzer_default_params(self):
        """Test FormantAnalyzer default parameters."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer()
        assert analyzer.n_formants == 5
        assert analyzer.pre_emphasis == 0.97
        assert analyzer.frame_length == 0.025
        assert analyzer.hop_length == 0.010

    def test_formant_analyzer_analyze_requires_sample_rate(self):
        """Test FormantAnalyzer.analyze raises error without sample_rate."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer()
        audio = np.random.randn(24000).astype(np.float32)

        with pytest.raises(ValueError, match="sample_rate required"):
            analyzer.analyze(audio)

    def test_formant_analyzer_compute_lpc(self):
        """Test FormantAnalyzer._compute_lpc method."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer()
        frame = np.random.randn(600).astype(np.float32)
        lpc_coeffs = analyzer._compute_lpc(frame, order=12)
        assert len(lpc_coeffs) == 13  # order + 1
        assert lpc_coeffs[0] == 1.0

    def test_formant_analyzer_lpc_formants(self):
        """Test FormantAnalyzer._lpc_formants method."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer(n_formants=4)
        # Create a simple signal
        t = np.linspace(0, 0.025, 600)
        frame = (np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
        formants = analyzer._lpc_formants(frame, sr=24000)
        assert len(formants) <= 4  # May extract fewer formants

    @patch('voice_soundboard.vocology.formants.FormantAnalyzer._load_audio')
    def test_formant_analyzer_analyze(self, mock_load):
        """Test FormantAnalyzer.analyze method."""
        from voice_soundboard.vocology.formants import FormantAnalyzer
        analyzer = FormantAnalyzer(n_formants=4)
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        result = analyzer.analyze(audio, sample_rate=24000)
        assert result is not None
        assert result.sample_rate == 24000
        assert result.n_formants == 4


# ============== FormantShifter Tests ==============

class TestFormantShifter:
    """Tests for FormantShifter class."""

    def test_formant_shifter_init_default(self):
        """Test FormantShifter default initialization."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter()
        assert shifter.method == "psola"

    def test_formant_shifter_init_lpc(self):
        """Test FormantShifter with LPC method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter(method="lpc")
        assert shifter.method == "lpc"

    def test_formant_shifter_init_phase_vocoder(self):
        """Test FormantShifter with phase vocoder method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter(method="phase_vocoder")
        assert shifter.method == "phase_vocoder"

    def test_formant_shifter_shift_requires_sample_rate(self):
        """Test FormantShifter.shift raises error without sample_rate."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter()
        audio = np.random.randn(24000).astype(np.float32)

        with pytest.raises(ValueError, match="sample_rate required"):
            shifter.shift(audio, ratio=1.2)

    @patch('voice_soundboard.vocology.formants.FormantShifter._shift_psola')
    def test_formant_shifter_shift_psola(self, mock_psola):
        """Test FormantShifter.shift with PSOLA method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter(method="psola")
        audio = np.random.randn(24000).astype(np.float32)
        mock_psola.return_value = (audio, 24000)

        result, sr = shifter.shift(audio, ratio=1.2, sample_rate=24000)
        mock_psola.assert_called_once()

    @patch('voice_soundboard.vocology.formants.FormantShifter._shift_lpc')
    def test_formant_shifter_shift_lpc(self, mock_lpc):
        """Test FormantShifter.shift with LPC method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter(method="lpc")
        audio = np.random.randn(24000).astype(np.float32)
        mock_lpc.return_value = (audio, 24000)

        result, sr = shifter.shift(audio, ratio=0.9, sample_rate=24000)
        mock_lpc.assert_called_once()

    @patch('voice_soundboard.vocology.formants.FormantShifter._shift_phase_vocoder')
    def test_formant_shifter_shift_phase_vocoder(self, mock_pv):
        """Test FormantShifter.shift with phase vocoder method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter(method="phase_vocoder")
        audio = np.random.randn(24000).astype(np.float32)
        mock_pv.return_value = (audio, 24000)

        result, sr = shifter.shift(audio, ratio=1.1, sample_rate=24000)
        mock_pv.assert_called_once()

    @patch('scipy.signal.resample')
    def test_formant_shifter_simple_resample(self, mock_resample):
        """Test FormantShifter._simple_resample method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter()
        audio = np.random.randn(24000).astype(np.float32)
        mock_resample.return_value = audio[:20000]

        result, sr = shifter._simple_resample(audio, 24000, ratio=1.2)
        mock_resample.assert_called_once()
        assert sr == 24000

    def test_formant_shifter_shift_selective(self):
        """Test FormantShifter.shift_selective method."""
        from voice_soundboard.vocology.formants import FormantShifter
        shifter = FormantShifter()
        audio = np.random.randn(24000).astype(np.float32)

        with patch.object(shifter, 'shift', return_value=(audio, 24000)) as mock_shift:
            result, sr = shifter.shift_selective(
                audio, f1_ratio=0.9, f2_ratio=1.0, f3_ratio=1.1, sample_rate=24000
            )
            # Average ratio: (0.9 + 1.0 + 1.1) / 3 = 1.0
            mock_shift.assert_called_once()


# ============== Convenience Functions Tests ==============

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch('voice_soundboard.vocology.formants.FormantAnalyzer')
    def test_analyze_formants(self, mock_analyzer_class):
        """Test analyze_formants convenience function."""
        from voice_soundboard.vocology.formants import analyze_formants, FormantAnalysis, FormantFrequencies

        mock_analysis = Mock(spec=FormantAnalysis)
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = mock_analysis
        mock_analyzer_class.return_value = mock_analyzer

        audio = np.random.randn(24000).astype(np.float32)
        result = analyze_formants(audio, sample_rate=24000)

        mock_analyzer_class.assert_called_once_with(n_formants=5)
        mock_analyzer.analyze.assert_called_once_with(audio, 24000)

    @patch('voice_soundboard.vocology.formants.FormantShifter')
    def test_shift_formants(self, mock_shifter_class):
        """Test shift_formants convenience function."""
        from voice_soundboard.vocology.formants import shift_formants

        audio = np.random.randn(24000).astype(np.float32)
        mock_shifter = Mock()
        mock_shifter.shift.return_value = (audio, 24000)
        mock_shifter_class.return_value = mock_shifter

        result, sr = shift_formants(audio, ratio=1.2, sample_rate=24000)

        mock_shifter_class.assert_called_once()
        mock_shifter.shift.assert_called_once_with(audio, 1.2, 24000, True)

    @patch('voice_soundboard.vocology.formants.FormantShifter')
    def test_shift_formants_no_preserve_pitch(self, mock_shifter_class):
        """Test shift_formants with preserve_pitch=False."""
        from voice_soundboard.vocology.formants import shift_formants

        audio = np.random.randn(24000).astype(np.float32)
        mock_shifter = Mock()
        mock_shifter.shift.return_value = (audio, 24000)
        mock_shifter_class.return_value = mock_shifter

        result, sr = shift_formants(audio, ratio=0.8, sample_rate=24000, preserve_pitch=False)

        mock_shifter.shift.assert_called_once_with(audio, 0.8, 24000, False)
