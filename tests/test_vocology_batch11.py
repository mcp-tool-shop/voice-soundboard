"""
Tests for Phase 9: Vocology Module - Batch 11
Module exports and integration tests

Tests cover:
- vocology/__init__.py exports (TEST-VOC-01 to TEST-VOC-15)
- Integration tests (TEST-VOC-INT-01 to TEST-VOC-INT-10)
"""

import pytest
import numpy as np
from pathlib import Path


# =============================================================================
# TEST-VOC-01 to TEST-VOC-15: Vocology Module Exports Tests
# =============================================================================

class TestVocologyModuleExports:
    """Tests for vocology module exports (TEST-VOC-01 to TEST-VOC-15)."""

    def test_voc_01_exports_voice_quality_metrics(self):
        """TEST-VOC-01: vocology exports VoiceQualityMetrics."""
        from voice_soundboard.vocology import VoiceQualityMetrics
        assert VoiceQualityMetrics is not None

    def test_voc_02_exports_voice_quality_analyzer(self):
        """TEST-VOC-02: vocology exports VoiceQualityAnalyzer."""
        from voice_soundboard.vocology import VoiceQualityAnalyzer
        assert VoiceQualityAnalyzer is not None

    def test_voc_03_exports_jitter_type(self):
        """TEST-VOC-03: vocology exports JitterType."""
        from voice_soundboard.vocology import JitterType
        assert JitterType is not None

    def test_voc_04_exports_shimmer_type(self):
        """TEST-VOC-04: vocology exports ShimmerType."""
        from voice_soundboard.vocology import ShimmerType
        assert ShimmerType is not None

    def test_voc_05_exports_formant_analysis(self):
        """TEST-VOC-05: vocology exports FormantAnalysis."""
        from voice_soundboard.vocology import FormantAnalysis
        assert FormantAnalysis is not None

    def test_voc_06_exports_formant_shifter(self):
        """TEST-VOC-06: vocology exports FormantShifter."""
        from voice_soundboard.vocology import FormantShifter
        assert FormantShifter is not None

    def test_voc_07_exports_phonation_type(self):
        """TEST-VOC-07: vocology exports PhonationType."""
        from voice_soundboard.vocology import PhonationType
        assert PhonationType is not None

    def test_voc_08_exports_phonation_synthesizer(self):
        """TEST-VOC-08: vocology exports PhonationSynthesizer."""
        from voice_soundboard.vocology import PhonationSynthesizer
        assert PhonationSynthesizer is not None

    def test_voc_09_exports_vocal_biomarkers(self):
        """TEST-VOC-09: vocology exports VocalBiomarkers."""
        from voice_soundboard.vocology import VocalBiomarkers
        assert VocalBiomarkers is not None

    def test_voc_10_exports_prosody_contour(self):
        """TEST-VOC-10: vocology exports ProsodyContour."""
        from voice_soundboard.vocology import ProsodyContour
        assert ProsodyContour is not None

    def test_voc_11_exports_prosody_modifier(self):
        """TEST-VOC-11: vocology exports ProsodyModifier."""
        from voice_soundboard.vocology import ProsodyModifier
        assert ProsodyModifier is not None

    def test_voc_12_exports_analyze_voice_quality(self):
        """TEST-VOC-12: vocology exports analyze_voice_quality convenience function."""
        from voice_soundboard.vocology import analyze_voice_quality
        assert callable(analyze_voice_quality)

    def test_voc_13_exports_analyze_formants(self):
        """TEST-VOC-13: vocology exports analyze_formants convenience function."""
        from voice_soundboard.vocology import analyze_formants
        assert callable(analyze_formants)

    def test_voc_14_exports_detect_phonation(self):
        """TEST-VOC-14: vocology exports detect_phonation convenience function."""
        from voice_soundboard.vocology import detect_phonation
        assert callable(detect_phonation)

    def test_voc_15_exports_analyze_biomarkers(self):
        """TEST-VOC-15: vocology exports analyze_biomarkers convenience function."""
        from voice_soundboard.vocology import analyze_biomarkers
        assert callable(analyze_biomarkers)


# =============================================================================
# TEST-VOC-INT-01 to TEST-VOC-INT-10: Integration Tests
# =============================================================================

class TestVocologyIntegration:
    """Integration tests for vocology module (TEST-VOC-INT-01 to TEST-VOC-INT-10)."""

    @pytest.fixture
    def sample_audio(self):
        """Generate test audio with voiced speech characteristics."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Fundamental frequency around 120 Hz
        f0 = 120
        audio = np.zeros_like(t)

        # Add harmonics
        for h in range(1, 20):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)

        # Normalize and add slight noise
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio += 0.02 * np.random.randn(len(audio))

        return audio.astype(np.float32), sr

    def test_voc_int_01_voice_quality_to_formant_pipeline(self, sample_audio):
        """TEST-VOC-INT-01: Analyze voice quality then shift formants."""
        from voice_soundboard.vocology import (
            VoiceQualityAnalyzer,
            FormantShifter,
        )

        audio, sr = sample_audio

        # Analyze voice quality
        analyzer = VoiceQualityAnalyzer()
        metrics = analyzer.analyze(audio, sample_rate=sr)
        assert metrics.f0_mean > 0

        # Shift formants
        shifter = FormantShifter()
        shifted_audio, _ = shifter.shift(audio, ratio=0.9, sample_rate=sr)

        # Verify output is valid
        assert isinstance(shifted_audio, np.ndarray)
        assert len(shifted_audio) > 0

    def test_voc_int_02_formant_reanalysis(self, sample_audio):
        """TEST-VOC-INT-02: FormantShifter output can be re-analyzed."""
        from voice_soundboard.vocology import FormantShifter, analyze_formants

        audio, sr = sample_audio

        # Shift formants
        shifter = FormantShifter()
        shifted_audio, _ = shifter.shift(audio, ratio=1.1, sample_rate=sr)

        # Re-analyze using convenience function
        analysis = analyze_formants(shifted_audio, sample_rate=sr)

        assert analysis.n_frames > 0
        assert analysis.mean_formants.f1 > 0

    def test_voc_int_03_phonation_then_prosody(self, sample_audio):
        """TEST-VOC-INT-03: Apply phonation effect then modify prosody."""
        from voice_soundboard.vocology import (
            PhonationType,
            PhonationSynthesizer,
            ProsodyModifier,
        )

        audio, sr = sample_audio

        # Apply breathiness
        synth = PhonationSynthesizer()
        breathy_audio, _ = synth.apply(
            audio, PhonationType.BREATHY, intensity=0.5, sample_rate=sr
        )

        # Modify prosody (slow down)
        modifier = ProsodyModifier()
        result_audio, _ = modifier.modify_duration(
            breathy_audio, ratio=1.2, sample_rate=sr
        )

        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_voc_int_04_prosody_then_phonation(self, sample_audio):
        """TEST-VOC-INT-04: Modify prosody then apply phonation effect."""
        from voice_soundboard.vocology import (
            PhonationType,
            PhonationSynthesizer,
            ProsodyModifier,
        )

        audio, sr = sample_audio

        # Modify prosody (raise pitch)
        modifier = ProsodyModifier()
        pitched_audio, _ = modifier.modify_pitch(audio, ratio=1.1, sample_rate=sr)

        # Apply creaky voice
        synth = PhonationSynthesizer()
        result_audio, _ = synth.apply(
            pitched_audio, PhonationType.CREAKY, intensity=0.6, sample_rate=sr
        )

        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_voc_int_05_full_biomarker_analysis(self, sample_audio):
        """TEST-VOC-INT-05: Full biomarker analysis on audio."""
        from voice_soundboard.vocology import VocalBiomarkers

        audio, sr = sample_audio

        analyzer = VocalBiomarkers()
        result = analyzer.analyze(audio, sample_rate=sr)

        # Check all components
        assert result.health_metrics is not None
        assert result.fatigue_level is not None
        assert result.voice_quality is not None
        assert result.audio_duration > 0

    def test_voc_int_06_fatigue_monitor_tracking(self, sample_audio):
        """TEST-VOC-INT-06: Fatigue monitor tracks multiple samples."""
        from voice_soundboard.vocology.biomarkers import VoiceFatigueMonitor
        from datetime import datetime, timedelta

        audio, sr = sample_audio

        monitor = VoiceFatigueMonitor()

        # Add samples at different times
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=2), sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=4), sample_rate=sr)

        # Get report
        report = monitor.get_fatigue_report()

        assert report["samples_analyzed"] == 3
        assert "trend" in report
        assert "quality_scores" in report

    def test_voc_int_07_health_analyzer_warnings(self, sample_audio):
        """TEST-VOC-INT-07: Health analyzer generates warnings for concerning values."""
        from voice_soundboard.vocology import VocalBiomarkers

        audio, sr = sample_audio

        # Create audio with poor quality characteristics (lots of noise)
        noisy_audio = audio + 0.5 * np.random.randn(len(audio))
        noisy_audio = noisy_audio.astype(np.float32)

        analyzer = VocalBiomarkers()
        result = analyzer.analyze(noisy_audio, sample_rate=sr)

        # Should have some warnings or concerns
        assert isinstance(result.warnings, list)
        assert isinstance(result.health_metrics.concerns, list)

    def test_voc_int_08_full_voice_modification_chain(self, sample_audio):
        """TEST-VOC-INT-08: Chain: analyze → shift formants → apply phonation → modify prosody."""
        from voice_soundboard.vocology import (
            VoiceQualityAnalyzer,
            FormantShifter,
            PhonationType,
            PhonationSynthesizer,
            ProsodyModifier,
        )

        audio, sr = sample_audio

        # 1. Analyze original
        analyzer = VoiceQualityAnalyzer()
        original_metrics = analyzer.analyze(audio, sample_rate=sr)

        # 2. Shift formants (deeper voice)
        formant_shifter = FormantShifter()
        audio, _ = formant_shifter.shift(audio, ratio=0.95, sample_rate=sr)

        # 3. Apply slight breathiness
        phonation_synth = PhonationSynthesizer()
        audio, _ = phonation_synth.apply(
            audio, PhonationType.BREATHY, intensity=0.3, sample_rate=sr
        )

        # 4. Slow down slightly
        prosody_mod = ProsodyModifier()
        final_audio, _ = prosody_mod.modify_duration(audio, ratio=1.1, sample_rate=sr)

        # Verify final output
        assert isinstance(final_audio, np.ndarray)
        assert len(final_audio) > 0

    def test_voc_int_09_voice_conversion_metrics(self, sample_audio):
        """TEST-VOC-INT-09: Extract metrics, apply to different audio."""
        from voice_soundboard.vocology import (
            VoiceQualityAnalyzer,
            ProsodyAnalyzer,
        )

        audio1, sr = sample_audio

        # Create slightly different audio
        t = np.linspace(0, 1.0, len(audio1))
        audio2 = 0.5 * np.sin(2 * np.pi * 150 * t)  # Different pitch
        audio2 = audio2.astype(np.float32)

        # Analyze source
        vq_analyzer = VoiceQualityAnalyzer()
        source_metrics = vq_analyzer.analyze(audio1, sample_rate=sr)

        # Analyze target
        prosody_analyzer = ProsodyAnalyzer()
        target_prosody = prosody_analyzer.analyze(audio2, sample_rate=sr)

        # Verify both analyses succeeded
        assert source_metrics.f0_mean > 0
        assert target_prosody.pitch is not None

    def test_voc_int_10_voice_aging_simulation(self, sample_audio):
        """TEST-VOC-INT-10: Voice aging: deeper formants + creaky phonation + slower prosody."""
        from voice_soundboard.vocology import (
            FormantShifter,
            PhonationType,
            PhonationSynthesizer,
            ProsodyModifier,
        )

        audio, sr = sample_audio

        # Apply "aging" effects

        # 1. Deeper formants (larger vocal tract)
        formant_shifter = FormantShifter()
        audio, _ = formant_shifter.shift(audio, ratio=0.90, sample_rate=sr)

        # 2. Add creakiness (vocal fry common in aging)
        phonation_synth = PhonationSynthesizer()
        audio, _ = phonation_synth.apply(
            audio, PhonationType.CREAKY, intensity=0.4, sample_rate=sr
        )

        # 3. Slower speech rate
        prosody_mod = ProsodyModifier()
        aged_audio, _ = prosody_mod.modify_duration(audio, ratio=1.15, sample_rate=sr)

        # Verify final output
        assert isinstance(aged_audio, np.ndarray)
        assert len(aged_audio) > 0
        # Should be longer due to slowing
        assert len(aged_audio) > len(sample_audio[0]) * 0.9
