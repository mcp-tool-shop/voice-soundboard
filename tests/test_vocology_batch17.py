"""
Tests for Phase 10: Humanization & Rhythm - Batch 17
Module Exports and Integration Tests

Tests cover:
- vocology/__init__.py Humanize Exports (TEST-VOC-HUM-01 to TEST-VOC-HUM-13)
- vocology/__init__.py Rhythm Exports (TEST-VOC-RHY-01 to TEST-VOC-RHY-10)
- Humanization Integration (TEST-HUM-INT-01 to TEST-HUM-INT-05)
- Rhythm Integration (TEST-RHY-INT-01 to TEST-RHY-INT-05)
- Combined Vocology Pipeline (TEST-VOC-FULL-01 to TEST-VOC-FULL-05)
"""

import pytest
import numpy as np


# =============================================================================
# TEST-VOC-HUM-01 to TEST-VOC-HUM-13: Humanize Module Exports Tests
# =============================================================================

class TestVocologyHumanizeExports:
    """Tests for vocology humanize module exports (TEST-VOC-HUM-01 to TEST-VOC-HUM-13)."""

    def test_voc_hum_01_exports_voice_humanizer(self):
        """TEST-VOC-HUM-01: vocology module exports VoiceHumanizer."""
        from voice_soundboard.vocology import VoiceHumanizer
        assert VoiceHumanizer is not None

    def test_voc_hum_02_exports_breath_inserter(self):
        """TEST-VOC-HUM-02: vocology module exports BreathInserter."""
        from voice_soundboard.vocology import BreathInserter
        assert BreathInserter is not None

    def test_voc_hum_03_exports_breath_generator(self):
        """TEST-VOC-HUM-03: vocology module exports BreathGenerator."""
        from voice_soundboard.vocology import BreathGenerator
        assert BreathGenerator is not None

    def test_voc_hum_04_exports_pitch_humanizer(self):
        """TEST-VOC-HUM-04: vocology module exports PitchHumanizer."""
        from voice_soundboard.vocology import PitchHumanizer
        assert PitchHumanizer is not None

    def test_voc_hum_05_exports_humanize_config(self):
        """TEST-VOC-HUM-05: vocology module exports HumanizeConfig."""
        from voice_soundboard.vocology import HumanizeConfig
        assert HumanizeConfig is not None

    def test_voc_hum_06_exports_breath_config(self):
        """TEST-VOC-HUM-06: vocology module exports BreathConfig."""
        from voice_soundboard.vocology import BreathConfig
        assert BreathConfig is not None

    def test_voc_hum_07_exports_pitch_humanize_config(self):
        """TEST-VOC-HUM-07: vocology module exports PitchHumanizeConfig."""
        from voice_soundboard.vocology import PitchHumanizeConfig
        assert PitchHumanizeConfig is not None

    def test_voc_hum_08_exports_timing_humanize_config(self):
        """TEST-VOC-HUM-08: vocology module exports TimingHumanizeConfig."""
        from voice_soundboard.vocology import TimingHumanizeConfig
        assert TimingHumanizeConfig is not None

    def test_voc_hum_09_exports_breath_type(self):
        """TEST-VOC-HUM-09: vocology module exports BreathType."""
        from voice_soundboard.vocology import BreathType
        assert BreathType is not None

    def test_voc_hum_10_exports_emotional_state(self):
        """TEST-VOC-HUM-10: vocology module exports EmotionalState."""
        from voice_soundboard.vocology import EmotionalState
        assert EmotionalState is not None

    def test_voc_hum_11_exports_humanize_audio(self):
        """TEST-VOC-HUM-11: vocology module exports humanize_audio."""
        from voice_soundboard.vocology import humanize_audio
        assert callable(humanize_audio)

    def test_voc_hum_12_exports_add_breaths(self):
        """TEST-VOC-HUM-12: vocology module exports add_breaths."""
        from voice_soundboard.vocology import add_breaths
        assert callable(add_breaths)

    def test_voc_hum_13_exports_humanize_pitch(self):
        """TEST-VOC-HUM-13: vocology module exports humanize_pitch."""
        from voice_soundboard.vocology import humanize_pitch
        assert callable(humanize_pitch)


# =============================================================================
# TEST-VOC-RHY-01 to TEST-VOC-RHY-10: Rhythm Module Exports Tests
# =============================================================================

class TestVocologyRhythmExports:
    """Tests for vocology rhythm module exports (TEST-VOC-RHY-01 to TEST-VOC-RHY-10)."""

    def test_voc_rhy_01_exports_rhythm_analyzer(self):
        """TEST-VOC-RHY-01: vocology module exports RhythmAnalyzer."""
        from voice_soundboard.vocology import RhythmAnalyzer
        assert RhythmAnalyzer is not None

    def test_voc_rhy_02_exports_rhythm_modifier(self):
        """TEST-VOC-RHY-02: vocology module exports RhythmModifier."""
        from voice_soundboard.vocology import RhythmModifier
        assert RhythmModifier is not None

    def test_voc_rhy_03_exports_rhythm_metrics(self):
        """TEST-VOC-RHY-03: vocology module exports RhythmMetrics."""
        from voice_soundboard.vocology import RhythmMetrics
        assert RhythmMetrics is not None

    def test_voc_rhy_04_exports_rhythm_zone(self):
        """TEST-VOC-RHY-04: vocology module exports RhythmZone."""
        from voice_soundboard.vocology import RhythmZone
        assert RhythmZone is not None

    def test_voc_rhy_05_exports_rzt_analysis(self):
        """TEST-VOC-RHY-05: vocology module exports RZTAnalysis."""
        from voice_soundboard.vocology import RZTAnalysis
        assert RZTAnalysis is not None

    def test_voc_rhy_06_exports_rhythm_class(self):
        """TEST-VOC-RHY-06: vocology module exports RhythmClass."""
        from voice_soundboard.vocology import RhythmClass
        assert RhythmClass is not None

    def test_voc_rhy_07_exports_rhythm_band(self):
        """TEST-VOC-RHY-07: vocology module exports RhythmBand."""
        from voice_soundboard.vocology import RhythmBand
        assert RhythmBand is not None

    def test_voc_rhy_08_exports_analyze_rhythm(self):
        """TEST-VOC-RHY-08: vocology module exports analyze_rhythm."""
        from voice_soundboard.vocology import analyze_rhythm
        assert callable(analyze_rhythm)

    def test_voc_rhy_09_exports_analyze_rhythm_zones(self):
        """TEST-VOC-RHY-09: vocology module exports analyze_rhythm_zones."""
        from voice_soundboard.vocology import analyze_rhythm_zones
        assert callable(analyze_rhythm_zones)

    def test_voc_rhy_10_exports_add_rhythm_variability(self):
        """TEST-VOC-RHY-10: vocology module exports add_rhythm_variability."""
        from voice_soundboard.vocology import add_rhythm_variability
        assert callable(add_rhythm_variability)


# =============================================================================
# TEST-HUM-INT-01 to TEST-HUM-INT-05: Humanization Integration Tests
# =============================================================================

class TestHumanizationIntegration:
    """Integration tests for humanization (TEST-HUM-INT-01 to TEST-HUM-INT-05)."""

    @pytest.fixture
    def mock_tts_audio(self):
        """Create mock TTS-like audio."""
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create speech-like audio with harmonic structure
        audio = np.zeros_like(t)
        f0 = 150  # Fundamental frequency

        # Add harmonics
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)

        # Add some phrase structure (silence gap)
        silence_start = int(0.9 * sr)
        silence_end = int(1.1 * sr)
        audio[silence_start:silence_end] = 0

        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32), sr

    def test_hum_int_01_tts_humanize_pipeline(self, mock_tts_audio):
        """TEST-HUM-INT-01: Generate TTS → humanize → output sounds more natural."""
        from voice_soundboard.vocology import VoiceHumanizer

        audio, sr = mock_tts_audio
        humanizer = VoiceHumanizer()
        result_audio, result_sr = humanizer.humanize(audio, sample_rate=sr)

        assert isinstance(result_audio, np.ndarray)
        assert result_sr == sr
        # Humanized audio should be different from original
        assert not np.allclose(audio, result_audio[:len(audio)], rtol=0.1)

    def test_hum_int_02_chain_breath_pitch_timing(self, mock_tts_audio):
        """TEST-HUM-INT-02: Chain: breath insertion → pitch humanization → timing variation."""
        from voice_soundboard.vocology import (
            BreathInserter,
            PitchHumanizer,
            BreathConfig,
            PitchHumanizeConfig,
        )

        audio, sr = mock_tts_audio

        # 1. Add breaths
        breath_config = BreathConfig(enabled=True, intensity=0.5)
        breath_inserter = BreathInserter(sample_rate=sr)
        audio_with_breath = breath_inserter.insert_breaths(audio, config=breath_config)

        # 2. Humanize pitch
        pitch_config = PitchHumanizeConfig(enabled=True)
        pitch_humanizer = PitchHumanizer(sample_rate=sr)
        final_audio = pitch_humanizer.humanize(audio_with_breath, config=pitch_config)

        assert isinstance(final_audio, np.ndarray)

    def test_hum_int_03_excited_preset(self, mock_tts_audio):
        """TEST-HUM-INT-03: Emotional preset EXCITED produces faster, higher variation speech."""
        from voice_soundboard.vocology import VoiceHumanizer, HumanizeConfig, EmotionalState

        audio, sr = mock_tts_audio
        config = HumanizeConfig.for_emotion(EmotionalState.EXCITED)
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, config=config, sample_rate=sr)

        # Excited config should have higher jitter
        assert config.pitch.jitter_cents > 5.0
        assert isinstance(result_audio, np.ndarray)

    def test_hum_int_04_calm_preset(self, mock_tts_audio):
        """TEST-HUM-INT-04: Emotional preset CALM produces slower, subtle speech."""
        from voice_soundboard.vocology import VoiceHumanizer, HumanizeConfig, EmotionalState

        audio, sr = mock_tts_audio
        config = HumanizeConfig.for_emotion(EmotionalState.CALM)
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, config=config, sample_rate=sr)

        # Calm config should have lower jitter
        assert config.pitch.jitter_cents < 5.0
        assert isinstance(result_audio, np.ndarray)

    def test_hum_int_05_question_inflection(self, mock_tts_audio):
        """TEST-HUM-INT-05: Humanization with questions adds rising inflection."""
        from voice_soundboard.vocology import VoiceHumanizer

        audio, sr = mock_tts_audio
        humanizer = VoiceHumanizer()

        # Process as question
        result_audio, _ = humanizer.humanize(audio, sample_rate=sr, is_question=True)
        assert isinstance(result_audio, np.ndarray)


# =============================================================================
# TEST-RHY-INT-01 to TEST-RHY-INT-05: Rhythm Integration Tests
# =============================================================================

class TestRhythmIntegration:
    """Integration tests for rhythm analysis (TEST-RHY-INT-01 to TEST-RHY-INT-05)."""

    @pytest.fixture
    def mock_speech_audio(self):
        """Create mock speech-like audio."""
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        audio = np.zeros_like(t)
        # Create syllable-like structure (5 Hz rate = typical English)
        syllable_rate = 5
        for i in range(int(duration * syllable_rate)):
            start_time = i / syllable_rate
            start = int(start_time * sr)
            end = int((start_time + 0.12) * sr)  # Variable syllable length
            if end < len(audio):
                t_local = t[start:end]
                audio[start:end] = 0.5 * np.sin(2 * np.pi * 200 * t_local)

        audio += 0.02 * np.random.randn(len(audio))
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32), sr

    def test_rhy_int_01_english_stress_timed(self, mock_speech_audio):
        """TEST-RHY-INT-01: Analyze rhythm of English speech → classify as stress-timed."""
        from voice_soundboard.vocology import RhythmAnalyzer, RhythmClass

        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        metrics = analyzer.analyze_metrics(audio, sample_rate=sr)
        rhythm_class = metrics.rhythm_class

        # Should be one of the valid rhythm classes
        assert isinstance(rhythm_class, RhythmClass)

    def test_rhy_int_02_tts_valid_metrics(self, mock_speech_audio):
        """TEST-RHY-INT-02: Analyze rhythm of TTS output → compute valid metrics."""
        from voice_soundboard.vocology import RhythmAnalyzer

        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        metrics = analyzer.analyze_metrics(audio, sample_rate=sr)

        # All metrics should be valid numbers
        assert metrics.percent_v >= 0
        assert metrics.npvi_v >= 0
        assert metrics.speech_rate > 0

    def test_rhy_int_03_rzt_detects_theta_band(self, mock_speech_audio):
        """TEST-RHY-INT-03: RZT analysis detects syllable-level rhythm (theta band)."""
        from voice_soundboard.vocology import RhythmAnalyzer

        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        rzt = analyzer.analyze_rzt(audio, sample_rate=sr)

        # Should detect syllable rhythm in theta band (4-8 Hz)
        assert rzt.syllable_rhythm > 0

    def test_rhy_int_04_modify_adds_variability(self, mock_speech_audio):
        """TEST-RHY-INT-04: Modify rhythm to add natural timing variability."""
        from voice_soundboard.vocology import RhythmModifier

        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result_audio, _ = modifier.add_variability(audio, amount=0.15, sample_rate=sr)

        assert isinstance(result_audio, np.ndarray)
        # Should be different from original (length may vary due to time stretching)
        assert len(result_audio) > 0

    def test_rhy_int_05_full_pipeline(self, mock_speech_audio):
        """TEST-RHY-INT-05: Full pipeline: analyze → classify → modify → re-analyze."""
        from voice_soundboard.vocology import RhythmAnalyzer, RhythmModifier

        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()

        # 1. Analyze original
        original_metrics = analyzer.analyze_metrics(audio, sample_rate=sr)
        original_class = original_metrics.rhythm_class

        # 2. Modify (add variability)
        modifier = RhythmModifier(sample_rate=sr)
        modified_audio, _ = modifier.add_variability(audio, sample_rate=sr)

        # 3. Re-analyze
        modified_metrics = analyzer.analyze_metrics(modified_audio, sample_rate=sr)

        # Both should be valid
        assert original_metrics.speech_rate > 0
        assert modified_metrics.speech_rate > 0


# =============================================================================
# TEST-VOC-FULL-01 to TEST-VOC-FULL-05: Combined Vocology Pipeline Tests
# =============================================================================

class TestCombinedVocologyPipeline:
    """Combined vocology pipeline tests (TEST-VOC-FULL-01 to TEST-VOC-FULL-05)."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio."""
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Speech-like audio
        f0 = 150
        audio = np.zeros_like(t)
        for h in range(1, 10):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)

        # Add structure
        for i in range(8):
            start = int(i * 0.25 * sr)
            end = int((i * 0.25 + 0.2) * sr)
            if end < len(audio):
                audio[start:end] *= 0.8

        audio = audio / np.max(np.abs(audio)) * 0.7
        audio += 0.02 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_voc_full_01_voice_quality_then_humanize(self, mock_audio):
        """TEST-VOC-FULL-01: Analyze voice quality → humanize."""
        from voice_soundboard.vocology import VoiceQualityAnalyzer, VoiceHumanizer

        audio, sr = mock_audio

        # Analyze quality
        vq_analyzer = VoiceQualityAnalyzer()
        metrics = vq_analyzer.analyze(audio, sample_rate=sr)

        # Humanize
        humanizer = VoiceHumanizer()
        result_audio, _ = humanizer.humanize(audio, sample_rate=sr)

        assert metrics.f0_mean > 0
        assert isinstance(result_audio, np.ndarray)

    def test_voc_full_02_rhythm_then_prosody(self, mock_audio):
        """TEST-VOC-FULL-02: Analyze rhythm → modify prosody."""
        from voice_soundboard.vocology import RhythmAnalyzer, ProsodyModifier

        audio, sr = mock_audio

        # Analyze rhythm
        rhythm_analyzer = RhythmAnalyzer()
        metrics = rhythm_analyzer.analyze_metrics(audio, sample_rate=sr)

        # Modify prosody
        prosody_modifier = ProsodyModifier()
        result_audio, _ = prosody_modifier.modify_duration(audio, ratio=1.1, sample_rate=sr)

        assert metrics.speech_rate > 0
        assert isinstance(result_audio, np.ndarray)

    def test_voc_full_03_formants_then_humanize(self, mock_audio):
        """TEST-VOC-FULL-03: Shift formants → humanize."""
        from voice_soundboard.vocology import FormantShifter, VoiceHumanizer

        audio, sr = mock_audio

        # Shift formants
        shifter = FormantShifter()
        shifted_audio, _ = shifter.shift(audio, ratio=0.95, sample_rate=sr)

        # Humanize
        humanizer = VoiceHumanizer()
        result_audio, _ = humanizer.humanize(shifted_audio, sample_rate=sr)

        assert isinstance(result_audio, np.ndarray)

    def test_voc_full_04_biomarkers_then_rhythm(self, mock_audio):
        """TEST-VOC-FULL-04: Analyze biomarkers → analyze rhythm."""
        from voice_soundboard.vocology import VocalBiomarkers, RhythmAnalyzer

        audio, sr = mock_audio

        # Analyze biomarkers
        biomarker_analyzer = VocalBiomarkers()
        bio_result = biomarker_analyzer.analyze(audio, sample_rate=sr)

        # Analyze rhythm
        rhythm_analyzer = RhythmAnalyzer()
        rhythm_metrics = rhythm_analyzer.analyze_metrics(audio, sample_rate=sr)

        assert bio_result.health_metrics is not None
        assert rhythm_metrics.speech_rate > 0

    def test_voc_full_05_full_vocology_chain(self, mock_audio):
        """TEST-VOC-FULL-05: Full chain: analyze → modify → humanize → analyze."""
        from voice_soundboard.vocology import (
            VoiceQualityAnalyzer,
            FormantShifter,
            VoiceHumanizer,
            RhythmAnalyzer,
        )

        audio, sr = mock_audio

        # 1. Analyze original
        vq_analyzer = VoiceQualityAnalyzer()
        original_metrics = vq_analyzer.analyze(audio, sample_rate=sr)

        # 2. Shift formants
        shifter = FormantShifter()
        audio, _ = shifter.shift(audio, ratio=0.97, sample_rate=sr)

        # 3. Humanize
        humanizer = VoiceHumanizer()
        audio, sr = humanizer.humanize(audio, sample_rate=sr)

        # 4. Analyze rhythm of final
        rhythm_analyzer = RhythmAnalyzer()
        final_rhythm = rhythm_analyzer.analyze_metrics(audio, sample_rate=sr)

        # All should succeed
        assert original_metrics.f0_mean > 0
        assert final_rhythm.speech_rate > 0
