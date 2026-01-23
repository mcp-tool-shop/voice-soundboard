"""
Tests for Phase 9: Vocology Module - Batch 8
VoiceFatigueMonitor class and biomarker convenience functions

Tests cover:
- VoiceFatigueMonitor initialization (TEST-FAT-01 to TEST-FAT-03)
- VoiceFatigueMonitor.add_sample() (TEST-FAT-04 to TEST-FAT-09)
- VoiceFatigueMonitor.get_fatigue_report() (TEST-FAT-10 to TEST-FAT-16)
- VoiceFatigueMonitor history management (TEST-FAT-17 to TEST-FAT-18)
- Biomarker convenience functions (TEST-FAT-19 to TEST-FAT-25)
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-FAT-01 to TEST-FAT-03: VoiceFatigueMonitor Initialization Tests
# =============================================================================

from voice_soundboard.vocology.biomarkers import (
    VoiceFatigueMonitor,
    VocalBiomarkers,
    BiomarkerResult,
    FatigueLevel,
    analyze_biomarkers,
    assess_vocal_fatigue,
)


class TestVoiceFatigueMonitorInit:
    """Tests for VoiceFatigueMonitor initialization (TEST-FAT-01 to TEST-FAT-03)."""

    def test_fat_01_init(self):
        """TEST-FAT-01: VoiceFatigueMonitor initializes without error."""
        monitor = VoiceFatigueMonitor()
        assert monitor is not None

    def test_fat_02_has_empty_samples(self):
        """TEST-FAT-02: VoiceFatigueMonitor starts with empty samples list."""
        monitor = VoiceFatigueMonitor()
        assert hasattr(monitor, 'samples')
        assert monitor.samples == []

    def test_fat_03_has_analyzer(self):
        """TEST-FAT-03: VoiceFatigueMonitor has internal VocalBiomarkers analyzer."""
        monitor = VoiceFatigueMonitor()
        assert hasattr(monitor, 'analyzer')
        assert isinstance(monitor.analyzer, VocalBiomarkers)


# =============================================================================
# TEST-FAT-04 to TEST-FAT-09: VoiceFatigueMonitor.add_sample() Tests
# =============================================================================

class TestVoiceFatigueMonitorAddSample:
    """Tests for VoiceFatigueMonitor.add_sample() (TEST-FAT-04 to TEST-FAT-09)."""

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

    def test_fat_04_add_sample_returns_result(self, mock_audio_array):
        """TEST-FAT-04: add_sample() returns BiomarkerResult."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        result = monitor.add_sample(audio, sample_rate=sr)
        assert isinstance(result, BiomarkerResult)

    def test_fat_05_add_sample_stores_in_history(self, mock_audio_array):
        """TEST-FAT-05: add_sample() stores sample in history."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        assert len(monitor.samples) == 0
        monitor.add_sample(audio, sample_rate=sr)
        assert len(monitor.samples) == 1

    def test_fat_06_add_sample_with_timestamp_string(self, mock_audio_array):
        """TEST-FAT-06: add_sample() accepts timestamp as ISO string."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        result = monitor.add_sample(
            audio, timestamp="2026-01-23T10:00:00", sample_rate=sr
        )
        assert monitor.samples[0]["timestamp"].hour == 10

    def test_fat_07_add_sample_with_timestamp_datetime(self, mock_audio_array):
        """TEST-FAT-07: add_sample() accepts timestamp as datetime object."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        ts = datetime(2026, 1, 23, 14, 30)
        monitor.add_sample(audio, timestamp=ts, sample_rate=sr)
        assert monitor.samples[0]["timestamp"] == ts

    def test_fat_08_add_sample_with_label(self, mock_audio_array):
        """TEST-FAT-08: add_sample() accepts optional label."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        monitor.add_sample(audio, label="morning_recording", sample_rate=sr)
        assert monitor.samples[0]["label"] == "morning_recording"

    def test_fat_09_add_sample_multiple(self, mock_audio_array):
        """TEST-FAT-09: add_sample() can be called multiple times."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        monitor.add_sample(audio, sample_rate=sr)
        monitor.add_sample(audio, sample_rate=sr)
        monitor.add_sample(audio, sample_rate=sr)
        assert len(monitor.samples) == 3


# =============================================================================
# TEST-FAT-10 to TEST-FAT-16: VoiceFatigueMonitor.get_fatigue_report() Tests
# =============================================================================

class TestVoiceFatigueMonitorReport:
    """Tests for VoiceFatigueMonitor.get_fatigue_report() (TEST-FAT-10 to TEST-FAT-16)."""

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

    def test_fat_10_report_insufficient_data(self):
        """TEST-FAT-10: get_fatigue_report() returns insufficient_data with < 2 samples."""
        monitor = VoiceFatigueMonitor()
        report = monitor.get_fatigue_report()
        assert report["trend"] == "insufficient_data"

    def test_fat_11_report_returns_dict(self, mock_audio_array):
        """TEST-FAT-11: get_fatigue_report() returns dictionary."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=1), sample_rate=sr)
        report = monitor.get_fatigue_report()
        assert isinstance(report, dict)

    def test_fat_12_report_has_trend(self, mock_audio_array):
        """TEST-FAT-12: get_fatigue_report() includes trend field."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=1), sample_rate=sr)
        report = monitor.get_fatigue_report()
        assert "trend" in report
        assert report["trend"] in ["improving", "stable", "declining"]

    def test_fat_13_report_has_current_fatigue(self, mock_audio_array):
        """TEST-FAT-13: get_fatigue_report() includes current fatigue level."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=1), sample_rate=sr)
        report = monitor.get_fatigue_report()
        assert "current_fatigue" in report

    def test_fat_14_report_has_samples_count(self, mock_audio_array):
        """TEST-FAT-14: get_fatigue_report() includes samples_analyzed count."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=1), sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=2), sample_rate=sr)
        report = monitor.get_fatigue_report()
        assert report["samples_analyzed"] == 3

    def test_fat_15_report_has_recommendations(self, mock_audio_array):
        """TEST-FAT-15: get_fatigue_report() includes recommendations list."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=1), sample_rate=sr)
        report = monitor.get_fatigue_report()
        assert "recommendations" in report
        assert isinstance(report["recommendations"], list)

    def test_fat_16_report_has_quality_scores(self, mock_audio_array):
        """TEST-FAT-16: get_fatigue_report() includes quality_scores list."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        now = datetime.now()
        monitor.add_sample(audio, timestamp=now, sample_rate=sr)
        monitor.add_sample(audio, timestamp=now + timedelta(hours=1), sample_rate=sr)
        report = monitor.get_fatigue_report()
        assert "quality_scores" in report
        assert len(report["quality_scores"]) == 2


# =============================================================================
# TEST-FAT-17 to TEST-FAT-18: VoiceFatigueMonitor History Management Tests
# =============================================================================

class TestVoiceFatigueMonitorHistory:
    """Tests for VoiceFatigueMonitor history management (TEST-FAT-17 to TEST-FAT-18)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 120 * t)
        audio += 0.01 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_fat_17_clear_history(self, mock_audio_array):
        """TEST-FAT-17: clear_history() removes all samples."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        monitor.add_sample(audio, sample_rate=sr)
        monitor.add_sample(audio, sample_rate=sr)
        assert len(monitor.samples) == 2

        monitor.clear_history()
        assert len(monitor.samples) == 0

    def test_fat_18_clear_history_allows_new_samples(self, mock_audio_array):
        """TEST-FAT-18: clear_history() allows adding new samples afterward."""
        audio, sr = mock_audio_array
        monitor = VoiceFatigueMonitor()
        monitor.add_sample(audio, sample_rate=sr)
        monitor.clear_history()
        monitor.add_sample(audio, sample_rate=sr)
        assert len(monitor.samples) == 1


# =============================================================================
# TEST-FAT-19 to TEST-FAT-25: Biomarker Convenience Functions Tests
# =============================================================================

class TestBiomarkerConvenienceFunctions:
    """Tests for biomarker convenience functions (TEST-FAT-19 to TEST-FAT-25)."""

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

    def test_fat_19_analyze_biomarkers_returns_result(self, mock_audio_array):
        """TEST-FAT-19: analyze_biomarkers() returns BiomarkerResult."""
        audio, sr = mock_audio_array
        result = analyze_biomarkers(audio, sample_rate=sr)
        assert isinstance(result, BiomarkerResult)

    def test_fat_20_analyze_biomarkers_has_health(self, mock_audio_array):
        """TEST-FAT-20: analyze_biomarkers() result has health_metrics."""
        audio, sr = mock_audio_array
        result = analyze_biomarkers(audio, sample_rate=sr)
        assert hasattr(result, 'health_metrics')

    def test_fat_21_analyze_biomarkers_has_fatigue(self, mock_audio_array):
        """TEST-FAT-21: analyze_biomarkers() result has fatigue_level."""
        audio, sr = mock_audio_array
        result = analyze_biomarkers(audio, sample_rate=sr)
        assert hasattr(result, 'fatigue_level')
        assert isinstance(result.fatigue_level, FatigueLevel)

    def test_fat_22_assess_vocal_fatigue_returns_level(self, mock_audio_array):
        """TEST-FAT-22: assess_vocal_fatigue() returns FatigueLevel."""
        audio, sr = mock_audio_array
        result = assess_vocal_fatigue(audio, sample_rate=sr)
        assert isinstance(result, FatigueLevel)

    def test_fat_23_assess_vocal_fatigue_valid_level(self, mock_audio_array):
        """TEST-FAT-23: assess_vocal_fatigue() returns valid fatigue level."""
        audio, sr = mock_audio_array
        result = assess_vocal_fatigue(audio, sample_rate=sr)
        valid_levels = [FatigueLevel.NONE, FatigueLevel.LOW, FatigueLevel.MODERATE,
                        FatigueLevel.HIGH, FatigueLevel.SEVERE]
        assert result in valid_levels

    def test_fat_24_analyze_biomarkers_to_dict(self, mock_audio_array):
        """TEST-FAT-24: analyze_biomarkers() result can be converted to dict."""
        audio, sr = mock_audio_array
        result = analyze_biomarkers(audio, sample_rate=sr)
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "health" in result_dict
        assert "fatigue" in result_dict

    def test_fat_25_analyze_biomarkers_warnings_list(self, mock_audio_array):
        """TEST-FAT-25: analyze_biomarkers() result has warnings list."""
        audio, sr = mock_audio_array
        result = analyze_biomarkers(audio, sample_rate=sr)
        assert hasattr(result, 'warnings')
        assert isinstance(result.warnings, list)
