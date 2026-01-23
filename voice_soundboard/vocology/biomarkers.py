"""
Vocal Biomarkers

Voice analysis for health assessment and monitoring.

Vocal biomarkers are measurable voice characteristics that may indicate
health conditions. This module provides analysis tools for:
- Voice quality health assessment
- Vocal fatigue detection
- General voice health monitoring

IMPORTANT DISCLAIMER:
This module is for informational and research purposes only.
It is NOT a substitute for professional medical diagnosis.
Always consult healthcare providers for medical concerns.

Reference:
- Canary Speech Vocal Biomarkers: https://canaryspeech.com/voice-biomarkers/
- PMC Voice for Health: https://pmc.ncbi.nlm.nih.gov/articles/PMC8138221/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict
import numpy as np


class VoiceHealthStatus(Enum):
    """General voice health status."""
    HEALTHY = "healthy"
    MILD_STRAIN = "mild_strain"
    MODERATE_STRAIN = "moderate_strain"
    SIGNIFICANT_STRAIN = "significant_strain"
    NEEDS_ATTENTION = "needs_attention"


class FatigueLevel(Enum):
    """Vocal fatigue level."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class VoiceHealthMetrics:
    """
    Voice health assessment metrics.

    Attributes:
        status: Overall health status
        quality_score: Voice quality score (0-100)
        stability_score: Voice stability score (0-100)
        clarity_score: Voice clarity score (0-100)
        concerns: List of potential concerns
        recommendations: Health recommendations
    """
    status: VoiceHealthStatus
    quality_score: float
    stability_score: float
    clarity_score: float
    concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall voice health score."""
        return (self.quality_score + self.stability_score + self.clarity_score) / 3


@dataclass
class BiomarkerResult:
    """
    Complete biomarker analysis result.

    Attributes:
        health_metrics: Voice health assessment
        fatigue_level: Detected fatigue level
        voice_quality: Detailed voice quality metrics
        timestamp: Analysis timestamp
        audio_duration: Duration of analyzed audio
        warnings: Any warnings or concerns
    """
    health_metrics: VoiceHealthMetrics
    fatigue_level: FatigueLevel
    voice_quality: Dict[str, float]
    timestamp: datetime
    audio_duration: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "health": {
                "status": self.health_metrics.status.value,
                "quality_score": self.health_metrics.quality_score,
                "stability_score": self.health_metrics.stability_score,
                "clarity_score": self.health_metrics.clarity_score,
                "overall_score": self.health_metrics.overall_score,
                "concerns": self.health_metrics.concerns,
                "recommendations": self.health_metrics.recommendations,
            },
            "fatigue": {
                "level": self.fatigue_level.value,
            },
            "voice_quality": self.voice_quality,
            "timestamp": self.timestamp.isoformat(),
            "duration_s": self.audio_duration,
            "warnings": self.warnings,
        }


class VocalBiomarkers:
    """
    Analyze vocal biomarkers for health assessment.

    Extracts voice quality metrics and provides health-related
    assessments based on acoustic analysis.

    DISCLAIMER: This is for informational purposes only and is not
    a medical diagnostic tool.

    Example:
        analyzer = VocalBiomarkers()
        result = analyzer.analyze("speech.wav")
        print(f"Health Status: {result.health_metrics.status.value}")
        print(f"Fatigue Level: {result.fatigue_level.value}")
    """

    # Thresholds for health assessment
    HEALTHY_THRESHOLDS = {
        "jitter_max": 1.0,      # %
        "shimmer_max": 5.0,     # %
        "hnr_min": 15.0,        # dB
        "cpp_min": 5.0,         # dB
        "f0_std_max": 50.0,     # Hz (for stability)
    }

    CONCERN_THRESHOLDS = {
        "jitter_concern": 2.0,
        "shimmer_concern": 8.0,
        "hnr_concern": 10.0,
        "cpp_concern": 3.0,
    }

    def __init__(self):
        """Initialize the biomarker analyzer."""
        pass

    def analyze(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> BiomarkerResult:
        """
        Analyze vocal biomarkers from audio.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            BiomarkerResult with health assessment
        """
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer

        # Get voice quality metrics
        analyzer = VoiceQualityAnalyzer()
        metrics = analyzer.analyze(audio, sample_rate)

        # Store raw metrics
        voice_quality = metrics.to_dict()

        # Assess health
        health_metrics = self._assess_health(metrics)

        # Assess fatigue
        fatigue_level = self._assess_fatigue(metrics)

        # Generate warnings
        warnings = self._generate_warnings(metrics)

        return BiomarkerResult(
            health_metrics=health_metrics,
            fatigue_level=fatigue_level,
            voice_quality=voice_quality,
            timestamp=datetime.now(),
            audio_duration=metrics.duration,
            warnings=warnings,
        )

    def _assess_health(self, metrics) -> VoiceHealthMetrics:
        """Assess overall voice health from metrics."""
        concerns = []
        recommendations = []

        # Calculate component scores (0-100)
        quality_score = self._calculate_quality_score(metrics)
        stability_score = self._calculate_stability_score(metrics)
        clarity_score = self._calculate_clarity_score(metrics)

        # Check for concerns
        if metrics.jitter_local > self.HEALTHY_THRESHOLDS["jitter_max"]:
            concerns.append("Elevated pitch instability (jitter)")
            if metrics.jitter_local > self.CONCERN_THRESHOLDS["jitter_concern"]:
                recommendations.append("Consider voice rest and hydration")

        if metrics.shimmer_local > self.HEALTHY_THRESHOLDS["shimmer_max"]:
            concerns.append("Elevated amplitude variation (shimmer)")
            if metrics.shimmer_local > self.CONCERN_THRESHOLDS["shimmer_concern"]:
                recommendations.append("May indicate vocal strain - rest recommended")

        if metrics.hnr < self.HEALTHY_THRESHOLDS["hnr_min"]:
            concerns.append("Reduced voice clarity (low HNR)")
            if metrics.hnr < self.CONCERN_THRESHOLDS["hnr_concern"]:
                recommendations.append("Voice sounds hoarse - consider rest")

        if metrics.cpp < self.HEALTHY_THRESHOLDS["cpp_min"]:
            concerns.append("Reduced voice periodicity (low CPP)")
            if metrics.cpp < self.CONCERN_THRESHOLDS["cpp_concern"]:
                recommendations.append("Voice quality may benefit from vocal exercises")

        # Determine overall status
        overall = (quality_score + stability_score + clarity_score) / 3

        if overall >= 80 and len(concerns) == 0:
            status = VoiceHealthStatus.HEALTHY
        elif overall >= 70 and len(concerns) <= 1:
            status = VoiceHealthStatus.MILD_STRAIN
        elif overall >= 55:
            status = VoiceHealthStatus.MODERATE_STRAIN
            if not recommendations:
                recommendations.append("Consider reducing voice use")
        elif overall >= 40:
            status = VoiceHealthStatus.SIGNIFICANT_STRAIN
            recommendations.append("Voice rest strongly recommended")
        else:
            status = VoiceHealthStatus.NEEDS_ATTENTION
            recommendations.append("Consider consulting a voice specialist")

        # Add general recommendations
        if not recommendations and status == VoiceHealthStatus.HEALTHY:
            recommendations.append("Voice quality is within normal range")

        return VoiceHealthMetrics(
            status=status,
            quality_score=quality_score,
            stability_score=stability_score,
            clarity_score=clarity_score,
            concerns=concerns,
            recommendations=recommendations,
        )

    def _calculate_quality_score(self, metrics) -> float:
        """Calculate voice quality score (0-100)."""
        # Based on jitter, shimmer, and HNR
        jitter_score = max(0, 100 - metrics.jitter_local * 50)
        shimmer_score = max(0, 100 - metrics.shimmer_local * 10)
        hnr_score = min(100, metrics.hnr * 5)

        return (jitter_score * 0.3 + shimmer_score * 0.3 + hnr_score * 0.4)

    def _calculate_stability_score(self, metrics) -> float:
        """Calculate voice stability score (0-100)."""
        # Based on F0 variation and jitter
        f0_stability = max(0, 100 - metrics.f0_std * 2)
        jitter_stability = max(0, 100 - metrics.jitter_local * 40)

        return (f0_stability * 0.6 + jitter_stability * 0.4)

    def _calculate_clarity_score(self, metrics) -> float:
        """Calculate voice clarity score (0-100)."""
        # Based on HNR and CPP
        hnr_clarity = min(100, metrics.hnr * 4)
        cpp_clarity = min(100, metrics.cpp * 10)

        return (hnr_clarity * 0.5 + cpp_clarity * 0.5)

    def _assess_fatigue(self, metrics) -> FatigueLevel:
        """Assess vocal fatigue level."""
        # Fatigue indicators:
        # - Increased jitter and shimmer
        # - Decreased HNR
        # - Decreased pitch range
        # - Increased breathiness (spectral tilt)

        fatigue_score = 0

        # Jitter increase indicates fatigue
        if metrics.jitter_local > 1.0:
            fatigue_score += 1
        if metrics.jitter_local > 2.0:
            fatigue_score += 1

        # Shimmer increase
        if metrics.shimmer_local > 5.0:
            fatigue_score += 1
        if metrics.shimmer_local > 8.0:
            fatigue_score += 1

        # HNR decrease
        if metrics.hnr < 15.0:
            fatigue_score += 1
        if metrics.hnr < 10.0:
            fatigue_score += 1

        # Reduced pitch range might indicate fatigue
        if metrics.f0_range < 30:
            fatigue_score += 1

        # Spectral tilt (more negative = breathier = potentially fatigued)
        if metrics.spectral_tilt < -15:
            fatigue_score += 1

        # Map score to fatigue level
        if fatigue_score <= 1:
            return FatigueLevel.NONE
        elif fatigue_score <= 2:
            return FatigueLevel.LOW
        elif fatigue_score <= 4:
            return FatigueLevel.MODERATE
        elif fatigue_score <= 6:
            return FatigueLevel.HIGH
        else:
            return FatigueLevel.SEVERE

    def _generate_warnings(self, metrics) -> List[str]:
        """Generate health warnings based on metrics."""
        warnings = []

        # Critical thresholds
        if metrics.jitter_local > 3.0:
            warnings.append("Very high jitter detected - may indicate voice disorder")

        if metrics.shimmer_local > 10.0:
            warnings.append("Very high shimmer detected - significant voice instability")

        if metrics.hnr < 5.0:
            warnings.append("Very low HNR - significant noise in voice signal")

        if metrics.cpp < 2.0:
            warnings.append("Very low CPP - reduced voice periodicity")

        if warnings:
            warnings.append(
                "DISCLAIMER: This is not a medical diagnosis. "
                "Consult a healthcare provider for any voice concerns."
            )

        return warnings


class VoiceFatigueMonitor:
    """
    Monitor vocal fatigue over time.

    Tracks voice quality across multiple samples to detect
    trends and fatigue patterns.

    Example:
        monitor = VoiceFatigueMonitor()
        monitor.add_sample("morning.wav", "2026-01-23T09:00:00")
        monitor.add_sample("afternoon.wav", "2026-01-23T14:00:00")
        monitor.add_sample("evening.wav", "2026-01-23T18:00:00")

        report = monitor.get_fatigue_report()
        print(f"Trend: {report['trend']}")
    """

    def __init__(self):
        """Initialize the monitor."""
        self.samples: List[Dict] = []
        self.analyzer = VocalBiomarkers()

    def add_sample(
        self,
        audio: Union[str, Path, np.ndarray],
        timestamp: Optional[Union[str, datetime]] = None,
        sample_rate: Optional[int] = None,
        label: Optional[str] = None,
    ) -> BiomarkerResult:
        """
        Add a voice sample for monitoring.

        Args:
            audio: Audio file or array
            timestamp: Sample timestamp (ISO format or datetime)
            sample_rate: Sample rate if audio is array
            label: Optional label for the sample

        Returns:
            BiomarkerResult for this sample
        """
        # Parse timestamp
        if timestamp is None:
            ts = datetime.now()
        elif isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp)
        else:
            ts = timestamp

        # Analyze sample
        result = self.analyzer.analyze(audio, sample_rate)

        # Store in history
        self.samples.append({
            "timestamp": ts,
            "result": result,
            "label": label,
            "quality_score": result.health_metrics.quality_score,
            "fatigue_level": result.fatigue_level.value,
        })

        return result

    def get_fatigue_report(self) -> Dict:
        """
        Get fatigue trend report.

        Returns:
            Report with trend analysis
        """
        if len(self.samples) < 2:
            return {
                "trend": "insufficient_data",
                "samples": len(self.samples),
                "message": "Need at least 2 samples for trend analysis",
            }

        # Calculate quality trend
        scores = [s["quality_score"] for s in self.samples]
        times = [(s["timestamp"] - self.samples[0]["timestamp"]).total_seconds() / 3600
                 for s in self.samples]

        # Simple linear regression for trend
        if len(times) >= 2:
            slope = np.polyfit(times, scores, 1)[0]
        else:
            slope = 0

        # Determine trend
        if slope > 2:
            trend = "improving"
            trend_message = "Voice quality is improving over time"
        elif slope < -2:
            trend = "declining"
            trend_message = "Voice quality is declining - consider rest"
        else:
            trend = "stable"
            trend_message = "Voice quality is stable"

        # Current fatigue level
        current_fatigue = self.samples[-1]["fatigue_level"]

        # Recommendations
        recommendations = []
        if current_fatigue in ["high", "severe"]:
            recommendations.append("Immediate voice rest recommended")
        if trend == "declining":
            recommendations.append("Trend suggests increasing strain")
            recommendations.append("Consider reducing voice use")

        return {
            "trend": trend,
            "trend_message": trend_message,
            "slope": slope,
            "current_fatigue": current_fatigue,
            "samples_analyzed": len(self.samples),
            "time_span_hours": times[-1] if times else 0,
            "quality_scores": scores,
            "recommendations": recommendations,
        }

    def clear_history(self):
        """Clear sample history."""
        self.samples = []


# Convenience functions

def analyze_biomarkers(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> BiomarkerResult:
    """
    Analyze vocal biomarkers from audio.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate (required if audio is array)

    Returns:
        BiomarkerResult with health assessment

    Note:
        This is for informational purposes only and is not
        a medical diagnostic tool.
    """
    analyzer = VocalBiomarkers()
    return analyzer.analyze(audio, sample_rate)


def assess_vocal_fatigue(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> FatigueLevel:
    """
    Assess vocal fatigue level from audio.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate (required if audio is array)

    Returns:
        FatigueLevel enum value
    """
    result = analyze_biomarkers(audio, sample_rate)
    return result.fatigue_level
