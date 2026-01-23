"""
Voice Quality Parameters

Scientific voice quality measurements including jitter, shimmer, HNR, and CPP.

These parameters are used for:
- Voice quality assessment
- Naturalness tuning in synthesis
- Health monitoring (vocal biomarkers)
- Voice characteristic modification

Reference:
- Titze, I.R. (1994). Principles of Voice Production
- NCVS Voice Quality Tutorials: https://ncvs.org/tutorials/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np


class JitterType(Enum):
    """Types of jitter (pitch perturbation) measurements."""
    LOCAL = "local"           # Cycle-to-cycle variation (%)
    RAP = "rap"               # Relative Average Perturbation (3-point)
    PPQ5 = "ppq5"             # 5-point Period Perturbation Quotient
    DDP = "ddp"               # Difference of Differences of Periods


class ShimmerType(Enum):
    """Types of shimmer (amplitude perturbation) measurements."""
    LOCAL = "local"           # Cycle-to-cycle amplitude variation (%)
    APQ3 = "apq3"             # 3-point Amplitude Perturbation Quotient
    APQ5 = "apq5"             # 5-point Amplitude Perturbation Quotient
    APQ11 = "apq11"           # 11-point (smoothed) APQ
    DDA = "dda"               # Difference of Differences of Amplitudes


@dataclass
class VoiceQualityMetrics:
    """
    Comprehensive voice quality measurements.

    Attributes:
        f0_mean: Mean fundamental frequency (Hz)
        f0_std: Standard deviation of F0 (Hz)
        f0_range: Range of F0 (max - min, Hz)

        jitter_local: Local jitter (%)
        jitter_rap: Relative Average Perturbation
        jitter_ppq5: 5-point Period Perturbation Quotient

        shimmer_local: Local shimmer (%)
        shimmer_apq3: 3-point Amplitude Perturbation Quotient
        shimmer_apq5: 5-point Amplitude Perturbation Quotient
        shimmer_apq11: 11-point Amplitude Perturbation Quotient

        hnr: Harmonics-to-Noise Ratio (dB)
        nhr: Noise-to-Harmonics Ratio (inverse of HNR)
        cpp: Cepstral Peak Prominence (dB)

        spectral_tilt: Spectral tilt (dB/octave)
        spectral_centroid: Spectral centroid (Hz)

        voiced_fraction: Fraction of voiced frames
        duration: Total duration (seconds)
    """
    # Pitch metrics
    f0_mean: float
    f0_std: float
    f0_range: float

    # Jitter metrics
    jitter_local: float
    jitter_rap: float
    jitter_ppq5: float

    # Shimmer metrics
    shimmer_local: float
    shimmer_apq3: float
    shimmer_apq5: float
    shimmer_apq11: float

    # Noise metrics
    hnr: float
    nhr: float
    cpp: float

    # Spectral metrics
    spectral_tilt: float
    spectral_centroid: float

    # General
    voiced_fraction: float
    duration: float

    @property
    def jitter_percent(self) -> float:
        """Convenience alias for local jitter as percentage."""
        return self.jitter_local

    @property
    def shimmer_percent(self) -> float:
        """Convenience alias for local shimmer as percentage."""
        return self.shimmer_local

    @property
    def hnr_db(self) -> float:
        """Convenience alias for HNR in dB."""
        return self.hnr

    def is_healthy(self) -> bool:
        """
        Check if voice quality metrics are within healthy ranges.

        Returns:
            True if all metrics are within normal ranges.
        """
        return (
            self.jitter_local < 1.0 and
            self.shimmer_local < 5.0 and
            self.hnr > 15.0 and
            self.cpp > 5.0
        )

    def quality_assessment(self) -> str:
        """
        Get a qualitative assessment of voice quality.

        Returns:
            Assessment string: 'excellent', 'good', 'fair', 'poor'
        """
        if self.jitter_local < 0.5 and self.shimmer_local < 3.0 and self.hnr > 20:
            return "excellent"
        elif self.jitter_local < 1.0 and self.shimmer_local < 5.0 and self.hnr > 15:
            return "good"
        elif self.jitter_local < 2.0 and self.shimmer_local < 8.0 and self.hnr > 10:
            return "fair"
        else:
            return "poor"

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "f0": {
                "mean_hz": self.f0_mean,
                "std_hz": self.f0_std,
                "range_hz": self.f0_range,
            },
            "jitter": {
                "local_percent": self.jitter_local,
                "rap": self.jitter_rap,
                "ppq5": self.jitter_ppq5,
            },
            "shimmer": {
                "local_percent": self.shimmer_local,
                "apq3": self.shimmer_apq3,
                "apq5": self.shimmer_apq5,
                "apq11": self.shimmer_apq11,
            },
            "noise": {
                "hnr_db": self.hnr,
                "nhr": self.nhr,
                "cpp_db": self.cpp,
            },
            "spectral": {
                "tilt_db_octave": self.spectral_tilt,
                "centroid_hz": self.spectral_centroid,
            },
            "general": {
                "voiced_fraction": self.voiced_fraction,
                "duration_s": self.duration,
                "quality": self.quality_assessment(),
                "healthy": self.is_healthy(),
            },
        }


class VoiceQualityAnalyzer:
    """
    Analyze voice quality from audio files.

    Extracts scientific voice quality parameters including pitch,
    jitter, shimmer, HNR, and spectral characteristics.

    Example:
        analyzer = VoiceQualityAnalyzer()
        metrics = analyzer.analyze("speech.wav")
        print(f"Jitter: {metrics.jitter_percent:.2f}%")
        print(f"HNR: {metrics.hnr_db:.1f} dB")
    """

    def __init__(
        self,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        frame_length: float = 0.025,
        hop_length: float = 0.010,
    ):
        """
        Initialize the analyzer.

        Args:
            f0_min: Minimum F0 to consider (Hz)
            f0_max: Maximum F0 to consider (Hz)
            frame_length: Analysis frame length (seconds)
            hop_length: Hop between frames (seconds)
        """
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_length = frame_length
        self.hop_length = hop_length

    def analyze(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> VoiceQualityMetrics:
        """
        Analyze voice quality from audio.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            VoiceQualityMetrics with all measurements
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required when audio is array")
            y = audio
            sr = sample_rate

        # Extract F0 using PYIN or similar
        f0, voiced_flag = self._extract_f0(y, sr)

        # Calculate metrics
        f0_voiced = f0[voiced_flag]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_range = float(np.ptp(f0_voiced)) if len(f0_voiced) > 0 else 0.0

        # Jitter calculations
        jitter_local = self._calculate_jitter(f0_voiced, JitterType.LOCAL)
        jitter_rap = self._calculate_jitter(f0_voiced, JitterType.RAP)
        jitter_ppq5 = self._calculate_jitter(f0_voiced, JitterType.PPQ5)

        # Shimmer calculations (need amplitude per cycle)
        amplitudes = self._extract_cycle_amplitudes(y, sr, f0, voiced_flag)
        shimmer_local = self._calculate_shimmer(amplitudes, ShimmerType.LOCAL)
        shimmer_apq3 = self._calculate_shimmer(amplitudes, ShimmerType.APQ3)
        shimmer_apq5 = self._calculate_shimmer(amplitudes, ShimmerType.APQ5)
        shimmer_apq11 = self._calculate_shimmer(amplitudes, ShimmerType.APQ11)

        # HNR and CPP
        hnr = self._calculate_hnr(y, sr, f0_mean)
        nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else float('inf')
        cpp = self._calculate_cpp(y, sr)

        # Spectral characteristics
        spectral_tilt = self._calculate_spectral_tilt(y, sr)
        spectral_centroid = self._calculate_spectral_centroid(y, sr)

        # General metrics
        voiced_fraction = float(np.mean(voiced_flag))
        duration = len(y) / sr

        return VoiceQualityMetrics(
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_range=f0_range,
            jitter_local=jitter_local,
            jitter_rap=jitter_rap,
            jitter_ppq5=jitter_ppq5,
            shimmer_local=shimmer_local,
            shimmer_apq3=shimmer_apq3,
            shimmer_apq5=shimmer_apq5,
            shimmer_apq11=shimmer_apq11,
            hnr=hnr,
            nhr=nhr,
            cpp=cpp,
            spectral_tilt=spectral_tilt,
            spectral_centroid=spectral_centroid,
            voiced_fraction=voiced_fraction,
            duration=duration,
        )

    def _load_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            import soundfile as sf
            y, sr = sf.read(str(path))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)  # Convert to mono
            return y.astype(np.float32), sr
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")

    def _extract_f0(
        self, y: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using PYIN algorithm."""
        try:
            import librosa
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=sr,
                frame_length=int(self.frame_length * sr),
                hop_length=int(self.hop_length * sr),
            )
            # Replace NaN with 0 and create voiced flag
            voiced = ~np.isnan(f0)
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0, voiced
        except ImportError:
            # Fallback: simple autocorrelation-based F0
            return self._simple_f0_extraction(y, sr)

    def _simple_f0_extraction(
        self, y: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple autocorrelation-based F0 extraction."""
        frame_samples = int(self.frame_length * sr)
        hop_samples = int(self.hop_length * sr)
        n_frames = (len(y) - frame_samples) // hop_samples + 1

        f0 = np.zeros(n_frames)
        voiced = np.zeros(n_frames, dtype=bool)

        for i in range(n_frames):
            start = i * hop_samples
            frame = y[start:start + frame_samples]

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            # Find first peak after zero crossing
            min_lag = int(sr / self.f0_max)
            max_lag = int(sr / self.f0_min)

            if max_lag < len(corr):
                search_region = corr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    if corr[peak_idx] > 0.3 * corr[0]:  # Voicing threshold
                        f0[i] = sr / peak_idx
                        voiced[i] = True

        return f0, voiced

    def _extract_cycle_amplitudes(
        self,
        y: np.ndarray,
        sr: int,
        f0: np.ndarray,
        voiced: np.ndarray,
    ) -> np.ndarray:
        """Extract peak amplitudes per pitch cycle."""
        amplitudes = []
        hop_samples = int(self.hop_length * sr)

        for i, (freq, is_voiced) in enumerate(zip(f0, voiced)):
            if is_voiced and freq > 0:
                period = int(sr / freq)
                start = i * hop_samples
                end = start + period
                if end < len(y):
                    cycle = y[start:end]
                    amplitudes.append(np.max(np.abs(cycle)))

        return np.array(amplitudes) if amplitudes else np.array([0.0])

    def _calculate_jitter(
        self, f0_voiced: np.ndarray, jitter_type: JitterType
    ) -> float:
        """Calculate jitter (pitch perturbation)."""
        if len(f0_voiced) < 3:
            return 0.0

        periods = 1.0 / f0_voiced[f0_voiced > 0]
        if len(periods) < 3:
            return 0.0

        if jitter_type == JitterType.LOCAL:
            # Local jitter: average absolute difference between consecutive periods
            diffs = np.abs(np.diff(periods))
            jitter = np.mean(diffs) / np.mean(periods) * 100
        elif jitter_type == JitterType.RAP:
            # RAP: 3-point smoothed
            if len(periods) < 3:
                return 0.0
            smoothed = np.convolve(periods, np.ones(3) / 3, mode='valid')
            diffs = np.abs(periods[1:-1] - smoothed)
            jitter = np.mean(diffs) / np.mean(periods) * 100
        elif jitter_type == JitterType.PPQ5:
            # PPQ5: 5-point smoothed
            if len(periods) < 5:
                return 0.0
            smoothed = np.convolve(periods, np.ones(5) / 5, mode='valid')
            diffs = np.abs(periods[2:-2] - smoothed)
            jitter = np.mean(diffs) / np.mean(periods) * 100
        else:
            jitter = 0.0

        return float(np.clip(jitter, 0, 100))

    def _calculate_shimmer(
        self, amplitudes: np.ndarray, shimmer_type: ShimmerType
    ) -> float:
        """Calculate shimmer (amplitude perturbation)."""
        if len(amplitudes) < 3:
            return 0.0

        amps = amplitudes[amplitudes > 0]
        if len(amps) < 3:
            return 0.0

        if shimmer_type == ShimmerType.LOCAL:
            diffs = np.abs(np.diff(amps))
            shimmer = np.mean(diffs) / np.mean(amps) * 100
        elif shimmer_type == ShimmerType.APQ3:
            if len(amps) < 3:
                return 0.0
            smoothed = np.convolve(amps, np.ones(3) / 3, mode='valid')
            diffs = np.abs(amps[1:-1] - smoothed)
            shimmer = np.mean(diffs) / np.mean(amps) * 100
        elif shimmer_type == ShimmerType.APQ5:
            if len(amps) < 5:
                return 0.0
            smoothed = np.convolve(amps, np.ones(5) / 5, mode='valid')
            diffs = np.abs(amps[2:-2] - smoothed)
            shimmer = np.mean(diffs) / np.mean(amps) * 100
        elif shimmer_type == ShimmerType.APQ11:
            if len(amps) < 11:
                return 0.0
            smoothed = np.convolve(amps, np.ones(11) / 11, mode='valid')
            diffs = np.abs(amps[5:-5] - smoothed)
            shimmer = np.mean(diffs) / np.mean(amps) * 100
        else:
            shimmer = 0.0

        return float(np.clip(shimmer, 0, 100))

    def _calculate_hnr(
        self, y: np.ndarray, sr: int, f0_mean: float
    ) -> float:
        """Calculate Harmonics-to-Noise Ratio (HNR) in dB."""
        if f0_mean <= 0:
            return 0.0

        try:
            # Use autocorrelation method
            period = int(sr / f0_mean)
            n_periods = len(y) // period

            if n_periods < 2:
                return 15.0  # Default reasonable value

            # Calculate autocorrelation at period lag
            acf = np.correlate(y, y, mode='full')
            acf = acf[len(acf) // 2:]
            acf = acf / acf[0]  # Normalize

            if period < len(acf):
                r = acf[period]
                if r > 0 and r < 1:
                    hnr = 10 * np.log10(r / (1 - r))
                    return float(np.clip(hnr, 0, 40))

            return 15.0  # Default
        except Exception:
            return 15.0

    def _calculate_cpp(self, y: np.ndarray, sr: int) -> float:
        """Calculate Cepstral Peak Prominence (CPP) in dB."""
        try:
            # Compute cepstrum
            n_fft = 2048
            spectrum = np.fft.rfft(y[:n_fft] * np.hanning(min(len(y), n_fft)))
            log_spectrum = np.log(np.abs(spectrum) + 1e-10)
            cepstrum = np.fft.irfft(log_spectrum)

            # Find peak in expected F0 range
            min_quefrency = int(sr / 500)  # 500 Hz max
            max_quefrency = int(sr / 50)   # 50 Hz min

            if max_quefrency < len(cepstrum):
                search = np.abs(cepstrum[min_quefrency:max_quefrency])
                if len(search) > 0:
                    peak = np.max(search)

                    # Linear regression for baseline
                    x = np.arange(len(search))
                    slope, intercept = np.polyfit(x, search, 1)
                    baseline = slope * np.argmax(search) + intercept

                    cpp = 20 * np.log10(peak / (baseline + 1e-10))
                    return float(np.clip(cpp, 0, 20))

            return 8.0  # Default reasonable value
        except Exception:
            return 8.0

    def _calculate_spectral_tilt(self, y: np.ndarray, sr: int) -> float:
        """Calculate spectral tilt in dB/octave."""
        try:
            # Compute spectrum
            n_fft = 2048
            spectrum = np.abs(np.fft.rfft(y[:n_fft] * np.hanning(min(len(y), n_fft))))
            freqs = np.fft.rfftfreq(n_fft, 1 / sr)

            # Fit line in log-log space
            log_freqs = np.log2(freqs[1:] + 1e-10)
            log_spectrum = 20 * np.log10(spectrum[1:] + 1e-10)

            # Linear regression
            slope, _ = np.polyfit(log_freqs, log_spectrum, 1)

            return float(slope)  # dB per octave
        except Exception:
            return -12.0  # Typical value

    def _calculate_spectral_centroid(self, y: np.ndarray, sr: int) -> float:
        """Calculate spectral centroid in Hz."""
        try:
            n_fft = 2048
            spectrum = np.abs(np.fft.rfft(y[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1 / sr)

            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            return float(centroid)
        except Exception:
            return 1500.0  # Typical speech value


# Convenience functions

def analyze_voice_quality(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> VoiceQualityMetrics:
    """
    Analyze voice quality from audio.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate (required if audio is array)

    Returns:
        VoiceQualityMetrics with all measurements
    """
    analyzer = VoiceQualityAnalyzer()
    return analyzer.analyze(audio, sample_rate)


def get_jitter(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    jitter_type: JitterType = JitterType.LOCAL,
) -> float:
    """Get jitter value from audio."""
    metrics = analyze_voice_quality(audio, sample_rate)
    if jitter_type == JitterType.LOCAL:
        return metrics.jitter_local
    elif jitter_type == JitterType.RAP:
        return metrics.jitter_rap
    elif jitter_type == JitterType.PPQ5:
        return metrics.jitter_ppq5
    return metrics.jitter_local


def get_shimmer(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    shimmer_type: ShimmerType = ShimmerType.LOCAL,
) -> float:
    """Get shimmer value from audio."""
    metrics = analyze_voice_quality(audio, sample_rate)
    if shimmer_type == ShimmerType.LOCAL:
        return metrics.shimmer_local
    elif shimmer_type == ShimmerType.APQ3:
        return metrics.shimmer_apq3
    elif shimmer_type == ShimmerType.APQ5:
        return metrics.shimmer_apq5
    elif shimmer_type == ShimmerType.APQ11:
        return metrics.shimmer_apq11
    return metrics.shimmer_local


def get_hnr(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> float:
    """Get Harmonics-to-Noise Ratio in dB."""
    metrics = analyze_voice_quality(audio, sample_rate)
    return metrics.hnr


def get_cpp(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> float:
    """Get Cepstral Peak Prominence in dB."""
    metrics = analyze_voice_quality(audio, sample_rate)
    return metrics.cpp
