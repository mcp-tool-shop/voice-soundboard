"""
Prosody Analysis and Control

Prosody encompasses the suprasegmental features of speech:
- Pitch (F0) contours
- Duration patterns
- Intensity/energy
- Rhythm and phrasing

This module provides tools for analyzing and modifying speech prosody.

Reference:
- TTS Prosody Modeling: https://apxml.com/courses/speech-recognition-synthesis-asr-tts/
- SSML Specification: https://www.w3.org/TR/speech-synthesis/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import numpy as np


class BreakStrength(Enum):
    """Pause/break strength levels."""
    NONE = "none"           # No pause
    X_WEAK = "x-weak"       # Minimal pause
    WEAK = "weak"           # Short pause (comma)
    MEDIUM = "medium"       # Medium pause
    STRONG = "strong"       # Long pause (period)
    X_STRONG = "x-strong"   # Very long pause (paragraph)


@dataclass
class PitchContour:
    """
    Pitch (F0) contour data.

    Attributes:
        times: Time points (seconds)
        frequencies: F0 values (Hz)
        voiced: Voicing flag per frame
        confidence: Confidence per frame
    """
    times: np.ndarray
    frequencies: np.ndarray
    voiced: np.ndarray
    confidence: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return float(self.times[-1]) if len(self.times) > 0 else 0.0

    @property
    def mean_f0(self) -> float:
        """Mean F0 of voiced regions."""
        voiced_f0 = self.frequencies[self.voiced]
        return float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    @property
    def f0_range(self) -> Tuple[float, float]:
        """F0 range (min, max) of voiced regions."""
        voiced_f0 = self.frequencies[self.voiced]
        if len(voiced_f0) > 0:
            return (float(np.min(voiced_f0)), float(np.max(voiced_f0)))
        return (0.0, 0.0)

    def resample(self, n_points: int) -> 'PitchContour':
        """Resample to fixed number of points."""
        new_times = np.linspace(0, self.times[-1], n_points)
        new_freqs = np.interp(new_times, self.times, self.frequencies)
        new_voiced = np.interp(new_times, self.times, self.voiced.astype(float)) > 0.5
        return PitchContour(new_times, new_freqs, new_voiced)


@dataclass
class DurationPattern:
    """
    Duration pattern for phonemes/syllables.

    Attributes:
        units: List of units (phonemes, syllables, or words)
        durations: Duration for each unit (seconds)
        boundaries: Time boundaries
    """
    units: List[str]
    durations: np.ndarray
    boundaries: np.ndarray

    @property
    def total_duration(self) -> float:
        """Total duration."""
        return float(np.sum(self.durations))

    @property
    def mean_duration(self) -> float:
        """Mean unit duration."""
        return float(np.mean(self.durations))

    def scale(self, factor: float) -> 'DurationPattern':
        """Scale all durations by factor."""
        return DurationPattern(
            units=self.units,
            durations=self.durations * factor,
            boundaries=self.boundaries * factor,
        )


@dataclass
class ProsodyContour:
    """
    Complete prosody representation.

    Attributes:
        pitch: Pitch contour
        duration: Duration pattern
        energy: Energy contour (optional)
        pauses: Pause locations and durations
    """
    pitch: Optional[PitchContour] = None
    duration: Optional[DurationPattern] = None
    energy: Optional[np.ndarray] = None
    pauses: List[Tuple[float, float]] = field(default_factory=list)  # (time, duration)

    def set_pitch_range(self, low: float, high: float):
        """Set target pitch range."""
        if self.pitch is not None:
            # Normalize to [0, 1] then scale to range
            voiced = self.pitch.voiced
            f0 = self.pitch.frequencies.copy()
            f0_voiced = f0[voiced]
            if len(f0_voiced) > 0:
                f0_min, f0_max = np.min(f0_voiced), np.max(f0_voiced)
                if f0_max > f0_min:
                    f0_norm = (f0 - f0_min) / (f0_max - f0_min)
                    f0_scaled = f0_norm * (high - low) + low
                    f0[voiced] = f0_scaled[voiced]
                    self.pitch.frequencies = f0

    def add_emphasis(
        self,
        time: float,
        pitch_boost: float = 1.2,
        duration_boost: float = 1.1,
    ):
        """Add emphasis at specific time."""
        # This would modify pitch and duration at the specified time
        # Simplified: just record the emphasis point
        pass


class ProsodyAnalyzer:
    """
    Analyze prosody from speech audio.

    Extracts pitch contours, duration patterns, and energy profiles.

    Example:
        analyzer = ProsodyAnalyzer()
        prosody = analyzer.analyze("speech.wav")
        print(f"Mean F0: {prosody.pitch.mean_f0:.1f} Hz")
    """

    def __init__(
        self,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        hop_length: float = 0.010,
    ):
        """
        Initialize analyzer.

        Args:
            f0_min: Minimum F0 (Hz)
            f0_max: Maximum F0 (Hz)
            hop_length: Analysis hop (seconds)
        """
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.hop_length = hop_length

    def analyze(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> ProsodyContour:
        """
        Analyze prosody from audio.

        Args:
            audio: Audio file or array
            sample_rate: Sample rate if audio is array

        Returns:
            ProsodyContour with extracted prosody
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required for array input")
            y = audio
            sr = sample_rate

        # Extract pitch
        pitch = self._extract_pitch(y, sr)

        # Extract energy
        energy = self._extract_energy(y, sr)

        # Detect pauses
        pauses = self._detect_pauses(y, sr)

        return ProsodyContour(
            pitch=pitch,
            duration=None,  # Would need text alignment for durations
            energy=energy,
            pauses=pauses,
        )

    def _load_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            import soundfile as sf
            y, sr = sf.read(str(path))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            return y.astype(np.float32), sr
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")

    def _extract_pitch(self, y: np.ndarray, sr: int) -> PitchContour:
        """Extract pitch contour."""
        try:
            import librosa
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=self.f0_min, fmax=self.f0_max, sr=sr,
                hop_length=int(self.hop_length * sr),
            )
            voiced = ~np.isnan(f0)
            f0 = np.nan_to_num(f0, nan=0.0)
            times = np.arange(len(f0)) * self.hop_length
            return PitchContour(times=times, frequencies=f0, voiced=voiced)
        except ImportError:
            # Fallback
            return self._simple_pitch_extraction(y, sr)

    def _simple_pitch_extraction(self, y: np.ndarray, sr: int) -> PitchContour:
        """Simple pitch extraction fallback."""
        hop_samples = int(self.hop_length * sr)
        n_frames = len(y) // hop_samples

        f0 = np.zeros(n_frames)
        voiced = np.zeros(n_frames, dtype=bool)
        times = np.arange(n_frames) * self.hop_length

        for i in range(n_frames):
            start = i * hop_samples
            frame = y[start:start + hop_samples * 2]
            if len(frame) < hop_samples:
                continue

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]
            corr = corr / (corr[0] + 1e-10)

            min_lag = int(sr / self.f0_max)
            max_lag = min(int(sr / self.f0_min), len(corr) - 1)

            if max_lag > min_lag:
                search = corr[min_lag:max_lag]
                if len(search) > 0 and np.max(search) > 0.3:
                    peak = np.argmax(search) + min_lag
                    f0[i] = sr / peak
                    voiced[i] = True

        return PitchContour(times=times, frequencies=f0, voiced=voiced)

    def _extract_energy(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract energy contour."""
        hop_samples = int(self.hop_length * sr)
        n_frames = len(y) // hop_samples

        energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_samples
            frame = y[start:start + hop_samples]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        return energy

    def _detect_pauses(
        self, y: np.ndarray, sr: int, threshold: float = 0.02
    ) -> List[Tuple[float, float]]:
        """Detect pauses (silence regions)."""
        hop_samples = int(self.hop_length * sr)
        n_frames = len(y) // hop_samples

        # Calculate energy
        energy = self._extract_energy(y, sr)
        threshold_val = threshold * np.max(energy)

        # Find silent regions
        is_silent = energy < threshold_val

        pauses = []
        in_pause = False
        pause_start = 0

        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                in_pause = True
                pause_start = i * self.hop_length
            elif not silent and in_pause:
                in_pause = False
                pause_duration = i * self.hop_length - pause_start
                if pause_duration > 0.1:  # Minimum 100ms pause
                    pauses.append((pause_start, pause_duration))

        return pauses


class ProsodyModifier:
    """
    Modify prosody of speech audio.

    Allows changing pitch, duration, and energy while preserving
    voice quality.

    Example:
        modifier = ProsodyModifier()

        # Raise pitch by 20%
        higher = modifier.modify_pitch(audio, ratio=1.2)

        # Slow down by 30%
        slower = modifier.modify_duration(audio, ratio=1.3)
    """

    def __init__(self):
        """Initialize modifier."""
        pass

    def modify_pitch(
        self,
        audio: Union[str, Path, np.ndarray],
        ratio: float = 1.0,
        semitones: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Modify pitch while preserving duration.

        Args:
            audio: Audio input
            ratio: Pitch ratio (1.0 = unchanged, 1.2 = 20% higher)
            semitones: Alternative: shift in semitones
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required for array input")
            y = audio
            sr = sample_rate

        try:
            import librosa

            if semitones is not None:
                n_steps = semitones
            else:
                n_steps = 12 * np.log2(ratio)

            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            return y_shifted, sr

        except ImportError:
            # Without librosa, use simple resampling
            return self._simple_pitch_shift(y, sr, ratio)

    def modify_duration(
        self,
        audio: Union[str, Path, np.ndarray],
        ratio: float = 1.0,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Modify duration while preserving pitch.

        Args:
            audio: Audio input
            ratio: Duration ratio (1.0 = unchanged, 1.5 = 50% longer)
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required for array input")
            y = audio
            sr = sample_rate

        try:
            import librosa
            y_stretched = librosa.effects.time_stretch(y, rate=1.0 / ratio)
            return y_stretched, sr
        except ImportError:
            # Without librosa, use simple resampling (changes pitch too)
            from scipy import signal
            n_out = int(len(y) * ratio)
            y_resampled = signal.resample(y, n_out)
            return y_resampled.astype(np.float32), sr

    def modify_energy(
        self,
        audio: Union[str, Path, np.ndarray],
        ratio: float = 1.0,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Modify energy/loudness.

        Args:
            audio: Audio input
            ratio: Energy ratio (1.0 = unchanged)
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required for array input")
            y = audio
            sr = sample_rate

        # Simple amplitude scaling with soft clipping to avoid distortion
        y_scaled = y * ratio
        y_scaled = np.tanh(y_scaled)  # Soft clip
        y_scaled = y_scaled / (np.max(np.abs(y_scaled)) + 1e-10)
        y_scaled = y_scaled * np.max(np.abs(y)) * min(ratio, 1.5)

        return y_scaled, sr

    def apply_contour(
        self,
        audio: Union[str, Path, np.ndarray],
        pitch_contour: Optional[PitchContour] = None,
        energy_contour: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply custom prosody contours to audio.

        Args:
            audio: Audio input
            pitch_contour: Target pitch contour
            energy_contour: Target energy contour
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # This would require sophisticated PSOLA or vocoder-based processing
        # Simplified: apply average modifications

        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required for array input")
            y = audio
            sr = sample_rate

        result = y.copy()

        if pitch_contour is not None:
            # Get current pitch
            analyzer = ProsodyAnalyzer()
            current_prosody = analyzer.analyze(result, sr)
            if current_prosody.pitch is not None:
                current_mean = current_prosody.pitch.mean_f0
                target_mean = pitch_contour.mean_f0
                if current_mean > 0 and target_mean > 0:
                    ratio = target_mean / current_mean
                    result, sr = self.modify_pitch(result, ratio=ratio, sample_rate=sr)

        if energy_contour is not None:
            # Apply energy envelope
            current_energy = self._get_energy_envelope(result, sr)
            if len(current_energy) > 0 and len(energy_contour) > 0:
                # Resample target to match length
                target_resampled = np.interp(
                    np.linspace(0, 1, len(current_energy)),
                    np.linspace(0, 1, len(energy_contour)),
                    energy_contour
                )
                # Apply envelope ratio
                ratio = (target_resampled + 1e-10) / (current_energy + 1e-10)
                ratio = np.clip(ratio, 0.5, 2.0)
                ratio_upsampled = np.interp(
                    np.arange(len(result)),
                    np.linspace(0, len(result), len(ratio)),
                    ratio
                )
                result = result * ratio_upsampled

        return result, sr

    def _load_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            import soundfile as sf
            y, sr = sf.read(str(path))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            return y.astype(np.float32), sr
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")

    def _simple_pitch_shift(
        self, y: np.ndarray, sr: int, ratio: float
    ) -> Tuple[np.ndarray, int]:
        """Simple pitch shift via resampling."""
        from scipy import signal

        # Resample to change pitch
        n_out = int(len(y) / ratio)
        y_resampled = signal.resample(y, n_out)

        # Time stretch back to original duration
        y_stretched = signal.resample(y_resampled, len(y))

        return y_stretched.astype(np.float32), sr

    def _get_energy_envelope(
        self, y: np.ndarray, sr: int, hop_length: float = 0.010
    ) -> np.ndarray:
        """Get energy envelope."""
        hop_samples = int(hop_length * sr)
        n_frames = len(y) // hop_samples
        energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_samples
            frame = y[start:start + hop_samples]
            energy[i] = np.sqrt(np.mean(frame ** 2))
        return energy


# Convenience functions

def analyze_prosody(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> ProsodyContour:
    """
    Analyze prosody from audio.

    Args:
        audio: Audio file or array
        sample_rate: Sample rate if audio is array

    Returns:
        ProsodyContour with extracted prosody
    """
    analyzer = ProsodyAnalyzer()
    return analyzer.analyze(audio, sample_rate)


def modify_prosody(
    audio: Union[str, Path, np.ndarray],
    pitch_ratio: float = 1.0,
    duration_ratio: float = 1.0,
    energy_ratio: float = 1.0,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Modify prosody of audio.

    Args:
        audio: Audio file or array
        pitch_ratio: Pitch modification ratio
        duration_ratio: Duration modification ratio
        energy_ratio: Energy modification ratio
        sample_rate: Sample rate if audio is array

    Returns:
        Tuple of (modified_audio, sample_rate)
    """
    modifier = ProsodyModifier()

    # Load audio once
    if isinstance(audio, (str, Path)):
        import soundfile as sf
        y, sr = sf.read(str(audio))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate required for array input")
        y = audio
        sr = sample_rate

    # Apply modifications
    if pitch_ratio != 1.0:
        y, sr = modifier.modify_pitch(y, ratio=pitch_ratio, sample_rate=sr)

    if duration_ratio != 1.0:
        y, sr = modifier.modify_duration(y, ratio=duration_ratio, sample_rate=sr)

    if energy_ratio != 1.0:
        y, sr = modifier.modify_energy(y, ratio=energy_ratio, sample_rate=sr)

    return y, sr
