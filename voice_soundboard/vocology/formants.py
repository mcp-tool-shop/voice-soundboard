"""
Formant Analysis and Shifting

Formants are resonant frequencies of the vocal tract that shape vowel identity
and voice "color." This module provides formant analysis and manipulation.

Key formants:
- F1 (~500 Hz): Tongue height (open/close vowels)
- F2 (~1500 Hz): Tongue frontness/backness
- F3 (~2500 Hz): Lip rounding, voice color
- F4+ (~3500 Hz): Individual voice characteristics

Applications:
- Voice modification (deeper/brighter voice)
- Gender transformation
- Character voice creation
- Accent simulation

Reference:
- Fant, G. (1960). Acoustic Theory of Speech Production
- VoiceScienceWorks: http://www.voicescienceworks.org/harmonics-vs-formants.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np


@dataclass
class FormantFrequencies:
    """
    Formant frequency measurements.

    Attributes:
        f1: First formant frequency (Hz) - tongue height
        f2: Second formant frequency (Hz) - tongue position
        f3: Third formant frequency (Hz) - lip rounding
        f4: Fourth formant frequency (Hz) - voice color
        f5: Fifth formant frequency (Hz) - individual characteristics
        bandwidths: Bandwidth of each formant (Hz)
    """
    f1: float
    f2: float
    f3: float
    f4: float
    f5: Optional[float] = None
    bandwidths: Optional[List[float]] = None

    @property
    def as_list(self) -> List[float]:
        """Return formants as list."""
        formants = [self.f1, self.f2, self.f3, self.f4]
        if self.f5 is not None:
            formants.append(self.f5)
        return formants

    def singer_formant_present(self) -> bool:
        """
        Check if singer's formant (2800-3400 Hz cluster) is present.

        The singer's formant is a resonance peak around 3 kHz that gives
        trained singers their characteristic "ring" and projection.
        """
        # Check if F3, F4, F5 cluster around 3000 Hz
        if self.f5 is None:
            return False

        cluster = [self.f3, self.f4, self.f5]
        mean_freq = np.mean(cluster)
        spread = np.std(cluster)

        return 2800 <= mean_freq <= 3400 and spread < 300


@dataclass
class FormantAnalysis:
    """
    Complete formant analysis results.

    Attributes:
        formants: Time-varying formant frequencies [n_frames, n_formants]
        mean_formants: Mean formant frequencies
        std_formants: Standard deviation of formants
        sample_rate: Audio sample rate
        hop_length: Hop length in samples
    """
    formants: np.ndarray  # [n_frames, n_formants]
    mean_formants: FormantFrequencies
    std_formants: List[float]
    sample_rate: int
    hop_length: int

    @property
    def n_frames(self) -> int:
        """Number of analysis frames."""
        return self.formants.shape[0]

    @property
    def n_formants(self) -> int:
        """Number of formants tracked."""
        return self.formants.shape[1]

    def get_frame(self, frame_idx: int) -> FormantFrequencies:
        """Get formants at specific frame."""
        f = self.formants[frame_idx]
        return FormantFrequencies(
            f1=f[0], f2=f[1], f3=f[2], f4=f[3],
            f5=f[4] if len(f) > 4 else None
        )


class FormantAnalyzer:
    """
    Analyze formant frequencies from speech audio.

    Uses Linear Predictive Coding (LPC) for formant estimation.

    Example:
        analyzer = FormantAnalyzer()
        analysis = analyzer.analyze("speech.wav")
        print(f"Mean F1: {analysis.mean_formants.f1:.0f} Hz")
    """

    def __init__(
        self,
        n_formants: int = 5,
        lpc_order: Optional[int] = None,
        pre_emphasis: float = 0.97,
        frame_length: float = 0.025,
        hop_length: float = 0.010,
    ):
        """
        Initialize the analyzer.

        Args:
            n_formants: Number of formants to track
            lpc_order: LPC order (default: 2 * n_formants + 2)
            pre_emphasis: Pre-emphasis coefficient
            frame_length: Analysis frame length (seconds)
            hop_length: Hop between frames (seconds)
        """
        self.n_formants = n_formants
        self.lpc_order = lpc_order or (2 * n_formants + 2)
        self.pre_emphasis = pre_emphasis
        self.frame_length = frame_length
        self.hop_length = hop_length

    def analyze(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> FormantAnalysis:
        """
        Analyze formants in audio.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            FormantAnalysis with time-varying formant tracks
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required when audio is array")
            y = audio
            sr = sample_rate

        # Pre-emphasis
        y_emph = np.append(y[0], y[1:] - self.pre_emphasis * y[:-1])

        # Frame-by-frame analysis
        frame_samples = int(self.frame_length * sr)
        hop_samples = int(self.hop_length * sr)
        n_frames = (len(y_emph) - frame_samples) // hop_samples + 1

        formants = np.zeros((n_frames, self.n_formants))

        for i in range(n_frames):
            start = i * hop_samples
            frame = y_emph[start:start + frame_samples]
            frame = frame * np.hamming(len(frame))

            # LPC analysis
            frame_formants = self._lpc_formants(frame, sr)
            formants[i, :len(frame_formants)] = frame_formants[:self.n_formants]

        # Calculate statistics
        mean_f = np.mean(formants, axis=0)
        std_f = np.std(formants, axis=0).tolist()

        mean_formants = FormantFrequencies(
            f1=mean_f[0], f2=mean_f[1], f3=mean_f[2], f4=mean_f[3],
            f5=mean_f[4] if self.n_formants > 4 else None
        )

        return FormantAnalysis(
            formants=formants,
            mean_formants=mean_formants,
            std_formants=std_f,
            sample_rate=sr,
            hop_length=hop_samples,
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

    def _lpc_formants(self, frame: np.ndarray, sr: int) -> List[float]:
        """Extract formants using LPC."""
        # Compute LPC coefficients
        lpc_coeffs = self._compute_lpc(frame, self.lpc_order)

        # Find roots of LPC polynomial
        roots = np.roots(lpc_coeffs)

        # Convert to frequencies
        formants = []
        for root in roots:
            if np.imag(root) >= 0:  # Take positive frequencies only
                angle = np.angle(root)
                freq = angle * sr / (2 * np.pi)
                if 50 < freq < sr / 2 - 50:  # Valid frequency range
                    # Bandwidth from root magnitude
                    bw = -np.log(np.abs(root)) * sr / np.pi
                    if bw < 500:  # Reasonable bandwidth
                        formants.append(freq)

        # Sort and return top N formants
        formants.sort()
        return formants[:self.n_formants]

    def _compute_lpc(self, frame: np.ndarray, order: int) -> np.ndarray:
        """Compute LPC coefficients using Levinson-Durbin."""
        # Autocorrelation
        n = len(frame)
        r = np.correlate(frame, frame, mode='full')[n - 1:n + order]

        # Levinson-Durbin recursion
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            lam = np.sum(a[:i] * r[i:0:-1])
            gamma = lam / e if e != 0 else 0

            a_new = a.copy()
            for j in range(1, i + 1):
                a_new[j] = a[j] - gamma * a[i - j]

            a = a_new
            e = e * (1 - gamma ** 2)

        return a


class FormantShifter:
    """
    Shift formant frequencies for voice modification.

    Formant shifting changes the perceived vocal tract size:
    - Lower formants → Larger vocal tract → Deeper voice
    - Higher formants → Smaller vocal tract → Higher voice

    Example:
        shifter = FormantShifter()

        # Make voice deeper (lower formants)
        deeper = shifter.shift("voice.wav", ratio=0.9)

        # Make voice brighter (higher formants)
        brighter = shifter.shift("voice.wav", ratio=1.1)
    """

    def __init__(self, method: str = "psola"):
        """
        Initialize the shifter.

        Args:
            method: Shifting method ('psola', 'lpc', 'phase_vocoder')
        """
        self.method = method

    def shift(
        self,
        audio: Union[str, Path, np.ndarray],
        ratio: float,
        sample_rate: Optional[int] = None,
        preserve_pitch: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Shift formant frequencies.

        Args:
            audio: Audio file path or numpy array
            ratio: Formant shift ratio (0.8 = deeper, 1.2 = brighter)
            sample_rate: Sample rate (required if audio is array)
            preserve_pitch: Keep original pitch (True) or shift with formants

        Returns:
            Tuple of (shifted_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required when audio is array")
            y = audio
            sr = sample_rate

        if self.method == "psola":
            return self._shift_psola(y, sr, ratio, preserve_pitch)
        elif self.method == "lpc":
            return self._shift_lpc(y, sr, ratio)
        else:
            return self._shift_phase_vocoder(y, sr, ratio, preserve_pitch)

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

    def _shift_psola(
        self,
        y: np.ndarray,
        sr: int,
        ratio: float,
        preserve_pitch: bool,
    ) -> Tuple[np.ndarray, int]:
        """
        Formant shift using PSOLA-like resampling.

        This is a simple but effective method:
        1. Resample to change formants
        2. Time-stretch to restore duration
        3. Pitch-shift to restore pitch (if preserve_pitch)
        """
        try:
            import librosa

            # Resample to shift formants
            # Lower ratio = lower sample rate = lower formants
            target_sr = int(sr * ratio)
            y_shifted = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            # Time-stretch to restore original duration
            stretch_factor = ratio
            y_stretched = librosa.effects.time_stretch(y_shifted, rate=stretch_factor)

            if preserve_pitch:
                # Pitch shift to compensate
                # Resampling changes pitch by ratio, so shift back
                semitones = -12 * np.log2(ratio)
                y_final = librosa.effects.pitch_shift(
                    y_stretched, sr=sr, n_steps=semitones
                )
            else:
                y_final = y_stretched

            return y_final, sr

        except ImportError:
            # Fallback: simple resampling only
            return self._simple_resample(y, sr, ratio)

    def _shift_lpc(
        self,
        y: np.ndarray,
        sr: int,
        ratio: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Formant shift using LPC analysis-resynthesis.

        More accurate but computationally expensive.
        """
        # This would require full LPC vocoder implementation
        # Fall back to PSOLA for now
        return self._shift_psola(y, sr, ratio, preserve_pitch=True)

    def _shift_phase_vocoder(
        self,
        y: np.ndarray,
        sr: int,
        ratio: float,
        preserve_pitch: bool,
    ) -> Tuple[np.ndarray, int]:
        """Formant shift using phase vocoder."""
        # Similar to PSOLA but using phase vocoder for time-stretching
        return self._shift_psola(y, sr, ratio, preserve_pitch)

    def _simple_resample(
        self,
        y: np.ndarray,
        sr: int,
        ratio: float,
    ) -> Tuple[np.ndarray, int]:
        """Simple resampling fallback."""
        from scipy import signal

        # Number of output samples
        n_out = int(len(y) / ratio)
        y_resampled = signal.resample(y, n_out)

        return y_resampled.astype(np.float32), sr

    def shift_selective(
        self,
        audio: Union[str, Path, np.ndarray],
        f1_ratio: float = 1.0,
        f2_ratio: float = 1.0,
        f3_ratio: float = 1.0,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Shift individual formants selectively.

        Args:
            audio: Audio input
            f1_ratio: F1 shift ratio
            f2_ratio: F2 shift ratio
            f3_ratio: F3 shift ratio
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (shifted_audio, sample_rate)

        Note:
            Selective formant shifting requires more sophisticated
            signal processing (LPC resynthesis or spectral manipulation).
            This is a simplified implementation.
        """
        # For selective shifting, we'd need to:
        # 1. Decompose into source (glottal) and filter (vocal tract)
        # 2. Modify filter (formants) selectively
        # 3. Resynthesize

        # Simplified: use average ratio
        avg_ratio = (f1_ratio + f2_ratio + f3_ratio) / 3
        return self.shift(audio, avg_ratio, sample_rate)


# Convenience functions

def analyze_formants(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    n_formants: int = 5,
) -> FormantAnalysis:
    """
    Analyze formants in audio.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate (required if audio is array)
        n_formants: Number of formants to track

    Returns:
        FormantAnalysis with formant tracks
    """
    analyzer = FormantAnalyzer(n_formants=n_formants)
    return analyzer.analyze(audio, sample_rate)


def shift_formants(
    audio: Union[str, Path, np.ndarray],
    ratio: float,
    sample_rate: Optional[int] = None,
    preserve_pitch: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Shift formant frequencies.

    Args:
        audio: Audio file path or numpy array
        ratio: Formant shift ratio (0.8 = deeper, 1.2 = brighter)
        sample_rate: Sample rate (required if audio is array)
        preserve_pitch: Keep original pitch

    Returns:
        Tuple of (shifted_audio, sample_rate)
    """
    shifter = FormantShifter()
    return shifter.shift(audio, ratio, sample_rate, preserve_pitch)
