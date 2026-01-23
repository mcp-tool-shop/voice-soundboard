"""
Phonation Types

Detection and synthesis of different phonation types (voice qualities).

Phonation types on the continuum:
- Breathy: Open glottis, air escaping, soft/intimate
- Modal: Normal vibration, clear voice
- Creaky (Vocal Fry): Tight glottis, low frequency, gravelly
- Harsh/Pressed: Very tense, strained

Reference:
- Laver, J. (1980). The Phonetic Description of Voice Quality
- Gordon, M. & Ladefoged, P. (2001). Phonation types: a cross-linguistic overview
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import numpy as np


class PhonationType(Enum):
    """Types of phonation (voice quality)."""
    MODAL = "modal"           # Normal voice
    BREATHY = "breathy"       # Soft, airy
    CREAKY = "creaky"         # Vocal fry, gravelly
    HARSH = "harsh"           # Tense, pressed
    FALSETTO = "falsetto"     # High, light
    WHISPER = "whisper"       # No vocal fold vibration


# Acoustic parameters associated with each phonation type
PHONATION_PARAMS: Dict[PhonationType, Dict[str, float]] = {
    PhonationType.MODAL: {
        "jitter_target": 0.5,
        "shimmer_target": 2.0,
        "hnr_target": 20.0,
        "spectral_tilt": -12.0,
        "noise_level": 0.0,
        "f0_variation": 0.1,
    },
    PhonationType.BREATHY: {
        "jitter_target": 0.6,
        "shimmer_target": 6.0,
        "hnr_target": 12.0,
        "spectral_tilt": -18.0,
        "noise_level": 0.3,
        "f0_variation": 0.08,
    },
    PhonationType.CREAKY: {
        "jitter_target": 2.5,
        "shimmer_target": 4.0,
        "hnr_target": 15.0,
        "spectral_tilt": -8.0,
        "noise_level": 0.1,
        "f0_variation": 0.25,
        "subharmonics": True,
    },
    PhonationType.HARSH: {
        "jitter_target": 1.5,
        "shimmer_target": 5.0,
        "hnr_target": 10.0,
        "spectral_tilt": -6.0,
        "noise_level": 0.2,
        "f0_variation": 0.15,
    },
    PhonationType.FALSETTO: {
        "jitter_target": 0.4,
        "shimmer_target": 2.5,
        "hnr_target": 18.0,
        "spectral_tilt": -15.0,
        "noise_level": 0.1,
        "f0_variation": 0.05,
        "f0_shift": 1.5,  # Higher pitch
    },
    PhonationType.WHISPER: {
        "jitter_target": 0.0,  # No periodic vibration
        "shimmer_target": 0.0,
        "hnr_target": 0.0,
        "spectral_tilt": -3.0,
        "noise_level": 1.0,  # All noise
        "f0_variation": 0.0,
    },
}


@dataclass
class PhonationAnalysisResult:
    """
    Result of phonation type analysis.

    Attributes:
        detected_type: Most likely phonation type
        confidence: Confidence score (0-1)
        type_scores: Scores for each phonation type
        features: Extracted acoustic features
    """
    detected_type: PhonationType
    confidence: float
    type_scores: Dict[PhonationType, float]
    features: Dict[str, float]


class PhonationAnalyzer:
    """
    Detect phonation type from audio.

    Uses acoustic features (jitter, shimmer, HNR, spectral tilt)
    to classify voice quality.

    Example:
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze("speech.wav")
        print(f"Phonation: {result.detected_type.value}")
        print(f"Confidence: {result.confidence:.2%}")
    """

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> PhonationAnalysisResult:
        """
        Analyze phonation type in audio.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            PhonationAnalysisResult with detected type and scores
        """
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer

        # Get voice quality metrics
        analyzer = VoiceQualityAnalyzer()
        metrics = analyzer.analyze(audio, sample_rate)

        # Extract relevant features
        features = {
            "jitter": metrics.jitter_local,
            "shimmer": metrics.shimmer_local,
            "hnr": metrics.hnr,
            "spectral_tilt": metrics.spectral_tilt,
            "f0_variation": metrics.f0_std / metrics.f0_mean if metrics.f0_mean > 0 else 0,
        }

        # Score each phonation type
        scores = {}
        for ptype, params in PHONATION_PARAMS.items():
            score = self._calculate_similarity(features, params)
            scores[ptype] = score

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        # Find best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        return PhonationAnalysisResult(
            detected_type=best_type,
            confidence=confidence,
            type_scores=scores,
            features=features,
        )

    def _calculate_similarity(
        self,
        features: Dict[str, float],
        target_params: Dict[str, float],
    ) -> float:
        """Calculate similarity between features and target parameters."""
        score = 0.0
        weights = {
            "jitter": 1.0,
            "shimmer": 1.0,
            "hnr": 1.5,
            "spectral_tilt": 0.8,
            "f0_variation": 0.5,
        }

        for feature, value in features.items():
            if feature in target_params and feature in weights:
                target = target_params.get(f"{feature}_target", target_params.get(feature, 0))
                # Gaussian similarity
                diff = abs(value - target)
                sigma = target * 0.5 + 1.0  # Scale-dependent sigma
                similarity = np.exp(-0.5 * (diff / sigma) ** 2)
                score += weights[feature] * similarity

        return score


class PhonationSynthesizer:
    """
    Apply phonation type characteristics to audio.

    Modifies audio to sound more like a target phonation type
    by adjusting jitter, shimmer, noise, and spectral characteristics.

    Example:
        synthesizer = PhonationSynthesizer()

        # Make voice breathier
        breathy = synthesizer.apply(audio, PhonationType.BREATHY, intensity=0.7)

        # Add vocal fry
        creaky = synthesizer.apply(audio, PhonationType.CREAKY, intensity=0.5)
    """

    def __init__(self):
        """Initialize the synthesizer."""
        pass

    def apply(
        self,
        audio: Union[str, Path, np.ndarray],
        phonation_type: PhonationType,
        intensity: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply phonation type characteristics to audio.

        Args:
            audio: Audio file path or numpy array
            phonation_type: Target phonation type
            intensity: Effect intensity (0-1)
            sample_rate: Sample rate (required if audio is array)

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required when audio is array")
            y = audio.copy()
            sr = sample_rate

        # Get target parameters
        params = PHONATION_PARAMS[phonation_type]

        # Apply modifications based on phonation type
        if phonation_type == PhonationType.BREATHY:
            y = self._add_breathiness(y, sr, intensity, params)
        elif phonation_type == PhonationType.CREAKY:
            y = self._add_creakiness(y, sr, intensity, params)
        elif phonation_type == PhonationType.HARSH:
            y = self._add_harshness(y, sr, intensity, params)
        elif phonation_type == PhonationType.FALSETTO:
            y = self._apply_falsetto(y, sr, intensity, params)
        elif phonation_type == PhonationType.WHISPER:
            y = self._apply_whisper(y, sr, intensity)
        # MODAL is default, no modification needed

        return y, sr

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

    def _add_breathiness(
        self,
        y: np.ndarray,
        sr: int,
        intensity: float,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Add breathiness to audio."""
        # Add aspiration noise
        noise_level = params["noise_level"] * intensity
        noise = np.random.randn(len(y)) * noise_level * np.std(y)

        # Apply spectral shaping to noise (high-pass for breathiness)
        from scipy import signal
        b, a = signal.butter(2, 1000 / (sr / 2), btype='high')
        noise = signal.filtfilt(b, a, noise)

        # Mix with original
        y_breathy = y + noise

        # Apply spectral tilt (reduce high frequencies less)
        # This is simplified - full implementation would use spectral envelope
        tilt_factor = 1.0 - intensity * 0.3
        b, a = signal.butter(1, 3000 / (sr / 2), btype='low')
        y_filtered = signal.filtfilt(b, a, y)
        y_breathy = tilt_factor * y_filtered + (1 - tilt_factor) * y_breathy

        # Normalize
        y_breathy = y_breathy / (np.max(np.abs(y_breathy)) + 1e-10)
        y_breathy = y_breathy * np.max(np.abs(y))

        return y_breathy

    def _add_creakiness(
        self,
        y: np.ndarray,
        sr: int,
        intensity: float,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Add creaky voice (vocal fry) characteristics."""
        # Add irregular amplitude modulation
        # Simulate irregular glottal pulses
        mod_freq = 30 + np.random.randn(len(y)) * 10 * intensity
        mod_freq = np.clip(mod_freq, 10, 80)

        # Cumulative phase for irregular modulation
        phase = np.cumsum(2 * np.pi * mod_freq / sr)
        modulation = 0.5 + 0.5 * np.cos(phase)
        modulation = modulation ** (1 + intensity)  # Make pulses sharper

        # Add jitter (pitch variation)
        jitter_amount = params["jitter_target"] * intensity / 100
        jitter = 1 + np.random.randn(len(y)) * jitter_amount
        jitter = np.clip(jitter, 0.8, 1.2)

        # Apply modulation
        y_creaky = y * modulation * jitter

        # Add slight noise
        noise = np.random.randn(len(y)) * 0.02 * intensity * np.std(y)
        y_creaky = y_creaky + noise

        # Normalize
        y_creaky = y_creaky / (np.max(np.abs(y_creaky)) + 1e-10)
        y_creaky = y_creaky * np.max(np.abs(y))

        return y_creaky

    def _add_harshness(
        self,
        y: np.ndarray,
        sr: int,
        intensity: float,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Add harsh/pressed voice characteristics."""
        # Soft clipping for harmonic distortion
        threshold = 1.0 - 0.3 * intensity
        y_normalized = y / (np.max(np.abs(y)) + 1e-10)
        y_clipped = np.tanh(y_normalized / threshold) * threshold

        # Boost high frequencies (brighter, more strident)
        from scipy import signal
        b, a = signal.butter(1, 2000 / (sr / 2), btype='high')
        high_freq = signal.filtfilt(b, a, y_clipped)
        y_harsh = y_clipped + high_freq * intensity * 0.3

        # Add noise
        noise = np.random.randn(len(y)) * params["noise_level"] * intensity * np.std(y)
        y_harsh = y_harsh + noise

        # Normalize
        y_harsh = y_harsh / (np.max(np.abs(y_harsh)) + 1e-10)
        y_harsh = y_harsh * np.max(np.abs(y))

        return y_harsh

    def _apply_falsetto(
        self,
        y: np.ndarray,
        sr: int,
        intensity: float,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Apply falsetto characteristics."""
        try:
            import librosa

            # Pitch shift up
            semitones = 7 * intensity  # Up to a fifth
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

            # Reduce low frequencies (thinner sound)
            from scipy import signal
            b, a = signal.butter(2, 200 / (sr / 2), btype='high')
            y_falsetto = signal.filtfilt(b, a, y_shifted)

            # Add slight breathiness
            noise = np.random.randn(len(y_falsetto)) * 0.05 * intensity * np.std(y_falsetto)
            y_falsetto = y_falsetto + noise

            # Normalize
            y_falsetto = y_falsetto / (np.max(np.abs(y_falsetto)) + 1e-10)
            y_falsetto = y_falsetto * np.max(np.abs(y))

            return y_falsetto

        except ImportError:
            # Without librosa, just apply filtering
            from scipy import signal
            b, a = signal.butter(2, 200 / (sr / 2), btype='high')
            return signal.filtfilt(b, a, y)

    def _apply_whisper(
        self,
        y: np.ndarray,
        sr: int,
        intensity: float,
    ) -> np.ndarray:
        """Convert to whispered speech."""
        # Extract envelope
        from scipy import signal

        # Get amplitude envelope
        analytic = signal.hilbert(y)
        envelope = np.abs(analytic)

        # Smooth envelope
        window = int(0.01 * sr)  # 10ms window
        envelope = np.convolve(envelope, np.ones(window) / window, mode='same')

        # Generate noise shaped by envelope
        noise = np.random.randn(len(y))

        # Formant-like filtering (preserve some speech characteristics)
        # Apply bandpass filters for formant regions
        formant_centers = [500, 1500, 2500]
        formant_bw = 200

        filtered_noise = np.zeros_like(noise)
        for fc in formant_centers:
            low = max(50, fc - formant_bw) / (sr / 2)
            high = min(0.99, (fc + formant_bw) / (sr / 2))
            b, a = signal.butter(2, [low, high], btype='band')
            filtered_noise += signal.filtfilt(b, a, noise) * 0.5

        # Apply envelope to noise
        y_whisper = filtered_noise * envelope

        # Mix with original based on intensity
        y_mixed = (1 - intensity) * y + intensity * y_whisper

        # Normalize
        y_mixed = y_mixed / (np.max(np.abs(y_mixed)) + 1e-10)
        y_mixed = y_mixed * np.max(np.abs(y))

        return y_mixed


# Convenience functions

def detect_phonation(
    audio: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
) -> PhonationAnalysisResult:
    """
    Detect phonation type from audio.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate (required if audio is array)

    Returns:
        PhonationAnalysisResult with detected type
    """
    analyzer = PhonationAnalyzer()
    return analyzer.analyze(audio, sample_rate)


def apply_phonation(
    audio: Union[str, Path, np.ndarray],
    phonation_type: PhonationType,
    intensity: float = 0.5,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Apply phonation type characteristics to audio.

    Args:
        audio: Audio file path or numpy array
        phonation_type: Target phonation type
        intensity: Effect intensity (0-1)
        sample_rate: Sample rate (required if audio is array)

    Returns:
        Tuple of (modified_audio, sample_rate)
    """
    synthesizer = PhonationSynthesizer()
    return synthesizer.apply(audio, phonation_type, intensity, sample_rate)
