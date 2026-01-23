"""
Speech Rhythm Analysis and Modification

Implements:
- Rhythm Zone Theory (RZT) - Gibbon & Lin's signal-based rhythm detection
- Classical rhythm metrics (nPVI, %V, ΔC, rPVI)
- Rhythm typology classification (stress-timed, syllable-timed, mora-timed)
- Neural oscillation-inspired rhythm processing

Based on research:
- Gibbon & Lin (2019): Rhythm Zone Theory - https://arxiv.org/abs/1902.01267
- Ramus et al. (1999): Correlates of linguistic rhythm
- Grabe & Low (2002): Durational variability in speech

Key concepts:
- Delta band (0.5-2 Hz): Phrase/sentence rhythm
- Theta band (4-8 Hz): Syllable rate (~5 Hz typical)
- Rhythm zones: Segments bounded by amplitude envelope edges
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import numpy as np


class RhythmClass(Enum):
    """Speech rhythm typology classification."""
    STRESS_TIMED = "stress_timed"      # English, German, Dutch
    SYLLABLE_TIMED = "syllable_timed"  # Spanish, French, Italian
    MORA_TIMED = "mora_timed"          # Japanese, Finnish
    MIXED = "mixed"                     # Exhibits features of multiple classes


class RhythmBand(Enum):
    """Frequency bands for rhythm analysis (based on neural oscillations)."""
    DELTA = "delta"    # 0.5-2 Hz - Phrase/prosodic grouping
    THETA = "theta"    # 4-8 Hz - Syllable rate
    ALPHA = "alpha"    # 8-12 Hz - Phoneme rate
    BETA = "beta"      # 15-30 Hz - Fast articulation


@dataclass
class RhythmMetrics:
    """
    Classical rhythm metrics for speech analysis.

    Attributes:
        percent_v: Percentage of vocalic intervals (%V)
        delta_v: Std dev of vocalic interval durations (ΔV)
        delta_c: Std dev of consonantal interval durations (ΔC)
        npvi_v: Normalized Pairwise Variability Index for vowels
        rpvi_c: Raw Pairwise Variability Index for consonants
        varco_v: Variation coefficient for vowels
        varco_c: Variation coefficient for consonants
        speech_rate: Estimated syllables per second
        articulation_rate: Speech rate excluding pauses
    """
    percent_v: float           # Higher = more syllable-timed
    delta_v: float             # Vocalic duration variability
    delta_c: float             # Consonantal duration variability
    npvi_v: float              # Normalized vowel variability (lower = syllable-timed)
    rpvi_c: float              # Raw consonant variability
    varco_v: float             # Vowel variation coefficient
    varco_c: float             # Consonant variation coefficient
    speech_rate: float         # Syllables per second
    articulation_rate: float   # Rate excluding pauses

    @property
    def rhythm_class(self) -> RhythmClass:
        """Estimate rhythm class from metrics."""
        # Based on Grabe & Low (2002) thresholds
        if self.npvi_v > 55 and self.percent_v < 45:
            return RhythmClass.STRESS_TIMED
        elif self.npvi_v < 45 and self.percent_v > 45:
            return RhythmClass.SYLLABLE_TIMED
        elif self.npvi_v < 35:
            return RhythmClass.MORA_TIMED
        else:
            return RhythmClass.MIXED

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "percent_v": self.percent_v,
            "delta_v": self.delta_v,
            "delta_c": self.delta_c,
            "npvi_v": self.npvi_v,
            "rpvi_c": self.rpvi_c,
            "varco_v": self.varco_v,
            "varco_c": self.varco_c,
            "speech_rate": self.speech_rate,
            "articulation_rate": self.articulation_rate,
            "rhythm_class": self.rhythm_class.value,
        }


@dataclass
class RhythmZone:
    """
    A rhythm zone identified by RZT analysis.

    Attributes:
        start_time: Zone start time (seconds)
        end_time: Zone end time (seconds)
        dominant_frequency: Primary rhythm frequency in zone (Hz)
        energy: Relative energy in zone
        band: Which frequency band this zone belongs to
    """
    start_time: float
    end_time: float
    dominant_frequency: float
    energy: float
    band: RhythmBand

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class RZTAnalysis:
    """
    Rhythm Zone Theory analysis results.

    Attributes:
        zones: List of identified rhythm zones
        envelope_spectrum: Amplitude envelope frequency spectrum
        rhythm_frequencies: Detected rhythm frequencies (Hz)
        phrase_rhythm: Dominant phrase-level rhythm (delta band)
        syllable_rhythm: Dominant syllable-level rhythm (theta band)
        sample_rate: Audio sample rate
    """
    zones: List[RhythmZone]
    envelope_spectrum: np.ndarray
    rhythm_frequencies: List[float]
    phrase_rhythm: float          # Hz, typically 0.5-2
    syllable_rhythm: float        # Hz, typically 4-7
    sample_rate: int

    @property
    def estimated_syllable_rate(self) -> float:
        """Estimated syllables per second from theta band."""
        return self.syllable_rhythm

    @property
    def phrase_duration(self) -> float:
        """Estimated phrase duration from delta band."""
        if self.phrase_rhythm > 0:
            return 1.0 / self.phrase_rhythm
        return 0.0


class RhythmAnalyzer:
    """
    Analyze speech rhythm using multiple approaches.

    Combines:
    - Classical metrics (nPVI, %V, ΔC)
    - Rhythm Zone Theory (envelope spectrum analysis)
    - Neural oscillation band analysis

    Example:
        analyzer = RhythmAnalyzer()

        # Get classical metrics
        metrics = analyzer.analyze_metrics(audio, sample_rate)
        print(f"Rhythm class: {metrics.rhythm_class}")

        # Get RZT analysis
        rzt = analyzer.analyze_rzt(audio, sample_rate)
        print(f"Syllable rate: {rzt.syllable_rhythm} Hz")
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def analyze_metrics(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> RhythmMetrics:
        """
        Compute classical rhythm metrics.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            RhythmMetrics with nPVI, %V, ΔC, etc.
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            y = audio
            sr = sample_rate or self.sample_rate

        # Detect vocalic and consonantal intervals
        v_intervals, c_intervals = self._detect_intervals(y, sr)

        # Calculate metrics
        percent_v = self._calculate_percent_v(v_intervals, c_intervals)
        delta_v = np.std(v_intervals) if len(v_intervals) > 1 else 0.0
        delta_c = np.std(c_intervals) if len(c_intervals) > 1 else 0.0
        npvi_v = self._calculate_npvi(v_intervals)
        rpvi_c = self._calculate_rpvi(c_intervals)
        varco_v = self._calculate_varco(v_intervals)
        varco_c = self._calculate_varco(c_intervals)

        # Estimate speech rate
        total_duration = len(y) / sr
        n_syllables = len(v_intervals)  # Approximate: 1 vowel = 1 syllable
        speech_rate = n_syllables / total_duration if total_duration > 0 else 0.0

        # Articulation rate (excluding pauses)
        voiced_duration = sum(v_intervals) + sum(c_intervals)
        articulation_rate = n_syllables / voiced_duration if voiced_duration > 0 else 0.0

        return RhythmMetrics(
            percent_v=percent_v,
            delta_v=delta_v,
            delta_c=delta_c,
            npvi_v=npvi_v,
            rpvi_c=rpvi_c,
            varco_v=varco_v,
            varco_c=varco_c,
            speech_rate=speech_rate,
            articulation_rate=articulation_rate,
        )

    def analyze_rzt(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> RZTAnalysis:
        """
        Perform Rhythm Zone Theory analysis.

        Extracts amplitude envelope, computes spectrum, detects
        rhythm zones bounded by spectral edges.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            RZTAnalysis with zones, frequencies, and band information
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            y = audio
            sr = sample_rate or self.sample_rate

        # Extract amplitude envelope
        envelope = self._extract_envelope(y, sr)

        # Compute envelope spectrum
        spectrum, freqs = self._compute_envelope_spectrum(envelope, sr)

        # Detect rhythm zones using edge detection
        zones = self._detect_rhythm_zones(envelope, spectrum, freqs, sr)

        # Find dominant frequencies in each band
        phrase_rhythm = self._find_dominant_frequency(spectrum, freqs, 0.5, 2.0)
        syllable_rhythm = self._find_dominant_frequency(spectrum, freqs, 4.0, 8.0)

        # Collect all significant rhythm frequencies
        rhythm_frequencies = self._find_rhythm_peaks(spectrum, freqs)

        return RZTAnalysis(
            zones=zones,
            envelope_spectrum=spectrum,
            rhythm_frequencies=rhythm_frequencies,
            phrase_rhythm=phrase_rhythm,
            syllable_rhythm=syllable_rhythm,
            sample_rate=sr,
        )

    def analyze_band_energy(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Dict[RhythmBand, float]:
        """
        Analyze energy in each rhythm band.

        Based on neural oscillation research showing different
        frequency bands correspond to different linguistic units.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate

        Returns:
            Dictionary mapping RhythmBand to relative energy
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            y = audio
            sr = sample_rate or self.sample_rate

        # Extract envelope and spectrum
        envelope = self._extract_envelope(y, sr)
        spectrum, freqs = self._compute_envelope_spectrum(envelope, sr)

        # Define band ranges
        bands = {
            RhythmBand.DELTA: (0.5, 2.0),
            RhythmBand.THETA: (4.0, 8.0),
            RhythmBand.ALPHA: (8.0, 12.0),
            RhythmBand.BETA: (15.0, 30.0),
        }

        # Calculate energy in each band
        band_energy = {}
        total_energy = np.sum(spectrum ** 2)

        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            energy = np.sum(spectrum[mask] ** 2)
            band_energy[band] = energy / total_energy if total_energy > 0 else 0.0

        return band_energy

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

    def _extract_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract amplitude envelope using Hilbert transform.
        """
        from scipy.signal import hilbert

        # Apply Hilbert transform to get analytic signal
        analytic = hilbert(audio)
        envelope = np.abs(analytic)

        # Low-pass filter to smooth envelope (keep < 30 Hz)
        envelope = self._lowpass_filter(envelope, sr, cutoff=30.0)

        return envelope

    def _lowpass_filter(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff: float
    ) -> np.ndarray:
        """Apply low-pass filter."""
        from scipy.signal import butter, filtfilt

        nyquist = sr / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)

        b, a = butter(4, normalized_cutoff, btype='low')
        filtered = filtfilt(b, a, signal)

        return filtered

    def _compute_envelope_spectrum(
        self,
        envelope: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute frequency spectrum of amplitude envelope."""
        # Downsample envelope for efficiency (100 Hz is enough for rhythm)
        downsample_rate = 100
        downsample_factor = sr // downsample_rate
        envelope_ds = envelope[::downsample_factor]

        # FFT
        n = len(envelope_ds)
        spectrum = np.abs(np.fft.rfft(envelope_ds))
        freqs = np.fft.rfftfreq(n, 1.0 / downsample_rate)

        # Normalize
        spectrum = spectrum / (np.max(spectrum) + 1e-10)

        return spectrum, freqs

    def _detect_rhythm_zones(
        self,
        envelope: np.ndarray,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        sr: int,
    ) -> List[RhythmZone]:
        """
        Detect rhythm zones using edge detection on envelope.
        """
        from scipy.signal import find_peaks

        # Find peaks in envelope (zone boundaries)
        # Downsample for efficiency
        ds_factor = sr // 100
        env_ds = envelope[::ds_factor]

        # Find local minima as zone boundaries
        inv_env = -env_ds
        peaks, properties = find_peaks(inv_env, distance=10, prominence=0.1)

        # Convert peak indices to times
        boundary_times = peaks * ds_factor / sr

        # Create zones between boundaries
        zones = []
        all_boundaries = [0.0] + list(boundary_times) + [len(envelope) / sr]

        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]

            if end - start < 0.1:  # Skip very short zones
                continue

            # Find dominant frequency in this zone
            zone_start_sample = int(start * sr)
            zone_end_sample = int(end * sr)
            zone_envelope = envelope[zone_start_sample:zone_end_sample]

            if len(zone_envelope) < 10:
                continue

            # Local spectrum analysis
            zone_spectrum, zone_freqs = self._compute_envelope_spectrum(zone_envelope, sr)
            dom_freq = self._find_dominant_frequency(zone_spectrum, zone_freqs, 0.5, 10.0)

            # Determine band
            if dom_freq < 2.0:
                band = RhythmBand.DELTA
            elif dom_freq < 8.0:
                band = RhythmBand.THETA
            elif dom_freq < 12.0:
                band = RhythmBand.ALPHA
            else:
                band = RhythmBand.BETA

            # Calculate energy
            energy = np.mean(zone_envelope ** 2)

            zones.append(RhythmZone(
                start_time=start,
                end_time=end,
                dominant_frequency=dom_freq,
                energy=energy,
                band=band,
            ))

        return zones

    def _find_dominant_frequency(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        low: float,
        high: float,
    ) -> float:
        """Find dominant frequency in a frequency range."""
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0

        masked_spectrum = spectrum[mask]
        masked_freqs = freqs[mask]

        if len(masked_spectrum) == 0:
            return 0.0

        peak_idx = np.argmax(masked_spectrum)
        return float(masked_freqs[peak_idx])

    def _find_rhythm_peaks(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        threshold: float = 0.3,
    ) -> List[float]:
        """Find all significant rhythm frequencies."""
        from scipy.signal import find_peaks

        # Find peaks above threshold
        peaks, _ = find_peaks(spectrum, height=threshold)

        # Filter to rhythm-relevant range (0.5-15 Hz)
        rhythm_peaks = []
        for p in peaks:
            if 0.5 <= freqs[p] <= 15.0:
                rhythm_peaks.append(float(freqs[p]))

        return sorted(rhythm_peaks)

    def _detect_intervals(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Detect vocalic and consonantal intervals.

        Uses energy-based segmentation as approximation.
        True V/C detection would require phonetic analysis.
        """
        # Frame-based energy analysis
        frame_size = int(0.01 * sr)  # 10ms frames
        hop_size = frame_size // 2

        n_frames = (len(audio) - frame_size) // hop_size + 1
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize energy
        energy = energy / (np.max(energy) + 1e-10)

        # Threshold for voiced/unvoiced
        threshold = 0.15
        is_voiced = energy > threshold

        # Find intervals
        v_intervals = []  # Voiced (vocalic-like)
        c_intervals = []  # Unvoiced (consonantal-like)

        in_voiced = False
        interval_start = 0

        for i, voiced in enumerate(is_voiced):
            if voiced and not in_voiced:
                # Start of voiced interval
                if i > 0:
                    # End of consonantal interval
                    duration = (i - interval_start) * hop_size / sr
                    if duration > 0.02:  # Min 20ms
                        c_intervals.append(duration)
                interval_start = i
                in_voiced = True
            elif not voiced and in_voiced:
                # End of voiced interval
                duration = (i - interval_start) * hop_size / sr
                if duration > 0.02:  # Min 20ms
                    v_intervals.append(duration)
                interval_start = i
                in_voiced = False

        # Handle final interval
        if in_voiced:
            duration = (len(is_voiced) - interval_start) * hop_size / sr
            if duration > 0.02:
                v_intervals.append(duration)

        return v_intervals, c_intervals

    def _calculate_percent_v(
        self,
        v_intervals: List[float],
        c_intervals: List[float],
    ) -> float:
        """Calculate %V (percentage of vocalic intervals)."""
        total_v = sum(v_intervals)
        total_c = sum(c_intervals)
        total = total_v + total_c

        if total == 0:
            return 0.0

        return 100.0 * total_v / total

    def _calculate_npvi(self, intervals: List[float]) -> float:
        """
        Calculate normalized Pairwise Variability Index.

        nPVI = 100 * Σ|dk - dk+1| / ((dk + dk+1)/2) / (n-1)
        """
        if len(intervals) < 2:
            return 0.0

        total = 0.0
        for i in range(len(intervals) - 1):
            d1, d2 = intervals[i], intervals[i + 1]
            mean = (d1 + d2) / 2
            if mean > 0:
                total += abs(d1 - d2) / mean

        return 100.0 * total / (len(intervals) - 1)

    def _calculate_rpvi(self, intervals: List[float]) -> float:
        """
        Calculate raw Pairwise Variability Index.

        rPVI = Σ|dk - dk+1| / (n-1)
        """
        if len(intervals) < 2:
            return 0.0

        total = 0.0
        for i in range(len(intervals) - 1):
            total += abs(intervals[i] - intervals[i + 1])

        return total / (len(intervals) - 1)

    def _calculate_varco(self, intervals: List[float]) -> float:
        """
        Calculate variation coefficient.

        VarcoV/C = 100 * std / mean
        """
        if len(intervals) < 2:
            return 0.0

        mean = np.mean(intervals)
        if mean == 0:
            return 0.0

        return 100.0 * np.std(intervals) / mean


class RhythmModifier:
    """
    Modify speech rhythm for naturalness or style transfer.

    Uses rhythm analysis to:
    - Adjust timing for target rhythm class
    - Smooth or add variability
    - Transfer rhythm patterns between utterances

    Example:
        modifier = RhythmModifier()

        # Make speech more syllable-timed (like Spanish)
        modified = modifier.adjust_rhythm_class(
            audio,
            target=RhythmClass.SYLLABLE_TIMED
        )

        # Add natural timing variability
        natural = modifier.add_variability(audio, amount=0.15)
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.analyzer = RhythmAnalyzer(sample_rate)

    def add_variability(
        self,
        audio: Union[str, Path, np.ndarray],
        amount: float = 0.1,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Add natural timing variability to speech.

        Prevents metronomic delivery by adding subtle
        timing fluctuations aligned with rhythm zones.

        Args:
            audio: Audio file path or numpy array
            amount: Variability amount (0-1, default 0.1 = 10%)
            sample_rate: Sample rate

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            y = audio
            sr = sample_rate or self.sample_rate

        # Analyze rhythm zones
        rzt = self.analyzer.analyze_rzt(y, sr)

        # Create time-varying stretch map
        stretch_map = self._create_variability_map(y, sr, rzt.zones, amount)

        # Apply time stretching
        output = self._apply_stretch_map(y, sr, stretch_map)

        return output, sr

    def adjust_rhythm_class(
        self,
        audio: Union[str, Path, np.ndarray],
        target: RhythmClass,
        strength: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Adjust speech rhythm toward target rhythm class.

        Args:
            audio: Audio file path or numpy array
            target: Target rhythm class
            strength: Adjustment strength (0-1)
            sample_rate: Sample rate

        Returns:
            Tuple of (modified_audio, sample_rate)
        """
        # Load audio
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            y = audio
            sr = sample_rate or self.sample_rate

        # Get current rhythm metrics
        metrics = self.analyzer.analyze_metrics(y, sr)
        current_class = metrics.rhythm_class

        if current_class == target:
            return y, sr  # Already at target

        # Determine adjustment direction
        if target == RhythmClass.SYLLABLE_TIMED:
            # Equalize interval durations
            output = self._equalize_rhythm(y, sr, strength)
        elif target == RhythmClass.STRESS_TIMED:
            # Increase contrast between stressed/unstressed
            output = self._increase_contrast(y, sr, strength)
        else:
            # For MORA_TIMED, aim for very regular timing
            output = self._equalize_rhythm(y, sr, strength * 1.5)

        return output, sr

    def transfer_rhythm(
        self,
        source: Union[str, Path, np.ndarray],
        target: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Transfer rhythm pattern from source to target audio.

        Args:
            source: Audio to get rhythm pattern from
            target: Audio to apply rhythm pattern to
            sample_rate: Sample rate

        Returns:
            Target audio with source rhythm pattern
        """
        # Load audio
        if isinstance(source, (str, Path)):
            y_src, sr = self._load_audio(source)
        else:
            y_src = source
            sr = sample_rate or self.sample_rate

        if isinstance(target, (str, Path)):
            y_tgt, _ = self._load_audio(target)
        else:
            y_tgt = target

        # Analyze both
        rzt_src = self.analyzer.analyze_rzt(y_src, sr)
        rzt_tgt = self.analyzer.analyze_rzt(y_tgt, sr)

        # Create stretch map to match source rhythm
        stretch_map = self._create_transfer_map(
            y_tgt, sr, rzt_src.zones, rzt_tgt.zones
        )

        # Apply
        output = self._apply_stretch_map(y_tgt, sr, stretch_map)

        return output, sr

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

    def _create_variability_map(
        self,
        audio: np.ndarray,
        sr: int,
        zones: List[RhythmZone],
        amount: float,
    ) -> np.ndarray:
        """Create time-varying stretch factors for variability."""
        n_samples = len(audio)
        stretch_map = np.ones(n_samples)

        for zone in zones:
            start_sample = int(zone.start_time * sr)
            end_sample = int(zone.end_time * sr)

            # Random stretch factor for this zone
            stretch = 1.0 + np.random.uniform(-amount, amount)

            # Smooth transition
            zone_len = end_sample - start_sample
            if zone_len > 0:
                # Fade in/out the stretch
                fade_len = min(zone_len // 4, int(0.05 * sr))

                zone_stretch = np.ones(zone_len) * stretch

                # Fade in
                if fade_len > 0:
                    zone_stretch[:fade_len] = np.linspace(1.0, stretch, fade_len)
                    zone_stretch[-fade_len:] = np.linspace(stretch, 1.0, fade_len)

                stretch_map[start_sample:end_sample] = zone_stretch[:end_sample - start_sample]

        return stretch_map

    def _create_transfer_map(
        self,
        audio: np.ndarray,
        sr: int,
        src_zones: List[RhythmZone],
        tgt_zones: List[RhythmZone],
    ) -> np.ndarray:
        """Create stretch map for rhythm transfer."""
        n_samples = len(audio)
        stretch_map = np.ones(n_samples)

        # Match zones by position
        n_zones = min(len(src_zones), len(tgt_zones))

        for i in range(n_zones):
            src_zone = src_zones[i]
            tgt_zone = tgt_zones[i]

            # Calculate stretch factor
            if tgt_zone.duration > 0:
                stretch = src_zone.duration / tgt_zone.duration
                stretch = np.clip(stretch, 0.5, 2.0)  # Limit extreme stretching

                start_sample = int(tgt_zone.start_time * sr)
                end_sample = int(tgt_zone.end_time * sr)

                stretch_map[start_sample:end_sample] = stretch

        return stretch_map

    def _apply_stretch_map(
        self,
        audio: np.ndarray,
        sr: int,
        stretch_map: np.ndarray,
    ) -> np.ndarray:
        """Apply time-varying stretch to audio."""
        try:
            import librosa

            # Average stretch for simple implementation
            # Full implementation would use WSOLA or similar
            avg_stretch = np.mean(stretch_map)

            if abs(avg_stretch - 1.0) < 0.01:
                return audio

            # Time stretch
            output = librosa.effects.time_stretch(audio, rate=1.0/avg_stretch)

            return output.astype(np.float32)

        except ImportError:
            # Fallback: simple resampling
            return audio

    def _equalize_rhythm(
        self,
        audio: np.ndarray,
        sr: int,
        strength: float,
    ) -> np.ndarray:
        """Make rhythm more equal/syllable-timed."""
        # This would require sophisticated time-stretching
        # Simplified: reduce variability in envelope
        from scipy.signal import hilbert

        envelope = np.abs(hilbert(audio))

        # Smooth envelope more to reduce variability
        kernel_size = int(0.1 * sr * strength)
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_env = np.convolve(envelope, kernel, mode='same')

            # Apply envelope modification
            ratio = smoothed_env / (envelope + 1e-10)
            ratio = np.clip(ratio, 0.5, 2.0)

            # Blend based on strength
            ratio = 1.0 + strength * (ratio - 1.0)

            output = audio * ratio

            # Normalize
            output = output / (np.max(np.abs(output)) + 1e-10) * np.max(np.abs(audio))

            return output.astype(np.float32)

        return audio

    def _increase_contrast(
        self,
        audio: np.ndarray,
        sr: int,
        strength: float,
    ) -> np.ndarray:
        """Make rhythm more stress-timed with greater contrast."""
        from scipy.signal import hilbert

        envelope = np.abs(hilbert(audio))

        # Enhance peaks, reduce valleys
        mean_env = np.mean(envelope)
        enhanced = envelope.copy()

        above_mean = envelope > mean_env
        below_mean = ~above_mean

        # Boost peaks
        enhanced[above_mean] = envelope[above_mean] * (1.0 + 0.5 * strength)
        # Reduce valleys
        enhanced[below_mean] = envelope[below_mean] * (1.0 - 0.3 * strength)

        # Apply
        ratio = enhanced / (envelope + 1e-10)
        ratio = np.clip(ratio, 0.3, 3.0)

        output = audio * ratio

        # Normalize
        output = output / (np.max(np.abs(output)) + 1e-10) * np.max(np.abs(audio))

        return output.astype(np.float32)


# Convenience functions

def analyze_rhythm(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
) -> RhythmMetrics:
    """
    Convenience function to analyze speech rhythm.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate

    Returns:
        RhythmMetrics with classical metrics and rhythm class
    """
    analyzer = RhythmAnalyzer(sample_rate)
    return analyzer.analyze_metrics(audio, sample_rate)


def analyze_rhythm_zones(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
) -> RZTAnalysis:
    """
    Convenience function for Rhythm Zone Theory analysis.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate

    Returns:
        RZTAnalysis with zones and rhythm frequencies
    """
    analyzer = RhythmAnalyzer(sample_rate)
    return analyzer.analyze_rzt(audio, sample_rate)


def add_rhythm_variability(
    audio: Union[str, Path, np.ndarray],
    amount: float = 0.1,
    sample_rate: int = 24000,
) -> Tuple[np.ndarray, int]:
    """
    Add natural timing variability to prevent robotic delivery.

    Args:
        audio: Audio file path or numpy array
        amount: Variability amount (0-1)
        sample_rate: Sample rate

    Returns:
        Tuple of (modified_audio, sample_rate)
    """
    modifier = RhythmModifier(sample_rate)
    return modifier.add_variability(audio, amount, sample_rate)
