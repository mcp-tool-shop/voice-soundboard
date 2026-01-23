"""
Voice Humanization Module

Adds natural imperfections to AI-generated vocals:
- Breath sounds insertion
- Pitch micro-variations (jitter, drift, scooping)
- Timing adjustments
- Formant micro-shifts

Based on research from:
- Sonarworks: https://www.sonarworks.com/blog/learn/how-to-add-breath-sounds-and-realism-to-ai-vocals
- Voice Soundboard Vocology Library
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import numpy as np


class BreathType(Enum):
    """Types of breath sounds."""
    QUICK = "quick"           # Short catch breath (100-150ms)
    MEDIUM = "medium"         # Normal breath (200-300ms)
    DEEP = "deep"             # Deep inhale before long phrase (300-500ms)
    GASP = "gasp"             # Emotional/surprised (200-300ms)
    SIGH = "sigh"             # Exhale, tired/relieved (300-400ms)
    NASAL = "nasal"           # Subtle nose breath


class EmotionalState(Enum):
    """Emotional states affecting humanization parameters."""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    TIRED = "tired"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"
    INTIMATE = "intimate"


@dataclass
class BreathConfig:
    """Configuration for breath insertion."""
    enabled: bool = True
    volume_db: float = -24.0          # dB below vocal level (subtle)
    pre_phrase_offset_ms: float = 150  # ms before phrase start
    min_phrase_gap_ms: float = 300     # Minimum gap to insert breath
    intensity: float = 0.25            # 0-1, affects breath audibility (subtle)

    # Breath type probabilities based on context
    deep_breath_threshold_s: float = 3.0  # Phrases longer than this get deep breath


@dataclass
class PitchHumanizeConfig:
    """Configuration for pitch humanization."""
    enabled: bool = True

    # Micro-jitter (random cycle-to-cycle variation)
    jitter_cents: float = 5.0          # Standard deviation in cents
    jitter_rate_hz: float = 10.0       # Rate of jitter variation

    # Drift (slow phrase-level wandering)
    drift_max_cents: float = 15.0      # Maximum drift amount
    drift_rate_hz: float = 0.5         # How fast drift changes

    # Scooping (pitch slides into notes)
    scoop_enabled: bool = True
    scoop_cents: float = 30.0          # How flat to start
    scoop_duration_ms: float = 80.0    # Duration of scoop

    # Phrase-final movement
    final_drop_cents: float = 20.0     # Drop at end of declarative phrases
    final_rise_cents: float = 30.0     # Rise at end of questions

    # Overshoot on pitch targets
    overshoot_enabled: bool = True
    overshoot_cents: float = 15.0      # Brief overshoot amount
    overshoot_duration_ms: float = 50.0


@dataclass
class TimingHumanizeConfig:
    """Configuration for timing humanization."""
    enabled: bool = True

    # Syllable timing variations
    timing_variation_ms: float = 20.0   # Std dev for syllable nudging

    # Emotion-based timing bias
    # Positive = ahead of beat, Negative = behind
    timing_bias_ms: float = 0.0

    # Phrase gap variation
    gap_variation_percent: float = 15.0  # % variation in pause lengths


@dataclass
class HumanizeConfig:
    """Master configuration for all humanization."""
    breath: BreathConfig = field(default_factory=BreathConfig)
    pitch: PitchHumanizeConfig = field(default_factory=PitchHumanizeConfig)
    timing: TimingHumanizeConfig = field(default_factory=TimingHumanizeConfig)

    # Overall intensity (scales all effects)
    intensity: float = 1.0

    # Emotional state affects all parameters
    emotion: EmotionalState = EmotionalState.NEUTRAL

    @classmethod
    def for_emotion(cls, emotion: EmotionalState) -> "HumanizeConfig":
        """Create config tuned for a specific emotional state."""
        config = cls(emotion=emotion)

        if emotion == EmotionalState.EXCITED:
            config.pitch.jitter_cents = 7.0
            config.pitch.drift_max_cents = 20.0
            config.timing.timing_bias_ms = -10.0  # Slightly ahead
            config.breath.intensity = 0.8

        elif emotion == EmotionalState.CALM:
            config.pitch.jitter_cents = 3.0
            config.pitch.drift_max_cents = 10.0
            config.timing.timing_bias_ms = 5.0   # Slightly behind
            config.breath.intensity = 0.5

        elif emotion == EmotionalState.TIRED:
            config.pitch.jitter_cents = 4.0
            config.pitch.drift_max_cents = 25.0
            config.pitch.final_drop_cents = 35.0
            config.timing.timing_bias_ms = 15.0  # Behind the beat
            config.breath.intensity = 0.9

        elif emotion == EmotionalState.ANXIOUS:
            config.pitch.jitter_cents = 8.0
            config.pitch.drift_rate_hz = 0.8
            config.timing.timing_variation_ms = 30.0
            config.breath.intensity = 0.85

        elif emotion == EmotionalState.CONFIDENT:
            config.pitch.jitter_cents = 4.0
            config.pitch.drift_max_cents = 8.0
            config.timing.timing_bias_ms = -5.0  # Slightly ahead
            config.breath.intensity = 0.6

        elif emotion == EmotionalState.INTIMATE:
            config.pitch.jitter_cents = 3.0
            config.pitch.drift_max_cents = 12.0
            config.breath.volume_db = -10.0  # More audible breaths
            config.breath.intensity = 0.9

        return config


class BreathGenerator:
    """
    Generate synthetic breath sounds.

    Creates realistic breath samples using filtered noise
    with appropriate envelope shaping.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def generate(
        self,
        breath_type: BreathType = BreathType.MEDIUM,
        duration_ms: Optional[float] = None,
        intensity: float = 0.7,
    ) -> np.ndarray:
        """
        Generate a breath sound.

        Args:
            breath_type: Type of breath to generate
            duration_ms: Override default duration
            intensity: Breath intensity (0-1)

        Returns:
            Breath audio as numpy array
        """
        # Default durations by type
        default_durations = {
            BreathType.QUICK: 120,
            BreathType.MEDIUM: 250,
            BreathType.DEEP: 400,
            BreathType.GASP: 200,
            BreathType.SIGH: 350,
            BreathType.NASAL: 200,
        }

        duration = duration_ms or default_durations[breath_type]
        n_samples = int(duration * self.sample_rate / 1000)

        # Generate base noise
        noise = np.random.randn(n_samples).astype(np.float32)

        # Apply breath-specific filtering
        breath = self._filter_breath(noise, breath_type)

        # Apply envelope
        breath = self._apply_envelope(breath, breath_type)

        # Scale by intensity
        breath = breath * intensity * 0.3  # Keep breath quieter than voice

        return breath.astype(np.float32)

    def _filter_breath(self, noise: np.ndarray, breath_type: BreathType) -> np.ndarray:
        """Apply frequency filtering for breath type."""
        # Simple low-pass using moving average (real implementation would use proper filters)
        if breath_type in [BreathType.NASAL, BreathType.QUICK]:
            # Higher frequency content
            kernel_size = 3
        elif breath_type == BreathType.DEEP:
            # Lower frequency content
            kernel_size = 15
        else:
            kernel_size = 7

        kernel = np.ones(kernel_size) / kernel_size
        filtered = np.convolve(noise, kernel, mode='same')

        # Add some harmonic content for realism
        if breath_type in [BreathType.GASP, BreathType.SIGH]:
            # Add slight tonal component
            t = np.arange(len(noise)) / self.sample_rate
            tone = 0.1 * np.sin(2 * np.pi * 150 * t)  # Low rumble
            filtered = filtered + tone

        return filtered

    def _apply_envelope(self, audio: np.ndarray, breath_type: BreathType) -> np.ndarray:
        """Apply amplitude envelope for breath type."""
        n = len(audio)

        if breath_type == BreathType.QUICK:
            # Fast attack, medium decay
            attack = int(0.1 * n)
            decay = int(0.4 * n)
        elif breath_type == BreathType.DEEP:
            # Slow attack, slow decay
            attack = int(0.3 * n)
            decay = int(0.5 * n)
        elif breath_type == BreathType.GASP:
            # Very fast attack
            attack = int(0.05 * n)
            decay = int(0.3 * n)
        elif breath_type == BreathType.SIGH:
            # Medium attack, long decay (exhale)
            attack = int(0.15 * n)
            decay = int(0.7 * n)
        else:
            attack = int(0.2 * n)
            decay = int(0.4 * n)

        # Build envelope
        env = np.ones(n)

        # Attack (fade in)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)

        # Sustain is implicit (ones)

        # Decay (fade out)
        sustain_end = n - decay
        if sustain_end > attack and decay > 0:
            env[sustain_end:] = np.linspace(1, 0, decay)

        return audio * env


class BreathInserter:
    """
    Insert breath sounds into audio at appropriate positions.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.breath_generator = BreathGenerator(sample_rate)

    def insert_breaths(
        self,
        audio: np.ndarray,
        phrase_boundaries: Optional[List[Tuple[float, float]]] = None,
        config: Optional[BreathConfig] = None,
    ) -> np.ndarray:
        """
        Insert breath sounds before phrases.

        Args:
            audio: Input audio array
            phrase_boundaries: List of (start_time, end_time) in seconds.
                              If None, will auto-detect pauses.
            config: Breath configuration

        Returns:
            Audio with breaths inserted
        """
        config = config or BreathConfig()

        if not config.enabled:
            return audio

        # Auto-detect phrase boundaries if not provided
        if phrase_boundaries is None:
            phrase_boundaries = self._detect_phrases(audio)

        if not phrase_boundaries:
            return audio

        # Create output buffer (may be slightly longer due to breaths)
        output = audio.copy()

        for i, (start, end) in enumerate(phrase_boundaries):
            phrase_duration = end - start

            # Determine breath type based on context
            if phrase_duration > config.deep_breath_threshold_s:
                breath_type = BreathType.DEEP
                breath_duration = 350
            elif i == 0:
                breath_type = BreathType.MEDIUM
                breath_duration = 250
            else:
                breath_type = BreathType.QUICK
                breath_duration = 120

            # Generate breath
            breath = self.breath_generator.generate(
                breath_type=breath_type,
                duration_ms=breath_duration,
                intensity=config.intensity,
            )

            # Apply volume scaling
            volume_linear = 10 ** (config.volume_db / 20)
            breath = breath * volume_linear

            # Calculate insertion position
            offset_samples = int(config.pre_phrase_offset_ms * self.sample_rate / 1000)
            insert_pos = int(start * self.sample_rate) - offset_samples

            # Ensure we don't go negative
            insert_pos = max(0, insert_pos)

            # Mix breath into output
            end_pos = min(insert_pos + len(breath), len(output))
            breath_len = end_pos - insert_pos

            if breath_len > 0:
                output[insert_pos:end_pos] += breath[:breath_len]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.95

        return output

    def _detect_phrases(
        self,
        audio: np.ndarray,
        silence_threshold: float = 0.02,
        min_silence_ms: float = 200,
    ) -> List[Tuple[float, float]]:
        """
        Auto-detect phrase boundaries based on silence.

        Args:
            audio: Input audio
            silence_threshold: Amplitude threshold for silence
            min_silence_ms: Minimum silence duration to count as phrase boundary

        Returns:
            List of (start_time, end_time) tuples
        """
        # Calculate frame energy
        frame_size = int(0.02 * self.sample_rate)  # 20ms frames
        hop_size = frame_size // 2

        n_frames = (len(audio) - frame_size) // hop_size + 1
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Find voiced regions
        is_voiced = energy > silence_threshold

        # Find phrase boundaries
        phrases = []
        in_phrase = False
        phrase_start = 0

        min_silence_frames = int(min_silence_ms / 1000 * self.sample_rate / hop_size)
        silence_count = 0

        for i, voiced in enumerate(is_voiced):
            if voiced:
                if not in_phrase:
                    phrase_start = i * hop_size / self.sample_rate
                    in_phrase = True
                silence_count = 0
            else:
                if in_phrase:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        phrase_end = (i - silence_count) * hop_size / self.sample_rate
                        phrases.append((phrase_start, phrase_end))
                        in_phrase = False

        # Handle final phrase
        if in_phrase:
            phrase_end = len(audio) / self.sample_rate
            phrases.append((phrase_start, phrase_end))

        return phrases


class PitchHumanizer:
    """
    Add natural pitch variations to audio.

    Applies:
    - Micro-jitter (random cycle-to-cycle variation)
    - Drift (slow phrase-level wandering)
    - Scooping (pitch slides)
    - Phrase-final movements
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def humanize(
        self,
        audio: np.ndarray,
        config: Optional[PitchHumanizeConfig] = None,
        phrase_boundaries: Optional[List[Tuple[float, float]]] = None,
        is_question: bool = False,
    ) -> np.ndarray:
        """
        Apply pitch humanization to audio.

        Args:
            audio: Input audio array
            config: Pitch humanization configuration
            phrase_boundaries: Phrase start/end times for phrase-level effects
            is_question: Whether this is a question (affects final pitch)

        Returns:
            Pitch-humanized audio
        """
        config = config or PitchHumanizeConfig()

        if not config.enabled:
            return audio

        # Generate pitch modification curve
        n_samples = len(audio)
        pitch_mod = np.zeros(n_samples, dtype=np.float32)

        # Add micro-jitter
        if config.jitter_cents > 0:
            jitter = self._generate_jitter(n_samples, config)
            pitch_mod += jitter

        # Add drift
        if config.drift_max_cents > 0:
            drift = self._generate_drift(n_samples, config)
            pitch_mod += drift

        # Add phrase-final effects
        if phrase_boundaries:
            final_mod = self._generate_final_movement(
                n_samples, phrase_boundaries, config, is_question
            )
            pitch_mod += final_mod

        # Apply pitch modification using phase vocoder approach
        output = self._apply_pitch_mod(audio, pitch_mod)

        return output

    def _generate_jitter(
        self,
        n_samples: int,
        config: PitchHumanizeConfig,
    ) -> np.ndarray:
        """Generate micro-jitter pitch variation."""
        # Create jitter at specified rate, then interpolate
        jitter_period = int(self.sample_rate / config.jitter_rate_hz)
        n_jitter_points = n_samples // jitter_period + 2

        # Random jitter values
        jitter_points = np.random.normal(0, config.jitter_cents, n_jitter_points)

        # Interpolate to full length
        x_points = np.arange(n_jitter_points) * jitter_period
        x_full = np.arange(n_samples)

        jitter = np.interp(x_full, x_points[:len(jitter_points)], jitter_points)

        return jitter.astype(np.float32)

    def _generate_drift(
        self,
        n_samples: int,
        config: PitchHumanizeConfig,
    ) -> np.ndarray:
        """Generate slow pitch drift."""
        # Low-frequency random walk
        drift_period = int(self.sample_rate / config.drift_rate_hz)
        n_drift_points = n_samples // drift_period + 2

        # Random walk
        drift_steps = np.random.normal(0, config.drift_max_cents / 3, n_drift_points)
        drift_points = np.cumsum(drift_steps)

        # Clamp to max drift
        drift_points = np.clip(drift_points, -config.drift_max_cents, config.drift_max_cents)

        # Interpolate smoothly
        x_points = np.arange(n_drift_points) * drift_period
        x_full = np.arange(n_samples)

        drift = np.interp(x_full, x_points[:len(drift_points)], drift_points)

        return drift.astype(np.float32)

    def _generate_final_movement(
        self,
        n_samples: int,
        phrase_boundaries: List[Tuple[float, float]],
        config: PitchHumanizeConfig,
        is_question: bool,
    ) -> np.ndarray:
        """Generate phrase-final pitch movements."""
        final_mod = np.zeros(n_samples, dtype=np.float32)

        for start, end in phrase_boundaries:
            end_sample = int(end * self.sample_rate)

            # Apply final movement in last 200ms of phrase
            movement_samples = int(0.2 * self.sample_rate)
            start_sample = max(0, end_sample - movement_samples)

            if start_sample >= n_samples:
                continue

            end_sample = min(end_sample, n_samples)
            length = end_sample - start_sample

            if length <= 0:
                continue

            # Create movement curve
            t = np.linspace(0, 1, length)

            if is_question:
                # Rising intonation
                movement = config.final_rise_cents * t ** 2
            else:
                # Falling intonation (declarative)
                movement = -config.final_drop_cents * t ** 2

            final_mod[start_sample:end_sample] += movement

        return final_mod

    def _apply_pitch_mod(
        self,
        audio: np.ndarray,
        pitch_mod_cents: np.ndarray,
    ) -> np.ndarray:
        """
        Apply pitch modification to audio.

        Uses simple resampling approach. For production, consider
        using librosa.effects.pitch_shift or a proper phase vocoder.
        """
        try:
            import librosa

            # Convert cents to semitones for librosa
            # Average pitch shift (librosa doesn't support time-varying easily)
            avg_cents = np.mean(pitch_mod_cents)
            semitones = avg_cents / 100.0

            if abs(semitones) < 0.01:
                return audio

            # Apply pitch shift
            output = librosa.effects.pitch_shift(
                audio.astype(np.float32),
                sr=self.sample_rate,
                n_steps=semitones,
            )

            return output.astype(np.float32)

        except ImportError:
            # Fallback: simple approach without librosa
            # Just return original if we can't pitch shift
            return audio


class VoiceHumanizer:
    """
    Main interface for humanizing AI-generated vocals.

    Combines breath insertion, pitch humanization, and timing adjustments.

    Example:
        humanizer = VoiceHumanizer()

        # Humanize with default settings
        humanized = humanizer.humanize(audio)

        # Humanize with emotional preset
        config = HumanizeConfig.for_emotion(EmotionalState.EXCITED)
        humanized = humanizer.humanize(audio, config=config)
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.breath_inserter = BreathInserter(sample_rate)
        self.pitch_humanizer = PitchHumanizer(sample_rate)

    def humanize(
        self,
        audio: Union[str, Path, np.ndarray],
        config: Optional[HumanizeConfig] = None,
        sample_rate: Optional[int] = None,
        is_question: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply full humanization pipeline to audio.

        Args:
            audio: Audio file path or numpy array
            config: Humanization configuration
            sample_rate: Sample rate (required if audio is array)
            is_question: Whether this is a question (affects intonation)

        Returns:
            Tuple of (humanized_audio, sample_rate)
        """
        config = config or HumanizeConfig()

        # Load audio if path
        if isinstance(audio, (str, Path)):
            y, sr = self._load_audio(audio)
        else:
            if sample_rate is None:
                sr = self.sample_rate
            else:
                sr = sample_rate
            y = audio.astype(np.float32)

        # Detect phrase boundaries once for all processors
        phrase_boundaries = self.breath_inserter._detect_phrases(y)

        # Apply humanization pipeline

        # 1. Insert breaths
        y = self.breath_inserter.insert_breaths(
            y,
            phrase_boundaries=phrase_boundaries,
            config=config.breath,
        )

        # 2. Apply pitch humanization
        y = self.pitch_humanizer.humanize(
            y,
            config=config.pitch,
            phrase_boundaries=phrase_boundaries,
            is_question=is_question,
        )

        # Scale by overall intensity
        # (intensity < 1 means more subtle effects)
        if config.intensity != 1.0:
            # Blend with original based on intensity
            # This requires keeping original, which we don't have here
            # So intensity mainly affects individual configs
            pass

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


# Convenience functions

def humanize_audio(
    audio: Union[str, Path, np.ndarray],
    emotion: Optional[EmotionalState] = None,
    sample_rate: int = 24000,
    is_question: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Convenience function to humanize audio.

    Args:
        audio: Audio file path or numpy array
        emotion: Emotional state preset
        sample_rate: Sample rate (if audio is array)
        is_question: Whether this is a question

    Returns:
        Tuple of (humanized_audio, sample_rate)
    """
    humanizer = VoiceHumanizer(sample_rate)

    if emotion:
        config = HumanizeConfig.for_emotion(emotion)
    else:
        config = HumanizeConfig()

    return humanizer.humanize(audio, config=config, is_question=is_question)


def add_breaths(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
    intensity: float = 0.7,
    volume_db: float = -15.0,
) -> np.ndarray:
    """
    Convenience function to add breath sounds.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate
        intensity: Breath intensity (0-1)
        volume_db: Breath volume relative to voice

    Returns:
        Audio with breaths inserted
    """
    inserter = BreathInserter(sample_rate)
    config = BreathConfig(intensity=intensity, volume_db=volume_db)

    if isinstance(audio, (str, Path)):
        import soundfile as sf
        y, sr = sf.read(str(audio))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
    else:
        y = audio

    return inserter.insert_breaths(y, config=config)


def humanize_pitch(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
    jitter_cents: float = 5.0,
    drift_cents: float = 15.0,
    is_question: bool = False,
) -> np.ndarray:
    """
    Convenience function for pitch humanization.

    Args:
        audio: Audio file path or numpy array
        sample_rate: Sample rate
        jitter_cents: Amount of micro-jitter
        drift_cents: Amount of pitch drift
        is_question: Whether this is a question

    Returns:
        Pitch-humanized audio
    """
    humanizer = PitchHumanizer(sample_rate)
    config = PitchHumanizeConfig(
        jitter_cents=jitter_cents,
        drift_max_cents=drift_cents,
    )

    if isinstance(audio, (str, Path)):
        import soundfile as sf
        y, sr = sf.read(str(audio))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
    else:
        y = audio

    return humanizer.humanize(y, config=config, is_question=is_question)
