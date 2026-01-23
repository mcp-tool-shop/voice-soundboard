"""
Voice Preset Library

Curated voice presets combining formant shifting and subtle humanization.
Based on acoustic research of notable speakers and voice archetypes.

Reference Sources:
- Morgan Freeman, James Earl Jones, Obama: ~96 Hz F0 (deep male)
- Average male: 100-130 Hz F0
- Average female: 180-230 Hz F0
- Husky female (Scarlett Johansson): lower end ~165-180 Hz
- Children: higher F0, shorter vocal tract = higher formants
- Elderly: F0 increases in males, decreases in females with age

Formant Shifting Reference:
- Male vocal tract ~15% longer than female = lower formants
- Deeper voice: ratio < 1.0 (shift formants down)
- Brighter voice: ratio > 1.0 (shift formants up)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np

from .humanize import (
    HumanizeConfig,
    BreathConfig,
    PitchHumanizeConfig,
    TimingHumanizeConfig,
    EmotionalState,
)
from .formants import FormantShifter


class VoicePreset(Enum):
    """
    Curated voice character presets.

    Each preset is based on acoustic analysis of notable speakers
    or voice archetypes from research literature.
    """

    # === NARRATOR VOICES ===
    WARM_NARRATOR = "warm_narrator"
    """
    Warm, trustworthy narrator voice.
    Based on: David Attenborough's documentary style
    F0 reference: ~110 Hz, slight formant lowering for warmth
    """

    DEEP_AUTHORITY = "deep_authority"
    """
    Deep, commanding authority voice.
    Based on: Morgan Freeman, James Earl Jones (~96 Hz F0)
    The "voice of God" archetype - deep, resonant, authoritative
    """

    ENERGETIC_HOST = "energetic_host"
    """
    Bright, engaging podcast/YouTube host voice.
    Slightly higher formants, more pitch variation
    """

    # === CHARACTER VOICES ===
    YOUNG_BRIGHT = "young_bright"
    """
    Youthful, energetic voice.
    Higher formants simulating younger/smaller vocal tract
    """

    ELDERLY_WISE = "elderly_wise"
    """
    Aged, wise character voice.
    Based on research: elderly males show F0 increase, more jitter
    Deeper formants with subtle instability
    """

    HUSKY_INTIMATE = "husky_intimate"
    """
    Husky, intimate voice.
    Based on: Scarlett Johansson's characteristic lower female voice
    Slightly lower formants, breathier quality
    """

    CHILD_LIKE = "child_like"
    """
    Childlike voice character.
    Based on research: children have shorter vocal tracts = higher formants
    More pitch variation and faster timing
    """

    # === PROFESSIONAL VOICES ===
    NEWS_ANCHOR = "news_anchor"
    """
    Professional broadcast voice.
    Based on research: optimal broadcast F0 ~126 Hz
    Clear, authoritative, minimal variation
    """

    AUDIOBOOK = "audiobook"
    """
    Engaging audiobook narrator.
    Balanced warmth with clear articulation
    Subtle humanization for long-form listening
    """

    # === MOOD VOICES ===
    CALM_MEDITATION = "calm_meditation"
    """
    Calm, soothing meditation guide voice.
    Slower, lower energy, very subtle humanization
    """

    CONFIDENT_PRESENTER = "confident_presenter"
    """
    Confident business presenter voice.
    Clear, assured, moderate depth
    """

    FRIENDLY_ASSISTANT = "friendly_assistant"
    """
    Friendly AI assistant voice.
    Neutral with slight warmth, approachable
    """


@dataclass
class PresetConfig:
    """Configuration for a voice preset."""

    name: str
    description: str

    # Formant shifting
    formant_ratio: float = 1.0  # < 1.0 = deeper, > 1.0 = brighter

    # Humanization (kept subtle by default)
    breath_enabled: bool = True
    breath_intensity: float = 0.15  # Very subtle
    breath_volume_db: float = -28.0  # Quiet

    pitch_jitter_cents: float = 3.0  # Subtle micro-variation
    pitch_drift_cents: float = 8.0  # Gentle drift

    timing_variation_ms: float = 10.0  # Subtle timing

    # Emotional base (mostly neutral for presets)
    emotion: EmotionalState = EmotionalState.NEUTRAL

    # Overall intensity (scales all humanization)
    intensity: float = 0.5  # 50% of full effect

    def to_humanize_config(self) -> HumanizeConfig:
        """Convert preset to HumanizeConfig."""
        return HumanizeConfig(
            breath=BreathConfig(
                enabled=self.breath_enabled,
                intensity=self.breath_intensity * self.intensity,
                volume_db=self.breath_volume_db,
            ),
            pitch=PitchHumanizeConfig(
                enabled=True,
                jitter_cents=self.pitch_jitter_cents * self.intensity,
                drift_max_cents=self.pitch_drift_cents * self.intensity,
            ),
            timing=TimingHumanizeConfig(
                enabled=True,
                timing_variation_ms=self.timing_variation_ms * self.intensity,
            ),
            emotion=self.emotion,
            intensity=self.intensity,
        )


# === PRESET DEFINITIONS ===
# Based on acoustic research and notable speaker analysis

PRESET_CONFIGS = {
    # --- NARRATOR VOICES ---

    VoicePreset.WARM_NARRATOR: PresetConfig(
        name="Warm Narrator",
        description="Warm, trustworthy documentary narrator (David Attenborough style)",
        formant_ratio=0.96,  # Slightly deeper for warmth
        breath_intensity=0.12,
        breath_volume_db=-30.0,
        pitch_jitter_cents=2.5,
        pitch_drift_cents=6.0,
        timing_variation_ms=8.0,
        emotion=EmotionalState.CALM,
        intensity=0.4,
    ),

    VoicePreset.DEEP_AUTHORITY: PresetConfig(
        name="Deep Authority",
        description="Deep, authoritative voice (Neil deGrasse Tyson style - engaging and gravelly)",
        formant_ratio=0.86,  # Deep but graceful
        breath_intensity=0.12,
        breath_volume_db=-30.0,
        pitch_jitter_cents=4.0,  # Gravelly texture
        pitch_drift_cents=8.0,   # Graceful pitch movement
        timing_variation_ms=10.0,  # Relaxed pacing
        emotion=EmotionalState.CALM,
        intensity=0.45,
    ),

    VoicePreset.ENERGETIC_HOST: PresetConfig(
        name="Energetic Host",
        description="Bright, engaging podcast/YouTube host",
        formant_ratio=1.04,  # Slightly brighter
        breath_intensity=0.18,
        breath_volume_db=-26.0,
        pitch_jitter_cents=4.0,
        pitch_drift_cents=12.0,
        timing_variation_ms=15.0,
        emotion=EmotionalState.EXCITED,
        intensity=0.5,
    ),

    # --- CHARACTER VOICES ---

    VoicePreset.YOUNG_BRIGHT: PresetConfig(
        name="Young Bright",
        description="Youthful, energetic voice with higher formants",
        formant_ratio=1.08,  # Higher formants = younger
        breath_intensity=0.15,
        breath_volume_db=-28.0,
        pitch_jitter_cents=4.5,
        pitch_drift_cents=10.0,
        timing_variation_ms=12.0,
        emotion=EmotionalState.EXCITED,
        intensity=0.45,
    ),

    VoicePreset.ELDERLY_WISE: PresetConfig(
        name="Elderly Wise",
        description="Aged, wise character (research: elderly males have higher F0, more jitter)",
        formant_ratio=0.94,  # Slightly deeper
        breath_intensity=0.20,  # More breaths for age
        breath_volume_db=-26.0,
        pitch_jitter_cents=5.5,  # More instability with age
        pitch_drift_cents=10.0,
        timing_variation_ms=18.0,  # Slower, more varied
        emotion=EmotionalState.TIRED,
        intensity=0.55,
    ),

    VoicePreset.HUSKY_INTIMATE: PresetConfig(
        name="Husky Intimate",
        description="Husky, intimate voice (Scarlett Johansson style)",
        formant_ratio=0.97,  # Slightly lower for huskiness
        breath_intensity=0.18,
        breath_volume_db=-26.0,
        pitch_jitter_cents=3.5,
        pitch_drift_cents=8.0,
        timing_variation_ms=10.0,
        emotion=EmotionalState.INTIMATE,
        intensity=0.5,
    ),

    VoicePreset.CHILD_LIKE: PresetConfig(
        name="Child-like",
        description="Childlike voice (shorter vocal tract = higher formants)",
        formant_ratio=1.15,  # Significantly higher formants
        breath_intensity=0.12,
        breath_volume_db=-28.0,
        pitch_jitter_cents=5.0,  # More variation in children
        pitch_drift_cents=15.0,
        timing_variation_ms=20.0,  # More timing variation
        emotion=EmotionalState.EXCITED,
        intensity=0.5,
    ),

    # --- PROFESSIONAL VOICES ---

    VoicePreset.NEWS_ANCHOR: PresetConfig(
        name="News Anchor",
        description="Professional broadcast voice (research: optimal ~126Hz)",
        formant_ratio=0.98,  # Very slight depth
        breath_intensity=0.08,  # Minimal breaths
        breath_volume_db=-34.0,
        pitch_jitter_cents=1.5,  # Very controlled
        pitch_drift_cents=4.0,
        timing_variation_ms=5.0,  # Precise timing
        emotion=EmotionalState.CONFIDENT,
        intensity=0.3,
    ),

    VoicePreset.AUDIOBOOK: PresetConfig(
        name="Audiobook",
        description="Engaging long-form audiobook narrator",
        formant_ratio=0.97,
        breath_intensity=0.15,
        breath_volume_db=-28.0,
        pitch_jitter_cents=3.0,
        pitch_drift_cents=8.0,
        timing_variation_ms=10.0,
        emotion=EmotionalState.CALM,
        intensity=0.45,
    ),

    # --- MOOD VOICES ---

    VoicePreset.CALM_MEDITATION: PresetConfig(
        name="Calm Meditation",
        description="Soothing meditation guide voice",
        formant_ratio=0.95,  # Slightly deeper for calm
        breath_intensity=0.20,  # More audible breaths for meditation
        breath_volume_db=-24.0,
        pitch_jitter_cents=2.0,
        pitch_drift_cents=5.0,
        timing_variation_ms=15.0,  # Slower pacing
        emotion=EmotionalState.CALM,
        intensity=0.4,
    ),

    VoicePreset.CONFIDENT_PRESENTER: PresetConfig(
        name="Confident Presenter",
        description="Confident business presenter",
        formant_ratio=0.98,
        breath_intensity=0.10,
        breath_volume_db=-30.0,
        pitch_jitter_cents=2.5,
        pitch_drift_cents=6.0,
        timing_variation_ms=8.0,
        emotion=EmotionalState.CONFIDENT,
        intensity=0.4,
    ),

    VoicePreset.FRIENDLY_ASSISTANT: PresetConfig(
        name="Friendly Assistant",
        description="Friendly AI assistant voice",
        formant_ratio=1.0,  # Neutral
        breath_intensity=0.12,
        breath_volume_db=-30.0,
        pitch_jitter_cents=3.0,
        pitch_drift_cents=7.0,
        timing_variation_ms=10.0,
        emotion=EmotionalState.NEUTRAL,
        intensity=0.4,
    ),
}


def get_preset_config(preset: VoicePreset) -> PresetConfig:
    """Get the configuration for a preset."""
    return PRESET_CONFIGS[preset]


def list_presets() -> list:
    """List all available presets with descriptions."""
    return [
        {
            "preset": preset.value,
            "name": config.name,
            "description": config.description,
            "formant_ratio": config.formant_ratio,
        }
        for preset, config in PRESET_CONFIGS.items()
    ]


def apply_preset(
    audio: Union[str, Path, np.ndarray],
    preset: VoicePreset,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Apply a voice preset to audio.

    Combines formant shifting and subtle humanization based on
    acoustic research of notable speakers.

    Args:
        audio: Audio file path or numpy array
        preset: VoicePreset to apply
        sample_rate: Sample rate (required if audio is array)

    Returns:
        Tuple of (processed_audio, sample_rate)

    Example:
        >>> audio, sr = apply_preset(tts_audio, VoicePreset.WARM_NARRATOR, sample_rate=24000)
        >>> audio, sr = apply_preset("speech.wav", VoicePreset.DEEP_AUTHORITY)
    """
    from .humanize import VoiceHumanizer

    config = PRESET_CONFIGS[preset]

    # Load audio if path
    if isinstance(audio, (str, Path)):
        try:
            import soundfile as sf
            y, sr = sf.read(str(audio))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")
    else:
        if sample_rate is None:
            raise ValueError("sample_rate required when audio is array")
        y = audio.astype(np.float32)
        sr = sample_rate

    # Apply formant shifting if ratio != 1.0
    if config.formant_ratio != 1.0:
        shifter = FormantShifter()
        y, sr = shifter.shift(y, ratio=config.formant_ratio, sample_rate=sr)

    # Apply humanization
    humanize_config = config.to_humanize_config()
    humanizer = VoiceHumanizer()
    y, sr = humanizer.humanize(y, config=humanize_config, sample_rate=sr)

    return y, sr


# Convenience aliases for common presets
def apply_narrator(audio, sample_rate=None):
    """Apply warm narrator preset."""
    return apply_preset(audio, VoicePreset.WARM_NARRATOR, sample_rate)


def apply_authority(audio, sample_rate=None):
    """Apply deep authority preset (Morgan Freeman style)."""
    return apply_preset(audio, VoicePreset.DEEP_AUTHORITY, sample_rate)


def apply_young(audio, sample_rate=None):
    """Apply young bright preset."""
    return apply_preset(audio, VoicePreset.YOUNG_BRIGHT, sample_rate)


def apply_elderly(audio, sample_rate=None):
    """Apply elderly wise preset."""
    return apply_preset(audio, VoicePreset.ELDERLY_WISE, sample_rate)
