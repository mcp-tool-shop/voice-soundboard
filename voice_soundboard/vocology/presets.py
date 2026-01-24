"""
Voice Preset Library

Curated voice presets combining formant shifting and subtle humanization.
Based on acoustic research of notable speakers and voice archetypes.

.. deprecated:: 1.2.0
    This module is deprecated. Use `voice_soundboard.presets` instead.
    The new preset system provides 70+ presets with semantic search,
    demographic coverage, and multi-source support.

    Migration guide:
        Old: from voice_soundboard.vocology.presets import VoicePreset, apply_preset
        New: from voice_soundboard.presets import get_catalog, VoicePreset as NewPreset

Reference Sources:
- Deep authoritative male voices: ~96 Hz F0
- Average male: 100-130 Hz F0
- Average female: 180-230 Hz F0
- Husky female voices: lower end ~165-180 Hz
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
import warnings

from .humanize import (
    HumanizeConfig,
    BreathConfig,
    PitchHumanizeConfig,
    TimingHumanizeConfig,
    EmotionalState,
)
from .formants import FormantShifter


# Emit deprecation warning on module import
warnings.warn(
    "voice_soundboard.vocology.presets is deprecated. "
    "Use voice_soundboard.presets instead for 70+ presets with semantic search.",
    DeprecationWarning,
    stacklevel=2,
)


# Mapping from old VoicePreset enum values to new catalog IDs
_OLD_TO_NEW_PRESET_MAP = {
    "warm_narrator": "vocology:warm_narrator",
    "deep_authority": "vocology:deep_authority",
    "energetic_host": "vocology:energetic_host",
    "young_bright": "vocology:young_bright",
    "elderly_wise": "vocology:elderly_wise",
    "husky_intimate": "vocology:husky_intimate",
    "child_like": "vocology:child_like",
    "news_anchor": "vocology:news_anchor",
    "audiobook": "vocology:audiobook",
    "calm_meditation": "vocology:calm_meditation",
    "confident_presenter": "vocology:confident_presenter",
    "friendly_assistant": "vocology:friendly_assistant",
}


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
    Documentary narrator style with slight warmth.
    F0 reference: ~110 Hz, slight formant lowering for warmth
    """

    DEEP_AUTHORITY = "deep_authority"
    """
    Deep, commanding authority voice.
    Deep resonant voice (~96 Hz F0) - authoritative and engaging.
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
    Lower register with slight breathiness - works best with female voices.
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
        description="Warm, trustworthy documentary narrator",
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
        description="Deep, authoritative voice - engaging and gravelly",
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
        description="Husky, intimate voice - best with female speakers",
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
    """Apply warm narrator preset.

    .. deprecated:: 1.2.0
        Use `voice_soundboard.presets.get_catalog().get("vocology:warm_narrator")` instead.
    """
    warnings.warn(
        "apply_narrator is deprecated. Use voice_soundboard.presets instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_preset(audio, VoicePreset.WARM_NARRATOR, sample_rate)


def apply_authority(audio, sample_rate=None):
    """Apply deep authority preset.

    .. deprecated:: 1.2.0
        Use `voice_soundboard.presets.get_catalog().get("vocology:deep_authority")` instead.
    """
    warnings.warn(
        "apply_authority is deprecated. Use voice_soundboard.presets instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_preset(audio, VoicePreset.DEEP_AUTHORITY, sample_rate)


def apply_young(audio, sample_rate=None):
    """Apply young bright preset.

    .. deprecated:: 1.2.0
        Use `voice_soundboard.presets.get_catalog().get("vocology:young_bright")` instead.
    """
    warnings.warn(
        "apply_young is deprecated. Use voice_soundboard.presets instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_preset(audio, VoicePreset.YOUNG_BRIGHT, sample_rate)


def apply_elderly(audio, sample_rate=None):
    """Apply elderly wise preset.

    .. deprecated:: 1.2.0
        Use `voice_soundboard.presets.get_catalog().get("vocology:elderly_wise")` instead.
    """
    warnings.warn(
        "apply_elderly is deprecated. Use voice_soundboard.presets instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_preset(audio, VoicePreset.ELDERLY_WISE, sample_rate)


# === MIGRATION HELPERS ===

def get_new_preset_id(old_preset: VoicePreset) -> str:
    """
    Get the new catalog ID for an old VoicePreset enum value.

    Args:
        old_preset: Old VoicePreset enum value

    Returns:
        New catalog ID (e.g., "vocology:warm_narrator")

    Example:
        >>> from voice_soundboard.vocology.presets import VoicePreset, get_new_preset_id
        >>> new_id = get_new_preset_id(VoicePreset.WARM_NARRATOR)
        >>> # new_id == "vocology:warm_narrator"
        >>> from voice_soundboard.presets import get_catalog
        >>> new_preset = get_catalog().get(new_id)
    """
    return _OLD_TO_NEW_PRESET_MAP.get(old_preset.value, f"vocology:{old_preset.value}")


def migrate_to_new_catalog():
    """
    Helper to migrate from old presets to new catalog system.

    Returns a dictionary mapping old enum values to new VoicePreset objects.

    Example:
        >>> mapping = migrate_to_new_catalog()
        >>> new_preset = mapping[VoicePreset.WARM_NARRATOR]
        >>> print(new_preset.description)
    """
    try:
        from voice_soundboard.presets import get_catalog
        catalog = get_catalog()

        result = {}
        for old_preset in VoicePreset:
            new_id = get_new_preset_id(old_preset)
            new_preset = catalog.get(new_id)
            if new_preset:
                result[old_preset] = new_preset
            else:
                warnings.warn(f"No migration found for {old_preset.value}")

        return result
    except ImportError:
        warnings.warn(
            "voice_soundboard.presets not available. "
            "Install sentence-transformers for full features."
        )
        return {}
