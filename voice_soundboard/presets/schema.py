"""
Voice Preset Schema

Unified schema for voice presets from multiple sources (Qwen3, VibeVoice, Hume, etc.)
Designed for both human and AI-friendly selection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
from pathlib import Path


class Gender(str, Enum):
    """Voice gender classification."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AgeRange(str, Enum):
    """Voice age range classification."""
    CHILD = "child"          # Under 12
    TEEN = "teen"            # 12-19
    YOUNG_ADULT = "young_adult"  # 20-35
    ADULT = "adult"          # 35-50
    MIDDLE_AGED = "middle_aged"  # 50-65
    ELDERLY = "elderly"      # 65+


class VoiceEnergy(str, Enum):
    """Voice energy level."""
    CALM = "calm"
    NEUTRAL = "neutral"
    ENERGETIC = "energetic"
    INTENSE = "intense"


class VoiceTone(str, Enum):
    """Voice tonal quality."""
    WARM = "warm"
    NEUTRAL = "neutral"
    COOL = "cool"
    BRIGHT = "bright"
    DARK = "dark"


class PresetSource(str, Enum):
    """Source of the preset definition."""
    QWEN3 = "qwen3"          # Alibaba Qwen3-TTS
    VIBEVOICE = "vibevoice"  # Microsoft VibeVoice
    HUME = "hume"            # Hume AI Octave
    FISH = "fish"            # Fish Audio
    VOCOLOGY = "vocology"    # Our acoustic research presets
    KOKORO = "kokoro"        # Kokoro TTS base voices
    DIVERSE = "diverse"      # Diverse demographic coverage presets
    CUSTOM = "custom"        # User-created


@dataclass
class AcousticParams:
    """
    Acoustic parameters for voice post-processing.

    Based on vocology research - these modify the voice characteristics
    through formant shifting, humanization, and phonation effects.
    """
    # Formant shifting (< 1.0 = deeper, > 1.0 = brighter)
    formant_ratio: float = 1.0

    # Pitch characteristics
    pitch_shift_semitones: float = 0.0
    jitter_percent: float = 0.5      # Pitch instability (0-2%)

    # Amplitude characteristics
    shimmer_percent: float = 2.0     # Amplitude instability (0-10%)

    # Humanization
    breath_intensity: float = 0.15   # Breath sound level (0-1)
    breath_volume_db: float = -28.0  # Breath volume relative to voice
    pitch_drift_cents: float = 8.0   # Slow pitch drift (0-20 cents)
    timing_variation_ms: float = 10.0  # Timing jitter (0-30ms)

    # Speed modification
    speed_factor: float = 1.0        # Speech rate multiplier

    # === Vocology Parameters ===
    # Emotional state (affects multiple params automatically)
    emotional_state: str = "neutral"  # neutral, excited, calm, tired, confident, intimate

    # Pitch scooping (natural slide into notes)
    scoop_cents: float = 30.0        # How flat to start before sliding up (0-60)

    # Phrase-final intonation
    final_drop_cents: float = 20.0   # Negative = drop (declarative), positive = rise (question)

    # Pitch overshoot on stressed words
    overshoot_cents: float = 15.0    # Brief overshoot amount (0-30)

    # Timing bias (confident = ahead, tired = behind)
    timing_bias_ms: float = 0.0      # -20 to +20 ms

    # === Research Lab Parameters ===
    # Phonation type (voice quality)
    phonation_type: str = "modal"    # modal, breathy, creaky, harsh
    phonation_intensity: float = 0.5 # How strongly to apply phonation (0-1)

    # Jitter/drift rate control
    jitter_rate_hz: float = 10.0     # Rate of pitch micro-variation (5-20 Hz)
    drift_rate_hz: float = 0.5       # Rate of phrase-level pitch drift (0.1-1.5 Hz)

    # Intonation Unit duration (based on 2024 neuroscience research)
    iu_duration_s: float = 1.6       # Universal speech rhythm ~1.6s (1.0-2.5s)

    # Voice clarity
    hnr_target_db: float = 20.0      # Harmonic-to-noise ratio target (5-30 dB)

    # Spectral tilt (high frequency roll-off)
    spectral_tilt_db: float = -12.0  # Steeper = darker/breathier (-20 to -4)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "formant_ratio": self.formant_ratio,
            "pitch_shift_semitones": self.pitch_shift_semitones,
            "jitter_percent": self.jitter_percent,
            "shimmer_percent": self.shimmer_percent,
            "breath_intensity": self.breath_intensity,
            "breath_volume_db": self.breath_volume_db,
            "pitch_drift_cents": self.pitch_drift_cents,
            "timing_variation_ms": self.timing_variation_ms,
            "speed_factor": self.speed_factor,
            # Vocology params
            "emotional_state": self.emotional_state,
            "scoop_cents": self.scoop_cents,
            "final_drop_cents": self.final_drop_cents,
            "overshoot_cents": self.overshoot_cents,
            "timing_bias_ms": self.timing_bias_ms,
            # Research Lab params
            "phonation_type": self.phonation_type,
            "phonation_intensity": self.phonation_intensity,
            "jitter_rate_hz": self.jitter_rate_hz,
            "drift_rate_hz": self.drift_rate_hz,
            "iu_duration_s": self.iu_duration_s,
            "hnr_target_db": self.hnr_target_db,
            "spectral_tilt_db": self.spectral_tilt_db,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VoicePreset:
    """
    A voice preset definition.

    Designed to be:
    1. Human-readable (clear names and descriptions)
    2. AI-friendly (rich tags and searchable descriptions)
    3. Engine-agnostic (works with multiple TTS backends)
    4. Demographically inclusive (covers diverse voices)

    Example:
        >>> preset = VoicePreset(
        ...     id="qwen3:ryan",
        ...     name="Ryan",
        ...     source=PresetSource.QWEN3,
        ...     description="Dynamic male voice with strong rhythmic drive",
        ...     tags=["male", "energetic", "american", "young"],
        ...     use_cases=["podcast", "gaming", "advertisement"],
        ...     gender=Gender.MALE,
        ...     age_range=AgeRange.YOUNG_ADULT,
        ...     voice_prompt="Dynamic male voice with strong rhythmic drive, confident delivery",
        ...     language="en",
        ... )
    """

    # === Identity ===
    id: str                              # Unique ID: "source:name" format
    name: str                            # Human-readable name
    source: PresetSource                 # Where this preset comes from

    # === Description (for AI selection) ===
    description: str                     # Rich natural language description
    tags: list[str] = field(default_factory=list)  # Searchable tags
    use_cases: list[str] = field(default_factory=list)  # Recommended applications

    # === Demographics ===
    gender: Optional[Gender] = None
    age_range: Optional[AgeRange] = None
    accent: Optional[str] = None         # "american", "british", "indian", etc.
    ethnicity_hint: Optional[str] = None # Cultural/regional hint for voice character

    # === Voice Characteristics ===
    energy: VoiceEnergy = VoiceEnergy.NEUTRAL
    tone: VoiceTone = VoiceTone.NEUTRAL
    voice_prompt: Optional[str] = None   # Natural language voice description (Qwen3 style)
    acting_instructions: Optional[str] = None  # How to deliver (Hume style)

    # === Technical Parameters ===
    acoustic: Optional[AcousticParams] = None  # Post-processing parameters

    # === Engine Hints ===
    preferred_engine: Optional[str] = None  # "kokoro", "chatterbox", "qwen3", etc.
    kokoro_voice: Optional[str] = None      # Direct Kokoro voice ID mapping
    chatterbox_voice: Optional[str] = None  # Chatterbox reference voice
    language: str = "en"                    # Primary language code
    languages: list[str] = field(default_factory=lambda: ["en"])  # All supported languages

    # === Metadata ===
    version: str = "1.0"
    author: Optional[str] = None
    license: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize preset data."""
        # Ensure source is enum
        if isinstance(self.source, str):
            self.source = PresetSource(self.source)

        # Ensure gender is enum or None
        if isinstance(self.gender, str):
            self.gender = Gender(self.gender)

        # Ensure age_range is enum or None
        if isinstance(self.age_range, str):
            self.age_range = AgeRange(self.age_range)

        # Ensure energy is enum
        if isinstance(self.energy, str):
            self.energy = VoiceEnergy(self.energy)

        # Ensure tone is enum
        if isinstance(self.tone, str):
            self.tone = VoiceTone(self.tone)

        # Convert acoustic dict to AcousticParams
        if isinstance(self.acoustic, dict):
            self.acoustic = AcousticParams.from_dict(self.acoustic)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source.value if isinstance(self.source, PresetSource) else self.source,
            "description": self.description,
            "tags": self.tags,
            "use_cases": self.use_cases,
            "gender": self.gender.value if self.gender else None,
            "age_range": self.age_range.value if self.age_range else None,
            "accent": self.accent,
            "ethnicity_hint": self.ethnicity_hint,
            "energy": self.energy.value if isinstance(self.energy, VoiceEnergy) else self.energy,
            "tone": self.tone.value if isinstance(self.tone, VoiceTone) else self.tone,
            "voice_prompt": self.voice_prompt,
            "acting_instructions": self.acting_instructions,
            "acoustic": self.acoustic.to_dict() if self.acoustic else None,
            "preferred_engine": self.preferred_engine,
            "kokoro_voice": self.kokoro_voice,
            "chatterbox_voice": self.chatterbox_voice,
            "language": self.language,
            "languages": self.languages,
            "version": self.version,
            "author": self.author,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoicePreset":
        """Create from dictionary (e.g., loaded from JSON)."""
        # Handle acoustic params
        if data.get("acoustic") and isinstance(data["acoustic"], dict):
            data["acoustic"] = AcousticParams.from_dict(data["acoustic"])

        # Filter to only valid fields
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def get_search_text(self) -> str:
        """
        Generate searchable text for semantic search.

        Combines name, description, tags, and voice prompt into
        a single searchable string optimized for embedding.
        """
        parts = [
            self.name,
            self.description,
            f"Tags: {', '.join(self.tags)}" if self.tags else "",
            f"Use cases: {', '.join(self.use_cases)}" if self.use_cases else "",
            self.voice_prompt or "",
            self.acting_instructions or "",
            f"Gender: {self.gender.value}" if self.gender else "",
            f"Age: {self.age_range.value}" if self.age_range else "",
            f"Accent: {self.accent}" if self.accent else "",
            f"Energy: {self.energy.value}" if self.energy else "",
            f"Tone: {self.tone.value}" if self.tone else "",
        ]
        return " ".join(p for p in parts if p)

    def matches_filters(
        self,
        gender: Optional[str] = None,
        age_range: Optional[str] = None,
        accent: Optional[str] = None,
        energy: Optional[str] = None,
        tone: Optional[str] = None,
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_case: Optional[str] = None,
    ) -> bool:
        """Check if preset matches the given filters."""
        if gender and self.gender and self.gender.value != gender:
            return False
        if age_range and self.age_range and self.age_range.value != age_range:
            return False
        if accent and self.accent and self.accent.lower() != accent.lower():
            return False
        if energy and self.energy and self.energy.value != energy:
            return False
        if tone and self.tone and self.tone.value != tone:
            return False
        if language and language not in self.languages:
            return False
        if tags and not all(t in self.tags for t in tags):
            return False
        if use_case and use_case not in self.use_cases:
            return False
        return True

    def summary(self) -> str:
        """Return a concise summary for display."""
        parts = [self.name]
        if self.gender:
            parts.append(self.gender.value)
        if self.age_range:
            parts.append(self.age_range.value.replace("_", " "))
        if self.accent:
            parts.append(self.accent)
        return f"{' | '.join(parts)}: {self.description[:80]}..."


@dataclass
class PresetCatalogMetadata:
    """Metadata for a preset catalog file."""
    source: str
    version: str
    description: str
    author: str
    last_updated: str
    preset_count: int = 0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "last_updated": self.last_updated,
            "preset_count": self.preset_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PresetCatalogMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
