"""
Voice Preset Library

A curated catalog of 50+ voice presets from multiple sources:
- Qwen3-TTS (Alibaba)
- Microsoft VibeVoice
- Hume AI Octave
- Vocology research presets
- Diverse coverage presets

Example:
    >>> from voice_soundboard.presets import get_catalog
    >>>
    >>> # Get the global catalog
    >>> catalog = get_catalog()
    >>>
    >>> # Get a specific preset
    >>> preset = catalog.get("qwen3:ryan")
    >>> print(preset.description)
    >>>
    >>> # Search for presets
    >>> results = catalog.search("warm narrator for meditation")
    >>> for r in results:
    ...     print(f"{r.preset.name}: {r.score:.0%}")
    >>>
    >>> # Filter by attributes
    >>> calm_females = catalog.filter(gender="female", energy="calm")
"""

from .schema import (
    VoicePreset,
    AcousticParams,
    PresetCatalogMetadata,
    PresetSource,
    Gender,
    AgeRange,
    VoiceEnergy,
    VoiceTone,
)

from .catalog import (
    PresetCatalog,
    SearchResult,
    get_catalog,
    reload_catalog,
)

__all__ = [
    # Schema
    "VoicePreset",
    "AcousticParams",
    "PresetCatalogMetadata",
    "PresetSource",
    "Gender",
    "AgeRange",
    "VoiceEnergy",
    "VoiceTone",
    # Catalog
    "PresetCatalog",
    "SearchResult",
    "get_catalog",
    "reload_catalog",
]
