"""
Voice Soundboard Vocology Module.

.. warning:: **Experimental.** This module may change or be removed between
   minor releases. See docs/API_STABILITY.md and docs/FEATURE_FLAGS.md.

Scientific voice analysis and manipulation based on vocology research.

Provides:
- Voice quality parameters (jitter, shimmer, HNR, etc.)
- Formant analysis and shifting
- Phonation type detection and synthesis
- Vocal biomarker analysis
- Prosody control

Example:
    from voice_soundboard.vocology import (
        VoiceQualityAnalyzer,
        FormantShifter,
        VocalBiomarkers,
    )

    # Analyze voice quality
    analyzer = VoiceQualityAnalyzer()
    metrics = analyzer.analyze("audio.wav")
    print(f"Jitter: {metrics.jitter_percent}%")

    # Shift formants for voice modification
    shifter = FormantShifter()
    modified = shifter.shift("audio.wav", ratio=0.95)  # Deeper voice
"""

from voice_soundboard.vocology.parameters import (
    VoiceQualityMetrics,
    VoiceQualityAnalyzer,
    JitterType,
    ShimmerType,
    analyze_voice_quality,
    get_jitter,
    get_shimmer,
    get_hnr,
    get_cpp,
)

from voice_soundboard.vocology.formants import (
    FormantAnalysis,
    FormantShifter,
    analyze_formants,
    shift_formants,
    FormantFrequencies,
)

from voice_soundboard.vocology.phonation import (
    PhonationType,
    PhonationAnalyzer,
    PhonationSynthesizer,
    detect_phonation,
    apply_phonation,
    PHONATION_PARAMS,
)

from voice_soundboard.vocology.biomarkers import (
    VocalBiomarkers,
    BiomarkerResult,
    VoiceHealthMetrics,
    analyze_biomarkers,
    assess_vocal_fatigue,
)

from voice_soundboard.vocology.prosody import (
    ProsodyContour,
    ProsodyAnalyzer,
    ProsodyModifier,
    analyze_prosody,
    modify_prosody,
    PitchContour,
    DurationPattern,
)

from voice_soundboard.vocology.humanize import (
    VoiceHumanizer,
    BreathInserter,
    BreathGenerator,
    PitchHumanizer,
    HumanizeConfig,
    BreathConfig,
    PitchHumanizeConfig,
    TimingHumanizeConfig,
    BreathType,
    EmotionalState,
    humanize_audio,
    add_breaths,
    humanize_pitch,
)

from voice_soundboard.vocology.rhythm import (
    RhythmAnalyzer,
    RhythmModifier,
    RhythmMetrics,
    RhythmZone,
    RZTAnalysis,
    RhythmClass,
    RhythmBand,
    analyze_rhythm,
    analyze_rhythm_zones,
    add_rhythm_variability,
)

from voice_soundboard.vocology.presets import (
    VoicePreset,
    PresetConfig,
    PRESET_CONFIGS,
    apply_preset,
    get_preset_config,
    list_presets,
    apply_narrator,
    apply_authority,
    apply_young,
    apply_elderly,
)

from voice_soundboard.vocology.naturalness import (
    NaturalSpeechProcessor,
    NaturalnessConfig,
    ProcessedText,
    FillerType,
    humanize_text,
)

__all__ = [
    # Voice Quality Parameters
    "VoiceQualityMetrics",
    "VoiceQualityAnalyzer",
    "JitterType",
    "ShimmerType",
    "analyze_voice_quality",
    "get_jitter",
    "get_shimmer",
    "get_hnr",
    "get_cpp",
    # Formants
    "FormantAnalysis",
    "FormantShifter",
    "analyze_formants",
    "shift_formants",
    "FormantFrequencies",
    # Phonation
    "PhonationType",
    "PhonationAnalyzer",
    "PhonationSynthesizer",
    "detect_phonation",
    "apply_phonation",
    "PHONATION_PARAMS",
    # Biomarkers
    "VocalBiomarkers",
    "BiomarkerResult",
    "VoiceHealthMetrics",
    "analyze_biomarkers",
    "assess_vocal_fatigue",
    # Prosody
    "ProsodyContour",
    "ProsodyAnalyzer",
    "ProsodyModifier",
    "analyze_prosody",
    "modify_prosody",
    "PitchContour",
    "DurationPattern",
    # Humanization
    "VoiceHumanizer",
    "BreathInserter",
    "BreathGenerator",
    "PitchHumanizer",
    "HumanizeConfig",
    "BreathConfig",
    "PitchHumanizeConfig",
    "TimingHumanizeConfig",
    "BreathType",
    "EmotionalState",
    "humanize_audio",
    "add_breaths",
    "humanize_pitch",
    # Rhythm Analysis (RZT)
    "RhythmAnalyzer",
    "RhythmModifier",
    "RhythmMetrics",
    "RhythmZone",
    "RZTAnalysis",
    "RhythmClass",
    "RhythmBand",
    "analyze_rhythm",
    "analyze_rhythm_zones",
    "add_rhythm_variability",
    # Voice Presets
    "VoicePreset",
    "PresetConfig",
    "PRESET_CONFIGS",
    "apply_preset",
    "get_preset_config",
    "list_presets",
    "apply_narrator",
    "apply_authority",
    "apply_young",
    "apply_elderly",
    # Natural Speech Humanization
    "NaturalSpeechProcessor",
    "NaturalnessConfig",
    "ProcessedText",
    "FillerType",
    "humanize_text",
]
