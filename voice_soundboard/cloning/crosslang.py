"""
Cross-Language Voice Cloning.

Enables cloning a voice from one language and
synthesizing speech in another language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

import numpy as np

from voice_soundboard.cloning.extractor import VoiceEmbedding


class Language(Enum):
    """Supported languages for cross-language cloning."""

    # Major languages
    ENGLISH = "en"
    CHINESE_MANDARIN = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"

    # Additional languages
    HINDI = "hi"
    ARABIC = "ar"
    DUTCH = "nl"
    POLISH = "pl"
    SWEDISH = "sv"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"
    CZECH = "cs"


# Language code to Language enum mapping
SUPPORTED_LANGUAGES: Dict[str, Language] = {lang.value: lang for lang in Language}


@dataclass
class LanguageConfig:
    """Configuration for a specific language."""

    code: str
    name: str
    native_name: str

    # Phonetic characteristics
    phoneme_set: str = "ipa"  # IPA or language-specific
    has_tones: bool = False
    syllable_timed: bool = False
    stress_timed: bool = True

    # TTS backend support
    supported_backends: List[str] = field(default_factory=lambda: ["kokoro"])

    # Language-specific settings
    default_speed: float = 1.0
    typical_speaking_rate_wpm: int = 150  # Words per minute

    # Romanization/transliteration
    requires_romanization: bool = False
    romanization_system: Optional[str] = None


# Language configurations
LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        code="en",
        name="English",
        native_name="English",
        stress_timed=True,
        typical_speaking_rate_wpm=150,
    ),
    "zh": LanguageConfig(
        code="zh",
        name="Chinese (Mandarin)",
        native_name="中文",
        has_tones=True,
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=160,
        requires_romanization=True,
        romanization_system="pinyin",
    ),
    "es": LanguageConfig(
        code="es",
        name="Spanish",
        native_name="Español",
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=180,
    ),
    "fr": LanguageConfig(
        code="fr",
        name="French",
        native_name="Français",
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=170,
    ),
    "de": LanguageConfig(
        code="de",
        name="German",
        native_name="Deutsch",
        stress_timed=True,
        typical_speaking_rate_wpm=140,
    ),
    "ja": LanguageConfig(
        code="ja",
        name="Japanese",
        native_name="日本語",
        has_tones=True,  # Pitch accent
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=200,
        requires_romanization=True,
        romanization_system="romaji",
    ),
    "ko": LanguageConfig(
        code="ko",
        name="Korean",
        native_name="한국어",
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=180,
        requires_romanization=True,
        romanization_system="revised_romanization",
    ),
    "pt": LanguageConfig(
        code="pt",
        name="Portuguese",
        native_name="Português",
        stress_timed=True,
        typical_speaking_rate_wpm=170,
    ),
    "it": LanguageConfig(
        code="it",
        name="Italian",
        native_name="Italiano",
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=170,
    ),
    "ru": LanguageConfig(
        code="ru",
        name="Russian",
        native_name="Русский",
        stress_timed=True,
        typical_speaking_rate_wpm=130,
    ),
    "hi": LanguageConfig(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        syllable_timed=True,
        stress_timed=False,
        typical_speaking_rate_wpm=160,
        requires_romanization=True,
    ),
    "ar": LanguageConfig(
        code="ar",
        name="Arabic",
        native_name="العربية",
        stress_timed=True,
        typical_speaking_rate_wpm=130,
        requires_romanization=True,
    ),
}


@dataclass
class CrossLanguageResult:
    """Result of cross-language synthesis."""

    success: bool
    source_language: str
    target_language: str

    # Audio output (when using with TTS)
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000

    # Metrics
    timbre_preservation_score: float = 0.0  # How well voice character preserved
    accent_transfer_score: float = 0.0  # How natural accent sounds

    # Issues
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class CrossLanguageCloner:
    """
    Handles cross-language voice cloning.

    Enables using a voice cloned from one language to
    synthesize speech in another language.
    """

    def __init__(
        self,
        source_language: str = "en",
        preserve_accent: bool = False,
    ):
        """
        Initialize cross-language cloner.

        Args:
            source_language: Language of the source voice
            preserve_accent: Whether to preserve source accent in target
        """
        self.source_language = source_language
        self.preserve_accent = preserve_accent

    @property
    def source_config(self) -> LanguageConfig:
        """Get source language configuration."""
        return LANGUAGE_CONFIGS.get(
            self.source_language,
            LANGUAGE_CONFIGS["en"],
        )

    def get_target_config(self, target_language: str) -> LanguageConfig:
        """Get target language configuration."""
        return LANGUAGE_CONFIGS.get(
            target_language,
            LANGUAGE_CONFIGS["en"],
        )

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in LANGUAGE_CONFIGS

    def list_supported_languages(self) -> List[Dict[str, str]]:
        """List all supported languages with metadata."""
        return [
            {
                "code": config.code,
                "name": config.name,
                "native_name": config.native_name,
            }
            for config in LANGUAGE_CONFIGS.values()
        ]

    def get_language_pair_compatibility(
        self,
        source: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Assess compatibility between language pair.

        Args:
            source: Source language code
            target: Target language code

        Returns:
            Compatibility assessment
        """
        source_config = LANGUAGE_CONFIGS.get(source)
        target_config = LANGUAGE_CONFIGS.get(target)

        if not source_config or not target_config:
            return {
                "compatible": False,
                "reason": "One or both languages not supported",
            }

        # Assess phonetic similarity
        phonetic_issues = []
        expected_quality = 1.0

        # Tonal language considerations
        if target_config.has_tones and not source_config.has_tones:
            phonetic_issues.append(
                f"{target_config.name} is tonal; source voice may lack tonal variation"
            )
            expected_quality -= 0.2

        # Timing system differences
        if source_config.stress_timed != target_config.stress_timed:
            phonetic_issues.append(
                "Different timing systems may affect rhythm naturalness"
            )
            expected_quality -= 0.1

        # Same language family bonuses
        family_groups = {
            "romance": ["es", "fr", "it", "pt"],
            "germanic": ["en", "de", "nl", "sv"],
            "slavic": ["ru", "pl", "cs"],
            "cjk": ["zh", "ja", "ko"],
        }

        same_family = False
        for languages in family_groups.values():
            if source in languages and target in languages:
                same_family = True
                expected_quality = min(1.0, expected_quality + 0.1)
                break

        return {
            "compatible": True,
            "source": source_config.name,
            "target": target_config.name,
            "same_language_family": same_family,
            "expected_quality": expected_quality,
            "phonetic_issues": phonetic_issues,
            "recommendations": self._get_recommendations(source_config, target_config),
        }

    def _get_recommendations(
        self,
        source: LanguageConfig,
        target: LanguageConfig,
    ) -> List[str]:
        """Get recommendations for language pair."""
        recommendations = []

        if target.has_tones:
            recommendations.append(
                f"Use expressive source audio for better {target.name} tonal rendering"
            )

        if abs(source.typical_speaking_rate_wpm - target.typical_speaking_rate_wpm) > 30:
            if target.typical_speaking_rate_wpm > source.typical_speaking_rate_wpm:
                recommendations.append(
                    f"{target.name} typically faster; may want to increase speed"
                )
            else:
                recommendations.append(
                    f"{target.name} typically slower; may want to decrease speed"
                )

        if target.requires_romanization:
            recommendations.append(
                f"Ensure {target.name} text is properly formatted"
            )

        return recommendations

    def prepare_embedding_for_language(
        self,
        embedding: VoiceEmbedding,
        target_language: str,
    ) -> Tuple[VoiceEmbedding, Dict[str, Any]]:
        """
        Prepare an embedding for cross-language synthesis.

        This may involve:
        - Adjusting speaker characteristics
        - Normalizing phonetic features
        - Language-specific transformations

        Args:
            embedding: Source voice embedding
            target_language: Target language code

        Returns:
            Tuple of (prepared embedding, metadata)
        """
        # For now, embeddings are language-agnostic in most models
        # Future: Apply language-specific transformations

        target_config = self.get_target_config(target_language)
        source_config = self.source_config

        metadata = {
            "source_language": source_config.code,
            "target_language": target_config.code,
            "transformations_applied": [],
        }

        # Speed adjustment recommendation
        speed_ratio = (
            target_config.typical_speaking_rate_wpm
            / source_config.typical_speaking_rate_wpm
        )
        metadata["recommended_speed_multiplier"] = speed_ratio

        # The embedding itself is returned unchanged for most backends
        # Advanced backends would apply accent transfer here
        return embedding, metadata

    def estimate_quality(
        self,
        embedding: VoiceEmbedding,
        target_language: str,
    ) -> float:
        """
        Estimate synthesis quality for target language.

        Args:
            embedding: Voice embedding
            target_language: Target language

        Returns:
            Estimated quality score 0-1
        """
        # Start with embedding quality
        quality = embedding.quality_score

        # Get compatibility
        compat = self.get_language_pair_compatibility(
            self.source_language,
            target_language,
        )

        # Factor in expected quality from language pair
        quality *= compat.get("expected_quality", 0.8)

        # Duration affects quality (longer = better characterization)
        if embedding.source_duration_seconds < 3:
            quality *= 0.8
        elif embedding.source_duration_seconds > 10:
            quality = min(1.0, quality * 1.1)

        return min(1.0, max(0.0, quality))


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of text.

    Args:
        text: Text to analyze

    Returns:
        Language code or None if uncertain
    """
    # Simple heuristic detection
    # In production, use a proper library like langdetect or fasttext

    # Check for CJK characters
    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0xAC00, 0xD7AF),  # Hangul
    ]

    for char in text:
        code = ord(char)

        # Japanese (has hiragana/katakana)
        if 0x3040 <= code <= 0x30FF:
            return "ja"

        # Korean (Hangul)
        if 0xAC00 <= code <= 0xD7AF:
            return "ko"

        # Chinese (if only CJK unified, likely Chinese)
        if 0x4E00 <= code <= 0x9FFF:
            # Could be Chinese, Japanese, or Korean
            # Default to Chinese if no kana/hangul found
            return "zh"

        # Cyrillic (Russian, etc.)
        if 0x0400 <= code <= 0x04FF:
            return "ru"

        # Arabic
        if 0x0600 <= code <= 0x06FF:
            return "ar"

        # Devanagari (Hindi)
        if 0x0900 <= code <= 0x097F:
            return "hi"

        # Thai
        if 0x0E00 <= code <= 0x0E7F:
            return "th"

    # For Latin-based scripts, we'd need more sophisticated detection
    # Default to English
    return "en"
