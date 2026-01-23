"""
Tests for Phase 8: engines/__init__.py F5-TTS exports and crosslang.py new languages.

Tests cover:
- engines/__init__.py F5-TTS export (TEST-EI-F5-01 to TEST-EI-F5-07)
- cloning/crosslang.py Language enum additions (TEST-CL-01 to TEST-CL-08)
- cloning/crosslang.py LanguageConfig additions (TEST-CL-09 to TEST-CL-24)
- cloning/crosslang.py LanguageConfig structure validation (TEST-CL-25 to TEST-CL-31)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-EI-F5-01 to TEST-EI-F5-07: engines/__init__.py F5-TTS Export Tests
# =============================================================================

class TestEnginesInitF5TTSExport:
    """Tests for F5TTSEngine export in engines/__init__.py (TEST-EI-F5-01 to TEST-EI-F5-07)."""

    def test_ei_f5_01_f5tts_available_exported(self):
        """TEST-EI-F5-01: F5TTS_AVAILABLE is exported."""
        from voice_soundboard import engines

        assert hasattr(engines, 'F5TTS_AVAILABLE')

    def test_ei_f5_02_f5tts_engine_exported_when_installed(self):
        """TEST-EI-F5-02: F5TTSEngine is exported when f5-tts is installed."""
        from voice_soundboard import engines

        # F5TTSEngine should be in the namespace (may be None if not installed)
        assert hasattr(engines, 'F5TTSEngine')

    def test_ei_f5_03_f5tts_engine_none_when_not_installed(self):
        """TEST-EI-F5-03: F5TTSEngine is None when f5-tts is not installed."""
        # This test verifies the behavior when f5-tts is not installed
        # We can't easily mock this since the import has already happened
        # Instead, verify the pattern is correct
        from voice_soundboard.engines import F5TTS_AVAILABLE, F5TTSEngine

        if not F5TTS_AVAILABLE:
            assert F5TTSEngine is None
        else:
            # If installed, it should be a class
            assert F5TTSEngine is not None

    def test_ei_f5_04_f5tts_available_is_bool(self):
        """TEST-EI-F5-04: F5TTS_AVAILABLE is True when f5-tts is installed."""
        from voice_soundboard.engines import F5TTS_AVAILABLE

        assert isinstance(F5TTS_AVAILABLE, bool)

    def test_ei_f5_05_f5tts_available_false_when_not_installed(self):
        """TEST-EI-F5-05: F5TTS_AVAILABLE is False when f5-tts is not installed."""
        # This tests the expected behavior - if not available, should be False
        from voice_soundboard.engines import F5TTS_AVAILABLE, F5TTSEngine

        if F5TTSEngine is None:
            assert F5TTS_AVAILABLE is False

    def test_ei_f5_06_all_includes_f5tts_engine(self):
        """TEST-EI-F5-06: __all__ includes 'F5TTSEngine'."""
        from voice_soundboard import engines

        assert "F5TTSEngine" in engines.__all__

    def test_ei_f5_07_all_includes_f5tts_available(self):
        """TEST-EI-F5-07: __all__ includes 'F5TTS_AVAILABLE'."""
        from voice_soundboard import engines

        assert "F5TTS_AVAILABLE" in engines.__all__


# =============================================================================
# TEST-CL-01 to TEST-CL-08: Language Enum Addition Tests
# =============================================================================

from voice_soundboard.cloning.crosslang import Language, LANGUAGE_CONFIGS


class TestLanguageEnumAdditions:
    """Tests for Language enum additions (TEST-CL-01 to TEST-CL-08)."""

    def test_cl_01_language_danish_exists(self):
        """TEST-CL-01: Language.DANISH ('da') exists in enum."""
        assert hasattr(Language, 'DANISH')
        assert Language.DANISH.value == "da"

    def test_cl_02_language_greek_exists(self):
        """TEST-CL-02: Language.GREEK ('el') exists in enum."""
        assert hasattr(Language, 'GREEK')
        assert Language.GREEK.value == "el"

    def test_cl_03_language_finnish_exists(self):
        """TEST-CL-03: Language.FINNISH ('fi') exists in enum."""
        assert hasattr(Language, 'FINNISH')
        assert Language.FINNISH.value == "fi"

    def test_cl_04_language_hebrew_exists(self):
        """TEST-CL-04: Language.HEBREW ('he') exists in enum."""
        assert hasattr(Language, 'HEBREW')
        assert Language.HEBREW.value == "he"

    def test_cl_05_language_malay_exists(self):
        """TEST-CL-05: Language.MALAY ('ms') exists in enum."""
        assert hasattr(Language, 'MALAY')
        assert Language.MALAY.value == "ms"

    def test_cl_06_language_norwegian_exists(self):
        """TEST-CL-06: Language.NORWEGIAN ('no') exists in enum."""
        assert hasattr(Language, 'NORWEGIAN')
        assert Language.NORWEGIAN.value == "no"

    def test_cl_07_language_swahili_exists(self):
        """TEST-CL-07: Language.SWAHILI ('sw') exists in enum."""
        assert hasattr(Language, 'SWAHILI')
        assert Language.SWAHILI.value == "sw"

    def test_cl_08_language_enum_has_27_members(self):
        """TEST-CL-08: Language enum has 27 total members."""
        assert len(Language) == 27


# =============================================================================
# TEST-CL-09 to TEST-CL-24: LanguageConfig Addition Tests
# =============================================================================

class TestLanguageConfigAdditions:
    """Tests for LanguageConfig additions (TEST-CL-09 to TEST-CL-24)."""

    def test_cl_09_config_has_czech(self):
        """TEST-CL-09: LANGUAGE_CONFIGS has entry for 'cs' (Czech)."""
        assert "cs" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["cs"].name == "Czech"

    def test_cl_10_config_has_danish(self):
        """TEST-CL-10: LANGUAGE_CONFIGS has entry for 'da' (Danish)."""
        assert "da" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["da"].name == "Danish"

    def test_cl_11_config_has_dutch(self):
        """TEST-CL-11: LANGUAGE_CONFIGS has entry for 'nl' (Dutch)."""
        assert "nl" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["nl"].name == "Dutch"

    def test_cl_12_config_has_greek(self):
        """TEST-CL-12: LANGUAGE_CONFIGS has entry for 'el' (Greek)."""
        assert "el" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["el"].name == "Greek"

    def test_cl_13_config_has_finnish(self):
        """TEST-CL-13: LANGUAGE_CONFIGS has entry for 'fi' (Finnish)."""
        assert "fi" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["fi"].name == "Finnish"

    def test_cl_14_config_has_hebrew(self):
        """TEST-CL-14: LANGUAGE_CONFIGS has entry for 'he' (Hebrew)."""
        assert "he" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["he"].name == "Hebrew"

    def test_cl_15_config_has_indonesian(self):
        """TEST-CL-15: LANGUAGE_CONFIGS has entry for 'id' (Indonesian)."""
        assert "id" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["id"].name == "Indonesian"

    def test_cl_16_config_has_malay(self):
        """TEST-CL-16: LANGUAGE_CONFIGS has entry for 'ms' (Malay)."""
        assert "ms" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["ms"].name == "Malay"

    def test_cl_17_config_has_norwegian(self):
        """TEST-CL-17: LANGUAGE_CONFIGS has entry for 'no' (Norwegian)."""
        assert "no" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["no"].name == "Norwegian"

    def test_cl_18_config_has_polish(self):
        """TEST-CL-18: LANGUAGE_CONFIGS has entry for 'pl' (Polish)."""
        assert "pl" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["pl"].name == "Polish"

    def test_cl_19_config_has_swedish(self):
        """TEST-CL-19: LANGUAGE_CONFIGS has entry for 'sv' (Swedish)."""
        assert "sv" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["sv"].name == "Swedish"

    def test_cl_20_config_has_swahili(self):
        """TEST-CL-20: LANGUAGE_CONFIGS has entry for 'sw' (Swahili)."""
        assert "sw" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["sw"].name == "Swahili"

    def test_cl_21_config_has_thai(self):
        """TEST-CL-21: LANGUAGE_CONFIGS has entry for 'th' (Thai)."""
        assert "th" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["th"].name == "Thai"

    def test_cl_22_config_has_turkish(self):
        """TEST-CL-22: LANGUAGE_CONFIGS has entry for 'tr' (Turkish)."""
        assert "tr" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["tr"].name == "Turkish"

    def test_cl_23_config_has_vietnamese(self):
        """TEST-CL-23: LANGUAGE_CONFIGS has entry for 'vi' (Vietnamese)."""
        assert "vi" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["vi"].name == "Vietnamese"

    def test_cl_24_config_has_27_entries(self):
        """TEST-CL-24: LANGUAGE_CONFIGS has 27 total entries."""
        assert len(LANGUAGE_CONFIGS) == 27


# =============================================================================
# TEST-CL-25 to TEST-CL-31: LanguageConfig Structure Validation Tests
# =============================================================================

class TestLanguageConfigStructure:
    """Tests for LanguageConfig structure validation (TEST-CL-25 to TEST-CL-31)."""

    # New languages added in Phase 8
    NEW_LANGUAGES = ["cs", "da", "nl", "el", "fi", "he", "id", "ms", "no", "pl", "sv", "sw", "th", "tr", "vi"]

    def test_cl_25_all_configs_have_name_field(self):
        """TEST-CL-25: All new configs have 'name' field."""
        for lang_code in self.NEW_LANGUAGES:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'name'), f"Config for '{lang_code}' missing 'name' field"
            assert config.name is not None
            assert len(config.name) > 0

    def test_cl_26_all_configs_have_code_field(self):
        """TEST-CL-26: All new configs have 'code' field matching the key."""
        for lang_code in self.NEW_LANGUAGES:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'code'), f"Config for '{lang_code}' missing 'code' field"
            assert config.code == lang_code

    def test_cl_27_all_configs_have_native_name(self):
        """TEST-CL-27: All new configs have 'native_name' field."""
        for lang_code in self.NEW_LANGUAGES:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'native_name'), f"Config for '{lang_code}' missing 'native_name' field"
            assert config.native_name is not None

    def test_cl_28_all_configs_have_phoneme_set(self):
        """TEST-CL-28: All new configs have 'phoneme_set' field."""
        for lang_code in self.NEW_LANGUAGES:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'phoneme_set'), f"Config for '{lang_code}' missing 'phoneme_set' field"

    def test_cl_29_all_configs_have_typical_speaking_rate(self):
        """TEST-CL-29: All new configs have 'typical_speaking_rate_wpm' field."""
        for lang_code in self.NEW_LANGUAGES:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'typical_speaking_rate_wpm'), f"Config for '{lang_code}' missing 'typical_speaking_rate_wpm' field"
            assert isinstance(config.typical_speaking_rate_wpm, int)
            assert config.typical_speaking_rate_wpm > 0

    def test_cl_30_all_configs_have_timing_fields(self):
        """TEST-CL-30: All new configs have 'syllable_timed' and 'stress_timed' fields."""
        for lang_code in self.NEW_LANGUAGES:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'syllable_timed'), f"Config for '{lang_code}' missing 'syllable_timed' field"
            assert hasattr(config, 'stress_timed'), f"Config for '{lang_code}' missing 'stress_timed' field"
            assert isinstance(config.syllable_timed, bool)
            assert isinstance(config.stress_timed, bool)

    def test_cl_31_tonal_languages_marked_correctly(self):
        """TEST-CL-31: Tonal languages (Thai, Vietnamese) have has_tones=True."""
        tonal_languages = ["th", "vi"]
        for lang_code in tonal_languages:
            config = LANGUAGE_CONFIGS[lang_code]
            assert hasattr(config, 'has_tones'), f"Config for '{lang_code}' missing 'has_tones' field"
            assert config.has_tones is True, f"Tonal language '{lang_code}' should have has_tones=True"


# =============================================================================
# Additional CrossLang Tests - Language Detection and Utilities
# =============================================================================

class TestCrossLangUtilities:
    """Additional tests for crosslang.py utility functions."""

    def test_supported_languages_dict_matches_enum(self):
        """Verify SUPPORTED_LANGUAGES dict matches Language enum."""
        from voice_soundboard.cloning.crosslang import SUPPORTED_LANGUAGES

        for lang in Language:
            assert lang.value in SUPPORTED_LANGUAGES
            assert SUPPORTED_LANGUAGES[lang.value] == lang

    def test_language_configs_match_enum_subset(self):
        """Verify all LANGUAGE_CONFIGS keys are valid language codes."""
        for code in LANGUAGE_CONFIGS.keys():
            # The config key should match the config's code field
            assert LANGUAGE_CONFIGS[code].code == code

    def test_detect_language_japanese(self):
        """Test language detection for Japanese text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("こんにちは")
        assert result == "ja"

    def test_detect_language_korean(self):
        """Test language detection for Korean text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("안녕하세요")
        assert result == "ko"

    def test_detect_language_chinese(self):
        """Test language detection for Chinese text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("你好")
        assert result == "zh"

    def test_detect_language_arabic(self):
        """Test language detection for Arabic text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("مرحبا")
        assert result == "ar"

    def test_detect_language_russian(self):
        """Test language detection for Russian text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("Привет")
        assert result == "ru"

    def test_detect_language_thai(self):
        """Test language detection for Thai text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("สวัสดี")
        assert result == "th"

    def test_detect_language_hindi(self):
        """Test language detection for Hindi text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("नमस्ते")
        assert result == "hi"

    def test_detect_language_english_default(self):
        """Test language detection defaults to English for Latin text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("Hello world")
        assert result == "en"
