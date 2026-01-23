"""
Tests for Chatterbox TTS Engine integration.

Tests cover:
- Paralinguistic tag parsing
- Emotion exaggeration control
- Voice cloning registration
- Engine capabilities
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from voice_soundboard.engines.base import TTSEngine, EngineResult, EngineCapabilities
from voice_soundboard.engines.chatterbox import (
    ChatterboxEngine,
    PARALINGUISTIC_TAGS,
    validate_paralinguistic_tags,
    has_paralinguistic_tags,
)


class TestParalinguisticTags:
    """Tests for paralinguistic tag parsing."""

    def test_all_tags_defined(self):
        """Verify all expected paralinguistic tags are defined."""
        expected_tags = [
            "laugh", "chuckle", "cough", "sigh", "gasp",
            "groan", "sniff", "shush", "clear throat"
        ]
        for tag in expected_tags:
            assert tag in PARALINGUISTIC_TAGS, f"Missing tag: {tag}"

    def test_validate_tags_single(self):
        """Test extracting a single tag."""
        text = "Hello [laugh] world"
        tags = validate_paralinguistic_tags(text)
        assert tags == ["laugh"]

    def test_validate_tags_multiple(self):
        """Test extracting multiple tags."""
        text = "That's hilarious! [laugh] Oh man, [sigh] I needed that."
        tags = validate_paralinguistic_tags(text)
        assert set(tags) == {"laugh", "sigh"}

    def test_validate_tags_case_insensitive(self):
        """Test that tag matching is case-insensitive."""
        text = "Hello [LAUGH] world [Sigh]"
        tags = validate_paralinguistic_tags(text)
        assert set(tags) == {"laugh", "sigh"}

    def test_validate_tags_none(self):
        """Test text with no tags."""
        text = "Hello world, no tags here!"
        tags = validate_paralinguistic_tags(text)
        assert tags == []

    def test_validate_tags_invalid_ignored(self):
        """Test that invalid tags are ignored."""
        text = "Hello [laugh] [invalid_tag] [sigh] world"
        tags = validate_paralinguistic_tags(text)
        assert set(tags) == {"laugh", "sigh"}
        assert "invalid_tag" not in tags

    def test_has_paralinguistic_tags_true(self):
        """Test detection of tags in text."""
        assert has_paralinguistic_tags("Hello [laugh] world")
        assert has_paralinguistic_tags("[cough] excuse me")
        assert has_paralinguistic_tags("Let me think... [sigh]")

    def test_has_paralinguistic_tags_false(self):
        """Test text without tags."""
        assert not has_paralinguistic_tags("Hello world")
        assert not has_paralinguistic_tags("No tags here [invalid]")
        assert not has_paralinguistic_tags("")

    def test_clear_throat_tag(self):
        """Test multi-word tag 'clear throat'."""
        text = "Ahem [clear throat] as I was saying..."
        tags = validate_paralinguistic_tags(text)
        assert "clear throat" in tags


class TestChatterboxEngineCapabilities:
    """Tests for engine capability reporting."""

    def test_engine_name(self):
        """Test engine name property."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.model_variant = "turbo"
        assert engine.name == "chatterbox-turbo"

        engine.model_variant = "standard"
        assert engine.name == "chatterbox-standard"

        engine.model_variant = "multilingual"
        assert engine.name == "chatterbox-multilingual"

    def test_capabilities_structure(self):
        """Test that capabilities return correct structure."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._is_multilingual = True  # Required for capabilities property
        caps = engine.capabilities

        assert isinstance(caps, EngineCapabilities)
        assert caps.supports_paralinguistic_tags is True
        assert caps.supports_emotion_exaggeration is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_streaming is True

    def test_capabilities_tags_list(self):
        """Test that capabilities include paralinguistic tags list."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._is_multilingual = True  # Required for capabilities property
        caps = engine.capabilities

        assert len(caps.paralinguistic_tags) > 0
        assert "laugh" in caps.paralinguistic_tags
        assert "sigh" in caps.paralinguistic_tags


class TestChatterboxEngineVoiceCloning:
    """Tests for voice cloning functionality."""

    def test_clone_voice_registers_voice(self, tmp_path):
        """Test that clone_voice registers the voice."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        # Create a dummy audio file
        audio_file = tmp_path / "reference.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header

        voice_id = engine.clone_voice(audio_file, "my_voice")

        assert voice_id == "my_voice"
        assert "my_voice" in engine._cloned_voices
        assert engine._cloned_voices["my_voice"] == audio_file

    def test_clone_voice_file_not_found(self):
        """Test clone_voice raises error for missing file."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        with pytest.raises(FileNotFoundError):
            engine.clone_voice(Path("/nonexistent/audio.wav"), "test")

    def test_list_cloned_voices(self, tmp_path):
        """Test listing cloned voices."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        # Register some voices
        audio1 = tmp_path / "voice1.wav"
        audio2 = tmp_path / "voice2.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 100)
        audio2.write_bytes(b"RIFF" + b"\x00" * 100)

        engine.clone_voice(audio1, "alice")
        engine.clone_voice(audio2, "bob")

        voices = engine.list_cloned_voices()

        assert len(voices) == 2
        assert "alice" in voices
        assert "bob" in voices

    def test_remove_cloned_voice(self, tmp_path):
        """Test removing a cloned voice."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        audio = tmp_path / "voice.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        engine.clone_voice(audio, "temp_voice")
        assert "temp_voice" in engine._cloned_voices

        result = engine.remove_cloned_voice("temp_voice")
        assert result is True
        assert "temp_voice" not in engine._cloned_voices

    def test_remove_nonexistent_voice(self):
        """Test removing a voice that doesn't exist."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        result = engine.remove_cloned_voice("nonexistent")
        assert result is False


class TestChatterboxEngineFormatWithTags:
    """Tests for the format_with_tags utility."""

    def test_format_with_single_tag(self):
        """Test inserting a single tag."""
        result = ChatterboxEngine.format_with_tags(
            "Hello how are you",
            {"laugh": [1]}
        )
        assert "[laugh]" in result
        assert result == "Hello how [laugh] are you"

    def test_format_with_multiple_tags(self):
        """Test inserting multiple tags."""
        result = ChatterboxEngine.format_with_tags(
            "Hello world this is great",
            {"laugh": [1], "sigh": [3]}
        )
        assert "[laugh]" in result
        assert "[sigh]" in result

    def test_format_with_no_tags(self):
        """Test with empty tags dict."""
        result = ChatterboxEngine.format_with_tags("Hello world", {})
        assert result == "Hello world"


class TestChatterboxEngineInit:
    """Tests for engine initialization."""

    def test_init_defaults(self):
        """Test default initialization values."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.__init__()

        assert engine.model_variant == "multilingual"  # Changed from "turbo" in Phase 8
        assert engine.device == "cuda"
        assert engine.default_exaggeration == 0.5
        assert engine.default_cfg_weight == 0.5
        assert engine._model is None
        assert engine._model_loaded is False

    def test_init_custom_variant(self):
        """Test initialization with custom model variant."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.__init__(model_variant="standard", device="cpu")

        assert engine.model_variant == "standard"
        assert engine.device == "cpu"

    def test_is_loaded_false_initially(self):
        """Test that model is not loaded initially."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._model_loaded = False

        assert engine.is_loaded() is False

    def test_unload_clears_model(self):
        """Test that unload clears the model."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._model = Mock()
        engine._model_loaded = True

        engine.unload()

        assert engine._model is None
        assert engine._model_loaded is False


class TestChatterboxEngineSpeakRaw:
    """Tests for speak_raw with mocked model."""

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_raw_returns_tuple(self, mock_load):
        """Test that speak_raw returns (samples, sample_rate) tuple."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.config = Mock()
        engine.config.default_voice = None
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._cloned_voices = {}
        engine.default_exaggeration = 0.5
        engine.default_cfg_weight = 0.5
        engine.default_language = "en"  # Required for multilingual support
        engine._is_multilingual = True  # Required for multilingual support

        samples, sr = engine.speak_raw("Hello world")

        assert isinstance(samples, np.ndarray)
        assert sr == 24000

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_raw_with_exaggeration(self, mock_load):
        """Test speak_raw passes exaggeration parameter."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.config = Mock()
        engine.config.default_voice = None
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._cloned_voices = {}
        engine.default_exaggeration = 0.5
        engine.default_cfg_weight = 0.5
        engine.default_language = "en"  # Required for multilingual support
        engine._is_multilingual = True  # Required for multilingual support

        engine.speak_raw("Hello", emotion_exaggeration=0.9, cfg_weight=0.3)

        # Verify generate was called with correct params
        engine._model.generate.assert_called_once()
        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["exaggeration"] == 0.9
        assert call_kwargs["cfg_weight"] == 0.3


class TestEngineResultMetadata:
    """Tests for EngineResult metadata handling."""

    def test_result_includes_paralinguistic_tags(self):
        """Test that result metadata includes detected tags."""
        result = EngineResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=1.0,
            engine_name="chatterbox-turbo",
            metadata={
                "paralinguistic_tags": ["laugh", "sigh"],
                "emotion_exaggeration": 0.7,
            }
        )

        assert "paralinguistic_tags" in result.metadata
        assert result.metadata["paralinguistic_tags"] == ["laugh", "sigh"]
        assert result.metadata["emotion_exaggeration"] == 0.7


class TestListParalinguisticTags:
    """Tests for the static list_paralinguistic_tags method."""

    def test_returns_copy(self):
        """Test that list returns a copy, not the original."""
        tags1 = ChatterboxEngine.list_paralinguistic_tags()
        tags2 = ChatterboxEngine.list_paralinguistic_tags()

        # Should be equal but not the same object
        assert tags1 == tags2
        assert tags1 is not tags2

        # Modifying one shouldn't affect the other
        tags1.append("test")
        assert "test" not in tags2

    def test_contains_expected_tags(self):
        """Test that all expected tags are included."""
        tags = ChatterboxEngine.list_paralinguistic_tags()

        expected = ["laugh", "chuckle", "cough", "sigh", "gasp"]
        for tag in expected:
            assert tag in tags


# =============================================================================
# TEST-CBM-01 to TEST-CBM-24: CHATTERBOX_LANGUAGES Constant Tests
# =============================================================================

from voice_soundboard.engines.chatterbox import CHATTERBOX_LANGUAGES, speak_chatterbox


class TestChatterboxLanguagesConstant:
    """Tests for CHATTERBOX_LANGUAGES constant (TEST-CBM-01 to TEST-CBM-24)."""

    def test_cbm_01_languages_has_23_entries(self):
        """TEST-CBM-01: CHATTERBOX_LANGUAGES is a list with 23 languages."""
        assert isinstance(CHATTERBOX_LANGUAGES, list)
        assert len(CHATTERBOX_LANGUAGES) == 23

    def test_cbm_02_includes_arabic(self):
        """TEST-CBM-02: CHATTERBOX_LANGUAGES includes 'ar' (Arabic)."""
        assert "ar" in CHATTERBOX_LANGUAGES

    def test_cbm_03_includes_danish(self):
        """TEST-CBM-03: CHATTERBOX_LANGUAGES includes 'da' (Danish)."""
        assert "da" in CHATTERBOX_LANGUAGES

    def test_cbm_04_includes_german(self):
        """TEST-CBM-04: CHATTERBOX_LANGUAGES includes 'de' (German)."""
        assert "de" in CHATTERBOX_LANGUAGES

    def test_cbm_05_includes_greek(self):
        """TEST-CBM-05: CHATTERBOX_LANGUAGES includes 'el' (Greek)."""
        assert "el" in CHATTERBOX_LANGUAGES

    def test_cbm_06_includes_english(self):
        """TEST-CBM-06: CHATTERBOX_LANGUAGES includes 'en' (English)."""
        assert "en" in CHATTERBOX_LANGUAGES

    def test_cbm_07_includes_spanish(self):
        """TEST-CBM-07: CHATTERBOX_LANGUAGES includes 'es' (Spanish)."""
        assert "es" in CHATTERBOX_LANGUAGES

    def test_cbm_08_includes_finnish(self):
        """TEST-CBM-08: CHATTERBOX_LANGUAGES includes 'fi' (Finnish)."""
        assert "fi" in CHATTERBOX_LANGUAGES

    def test_cbm_09_includes_french(self):
        """TEST-CBM-09: CHATTERBOX_LANGUAGES includes 'fr' (French)."""
        assert "fr" in CHATTERBOX_LANGUAGES

    def test_cbm_10_includes_hebrew(self):
        """TEST-CBM-10: CHATTERBOX_LANGUAGES includes 'he' (Hebrew)."""
        assert "he" in CHATTERBOX_LANGUAGES

    def test_cbm_11_includes_hindi(self):
        """TEST-CBM-11: CHATTERBOX_LANGUAGES includes 'hi' (Hindi)."""
        assert "hi" in CHATTERBOX_LANGUAGES

    def test_cbm_12_includes_italian(self):
        """TEST-CBM-12: CHATTERBOX_LANGUAGES includes 'it' (Italian)."""
        assert "it" in CHATTERBOX_LANGUAGES

    def test_cbm_13_includes_japanese(self):
        """TEST-CBM-13: CHATTERBOX_LANGUAGES includes 'ja' (Japanese)."""
        assert "ja" in CHATTERBOX_LANGUAGES

    def test_cbm_14_includes_korean(self):
        """TEST-CBM-14: CHATTERBOX_LANGUAGES includes 'ko' (Korean)."""
        assert "ko" in CHATTERBOX_LANGUAGES

    def test_cbm_15_includes_malay(self):
        """TEST-CBM-15: CHATTERBOX_LANGUAGES includes 'ms' (Malay)."""
        assert "ms" in CHATTERBOX_LANGUAGES

    def test_cbm_16_includes_dutch(self):
        """TEST-CBM-16: CHATTERBOX_LANGUAGES includes 'nl' (Dutch)."""
        assert "nl" in CHATTERBOX_LANGUAGES

    def test_cbm_17_includes_norwegian(self):
        """TEST-CBM-17: CHATTERBOX_LANGUAGES includes 'no' (Norwegian)."""
        assert "no" in CHATTERBOX_LANGUAGES

    def test_cbm_18_includes_polish(self):
        """TEST-CBM-18: CHATTERBOX_LANGUAGES includes 'pl' (Polish)."""
        assert "pl" in CHATTERBOX_LANGUAGES

    def test_cbm_19_includes_portuguese(self):
        """TEST-CBM-19: CHATTERBOX_LANGUAGES includes 'pt' (Portuguese)."""
        assert "pt" in CHATTERBOX_LANGUAGES

    def test_cbm_20_includes_russian(self):
        """TEST-CBM-20: CHATTERBOX_LANGUAGES includes 'ru' (Russian)."""
        assert "ru" in CHATTERBOX_LANGUAGES

    def test_cbm_21_includes_swedish(self):
        """TEST-CBM-21: CHATTERBOX_LANGUAGES includes 'sv' (Swedish)."""
        assert "sv" in CHATTERBOX_LANGUAGES

    def test_cbm_22_includes_swahili(self):
        """TEST-CBM-22: CHATTERBOX_LANGUAGES includes 'sw' (Swahili)."""
        assert "sw" in CHATTERBOX_LANGUAGES

    def test_cbm_23_includes_turkish(self):
        """TEST-CBM-23: CHATTERBOX_LANGUAGES includes 'tr' (Turkish)."""
        assert "tr" in CHATTERBOX_LANGUAGES

    def test_cbm_24_includes_chinese(self):
        """TEST-CBM-24: CHATTERBOX_LANGUAGES includes 'zh' (Chinese)."""
        assert "zh" in CHATTERBOX_LANGUAGES


# =============================================================================
# TEST-CBM-25 to TEST-CBM-28: Chatterbox Multilingual Initialization Tests
# =============================================================================

class TestChatterboxMultilingualInit:
    """Tests for Chatterbox multilingual initialization (TEST-CBM-25 to TEST-CBM-28)."""

    def test_cbm_25_default_variant_is_multilingual(self):
        """TEST-CBM-25: Default model_variant is 'multilingual' (not 'turbo')."""
        engine = ChatterboxEngine()

        assert engine.model_variant == "multilingual"

    def test_cbm_26_is_multilingual_true_for_multilingual_variant(self):
        """TEST-CBM-26: _is_multilingual is True when variant is 'multilingual'."""
        engine = ChatterboxEngine(model_variant="multilingual")

        assert engine._is_multilingual is True

    def test_cbm_27_is_multilingual_false_for_turbo_variant(self):
        """TEST-CBM-27: _is_multilingual is False when variant is 'turbo'."""
        engine = ChatterboxEngine(model_variant="turbo")

        assert engine._is_multilingual is False

    def test_cbm_28_default_language_is_en(self):
        """TEST-CBM-28: default_language is 'en'."""
        engine = ChatterboxEngine()

        assert engine.default_language == "en"


# =============================================================================
# TEST-CBM-29 to TEST-CBM-31: Chatterbox Capabilities Tests
# =============================================================================

class TestChatterboxMultilingualCapabilities:
    """Tests for Chatterbox multilingual capabilities (TEST-CBM-29 to TEST-CBM-31)."""

    def test_cbm_29_multilingual_reports_23_languages(self):
        """TEST-CBM-29: Multilingual model reports 23 languages in capabilities."""
        engine = ChatterboxEngine(model_variant="multilingual")
        caps = engine.capabilities

        assert len(caps.languages) == 23

    def test_cbm_30_turbo_reports_only_english(self):
        """TEST-CBM-30: Turbo model reports only ['en'] in capabilities."""
        engine = ChatterboxEngine(model_variant="turbo")
        caps = engine.capabilities

        assert caps.languages == ["en"]

    def test_cbm_31_capabilities_languages_is_copy(self):
        """TEST-CBM-31: capabilities.languages is a copy (modifying doesn't affect original)."""
        engine = ChatterboxEngine(model_variant="multilingual")
        caps = engine.capabilities

        original_len = len(caps.languages)
        caps.languages.append("test_lang")

        # Get capabilities again - should not include our modification
        caps2 = engine.capabilities
        assert len(caps2.languages) == original_len


# =============================================================================
# TEST-CBM-32 to TEST-CBM-36: list_languages() Tests
# =============================================================================

class TestChatterboxListLanguages:
    """Tests for list_languages() method (TEST-CBM-32 to TEST-CBM-36)."""

    def test_cbm_32_list_languages_multilingual_returns_23(self):
        """TEST-CBM-32: list_languages() returns 23 languages for multilingual."""
        engine = ChatterboxEngine(model_variant="multilingual")

        languages = engine.list_languages()

        assert len(languages) == 23

    def test_cbm_33_list_languages_turbo_returns_en_only(self):
        """TEST-CBM-33: list_languages() returns ['en'] for turbo."""
        engine = ChatterboxEngine(model_variant="turbo")

        languages = engine.list_languages()

        assert languages == ["en"]

    def test_cbm_34_list_languages_returns_copy(self):
        """TEST-CBM-34: list_languages() returns a copy (not original list)."""
        engine = ChatterboxEngine(model_variant="multilingual")

        list1 = engine.list_languages()
        list2 = engine.list_languages()

        # Should be equal but not the same object
        assert list1 == list2
        assert list1 is not list2

        # Modifying one shouldn't affect the other
        list1.append("test")
        assert "test" not in list2

    def test_cbm_35_list_all_languages_static_returns_23(self):
        """TEST-CBM-35: list_all_languages() static method returns 23 languages."""
        languages = ChatterboxEngine.list_all_languages()

        assert len(languages) == 23

    def test_cbm_36_list_all_languages_returns_copy(self):
        """TEST-CBM-36: list_all_languages() returns a copy."""
        list1 = ChatterboxEngine.list_all_languages()
        list2 = ChatterboxEngine.list_all_languages()

        assert list1 == list2
        assert list1 is not list2


# =============================================================================
# TEST-CBM-37 to TEST-CBM-45: speak() with Language Parameter Tests
# =============================================================================

class TestChatterboxSpeakWithLanguage:
    """Tests for speak() with language parameter (TEST-CBM-37 to TEST-CBM-45)."""

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_37_speak_language_en_passes_to_model(self, mock_load, tmp_path):
        """TEST-CBM-37: speak(language='en') passes language_id to multilingual model."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("Hello", language="en", save_path=tmp_path / "out.wav")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "en"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_38_speak_language_fr(self, mock_load, tmp_path):
        """TEST-CBM-38: speak(language='fr') generates French audio."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("Bonjour", language="fr", save_path=tmp_path / "out.wav")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "fr"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_39_speak_language_ja(self, mock_load, tmp_path):
        """TEST-CBM-39: speak(language='ja') generates Japanese audio."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("こんにちは", language="ja", save_path=tmp_path / "out.wav")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "ja"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_40_speak_language_zh(self, mock_load, tmp_path):
        """TEST-CBM-40: speak(language='zh') generates Chinese audio."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("你好", language="zh", save_path=tmp_path / "out.wav")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "zh"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_41_speak_language_de(self, mock_load, tmp_path):
        """TEST-CBM-41: speak(language='de') generates German audio."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("Guten Tag", language="de", save_path=tmp_path / "out.wav")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "de"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_42_speak_without_language_defaults_to_en(self, mock_load, tmp_path):
        """TEST-CBM-42: speak() without language parameter defaults to 'en'."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("Hello world", save_path=tmp_path / "out.wav")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "en"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_43_speak_invalid_language_falls_back_to_en(self, mock_load, tmp_path, capsys):
        """TEST-CBM-43: speak(language='invalid') falls back to 'en' with warning."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("Hello", language="invalid_lang", save_path=tmp_path / "out.wav")

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "en"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_44_speak_metadata_includes_language(self, mock_load, tmp_path):
        """TEST-CBM-44: speak() metadata includes language field."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        result = engine.speak("Bonjour", language="fr", save_path=tmp_path / "out.wav")

        assert "language" in result.metadata
        assert result.metadata["language"] == "fr"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_45_turbo_ignores_language_parameter(self, mock_load, tmp_path):
        """TEST-CBM-45: Turbo model ignores language parameter (English only)."""
        engine = ChatterboxEngine(model_variant="turbo")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak("Hello", language="fr", save_path=tmp_path / "out.wav")

        # Turbo model should NOT have language_id in call
        call_kwargs = engine._model.generate.call_args[1]
        assert "language_id" not in call_kwargs


# =============================================================================
# TEST-CBM-46 to TEST-CBM-47: speak_raw() with Language Tests
# =============================================================================

class TestChatterboxSpeakRawWithLanguage:
    """Tests for speak_raw() with language parameter (TEST-CBM-46 to TEST-CBM-47)."""

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_46_speak_raw_language_fr(self, mock_load):
        """TEST-CBM-46: speak_raw(language='fr') passes language_id to model."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak_raw("Bonjour", language="fr")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "fr"

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_cbm_47_speak_raw_without_language_uses_default(self, mock_load):
        """TEST-CBM-47: speak_raw() without language uses default_language."""
        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        engine.speak_raw("Hello")

        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["language_id"] == "en"


# =============================================================================
# TEST-CBM-48 to TEST-CBM-50: Model Loading Tests
# =============================================================================

class TestChatterboxMultilingualModelLoading:
    """Tests for Chatterbox multilingual model loading (TEST-CBM-48 to TEST-CBM-50)."""

    def test_cbm_48_multilingual_imports_from_mtl_tts(self):
        """TEST-CBM-48: Multilingual model imports from chatterbox.mtl_tts."""
        engine = ChatterboxEngine(model_variant="multilingual")

        # Verify the import path is referenced in the code
        import inspect
        source = inspect.getsource(engine._ensure_model_loaded)
        assert "chatterbox.mtl_tts" in source or "mtl_tts" in source

    def test_cbm_49_turbo_imports_from_tts_turbo(self):
        """TEST-CBM-49: Turbo model imports from chatterbox.tts_turbo."""
        engine = ChatterboxEngine(model_variant="turbo")

        import inspect
        source = inspect.getsource(engine._ensure_model_loaded)
        assert "chatterbox.tts_turbo" in source or "tts_turbo" in source

    def test_cbm_50_standard_imports_from_tts(self):
        """TEST-CBM-50: Standard model imports from chatterbox.tts."""
        engine = ChatterboxEngine(model_variant="standard")

        import inspect
        source = inspect.getsource(engine._ensure_model_loaded)
        assert "chatterbox.tts" in source


# =============================================================================
# TEST-CBM-51 to TEST-CBM-53: speak_chatterbox() Convenience Function Tests
# =============================================================================

class TestSpeakChatterboxConvenience:
    """Tests for speak_chatterbox() convenience function (TEST-CBM-51 to TEST-CBM-53)."""

    def test_cbm_51_speak_chatterbox_accepts_language(self):
        """TEST-CBM-51: speak_chatterbox() accepts language parameter."""
        import inspect
        sig = inspect.signature(speak_chatterbox)

        assert "language" in sig.parameters

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine')
    def test_cbm_52_speak_chatterbox_creates_multilingual_engine(self, mock_engine_class):
        """TEST-CBM-52: speak_chatterbox() creates multilingual engine by default."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_engine.speak.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        speak_chatterbox("Hello")

        mock_engine_class.assert_called_once_with(model_variant="multilingual")

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine')
    def test_cbm_53_speak_chatterbox_language_fr(self, mock_engine_class):
        """TEST-CBM-53: speak_chatterbox(language='fr') generates French audio."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_engine.speak.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        speak_chatterbox("Bonjour", language="fr")

        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs["language"] == "fr"


# =============================================================================
# Integration Tests (Requires chatterbox-tts installed)
# =============================================================================

@pytest.mark.skip(reason="Integration test - requires chatterbox-tts installed")
class TestChatterboxIntegration:
    """Integration tests that require actual chatterbox installation."""

    def test_full_generation(self, tmp_path):
        """Test full speech generation with Chatterbox."""
        engine = ChatterboxEngine(device="cuda")

        result = engine.speak(
            "Hello! [laugh] That's funny.",
            emotion_exaggeration=0.7,
            save_path=tmp_path / "output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert "laugh" in result.metadata.get("paralinguistic_tags", [])


# =============================================================================
# TEST-CBM-INT-01 to TEST-CBM-INT-08: Multilingual Integration Tests
# =============================================================================

@pytest.mark.skip(reason="Integration test - requires chatterbox-tts installed")
class TestChatterboxMultilingualIntegration:
    """Integration tests for Chatterbox multilingual (TEST-CBM-INT-01 to TEST-CBM-INT-08)."""

    def test_cbm_int_01_generate_french_speech(self, tmp_path):
        """TEST-CBM-INT-01: Generate French speech with correct pronunciation."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        result = engine.speak(
            "Bonjour, comment allez-vous aujourd'hui?",
            language="fr",
            save_path=tmp_path / "french_output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.metadata.get("language") == "fr"

    def test_cbm_int_02_generate_japanese_speech(self, tmp_path):
        """TEST-CBM-INT-02: Generate Japanese speech with correct characters."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        result = engine.speak(
            "こんにちは、お元気ですか？",
            language="ja",
            save_path=tmp_path / "japanese_output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.metadata.get("language") == "ja"

    def test_cbm_int_03_generate_german_speech_with_umlauts(self, tmp_path):
        """TEST-CBM-INT-03: Generate German speech with umlauts."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        result = engine.speak(
            "Guten Tag! Wie geht es Ihnen? Möchten Sie Käse oder Brötchen?",
            language="de",
            save_path=tmp_path / "german_output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.metadata.get("language") == "de"

    def test_cbm_int_04_generate_chinese_speech_with_tones(self, tmp_path):
        """TEST-CBM-INT-04: Generate Chinese speech with tones."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        result = engine.speak(
            "你好，今天天气怎么样？",
            language="zh",
            save_path=tmp_path / "chinese_output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.metadata.get("language") == "zh"

    def test_cbm_int_05_generate_arabic_speech_rtl(self, tmp_path):
        """TEST-CBM-INT-05: Generate Arabic speech with RTL text."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        result = engine.speak(
            "مرحبا، كيف حالك اليوم؟",
            language="ar",
            save_path=tmp_path / "arabic_output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.metadata.get("language") == "ar"

    def test_cbm_int_06_paralinguistic_tags_in_non_english(self, tmp_path):
        """TEST-CBM-INT-06: Paralinguistic tags work in non-English languages."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        result = engine.speak(
            "C'est très drôle! [laugh] J'adore ça!",
            language="fr",
            emotion_exaggeration=0.7,
            save_path=tmp_path / "french_with_laugh.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.metadata.get("language") == "fr"
        # Paralinguistic tags should be detected
        assert "laugh" in result.metadata.get("paralinguistic_tags", [])

    def test_cbm_int_07_voice_cloning_across_languages(self, tmp_path):
        """TEST-CBM-INT-07: Voice cloning works across languages."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        # Clone a voice
        ref_audio = tmp_path / "reference_voice.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1000)

        engine.clone_voice(ref_audio, voice_id="multilang_voice")

        # Generate in multiple languages with the same cloned voice
        result_en = engine.speak(
            "Hello, this is English.",
            voice="multilang_voice",
            language="en",
            save_path=tmp_path / "cloned_english.wav"
        )

        result_fr = engine.speak(
            "Bonjour, c'est français.",
            voice="multilang_voice",
            language="fr",
            save_path=tmp_path / "cloned_french.wav"
        )

        result_ja = engine.speak(
            "こんにちは、日本語です。",
            voice="multilang_voice",
            language="ja",
            save_path=tmp_path / "cloned_japanese.wav"
        )

        assert result_en.audio_path.exists()
        assert result_fr.audio_path.exists()
        assert result_ja.audio_path.exists()

        assert result_en.metadata.get("language") == "en"
        assert result_fr.metadata.get("language") == "fr"
        assert result_ja.metadata.get("language") == "ja"

    def test_cbm_int_08_emotion_exaggeration_in_all_languages(self, tmp_path):
        """TEST-CBM-INT-08: Emotion exaggeration works in all languages."""
        engine = ChatterboxEngine(model_variant="multilingual", device="cuda")

        languages_to_test = ["en", "fr", "de", "ja", "zh", "es", "ko"]
        test_texts = {
            "en": "This is so exciting!",
            "fr": "C'est tellement excitant!",
            "de": "Das ist so aufregend!",
            "ja": "これはとても興奮します！",
            "zh": "这太令人兴奋了！",
            "es": "¡Esto es muy emocionante!",
            "ko": "이것은 정말 흥미진진해요!",
        }

        for lang in languages_to_test:
            result = engine.speak(
                test_texts[lang],
                language=lang,
                emotion_exaggeration=0.9,  # High exaggeration
                save_path=tmp_path / f"emotion_{lang}.wav"
            )

            assert result.audio_path.exists(), f"Failed for language: {lang}"
            assert result.duration_seconds > 0, f"No duration for language: {lang}"
            assert result.metadata.get("language") == lang
            assert result.metadata.get("emotion_exaggeration") == 0.9
