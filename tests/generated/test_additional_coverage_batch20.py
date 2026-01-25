"""
Additional coverage tests - Batch 20.

Tests for:
- cloning/crosslang.py (Language, LanguageConfig, CrossLanguageCloner, detect_language)
- cloning/cloner.py (CloningConfig, CloningResult, VoiceCloner)
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil


# =============================================================================
# crosslang.py tests
# =============================================================================


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self):
        from voice_soundboard.cloning.crosslang import Language

        assert Language.ENGLISH.value == "en"
        assert Language.CHINESE_MANDARIN.value == "zh"
        assert Language.JAPANESE.value == "ja"
        assert Language.KOREAN.value == "ko"
        assert Language.RUSSIAN.value == "ru"

    def test_all_languages_have_unique_values(self):
        from voice_soundboard.cloning.crosslang import Language

        values = [lang.value for lang in Language]
        assert len(values) == len(set(values))

    def test_supported_languages_mapping(self):
        from voice_soundboard.cloning.crosslang import SUPPORTED_LANGUAGES, Language

        assert SUPPORTED_LANGUAGES["en"] == Language.ENGLISH
        assert SUPPORTED_LANGUAGES["zh"] == Language.CHINESE_MANDARIN
        assert SUPPORTED_LANGUAGES["ja"] == Language.JAPANESE


class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_default_values(self):
        from voice_soundboard.cloning.crosslang import LanguageConfig

        config = LanguageConfig(code="test", name="Test", native_name="Test")
        assert config.phoneme_set == "ipa"
        assert config.has_tones is False
        assert config.syllable_timed is False
        assert config.stress_timed is True
        assert config.default_speed == 1.0
        assert config.typical_speaking_rate_wpm == 150
        assert config.requires_romanization is False

    def test_tonal_language_config(self):
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        zh_config = LANGUAGE_CONFIGS["zh"]
        assert zh_config.has_tones is True
        assert zh_config.syllable_timed is True
        assert zh_config.requires_romanization is True
        assert zh_config.romanization_system == "pinyin"

    def test_japanese_config(self):
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        ja_config = LANGUAGE_CONFIGS["ja"]
        assert ja_config.has_tones is True  # Pitch accent
        assert ja_config.requires_romanization is True
        assert ja_config.romanization_system == "romaji"

    def test_stress_timed_languages(self):
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        assert LANGUAGE_CONFIGS["en"].stress_timed is True
        assert LANGUAGE_CONFIGS["de"].stress_timed is True
        assert LANGUAGE_CONFIGS["ru"].stress_timed is True

    def test_syllable_timed_languages(self):
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        assert LANGUAGE_CONFIGS["es"].syllable_timed is True
        assert LANGUAGE_CONFIGS["fr"].syllable_timed is True
        assert LANGUAGE_CONFIGS["it"].syllable_timed is True


class TestCrossLanguageResult:
    """Tests for CrossLanguageResult dataclass."""

    def test_default_values(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageResult

        result = CrossLanguageResult(
            success=True,
            source_language="en",
            target_language="zh",
        )
        assert result.audio is None
        assert result.sample_rate == 24000
        assert result.timbre_preservation_score == 0.0
        assert result.accent_transfer_score == 0.0
        assert result.error is None
        assert result.warnings == []

    def test_with_audio(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageResult

        audio = np.random.randn(24000).astype(np.float32)
        result = CrossLanguageResult(
            success=True,
            source_language="en",
            target_language="fr",
            audio=audio,
            sample_rate=44100,
        )
        assert result.audio is not None
        assert len(result.audio) == 24000
        assert result.sample_rate == 44100

    def test_with_error(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageResult

        result = CrossLanguageResult(
            success=False,
            source_language="en",
            target_language="xx",
            error="Unsupported language",
            warnings=["Language not in database"],
        )
        assert result.success is False
        assert result.error == "Unsupported language"
        assert len(result.warnings) == 1


class TestCrossLanguageCloner:
    """Tests for CrossLanguageCloner class."""

    def test_init_defaults(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        assert cloner.source_language == "en"
        assert cloner.preserve_accent is False

    def test_init_with_custom_settings(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="zh", preserve_accent=True)
        assert cloner.source_language == "zh"
        assert cloner.preserve_accent is True

    def test_source_config_property(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="ja")
        config = cloner.source_config
        assert config.code == "ja"
        assert config.name == "Japanese"

    def test_source_config_fallback_to_english(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="unknown")
        config = cloner.source_config
        assert config.code == "en"

    def test_get_target_config(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        config = cloner.get_target_config("fr")
        assert config.code == "fr"
        assert config.name == "French"

    def test_get_target_config_fallback(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        config = cloner.get_target_config("unknown")
        assert config.code == "en"

    def test_is_language_supported(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        assert cloner.is_language_supported("en") is True
        assert cloner.is_language_supported("zh") is True
        assert cloner.is_language_supported("unknown") is False

    def test_list_supported_languages(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        languages = cloner.list_supported_languages()
        assert len(languages) > 0
        assert all("code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)
        assert all("native_name" in lang for lang in languages)

    def test_language_pair_compatibility_unsupported(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "unknown")
        assert compat["compatible"] is False
        assert "not supported" in compat["reason"]

    def test_language_pair_compatibility_tonal_target(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "zh")
        assert compat["compatible"] is True
        assert len(compat["phonetic_issues"]) > 0
        assert compat["expected_quality"] < 1.0

    def test_language_pair_compatibility_same_family(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        # Romance languages
        compat = cloner.get_language_pair_compatibility("es", "fr")
        assert compat["same_language_family"] is True
        assert compat["expected_quality"] > 0.9

    def test_language_pair_compatibility_different_timing(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        # English (stress-timed) to Spanish (syllable-timed)
        compat = cloner.get_language_pair_compatibility("en", "es")
        assert compat["compatible"] is True
        # Should have timing issue noted
        assert any("timing" in issue.lower() for issue in compat["phonetic_issues"])

    def test_get_recommendations_for_tonal_target(self):
        from voice_soundboard.cloning.crosslang import (
            CrossLanguageCloner,
            LANGUAGE_CONFIGS,
        )

        cloner = CrossLanguageCloner()
        source = LANGUAGE_CONFIGS["en"]
        target = LANGUAGE_CONFIGS["zh"]
        recs = cloner._get_recommendations(source, target)
        assert any("expressive" in rec.lower() for rec in recs)

    def test_get_recommendations_speed_difference(self):
        from voice_soundboard.cloning.crosslang import (
            CrossLanguageCloner,
            LANGUAGE_CONFIGS,
        )

        cloner = CrossLanguageCloner()
        source = LANGUAGE_CONFIGS["ru"]  # 130 WPM
        target = LANGUAGE_CONFIGS["ja"]  # 200 WPM
        recs = cloner._get_recommendations(source, target)
        assert any("faster" in rec.lower() or "speed" in rec.lower() for rec in recs)

    def test_prepare_embedding_for_language(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )

        prepared, metadata = cloner.prepare_embedding_for_language(embedding, "zh")
        assert metadata["source_language"] == "en"
        assert metadata["target_language"] == "zh"
        assert "recommended_speed_multiplier" in metadata
        assert prepared is embedding  # Currently returns unchanged

    def test_estimate_quality(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
            quality_score=0.9,
            source_duration_seconds=5.0,
        )

        quality = cloner.estimate_quality(embedding, "fr")
        assert 0.0 <= quality <= 1.0

    def test_estimate_quality_short_audio(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner()
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
            quality_score=0.9,
            source_duration_seconds=2.0,  # Short
        )

        quality = cloner.estimate_quality(embedding, "en")
        # Should be penalized for short audio
        assert quality < 0.9

    def test_estimate_quality_long_audio(self):
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner()
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
            quality_score=0.8,
            source_duration_seconds=15.0,  # Long
        )

        quality = cloner.estimate_quality(embedding, "en")
        # Should get bonus for long audio
        assert quality >= 0.8


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_japanese(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("こんにちは") == "ja"
        assert detect_language("カタカナ") == "ja"

    def test_detect_korean(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("안녕하세요") == "ko"

    def test_detect_chinese(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("你好") == "zh"

    def test_detect_russian(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("Привет") == "ru"

    def test_detect_arabic(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("مرحبا") == "ar"

    def test_detect_hindi(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("नमस्ते") == "hi"

    def test_detect_thai(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("สวัสดี") == "th"

    def test_detect_english_default(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("Hello World") == "en"

    def test_detect_empty_string(self):
        from voice_soundboard.cloning.crosslang import detect_language

        assert detect_language("") == "en"


# =============================================================================
# cloner.py tests
# =============================================================================


class TestCloningConfig:
    """Tests for CloningConfig dataclass."""

    def test_default_values(self):
        from voice_soundboard.cloning.cloner import CloningConfig
        from voice_soundboard.cloning.extractor import ExtractorBackend

        config = CloningConfig()
        assert config.extractor_backend == ExtractorBackend.MOCK
        assert config.device == "cpu"
        assert config.min_audio_seconds == 1.0
        assert config.max_audio_seconds == 30.0
        assert config.optimal_audio_seconds == 5.0
        assert config.min_quality_score == 0.3
        assert config.min_snr_db == 10.0
        assert config.use_segment_averaging is True
        assert config.require_consent is True
        assert config.add_watermark is False

    def test_custom_values(self):
        from voice_soundboard.cloning.cloner import CloningConfig
        from voice_soundboard.cloning.extractor import ExtractorBackend

        config = CloningConfig(
            extractor_backend=ExtractorBackend.RESEMBLYZER,
            device="cuda",
            min_audio_seconds=2.0,
            require_consent=False,
        )
        assert config.extractor_backend == ExtractorBackend.RESEMBLYZER
        assert config.device == "cuda"
        assert config.min_audio_seconds == 2.0
        assert config.require_consent is False


class TestCloningResult:
    """Tests for CloningResult dataclass."""

    def test_default_values(self):
        from voice_soundboard.cloning.cloner import CloningResult

        result = CloningResult(success=True)
        assert result.voice_id == ""
        assert result.profile is None
        assert result.embedding is None
        assert result.extraction_time == 0.0
        assert result.total_time == 0.0
        assert result.error is None
        assert result.warnings == []
        assert result.recommendations == []

    def test_failure_result(self):
        from voice_soundboard.cloning.cloner import CloningResult

        result = CloningResult(
            success=False,
            voice_id="test",
            error="Something went wrong",
            warnings=["Warning 1"],
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert len(result.warnings) == 1


class TestVoiceCloner:
    """Tests for VoiceCloner class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def mock_library(self, temp_dir):
        """Create a mock library."""
        from voice_soundboard.cloning.library import VoiceLibrary

        return VoiceLibrary(library_path=temp_dir / "voices")

    @pytest.fixture
    def cloner(self, mock_library):
        """Create a VoiceCloner with mock library."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig

        config = CloningConfig(require_consent=False)
        return VoiceCloner(config=config, library=mock_library)

    def test_init_defaults(self):
        from voice_soundboard.cloning.cloner import VoiceCloner

        with patch(
            "voice_soundboard.cloning.cloner.get_default_library"
        ) as mock_get_lib:
            mock_get_lib.return_value = Mock()
            cloner = VoiceCloner()
            assert cloner.config is not None
            assert cloner.library is not None

    def test_init_with_custom_config(self, mock_library):
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig

        config = CloningConfig(device="cuda", min_audio_seconds=2.0)
        cloner = VoiceCloner(config=config, library=mock_library)
        assert cloner.config.device == "cuda"
        assert cloner.config.min_audio_seconds == 2.0

    def test_extractor_property(self, cloner):
        extractor = cloner.extractor
        assert extractor is not None
        # Should return same instance on second call
        assert cloner.extractor is extractor

    def test_clone_requires_consent(self, mock_library):
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig

        config = CloningConfig(require_consent=True)
        cloner = VoiceCloner(config=config, library=mock_library)

        result = cloner.clone(
            audio="/fake/audio.wav",
            voice_id="test",
            consent_given=False,
        )
        assert result.success is False
        assert "consent" in result.error.lower()

    def test_clone_voice_already_exists(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        # Add a voice first
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="existing",
            name="Existing Voice",
            embedding=embedding,
        )

        # Try to clone with same ID
        result = cloner.clone(
            audio="/fake/audio.wav",
            voice_id="existing",
            consent_given=True,
        )
        assert result.success is False
        assert "already exists" in result.error

    def test_clone_file_not_found(self, cloner):
        result = cloner.clone(
            audio="/nonexistent/audio.wav",
            voice_id="test",
            consent_given=True,
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_clone_quick(self, cloner, temp_dir):
        # Create a fake audio file
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        with patch.object(cloner.extractor, "extract") as mock_extract:
            from voice_soundboard.cloning.extractor import VoiceEmbedding

            mock_extract.return_value = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=str(audio_file),
                quality_score=0.9,
                source_duration_seconds=5.0,
            )

            result = cloner.clone_quick(audio=audio_file, name="Test Voice")
            assert result.success is True
            assert result.profile is not None
            assert result.voice_id.startswith("temp_")

    def test_clone_quick_extraction_fails(self, cloner):
        with patch.object(
            cloner.extractor, "extract", side_effect=Exception("Extraction error")
        ):
            result = cloner.clone_quick(audio="/fake/audio.wav", name="Test")
            assert result.success is False
            assert "Extraction failed" in result.error

    def test_update_voice_not_found(self, cloner):
        result = cloner.update_voice(voice_id="nonexistent")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_update_voice_with_new_audio(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        # Add a voice first
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
            quality_score=0.8,
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
        )

        audio_file = temp_dir / "new_audio.wav"
        audio_file.write_bytes(b"fake audio")

        with patch.object(cloner.extractor, "extract") as mock_extract:
            new_embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=str(audio_file),
                quality_score=0.95,
            )
            mock_extract.return_value = new_embedding

            result = cloner.update_voice(voice_id="test", audio=audio_file)
            assert result.success is True

    def test_update_voice_metadata_only(self, cloner, mock_library):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
            quality_score=0.8,
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
        )

        result = cloner.update_voice(voice_id="test", name="Updated Name")
        assert result.success is True
        assert result.profile.name == "Updated Name"

    def test_delete_voice(self, cloner, mock_library):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
        )

        assert cloner.delete_voice("test") is True
        assert cloner.delete_voice("nonexistent") is False

    def test_get_voice(self, cloner, mock_library):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
        )

        profile = cloner.get_voice("test")
        assert profile is not None
        assert profile.name == "Test Voice"

        assert cloner.get_voice("nonexistent") is None

    def test_list_voices(self, cloner, mock_library):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        for i in range(3):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=f"/fake/audio{i}.wav",
            )
            mock_library.add(
                voice_id=f"voice{i}",
                name=f"Voice {i}",
                embedding=embedding,
                tags=["test"],
            )

        voices = cloner.list_voices()
        assert len(voices) == 3

        voices_with_query = cloner.list_voices(query="Voice 1")
        assert len(voices_with_query) == 1

    def test_find_similar(self, cloner, mock_library):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        # Add some voices
        for i in range(3):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=f"/fake/audio{i}.wav",
            )
            mock_library.add(
                voice_id=f"voice{i}",
                name=f"Voice {i}",
                embedding=embedding,
            )

        # Find similar with an embedding
        query_embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/query.wav",
        )

        similar = cloner.find_similar(query_embedding, top_k=2)
        assert len(similar) <= 2

    def test_validate_audio(self, cloner, temp_dir):
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        with patch.object(cloner.extractor, "extract") as mock_extract:
            from voice_soundboard.cloning.extractor import VoiceEmbedding

            mock_extract.return_value = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=str(audio_file),
                quality_score=0.9,
                snr_db=25.0,
                source_duration_seconds=5.0,
            )

            result = cloner.validate_audio(audio_file)
            assert result["is_valid"] is True
            assert result["quality_score"] == 0.9

    def test_validate_audio_too_short(self, cloner, temp_dir):
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        with patch.object(cloner.extractor, "extract") as mock_extract:
            from voice_soundboard.cloning.extractor import VoiceEmbedding

            mock_extract.return_value = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=str(audio_file),
                source_duration_seconds=0.5,  # Too short
            )

            result = cloner.validate_audio(audio_file)
            assert result["is_valid"] is False
            assert any("short" in issue.lower() for issue in result["issues"])

    def test_validate_audio_low_quality(self, cloner, temp_dir):
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        with patch.object(cloner.extractor, "extract") as mock_extract:
            from voice_soundboard.cloning.extractor import VoiceEmbedding

            mock_extract.return_value = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=str(audio_file),
                quality_score=0.1,  # Too low
                source_duration_seconds=5.0,
            )

            result = cloner.validate_audio(audio_file)
            assert result["is_valid"] is False
            assert any("quality" in issue.lower() for issue in result["issues"])

    def test_validate_audio_extraction_fails(self, cloner):
        with patch.object(
            cloner.extractor, "extract", side_effect=Exception("Cannot process")
        ):
            result = cloner.validate_audio("/fake/audio.wav")
            assert result["is_valid"] is False
            assert any("cannot process" in issue.lower() for issue in result["issues"])

    def test_export_voice_json(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
        )

        output_path = temp_dir / "export.json"
        result = cloner.export_voice("test", output_path)
        assert result == output_path
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert data["voice_id"] == "test"
        assert data["name"] == "Test Voice"

    def test_export_voice_npz(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
        )

        output_path = temp_dir / "export.npz"
        result = cloner.export_voice("test", output_path)
        assert result == output_path
        assert output_path.exists()

        data = np.load(output_path, allow_pickle=True)
        assert "embedding" in data
        assert "metadata" in data

    def test_export_voice_without_source_audio(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/original/audio.wav",
        )
        mock_library.add(
            voice_id="test",
            name="Test Voice",
            embedding=embedding,
            source_audio_path="/original/audio.wav",
        )

        output_path = temp_dir / "export.json"
        cloner.export_voice("test", output_path, include_source_audio=False)

        with open(output_path) as f:
            data = json.load(f)
        assert data["source_audio_path"] is None

    def test_import_voice_json(self, cloner, mock_library, temp_dir):
        # Create an export file
        export_data = {
            "voice_id": "imported",
            "name": "Imported Voice",
            "description": "A test import",
            "tags": ["imported"],
            "gender": "neutral",
            "language": "en",
            "consent_given": True,
            "consent_notes": "Test",
            "embedding": {
                "embedding": np.random.randn(256).astype(np.float32).tolist(),
                "embedding_dim": 256,
            },
        }

        import_path = temp_dir / "import.json"
        with open(import_path, "w") as f:
            json.dump(export_data, f)

        profile = cloner.import_voice(import_path)
        assert profile.voice_id == "imported"
        assert profile.name == "Imported Voice"

    def test_import_voice_with_override_id(self, cloner, mock_library, temp_dir):
        export_data = {
            "voice_id": "original_id",
            "name": "Test Voice",
            "embedding": {
                "embedding": np.random.randn(256).astype(np.float32).tolist(),
                "embedding_dim": 256,
            },
        }

        import_path = temp_dir / "import.json"
        with open(import_path, "w") as f:
            json.dump(export_data, f)

        profile = cloner.import_voice(import_path, voice_id="new_id")
        assert profile.voice_id == "new_id"

    def test_import_voice_already_exists(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        # Add existing voice
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="existing",
            name="Existing Voice",
            embedding=embedding,
        )

        export_data = {
            "voice_id": "existing",
            "name": "Import Voice",
            "embedding": {
                "embedding": np.random.randn(256).astype(np.float32).tolist(),
                "embedding_dim": 256,
            },
        }

        import_path = temp_dir / "import.json"
        with open(import_path, "w") as f:
            json.dump(export_data, f)

        with pytest.raises(ValueError, match="already exists"):
            cloner.import_voice(import_path)

    def test_import_voice_overwrite(self, cloner, mock_library, temp_dir):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        # Add existing voice
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )
        mock_library.add(
            voice_id="existing",
            name="Existing Voice",
            embedding=embedding,
        )

        export_data = {
            "voice_id": "existing",
            "name": "New Name",
            "embedding": {
                "embedding": np.random.randn(256).astype(np.float32).tolist(),
                "embedding_dim": 256,
            },
        }

        import_path = temp_dir / "import.json"
        with open(import_path, "w") as f:
            json.dump(export_data, f)

        profile = cloner.import_voice(import_path, overwrite=True)
        assert profile.name == "New Name"

    def test_import_voice_npz(self, cloner, mock_library, temp_dir):
        # Create an npz export file
        embedding_array = np.random.randn(256).astype(np.float32)
        metadata = {
            "voice_id": "npz_imported",
            "name": "NPZ Imported",
            "tags": [],
            "language": "en",
            "consent_given": True,
            "consent_notes": "",
        }

        import_path = temp_dir / "import.npz"
        np.savez_compressed(
            import_path,
            embedding=embedding_array,
            metadata=json.dumps(metadata),
        )

        profile = cloner.import_voice(import_path)
        assert profile.voice_id == "npz_imported"
        assert profile.name == "NPZ Imported"
