"""
Additional test coverage batch 13: cloning/crosslang.py, cloning/cloner.py, cloning/library.py.

Tests for cross-language cloning, voice cloner API, and voice library management.
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np

from voice_soundboard.cloning.crosslang import (
    Language,
    SUPPORTED_LANGUAGES,
    LanguageConfig,
    LANGUAGE_CONFIGS,
    CrossLanguageResult,
    CrossLanguageCloner,
    detect_language,
)
from voice_soundboard.cloning.cloner import (
    CloningConfig,
    CloningResult,
    VoiceCloner,
)
from voice_soundboard.cloning.library import (
    VoiceProfile,
    VoiceLibrary,
    DEFAULT_LIBRARY_PATH,
    get_default_library,
)
from voice_soundboard.cloning.extractor import (
    VoiceEmbedding,
    VoiceExtractor,
    ExtractorBackend,
)


# =============================================================================
# Language Enum Tests
# =============================================================================

class TestLanguageEnum:
    """Tests for Language enum."""

    def test_english_value(self):
        """Test ENGLISH value."""
        assert Language.ENGLISH.value == "en"

    def test_chinese_value(self):
        """Test CHINESE_MANDARIN value."""
        assert Language.CHINESE_MANDARIN.value == "zh"

    def test_japanese_value(self):
        """Test JAPANESE value."""
        assert Language.JAPANESE.value == "ja"

    def test_korean_value(self):
        """Test KOREAN value."""
        assert Language.KOREAN.value == "ko"

    def test_spanish_value(self):
        """Test SPANISH value."""
        assert Language.SPANISH.value == "es"

    def test_all_languages_in_supported(self):
        """Test all enum values are in SUPPORTED_LANGUAGES."""
        for lang in Language:
            assert lang.value in SUPPORTED_LANGUAGES


# =============================================================================
# LanguageConfig Tests
# =============================================================================

class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        config = LanguageConfig(
            code="en",
            name="English",
            native_name="English",
        )
        assert config.code == "en"
        assert config.name == "English"
        assert config.has_tones is False
        assert config.stress_timed is True

    def test_creation_tonal_language(self):
        """Test creating tonal language config."""
        config = LanguageConfig(
            code="zh",
            name="Chinese",
            native_name="中文",
            has_tones=True,
            syllable_timed=True,
            stress_timed=False,
        )
        assert config.has_tones is True
        assert config.syllable_timed is True
        assert config.stress_timed is False

    def test_default_values(self):
        """Test default values."""
        config = LanguageConfig(
            code="test",
            name="Test",
            native_name="Test",
        )
        assert config.phoneme_set == "ipa"
        assert config.default_speed == 1.0
        assert config.typical_speaking_rate_wpm == 150
        assert config.requires_romanization is False


# =============================================================================
# LANGUAGE_CONFIGS Tests
# =============================================================================

class TestLanguageConfigs:
    """Tests for LANGUAGE_CONFIGS dictionary."""

    def test_contains_english(self):
        """Test English config exists."""
        assert "en" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["en"].name == "English"

    def test_contains_chinese(self):
        """Test Chinese config exists."""
        assert "zh" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["zh"].has_tones is True

    def test_contains_japanese(self):
        """Test Japanese config exists."""
        assert "ja" in LANGUAGE_CONFIGS
        assert LANGUAGE_CONFIGS["ja"].requires_romanization is True

    def test_all_configs_valid(self):
        """Test all configs are LanguageConfig instances."""
        for code, config in LANGUAGE_CONFIGS.items():
            assert isinstance(config, LanguageConfig)
            assert config.code == code


# =============================================================================
# CrossLanguageResult Tests
# =============================================================================

class TestCrossLanguageResult:
    """Tests for CrossLanguageResult dataclass."""

    def test_creation_success(self):
        """Test successful result creation."""
        result = CrossLanguageResult(
            success=True,
            source_language="en",
            target_language="es",
        )
        assert result.success is True
        assert result.source_language == "en"
        assert result.target_language == "es"

    def test_creation_with_audio(self):
        """Test result with audio."""
        audio = np.random.randn(24000).astype(np.float32)
        result = CrossLanguageResult(
            success=True,
            source_language="en",
            target_language="fr",
            audio=audio,
            sample_rate=24000,
        )
        assert result.audio is not None
        assert result.sample_rate == 24000

    def test_creation_failure(self):
        """Test failure result."""
        result = CrossLanguageResult(
            success=False,
            source_language="en",
            target_language="xyz",
            error="Unsupported language",
        )
        assert result.success is False
        assert result.error == "Unsupported language"


# =============================================================================
# CrossLanguageCloner Tests
# =============================================================================

class TestCrossLanguageCloner:
    """Tests for CrossLanguageCloner class."""

    def test_init_default(self):
        """Test default initialization."""
        cloner = CrossLanguageCloner()
        assert cloner.source_language == "en"
        assert cloner.preserve_accent is False

    def test_init_custom(self):
        """Test custom initialization."""
        cloner = CrossLanguageCloner(
            source_language="zh",
            preserve_accent=True,
        )
        assert cloner.source_language == "zh"
        assert cloner.preserve_accent is True

    def test_source_config(self):
        """Test source_config property."""
        cloner = CrossLanguageCloner(source_language="en")
        config = cloner.source_config
        assert isinstance(config, LanguageConfig)
        assert config.code == "en"

    def test_source_config_fallback(self):
        """Test source_config fallback for unknown language."""
        cloner = CrossLanguageCloner(source_language="xyz")
        config = cloner.source_config
        # Should fall back to English
        assert config.code == "en"

    def test_get_target_config(self):
        """Test get_target_config method."""
        cloner = CrossLanguageCloner()
        config = cloner.get_target_config("es")
        assert config.code == "es"

    def test_get_target_config_fallback(self):
        """Test get_target_config fallback."""
        cloner = CrossLanguageCloner()
        config = cloner.get_target_config("xyz")
        assert config.code == "en"

    def test_is_language_supported_true(self):
        """Test is_language_supported returns True."""
        cloner = CrossLanguageCloner()
        assert cloner.is_language_supported("en") is True
        assert cloner.is_language_supported("zh") is True

    def test_is_language_supported_false(self):
        """Test is_language_supported returns False."""
        cloner = CrossLanguageCloner()
        assert cloner.is_language_supported("xyz") is False

    def test_list_supported_languages(self):
        """Test list_supported_languages method."""
        cloner = CrossLanguageCloner()
        languages = cloner.list_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert all("code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)

    def test_get_language_pair_compatibility_supported(self):
        """Test compatibility check for supported pair."""
        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "es")
        assert compat["compatible"] is True
        assert "expected_quality" in compat

    def test_get_language_pair_compatibility_unsupported(self):
        """Test compatibility check for unsupported language."""
        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "xyz")
        assert compat["compatible"] is False

    def test_get_language_pair_compatibility_same_family(self):
        """Test same language family detection."""
        cloner = CrossLanguageCloner()
        # Romance languages
        compat = cloner.get_language_pair_compatibility("es", "fr")
        assert compat["same_language_family"] is True

    def test_get_language_pair_compatibility_different_family(self):
        """Test different language family detection."""
        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "zh")
        assert compat["same_language_family"] is False

    def test_get_language_pair_compatibility_tonal_target(self):
        """Test recommendations for tonal target language."""
        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "zh")
        # Should have phonetic issues noted
        assert len(compat["phonetic_issues"]) > 0

    def test_prepare_embedding_for_language(self):
        """Test preparing embedding for target language."""
        cloner = CrossLanguageCloner()
        emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        prepared, metadata = cloner.prepare_embedding_for_language(emb, "es")
        assert "source_language" in metadata
        assert "target_language" in metadata
        assert "recommended_speed_multiplier" in metadata

    def test_estimate_quality(self):
        """Test quality estimation."""
        cloner = CrossLanguageCloner()
        emb = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            quality_score=0.9,
            source_duration_seconds=5.0,
        )
        quality = cloner.estimate_quality(emb, "es")
        assert 0.0 <= quality <= 1.0

    def test_estimate_quality_short_audio(self):
        """Test quality penalty for short audio."""
        cloner = CrossLanguageCloner()
        emb = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            quality_score=0.9,
            source_duration_seconds=2.0,  # Short
        )
        quality = cloner.estimate_quality(emb, "es")
        assert quality < 0.9  # Should be penalized


# =============================================================================
# detect_language Function Tests
# =============================================================================

class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_english(self):
        """Test detecting English text."""
        result = detect_language("Hello world")
        assert result == "en"

    def test_detect_japanese(self):
        """Test detecting Japanese (hiragana)."""
        result = detect_language("こんにちは")
        assert result == "ja"

    def test_detect_korean(self):
        """Test detecting Korean (hangul)."""
        result = detect_language("안녕하세요")
        assert result == "ko"

    def test_detect_chinese(self):
        """Test detecting Chinese."""
        result = detect_language("你好世界")
        assert result == "zh"

    def test_detect_russian(self):
        """Test detecting Russian (cyrillic)."""
        result = detect_language("Привет мир")
        assert result == "ru"

    def test_detect_arabic(self):
        """Test detecting Arabic."""
        result = detect_language("مرحبا")
        assert result == "ar"

    def test_detect_hindi(self):
        """Test detecting Hindi (devanagari)."""
        result = detect_language("नमस्ते")
        assert result == "hi"

    def test_detect_thai(self):
        """Test detecting Thai."""
        result = detect_language("สวัสดี")
        assert result == "th"


# =============================================================================
# CloningConfig Tests
# =============================================================================

class TestCloningConfig:
    """Tests for CloningConfig dataclass."""

    def test_creation_default(self):
        """Test default creation."""
        config = CloningConfig()
        assert config.extractor_backend == ExtractorBackend.MOCK
        assert config.device == "cpu"
        assert config.min_audio_seconds == 1.0
        assert config.require_consent is True

    def test_creation_custom(self):
        """Test custom creation."""
        config = CloningConfig(
            device="cuda",
            min_audio_seconds=2.0,
            require_consent=False,
        )
        assert config.device == "cuda"
        assert config.min_audio_seconds == 2.0
        assert config.require_consent is False


# =============================================================================
# CloningResult Tests
# =============================================================================

class TestCloningResult:
    """Tests for CloningResult dataclass."""

    def test_creation_success(self):
        """Test successful result creation."""
        result = CloningResult(
            success=True,
            voice_id="test_voice",
            quality_score=0.9,
        )
        assert result.success is True
        assert result.voice_id == "test_voice"
        assert result.error is None

    def test_creation_failure(self):
        """Test failure result creation."""
        result = CloningResult(
            success=False,
            voice_id="test_voice",
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


# =============================================================================
# VoiceProfile Tests
# =============================================================================

class TestVoiceProfile:
    """Tests for VoiceProfile dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        profile = VoiceProfile(
            voice_id="test_voice",
            name="Test Voice",
        )
        assert profile.voice_id == "test_voice"
        assert profile.name == "Test Voice"
        assert profile.language == "en"

    def test_creation_with_embedding(self):
        """Test creation with embedding."""
        emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            embedding=emb,
        )
        assert profile.embedding is not None

    def test_to_dict(self):
        """Test serialization to dict."""
        profile = VoiceProfile(
            voice_id="test",
            name="Test Voice",
            description="A test voice",
            tags=["test", "demo"],
        )
        data = profile.to_dict()
        assert data["voice_id"] == "test"
        assert data["name"] == "Test Voice"
        assert "test" in data["tags"]

    def test_to_dict_with_embedding(self):
        """Test serialization with embedding."""
        emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            embedding=emb,
        )
        data = profile.to_dict()
        assert "embedding" in data
        assert isinstance(data["embedding"]["embedding"], list)

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "voice_id": "test",
            "name": "Test Voice",
            "description": "Test",
            "embedding": None,
            "source_audio_path": None,
            "source_duration_seconds": 0.0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "tags": ["test"],
            "gender": "male",
            "age_range": None,
            "accent": None,
            "language": "en",
            "default_speed": 1.0,
            "default_emotion": None,
            "quality_rating": 1.0,
            "usage_count": 0,
            "last_used_at": None,
            "consent_given": True,
            "consent_date": "2024-01-01",
            "consent_notes": "",
        }
        profile = VoiceProfile.from_dict(data)
        assert profile.voice_id == "test"
        assert profile.gender == "male"

    def test_created_date_property(self):
        """Test created_date property."""
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            created_at=1704067200.0,  # 2024-01-01 00:00:00 UTC
        )
        date_str = profile.created_date
        assert "2024" in date_str or "2023" in date_str  # Depends on timezone

    def test_record_usage(self):
        """Test record_usage method."""
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
        )
        initial_count = profile.usage_count
        profile.record_usage()
        assert profile.usage_count == initial_count + 1
        assert profile.last_used_at is not None


# =============================================================================
# VoiceLibrary Tests
# =============================================================================

class TestVoiceLibrary:
    """Tests for VoiceLibrary class."""

    def test_init_default(self):
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            assert lib.library_path == Path(tmpdir) / "voices"

    def test_load_empty(self):
        """Test loading empty library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            lib.load()
            assert len(lib) == 0

    def test_add_and_get(self):
        """Test adding and getting voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))

            profile = lib.add(
                voice_id="test_voice",
                name="Test Voice",
                embedding=emb,
            )

            assert profile.voice_id == "test_voice"

            retrieved = lib.get("test_voice")
            assert retrieved is not None
            assert retrieved.name == "Test Voice"

    def test_add_duplicate_raises(self):
        """Test adding duplicate voice raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))

            lib.add(voice_id="test", name="Test", embedding=emb)

            with pytest.raises(ValueError, match="already exists"):
                lib.add(voice_id="test", name="Test 2", embedding=emb)

    def test_get_or_raise_exists(self):
        """Test get_or_raise for existing voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="test", name="Test", embedding=emb)

            profile = lib.get_or_raise("test")
            assert profile.voice_id == "test"

    def test_get_or_raise_not_found(self):
        """Test get_or_raise for non-existent voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            lib.load()

            with pytest.raises(KeyError, match="not found"):
                lib.get_or_raise("nonexistent")

    def test_update(self):
        """Test updating voice profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="test", name="Original", embedding=emb)

            updated = lib.update("test", name="Updated Name", description="New desc")
            assert updated.name == "Updated Name"
            assert updated.description == "New desc"

    def test_remove(self):
        """Test removing voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="test", name="Test", embedding=emb)

            result = lib.remove("test")
            assert result is True
            assert lib.get("test") is None

    def test_remove_not_found(self):
        """Test removing non-existent voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            lib.load()

            result = lib.remove("nonexistent")
            assert result is False

    def test_list_all(self):
        """Test listing all voices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="voice1", name="Voice 1", embedding=emb)
            lib.add(voice_id="voice2", name="Voice 2", embedding=emb)

            all_voices = lib.list_all()
            assert len(all_voices) == 2

    def test_list_ids(self):
        """Test listing voice IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="voice1", name="Voice 1", embedding=emb)
            lib.add(voice_id="voice2", name="Voice 2", embedding=emb)

            ids = lib.list_ids()
            assert "voice1" in ids
            assert "voice2" in ids

    def test_search_by_query(self):
        """Test search by text query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="alice", name="Alice Voice", embedding=emb)
            lib.add(voice_id="bob", name="Bob Voice", embedding=emb)

            results = lib.search(query="alice")
            assert len(results) == 1
            assert results[0].voice_id == "alice"

    def test_search_by_tags(self):
        """Test search by tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="voice1", name="Voice 1", embedding=emb, tags=["female"])
            lib.add(voice_id="voice2", name="Voice 2", embedding=emb, tags=["male"])

            results = lib.search(tags=["female"])
            assert len(results) == 1
            assert results[0].voice_id == "voice1"

    def test_search_by_gender(self):
        """Test search by gender."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="voice1", name="Voice 1", embedding=emb, gender="female")
            lib.add(voice_id="voice2", name="Voice 2", embedding=emb, gender="male")

            results = lib.search(gender="female")
            assert len(results) == 1
            assert results[0].voice_id == "voice1"

    def test_find_similar(self):
        """Test finding similar voices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")

            # Add voice with known embedding
            emb1 = VoiceEmbedding(embedding=np.ones(256, dtype=np.float32))
            lib.add(voice_id="voice1", name="Voice 1", embedding=emb1)

            # Search with similar embedding
            search_emb = VoiceEmbedding(embedding=np.ones(256, dtype=np.float32) * 0.99)
            results = lib.find_similar(search_emb, top_k=5, min_similarity=0.5)

            assert len(results) >= 1
            assert results[0][0].voice_id == "voice1"
            assert results[0][1] > 0.9  # High similarity

    def test_contains(self):
        """Test __contains__ method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="test", name="Test", embedding=emb)

            assert "test" in lib
            assert "nonexistent" not in lib

    def test_len(self):
        """Test __len__ method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))

            lib.add(voice_id="voice1", name="Voice 1", embedding=emb)
            assert len(lib) == 1

            lib.add(voice_id="voice2", name="Voice 2", embedding=emb)
            assert len(lib) == 2

    def test_iter(self):
        """Test __iter__ method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib.add(voice_id="voice1", name="Voice 1", embedding=emb)
            lib.add(voice_id="voice2", name="Voice 2", embedding=emb)

            voice_ids = [v.voice_id for v in lib]
            assert "voice1" in voice_ids
            assert "voice2" in voice_ids

    def test_persistence(self):
        """Test library persistence across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = Path(tmpdir) / "voices"

            # Create and add voice
            lib1 = VoiceLibrary(lib_path)
            emb = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            lib1.add(voice_id="test", name="Test", embedding=emb)

            # Create new instance and check voice exists
            lib2 = VoiceLibrary(lib_path)
            lib2.load()

            assert "test" in lib2
            assert lib2.get("test").name == "Test"


# =============================================================================
# VoiceCloner Tests
# =============================================================================

class TestVoiceCloner:
    """Tests for VoiceCloner class."""

    def test_init_default(self):
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)
            assert cloner.config is not None
            assert cloner.library is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            config = CloningConfig(require_consent=False)
            cloner = VoiceCloner(config=config, library=lib)
            assert cloner.config.require_consent is False

    def test_extractor_property(self):
        """Test extractor property creates extractor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)
            extractor = cloner.extractor
            assert isinstance(extractor, VoiceExtractor)

    def test_clone_without_consent(self):
        """Test cloning fails without consent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            result = cloner.clone(
                audio=audio,
                voice_id="test",
                consent_given=False,
            )

            assert result.success is False
            assert "consent" in result.error.lower()

    def test_clone_with_consent(self):
        """Test successful cloning with consent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            result = cloner.clone(
                audio=audio,
                voice_id="test_voice",
                name="Test Voice",
                consent_given=True,
                sample_rate=16000,
            )

            assert result.success is True
            assert result.voice_id == "test_voice"
            assert result.profile is not None

    def test_clone_duplicate_fails(self):
        """Test cloning duplicate voice fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)

            # First clone
            cloner.clone(audio=audio, voice_id="test", consent_given=True)

            # Second clone with same ID
            result = cloner.clone(audio=audio, voice_id="test", consent_given=True)
            assert result.success is False
            assert "already exists" in result.error

    def test_clone_quick(self):
        """Test quick clone without saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            result = cloner.clone_quick(audio=audio, name="Quick Test")

            assert result.success is True
            assert result.embedding is not None
            # Should not be in library
            assert "Quick Test" not in [v.name for v in lib.list_all()]

    def test_delete_voice(self):
        """Test deleting voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            cloner.clone(audio=audio, voice_id="test", consent_given=True)

            result = cloner.delete_voice("test")
            assert result is True
            assert cloner.get_voice("test") is None

    def test_get_voice(self):
        """Test getting voice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            cloner.clone(audio=audio, voice_id="test", consent_given=True)

            voice = cloner.get_voice("test")
            assert voice is not None
            assert voice.voice_id == "test"

    def test_list_voices(self):
        """Test listing voices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            cloner.clone(audio=audio, voice_id="voice1", consent_given=True)
            cloner.clone(audio=audio, voice_id="voice2", consent_given=True)

            voices = cloner.list_voices()
            assert len(voices) == 2

    def test_validate_audio(self):
        """Test validating audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            cloner = VoiceCloner(library=lib)

            audio = np.random.randn(48000).astype(np.float32)
            validation = cloner.validate_audio(audio)

            assert "is_valid" in validation
            assert "quality_score" in validation
            assert "duration_seconds" in validation

    def test_validate_audio_too_short(self):
        """Test validation fails for short audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = VoiceLibrary(Path(tmpdir) / "voices")
            config = CloningConfig(min_audio_seconds=5.0)
            cloner = VoiceCloner(config=config, library=lib)

            # Only 0.5 seconds
            audio = np.random.randn(8000).astype(np.float32)
            validation = cloner.validate_audio(audio)

            assert validation["is_valid"] is False
            assert len(validation["issues"]) > 0


# =============================================================================
# get_default_library Function Tests
# =============================================================================

class TestGetDefaultLibrary:
    """Tests for get_default_library function."""

    def test_returns_library(self):
        """Test returns VoiceLibrary instance."""
        lib = get_default_library()
        assert isinstance(lib, VoiceLibrary)

    def test_returns_same_instance(self):
        """Test returns same instance on subsequent calls."""
        lib1 = get_default_library()
        lib2 = get_default_library()
        assert lib1 is lib2
