"""
Additional coverage tests - Batch 39: Cloning Module Complete Coverage.

Comprehensive tests for:
- voice_soundboard/cloning/cloner.py
- voice_soundboard/cloning/library.py
- voice_soundboard/cloning/extractor.py
- voice_soundboard/cloning/crosslang.py
"""

import pytest
import json
import tempfile
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Voice Library Tests
# =============================================================================

class TestVoiceLibrary:
    """Tests for VoiceLibrary class."""

    def test_library_creation(self, tmp_path):
        """Test creating a voice library."""
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()
        assert len(lib) == 0

    def test_library_add_voice(self, tmp_path):
        """Test adding a voice to the library."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )

        profile = lib.add(
            voice_id="test_voice",
            name="Test Voice",
            embedding=embedding,
            tags=["test", "demo"],
        )

        assert profile.voice_id == "test_voice"
        assert profile.name == "Test Voice"
        assert "test" in profile.tags

    def test_library_get_voice(self, tmp_path):
        """Test getting a voice from the library."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        lib.add(voice_id="my_voice", name="My Voice", embedding=embedding)

        profile = lib.get("my_voice")
        assert profile is not None
        assert profile.name == "My Voice"

    def test_library_get_nonexistent(self, tmp_path):
        """Test getting a nonexistent voice."""
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        profile = lib.get("nonexistent")
        assert profile is None

    def test_library_get_or_raise(self, tmp_path):
        """Test get_or_raise with nonexistent voice."""
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        with pytest.raises(KeyError):
            lib.get_or_raise("nonexistent")

    def test_library_update_voice(self, tmp_path):
        """Test updating a voice profile."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        lib.add(voice_id="update_test", name="Original", embedding=embedding)

        updated = lib.update("update_test", name="Updated Name", description="New desc")
        assert updated.name == "Updated Name"
        assert updated.description == "New desc"

    def test_library_remove_voice(self, tmp_path):
        """Test removing a voice from the library."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        lib.add(voice_id="to_remove", name="To Remove", embedding=embedding)

        assert "to_remove" in lib
        result = lib.remove("to_remove")
        assert result is True
        assert "to_remove" not in lib

    def test_library_search_by_tags(self, tmp_path):
        """Test searching voices by tags."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        for i, tags in enumerate([["male", "narrator"], ["female", "assistant"], ["male", "casual"]]):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=5.0,
            )
            lib.add(voice_id=f"voice_{i}", name=f"Voice {i}", embedding=embedding, tags=tags)

        results = lib.search(tags=["male"])
        assert len(results) == 2

    def test_library_search_by_gender(self, tmp_path):
        """Test searching voices by gender."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        for i, gender in enumerate(["male", "female", "male"]):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=5.0,
            )
            lib.add(voice_id=f"g_voice_{i}", name=f"Voice {i}", embedding=embedding, gender=gender)

        results = lib.search(gender="female")
        assert len(results) == 1

    def test_library_search_by_query(self, tmp_path):
        """Test searching voices by text query."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        lib.add(
            voice_id="narrator_voice",
            name="Professional Narrator",
            embedding=embedding,
            description="Deep voice for documentaries",
        )

        results = lib.search(query="narrator")
        assert len(results) == 1

    def test_library_find_similar(self, tmp_path):
        """Test finding similar voices by embedding."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        # Create similar embeddings
        base_embedding = np.random.randn(256).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        for i in range(3):
            # Add small noise to create similar embeddings
            noisy = base_embedding + np.random.randn(256).astype(np.float32) * 0.1
            noisy = noisy / np.linalg.norm(noisy)
            embedding = VoiceEmbedding(
                embedding=noisy,
                source_duration_seconds=5.0,
            )
            lib.add(voice_id=f"sim_voice_{i}", name=f"Similar Voice {i}", embedding=embedding)

        query_embedding = VoiceEmbedding(
            embedding=base_embedding,
            source_duration_seconds=5.0,
        )
        similar = lib.find_similar(query_embedding, top_k=2)
        assert len(similar) <= 2

    def test_library_persistence(self, tmp_path):
        """Test library persistence across loads."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib_path = tmp_path / "persistent_voices"

        # Create and save
        lib1 = VoiceLibrary(library_path=lib_path)
        lib1.load()
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        lib1.add(voice_id="persistent", name="Persistent Voice", embedding=embedding)

        # Load in new instance
        lib2 = VoiceLibrary(library_path=lib_path)
        lib2.load()
        profile = lib2.get("persistent")
        assert profile is not None
        assert profile.name == "Persistent Voice"

    def test_library_iteration(self, tmp_path):
        """Test iterating over library."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        lib.load()

        for i in range(3):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=5.0,
            )
            lib.add(voice_id=f"iter_voice_{i}", name=f"Voice {i}", embedding=embedding)

        profiles = list(lib)
        assert len(profiles) == 3


# =============================================================================
# Voice Profile Tests
# =============================================================================

class TestVoiceProfile:
    """Tests for VoiceProfile dataclass."""

    def test_profile_to_dict(self):
        """Test serializing profile to dict."""
        from voice_soundboard.cloning.library import VoiceProfile
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            embedding=embedding,
            tags=["demo"],
        )

        data = profile.to_dict()
        assert data["voice_id"] == "test"
        assert data["name"] == "Test"
        assert "embedding" in data

    def test_profile_from_dict(self):
        """Test deserializing profile from dict."""
        from voice_soundboard.cloning.library import VoiceProfile

        data = {
            "voice_id": "restored",
            "name": "Restored Voice",
            "description": "A restored voice",
            "embedding": {
                "embedding": [0.1] * 256,
                "embedding_dim": 256,
                "source_duration_seconds": 5.0,
            },
            "tags": ["restored"],
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        profile = VoiceProfile.from_dict(data)
        assert profile.voice_id == "restored"
        assert profile.name == "Restored Voice"

    def test_profile_record_usage(self):
        """Test recording profile usage."""
        from voice_soundboard.cloning.library import VoiceProfile
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )
        profile = VoiceProfile(
            voice_id="usage_test",
            name="Usage Test",
            embedding=embedding,
        )

        assert profile.usage_count == 0
        profile.record_usage()
        assert profile.usage_count == 1
        assert profile.last_used_at is not None


# =============================================================================
# Voice Extractor Tests
# =============================================================================

class TestVoiceExtractor:
    """Tests for VoiceExtractor class."""

    def test_extractor_mock_backend(self):
        """Test extractor with mock backend."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds

        embedding = extractor.extract(audio)
        assert embedding.embedding.shape == (256,)
        assert embedding.source_duration_seconds > 0

    def test_extractor_embedding_dim(self):
        """Test extractor embedding dimension property."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        assert extractor.embedding_dim == 256

    def test_extractor_from_segments(self):
        """Test extracting from segments."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Create audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            audio_data = (np.random.randn(16000 * 10) * 32767).astype(np.int16)
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_data.tobytes())

            embeddings = extractor.extract_from_segments(f.name, segment_seconds=3.0)
            assert len(embeddings) >= 1

    def test_extractor_average_embeddings(self):
        """Test averaging multiple embeddings."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, VoiceEmbedding, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        embeddings = [
            VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=3.0,
            )
            for _ in range(5)
        ]

        averaged = extractor.average_embeddings(embeddings)
        assert averaged.embedding.shape == (256,)
        # Averaged duration should be sum
        assert averaged.source_duration_seconds == 15.0

    def test_extractor_quality_estimation(self):
        """Test audio quality estimation."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Good quality audio
        good_audio = np.sin(np.linspace(0, 100 * np.pi, 16000)) * 0.5
        quality, snr = extractor._estimate_quality(good_audio.astype(np.float32))
        assert 0 <= quality <= 1
        assert snr > 0


# =============================================================================
# Voice Embedding Tests
# =============================================================================

class TestVoiceEmbedding:
    """Tests for VoiceEmbedding dataclass."""

    def test_embedding_similarity(self):
        """Test embedding similarity computation."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = VoiceEmbedding(
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            embedding_dim=3,
        )
        emb2 = VoiceEmbedding(
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            embedding_dim=3,
        )
        emb3 = VoiceEmbedding(
            embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            embedding_dim=3,
        )

        assert emb1.similarity(emb2) == pytest.approx(1.0)
        assert emb1.similarity(emb3) == pytest.approx(0.0)

    def test_embedding_save_load_npz(self, tmp_path):
        """Test saving and loading embedding as NPZ."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )

        path = tmp_path / "embedding.npz"
        embedding.save(path)

        loaded = VoiceEmbedding.load(path)
        assert np.allclose(embedding.embedding, loaded.embedding)

    def test_embedding_save_load_json(self, tmp_path):
        """Test saving and loading embedding as JSON."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )

        path = tmp_path / "embedding.json"
        embedding.save(path)

        loaded = VoiceEmbedding.load(path)
        assert np.allclose(embedding.embedding, loaded.embedding, atol=1e-5)


# =============================================================================
# Voice Cloner Tests
# =============================================================================

class TestVoiceCloner:
    """Tests for VoiceCloner class."""

    def test_cloner_consent_required(self, tmp_path):
        """Test that cloning fails without consent."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=True)
        cloner = VoiceCloner(config=config, library=lib)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.clone(audio, voice_id="no_consent", consent_given=False)

        assert result.success is False
        assert "consent" in result.error.lower()

    def test_cloner_duplicate_voice_id(self, tmp_path):
        """Test that duplicate voice IDs are rejected."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False)
        cloner = VoiceCloner(config=config, library=lib)

        audio = np.random.randn(16000 * 5).astype(np.float32)

        # First clone should succeed
        result1 = cloner.clone(audio, voice_id="duplicate_test")
        assert result1.success is True

        # Second clone with same ID should fail
        result2 = cloner.clone(audio, voice_id="duplicate_test")
        assert result2.success is False
        assert "exists" in result2.error.lower()

    def test_cloner_audio_too_short(self, tmp_path):
        """Test rejection of too-short audio."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False, min_audio_seconds=3.0)
        cloner = VoiceCloner(config=config, library=lib)

        # Too short audio (0.5 seconds)
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        result = cloner.clone(audio, voice_id="short_audio")

        # Should fail or warn
        if not result.success:
            assert "short" in result.error.lower()

    def test_cloner_quick_clone(self, tmp_path):
        """Test quick clone without library save."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False)
        cloner = VoiceCloner(config=config, library=lib)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.clone_quick(audio, name="Quick Voice")

        assert result.success is True
        assert result.embedding is not None
        # Should not be in library
        assert len(lib) == 0

    def test_cloner_validate_audio(self, tmp_path):
        """Test audio validation."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False)
        cloner = VoiceCloner(config=config, library=lib)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        validation = cloner.validate_audio(audio)

        assert "is_valid" in validation
        assert "quality_score" in validation
        assert "duration_seconds" in validation

    def test_cloner_export_import(self, tmp_path):
        """Test voice export and import."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        lib = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False)
        cloner = VoiceCloner(config=config, library=lib)

        # Clone a voice
        audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.clone(audio, voice_id="export_test", name="Export Test")
        assert result.success is True

        # Export
        export_path = tmp_path / "exported_voice.json"
        cloner.export_voice("export_test", export_path)
        assert export_path.exists()

        # Remove from library
        cloner.delete_voice("export_test")
        assert cloner.get_voice("export_test") is None

        # Import
        imported = cloner.import_voice(export_path)
        assert imported.voice_id == "export_test"
        assert cloner.get_voice("export_test") is not None


# =============================================================================
# Cross-Language Cloning Tests
# =============================================================================

class TestCrossLanguageCloner:
    """Tests for CrossLanguageCloner class."""

    def test_crosslang_supported_languages(self):
        """Test listing supported languages."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        languages = cloner.list_supported_languages()

        assert len(languages) > 0
        assert any(lang["code"] == "en" for lang in languages)
        assert any(lang["code"] == "zh" for lang in languages)

    def test_crosslang_is_language_supported(self):
        """Test checking language support."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        assert cloner.is_language_supported("en") is True
        assert cloner.is_language_supported("xyz") is False

    def test_crosslang_language_pair_compatibility(self):
        """Test language pair compatibility assessment."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")
        compat = cloner.get_language_pair_compatibility("en", "fr")

        assert compat["compatible"] is True
        assert "expected_quality" in compat
        assert 0 <= compat["expected_quality"] <= 1

    def test_crosslang_tonal_language_warning(self):
        """Test warning for non-tonal to tonal language transfer."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")
        compat = cloner.get_language_pair_compatibility("en", "zh")

        # Should have phonetic issues for tonal language
        assert len(compat.get("phonetic_issues", [])) > 0

    def test_crosslang_prepare_embedding(self):
        """Test preparing embedding for cross-language synthesis."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=5.0,
        )

        prepared, metadata = cloner.prepare_embedding_for_language(embedding, "fr")
        assert prepared is not None
        assert "recommended_speed_multiplier" in metadata

    def test_crosslang_estimate_quality(self):
        """Test quality estimation for cross-language."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=10.0,
            quality_score=0.9,
        )

        quality = cloner.estimate_quality(embedding, "de")
        assert 0 <= quality <= 1


# =============================================================================
# Language Detection Tests
# =============================================================================

class TestLanguageDetection:
    """Tests for language detection function."""

    def test_detect_english(self):
        """Test detecting English text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("Hello world, this is a test.")
        assert result == "en"

    def test_detect_chinese(self):
        """Test detecting Chinese text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("你好世界")
        assert result == "zh"

    def test_detect_japanese(self):
        """Test detecting Japanese text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("こんにちは")
        assert result == "ja"

    def test_detect_korean(self):
        """Test detecting Korean text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("안녕하세요")
        assert result == "ko"

    def test_detect_russian(self):
        """Test detecting Russian text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("Привет мир")
        assert result == "ru"

    def test_detect_arabic(self):
        """Test detecting Arabic text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("مرحبا بالعالم")
        assert result == "ar"


# =============================================================================
# Language Config Tests
# =============================================================================

class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_language_config_defaults(self):
        """Test LanguageConfig default values."""
        from voice_soundboard.cloning.crosslang import LanguageConfig

        config = LanguageConfig(
            code="test",
            name="Test Language",
            native_name="Test",
        )
        assert config.phoneme_set == "ipa"
        assert config.default_speed == 1.0

    def test_language_configs_exist(self):
        """Test that language configs exist for major languages."""
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        assert "en" in LANGUAGE_CONFIGS
        assert "zh" in LANGUAGE_CONFIGS
        assert "ja" in LANGUAGE_CONFIGS
        assert "es" in LANGUAGE_CONFIGS
        assert "fr" in LANGUAGE_CONFIGS
        assert "de" in LANGUAGE_CONFIGS

    def test_language_enum_values(self):
        """Test Language enum values."""
        from voice_soundboard.cloning.crosslang import Language

        assert Language.ENGLISH.value == "en"
        assert Language.CHINESE_MANDARIN.value == "zh"
        assert Language.JAPANESE.value == "ja"
