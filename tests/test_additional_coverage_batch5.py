"""
Additional tests - Batch 5.

Covers remaining unchecked items from TEST_PLAN.md:
- engines/base.py (TEST-EB01 to TEST-EB13)
- engines/kokoro.py (TEST-EK01 to TEST-EK12)
- cloning/extractor.py (VoiceEmbedding, VoiceExtractor)
- cloning/library.py (VoiceLibrary, VoiceProfile)
- conversion/__init__.py exports
- llm/__init__.py exports
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json


# =============================================================================
# Module: engines/base.py - TTSEngine Interface Tests (TEST-EB01 to TEST-EB13)
# =============================================================================

class TestTTSEngineAbstract:
    """Tests for TTSEngine abstract base class."""

    def test_tts_engine_is_abstract(self):
        """TEST-EB01: TTSEngine is abstract base class."""
        from voice_soundboard.engines.base import TTSEngine

        # Cannot instantiate directly
        with pytest.raises(TypeError):
            TTSEngine()

    def test_engine_result_has_required_fields(self):
        """TEST-EB02: EngineResult dataclass has all required fields."""
        from voice_soundboard.engines.base import EngineResult

        result = EngineResult()

        # Check all required fields exist
        assert hasattr(result, "audio_path")
        assert hasattr(result, "samples")
        assert hasattr(result, "sample_rate")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "generation_time")
        assert hasattr(result, "voice_used")
        assert hasattr(result, "realtime_factor")
        assert hasattr(result, "engine_name")
        assert hasattr(result, "metadata")

    def test_engine_capabilities_defaults(self):
        """TEST-EB03: EngineCapabilities dataclass defaults are correct."""
        from voice_soundboard.engines.base import EngineCapabilities

        caps = EngineCapabilities()

        assert caps.supports_streaming is False
        assert caps.supports_ssml is False
        assert caps.supports_voice_cloning is False
        assert caps.supports_emotion_control is False
        assert caps.supports_paralinguistic_tags is False
        assert caps.supports_emotion_exaggeration is False
        assert caps.paralinguistic_tags == []
        assert caps.languages == ["en"]
        assert caps.typical_rtf == 1.0
        assert caps.min_latency_ms == 200.0

    def test_name_property_is_abstract(self):
        """TEST-EB04: TTSEngine.name property is abstract."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class IncompleteEngine(TTSEngine):
            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, **kwargs):
                pass

            def speak_raw(self, text, **kwargs):
                pass

            def list_voices(self):
                return []

        # Missing name property should fail
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_capabilities_property_is_abstract(self):
        """TEST-EB05: TTSEngine.capabilities property is abstract."""
        from voice_soundboard.engines.base import TTSEngine

        class IncompleteEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            def speak(self, text, **kwargs):
                pass

            def speak_raw(self, text, **kwargs):
                pass

            def list_voices(self):
                return []

        # Missing capabilities property should fail
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_speak_is_abstract(self):
        """TEST-EB06: TTSEngine.speak() is abstract."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class IncompleteEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak_raw(self, text, **kwargs):
                pass

            def list_voices(self):
                return []

        # Missing speak should fail
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_speak_raw_is_abstract(self):
        """TEST-EB07: TTSEngine.speak_raw() is abstract."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class IncompleteEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, **kwargs):
                pass

            def list_voices(self):
                return []

        # Missing speak_raw should fail
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_list_voices_is_abstract(self):
        """TEST-EB08: TTSEngine.list_voices() is abstract."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class IncompleteEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, **kwargs):
                pass

            def speak_raw(self, text, **kwargs):
                pass

        # Missing list_voices should fail
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_get_voice_info_has_default(self):
        """TEST-EB09: TTSEngine.get_voice_info() has default implementation."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities, EngineResult

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, **kwargs):
                return EngineResult()

            def speak_raw(self, text, **kwargs):
                return np.zeros(1000), 24000

            def list_voices(self):
                return ["voice1"]

        engine = TestEngine()
        info = engine.get_voice_info("voice1")

        assert info["id"] == "voice1"
        assert info["name"] == "voice1"

    def test_clone_voice_raises_not_implemented(self):
        """TEST-EB11: TTSEngine.clone_voice() raises NotImplementedError when unsupported."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities, EngineResult

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities(supports_voice_cloning=False)

            def speak(self, text, **kwargs):
                return EngineResult()

            def speak_raw(self, text, **kwargs):
                return np.zeros(1000), 24000

            def list_voices(self):
                return []

        engine = TestEngine()
        with pytest.raises(NotImplementedError):
            engine.clone_voice(Path("test.wav"), "cloned")

    def test_is_loaded_default_false(self):
        """TEST-EB12: TTSEngine.is_loaded() default returns False."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities, EngineResult

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, **kwargs):
                return EngineResult()

            def speak_raw(self, text, **kwargs):
                return np.zeros(1000), 24000

            def list_voices(self):
                return []

        engine = TestEngine()
        assert engine.is_loaded() is False

    def test_unload_default_noop(self):
        """TEST-EB13: TTSEngine.unload() default does nothing."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities, EngineResult

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, **kwargs):
                return EngineResult()

            def speak_raw(self, text, **kwargs):
                return np.zeros(1000), 24000

            def list_voices(self):
                return []

        engine = TestEngine()
        # Should not raise
        engine.unload()


class TestEngineResultMetadata:
    """Additional tests for EngineResult."""

    def test_engine_result_with_samples(self):
        """Test EngineResult with sample data."""
        from voice_soundboard.engines.base import EngineResult

        samples = np.zeros(24000, dtype=np.float32)
        result = EngineResult(
            samples=samples,
            sample_rate=24000,
            duration_seconds=1.0,
            voice_used="test_voice",
            engine_name="test",
        )

        assert result.samples is not None
        assert len(result.samples) == 24000
        assert result.duration_seconds == 1.0

    def test_engine_result_metadata_dict(self):
        """Test EngineResult with custom metadata."""
        from voice_soundboard.engines.base import EngineResult

        result = EngineResult(
            metadata={"custom_key": "custom_value", "count": 42}
        )

        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["count"] == 42


# =============================================================================
# Module: engines/kokoro.py - KokoroEngine Tests (TEST-EK01 to TEST-EK12)
# =============================================================================

class TestKokoroEngineProperties:
    """Tests for KokoroEngine properties."""

    def test_kokoro_engine_name_is_kokoro(self):
        """TEST-EK01: KokoroEngine.name returns 'kokoro'."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        assert engine.name == "kokoro"

    def test_kokoro_engine_capabilities(self):
        """TEST-EK02: KokoroEngine.capabilities reports correct features."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        caps = engine.capabilities

        assert caps.supports_streaming is True
        assert caps.supports_ssml is True
        assert "en" in caps.languages

    def test_kokoro_no_paralinguistic_tags(self):
        """TEST-EK03: KokoroEngine.capabilities.supports_paralinguistic_tags is False."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        assert engine.capabilities.supports_paralinguistic_tags is False

    def test_kokoro_no_voice_cloning(self):
        """TEST-EK04: KokoroEngine.capabilities.supports_voice_cloning is False."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        assert engine.capabilities.supports_voice_cloning is False

    def test_kokoro_is_loaded_initially_false(self):
        """TEST-EK11: KokoroEngine.is_loaded() returns False initially."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        assert engine.is_loaded() is False

    def test_kokoro_unload_clears_model(self):
        """TEST-EK12: KokoroEngine.unload() clears model."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._model_loaded = True

        engine.unload()

        assert engine._kokoro is None
        assert engine._model_loaded is False


class TestKokoroEngineGetVoiceInfo:
    """Tests for KokoroEngine.get_voice_info()."""

    def test_get_voice_info_known_voice(self):
        """TEST-EK10: KokoroEngine.get_voice_info() returns metadata dict."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        info = engine.get_voice_info("af_bella")

        assert info["id"] == "af_bella"
        assert "gender" in info or "name" in info

    def test_get_voice_info_unknown_voice(self):
        """Test get_voice_info with unknown voice returns fallback."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        engine = KokoroEngine()
        info = engine.get_voice_info("unknown_voice_xyz")

        assert info["id"] == "unknown_voice_xyz"
        assert info["gender"] == "unknown"


# =============================================================================
# Module: cloning/extractor.py - VoiceEmbedding Tests
# =============================================================================

class TestVoiceEmbedding:
    """Tests for VoiceEmbedding dataclass."""

    def test_voice_embedding_creation(self):
        """Test VoiceEmbedding creation."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(embedding=embedding)

        assert ve.embedding is not None
        assert ve.embedding_dim == 256
        assert ve.embedding_id != ""  # Generated in __post_init__

    def test_voice_embedding_similarity(self):
        """Test VoiceEmbedding.similarity() cosine similarity."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        ve1 = VoiceEmbedding(embedding=emb1, embedding_dim=3)
        ve2 = VoiceEmbedding(embedding=emb2, embedding_dim=3)
        ve3 = VoiceEmbedding(embedding=emb3, embedding_dim=3)

        # Identical embeddings should have similarity 1.0
        assert ve1.similarity(ve2) == pytest.approx(1.0, abs=0.01)

        # Orthogonal embeddings should have similarity 0.0
        assert ve1.similarity(ve3) == pytest.approx(0.0, abs=0.01)

    def test_voice_embedding_similarity_dimension_mismatch(self):
        """Test similarity raises error on dimension mismatch."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        ve1 = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        ve2 = VoiceEmbedding(embedding=np.random.randn(128).astype(np.float32))

        with pytest.raises(ValueError, match="dimension mismatch"):
            ve1.similarity(ve2)

    def test_voice_embedding_to_dict(self):
        """Test VoiceEmbedding.to_dict() serialization."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        ve = VoiceEmbedding(
            embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            embedding_dim=3,
            source_path="/path/to/audio.wav",
        )

        data = ve.to_dict()

        assert data["embedding"] == [1.0, 2.0, 3.0]
        assert data["embedding_dim"] == 3
        assert data["source_path"] == "/path/to/audio.wav"

    def test_voice_embedding_from_dict(self):
        """Test VoiceEmbedding.from_dict() deserialization."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        data = {
            "embedding": [1.0, 2.0, 3.0],
            "embedding_dim": 3,
            "source_path": "/path/to/audio.wav",
            "source_duration_seconds": 5.0,
            "source_sample_rate": 16000,
            "extractor_backend": "mock",
            "extraction_time": 0.1,
            "created_at": 1234567890.0,
            "quality_score": 0.9,
            "snr_db": 25.0,
            "estimated_gender": "female",
            "estimated_age_range": None,
            "language_detected": "en",
            "embedding_id": "abc123",
        }

        ve = VoiceEmbedding.from_dict(data)

        assert np.array_equal(ve.embedding, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert ve.embedding_dim == 3
        assert ve.quality_score == 0.9

    def test_voice_embedding_save_load_json(self, tmp_path):
        """Test VoiceEmbedding save/load JSON round-trip."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        original = VoiceEmbedding(
            embedding=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            embedding_dim=4,
            source_path="test.wav",
        )

        json_path = tmp_path / "embedding.json"
        original.save(json_path)

        loaded = VoiceEmbedding.load(json_path)

        assert np.allclose(loaded.embedding, original.embedding)
        assert loaded.embedding_dim == original.embedding_dim

    def test_voice_embedding_save_load_npz(self, tmp_path):
        """Test VoiceEmbedding save/load NPZ round-trip."""
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        original = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256,
        )

        npz_path = tmp_path / "embedding.npz"
        original.save(npz_path)

        loaded = VoiceEmbedding.load(npz_path)

        assert np.allclose(loaded.embedding, original.embedding)


class TestVoiceExtractor:
    """Tests for VoiceExtractor class."""

    def test_extractor_backend_enum(self):
        """Test ExtractorBackend enum values."""
        from voice_soundboard.cloning.extractor import ExtractorBackend

        assert ExtractorBackend.MOCK.value == "mock"
        assert ExtractorBackend.RESEMBLYZER.value == "resemblyzer"
        assert ExtractorBackend.SPEECHBRAIN.value == "speechbrain"
        assert ExtractorBackend.WESPEAKER.value == "wespeaker"

    def test_extractor_embedding_dim(self):
        """Test VoiceExtractor.embedding_dim property."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        assert extractor.embedding_dim == 256

    def test_extractor_mock_extract(self):
        """Test VoiceExtractor.extract() with mock backend."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Create mock audio
        audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

        ve = extractor.extract(audio, sample_rate=16000)

        assert ve.embedding.shape == (256,)
        assert ve.extractor_backend == "mock"

    def test_extractor_extract_from_file(self, tmp_path):
        """Test VoiceExtractor.extract() from file."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend
        import soundfile as sf

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Create temp audio file
        audio = np.random.randn(16000).astype(np.float32)
        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), audio, 16000)

        ve = extractor.extract(audio_path)

        assert ve.source_path == str(audio_path)
        assert ve.embedding.shape == (256,)

    def test_extractor_file_not_found(self):
        """Test VoiceExtractor.extract() with missing file."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/path/audio.wav")

    def test_extractor_extract_from_segments(self, tmp_path):
        """Test VoiceExtractor.extract_from_segments()."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend
        import soundfile as sf

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Create 6 seconds of audio (should produce 2 segments at 3s each)
        audio = np.random.randn(16000 * 6).astype(np.float32)
        audio_path = tmp_path / "long_audio.wav"
        sf.write(str(audio_path), audio, 16000)

        embeddings = extractor.extract_from_segments(audio_path, segment_seconds=3.0)

        assert len(embeddings) == 2
        for e in embeddings:
            assert e.embedding.shape == (256,)

    def test_extractor_average_embeddings(self):
        """Test VoiceExtractor.average_embeddings()."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, VoiceEmbedding, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Create embeddings
        emb1 = VoiceEmbedding(
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            embedding_dim=3,
        )
        emb2 = VoiceEmbedding(
            embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            embedding_dim=3,
        )

        averaged = extractor.average_embeddings([emb1, emb2])

        # Should be normalized
        assert np.linalg.norm(averaged.embedding) == pytest.approx(1.0, abs=0.01)

    def test_extractor_average_empty_raises(self):
        """Test average_embeddings raises on empty list."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        with pytest.raises(ValueError, match="No embeddings"):
            extractor.average_embeddings([])

    def test_extractor_load_unload(self):
        """Test VoiceExtractor.load() and unload()."""
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        assert extractor._loaded is False

        extractor.load()
        assert extractor._loaded is True

        extractor.unload()
        assert extractor._loaded is False


class TestExtractEmbeddingFunction:
    """Tests for extract_embedding convenience function."""

    def test_extract_embedding_convenience(self):
        """Test extract_embedding() convenience function."""
        from voice_soundboard.cloning.extractor import extract_embedding

        audio = np.random.randn(16000).astype(np.float32)
        ve = extract_embedding(audio, backend="mock")

        assert ve.embedding.shape == (256,)

    def test_extract_embedding_with_enum(self):
        """Test extract_embedding() with ExtractorBackend enum."""
        from voice_soundboard.cloning.extractor import extract_embedding, ExtractorBackend

        audio = np.random.randn(16000).astype(np.float32)
        ve = extract_embedding(audio, backend=ExtractorBackend.MOCK)

        assert ve.embedding.shape == (256,)


# =============================================================================
# Module: cloning/library.py - VoiceLibrary Tests
# =============================================================================

class TestVoiceProfile:
    """Tests for VoiceProfile dataclass."""

    def test_voice_profile_creation(self):
        """Test VoiceProfile creation."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(
            voice_id="my_voice",
            name="My Voice",
            description="A test voice",
            gender="female",
            language="en",
        )

        assert profile.voice_id == "my_voice"
        assert profile.name == "My Voice"
        assert profile.gender == "female"

    def test_voice_profile_to_dict(self):
        """Test VoiceProfile.to_dict() serialization."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(
            voice_id="test",
            name="Test Voice",
            tags=["test", "demo"],
        )

        data = profile.to_dict()

        assert data["voice_id"] == "test"
        assert data["name"] == "Test Voice"
        assert data["tags"] == ["test", "demo"]

    def test_voice_profile_from_dict(self):
        """Test VoiceProfile.from_dict() deserialization."""
        from voice_soundboard.cloning.library import VoiceProfile

        data = {
            "voice_id": "loaded",
            "name": "Loaded Voice",
            "description": "",
            "embedding": None,
            "source_audio_path": None,
            "source_duration_seconds": 0.0,
            "created_at": 1234567890.0,
            "updated_at": 1234567890.0,
            "tags": [],
            "gender": "male",
            "age_range": None,
            "accent": None,
            "language": "en",
            "default_speed": 1.0,
            "default_emotion": None,
            "quality_rating": 1.0,
            "usage_count": 0,
            "last_used_at": None,
            "consent_given": False,
            "consent_date": None,
            "consent_notes": "",
        }

        profile = VoiceProfile.from_dict(data)

        assert profile.voice_id == "loaded"
        assert profile.gender == "male"

    def test_voice_profile_record_usage(self):
        """Test VoiceProfile.record_usage()."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(voice_id="test", name="Test")

        assert profile.usage_count == 0
        assert profile.last_used_at is None

        profile.record_usage()

        assert profile.usage_count == 1
        assert profile.last_used_at is not None


class TestVoiceLibrary:
    """Tests for VoiceLibrary class."""

    def test_library_initialization(self, tmp_path):
        """Test VoiceLibrary initialization."""
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")

        assert library.library_path == tmp_path / "voices"
        assert len(library) == 0

    def test_library_add_voice(self, tmp_path):
        """Test VoiceLibrary.add() voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256,
        )

        profile = library.add(
            voice_id="test_voice",
            name="Test Voice",
            embedding=embedding,
            gender="female",
        )

        assert profile.voice_id == "test_voice"
        assert len(library) == 1
        assert "test_voice" in library

    def test_library_add_duplicate_raises(self, tmp_path):
        """Test VoiceLibrary.add() raises on duplicate."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))

        library.add(voice_id="test", name="Test", embedding=embedding)

        with pytest.raises(ValueError, match="already exists"):
            library.add(voice_id="test", name="Test2", embedding=embedding)

    def test_library_get_voice(self, tmp_path):
        """Test VoiceLibrary.get() voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="test", name="Test", embedding=embedding)

        profile = library.get("test")
        assert profile is not None
        assert profile.voice_id == "test"

        # Non-existent returns None
        assert library.get("nonexistent") is None

    def test_library_get_or_raise(self, tmp_path):
        """Test VoiceLibrary.get_or_raise()."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="test", name="Test", embedding=embedding)

        profile = library.get_or_raise("test")
        assert profile.voice_id == "test"

        with pytest.raises(KeyError):
            library.get_or_raise("nonexistent")

    def test_library_remove_voice(self, tmp_path):
        """Test VoiceLibrary.remove() voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="test", name="Test", embedding=embedding)

        assert len(library) == 1

        result = library.remove("test")
        assert result is True
        assert len(library) == 0

        # Removing again returns False
        result = library.remove("test")
        assert result is False

    def test_library_list_all(self, tmp_path):
        """Test VoiceLibrary.list_all()."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="voice1", name="Voice 1", embedding=embedding)
        library.add(voice_id="voice2", name="Voice 2", embedding=embedding)

        all_profiles = library.list_all()
        assert len(all_profiles) == 2

    def test_library_list_ids(self, tmp_path):
        """Test VoiceLibrary.list_ids()."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="voice1", name="Voice 1", embedding=embedding)
        library.add(voice_id="voice2", name="Voice 2", embedding=embedding)

        ids = library.list_ids()
        assert "voice1" in ids
        assert "voice2" in ids

    def test_library_search_by_query(self, tmp_path):
        """Test VoiceLibrary.search() by query."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="alice", name="Alice Voice", embedding=embedding)
        library.add(voice_id="bob", name="Bob Voice", embedding=embedding)

        results = library.search(query="alice")
        assert len(results) == 1
        assert results[0].voice_id == "alice"

    def test_library_search_by_gender(self, tmp_path):
        """Test VoiceLibrary.search() by gender."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="alice", name="Alice", embedding=embedding, gender="female")
        library.add(voice_id="bob", name="Bob", embedding=embedding, gender="male")

        results = library.search(gender="female")
        assert len(results) == 1
        assert results[0].voice_id == "alice"

    def test_library_iteration(self, tmp_path):
        """Test VoiceLibrary iteration."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(library_path=tmp_path / "voices")

        embedding = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        library.add(voice_id="voice1", name="Voice 1", embedding=embedding)
        library.add(voice_id="voice2", name="Voice 2", embedding=embedding)

        ids = [p.voice_id for p in library]
        assert len(ids) == 2


class TestGetDefaultLibrary:
    """Tests for get_default_library() function."""

    def test_get_default_library_singleton(self):
        """Test get_default_library() returns singleton."""
        from voice_soundboard.cloning.library import get_default_library

        lib1 = get_default_library()
        lib2 = get_default_library()

        assert lib1 is lib2


# =============================================================================
# Module: conversion/__init__.py - Export Tests
# =============================================================================

class TestConversionModuleExports:
    """Tests for conversion module exports."""

    def test_voice_converter_exported(self):
        """Test VoiceConverter is exported."""
        from voice_soundboard.conversion import VoiceConverter

        assert VoiceConverter is not None

    def test_conversion_config_exported(self):
        """Test ConversionConfig is exported."""
        from voice_soundboard.conversion import ConversionConfig

        assert ConversionConfig is not None

    def test_latency_mode_exported(self):
        """Test LatencyMode is exported."""
        from voice_soundboard.conversion import LatencyMode

        assert LatencyMode is not None

    def test_streaming_converter_exported(self):
        """Test StreamingConverter is exported."""
        from voice_soundboard.conversion import StreamingConverter

        assert StreamingConverter is not None

    def test_audio_device_exported(self):
        """Test AudioDevice is exported."""
        from voice_soundboard.conversion import AudioDevice

        assert AudioDevice is not None

    def test_list_audio_devices_exported(self):
        """Test list_audio_devices is exported."""
        from voice_soundboard.conversion import list_audio_devices

        assert callable(list_audio_devices)

    def test_realtime_converter_exported(self):
        """Test RealtimeConverter is exported."""
        from voice_soundboard.conversion import RealtimeConverter

        assert RealtimeConverter is not None


# =============================================================================
# Module: llm/__init__.py - Export Tests
# =============================================================================

class TestLLMModuleExports:
    """Tests for LLM module exports."""

    def test_streaming_llm_speaker_exported(self):
        """Test StreamingLLMSpeaker is exported."""
        from voice_soundboard.llm import StreamingLLMSpeaker

        assert StreamingLLMSpeaker is not None

    def test_speech_pipeline_exported(self):
        """Test SpeechPipeline is exported."""
        from voice_soundboard.llm import SpeechPipeline

        assert SpeechPipeline is not None

    def test_llm_provider_exported(self):
        """Test LLMProvider is exported."""
        from voice_soundboard.llm import LLMProvider

        assert LLMProvider is not None

    def test_ollama_provider_exported(self):
        """Test OllamaProvider is exported."""
        from voice_soundboard.llm import OllamaProvider

        assert OllamaProvider is not None

    def test_interruption_handler_exported(self):
        """Test InterruptionHandler is exported."""
        from voice_soundboard.llm import InterruptionHandler

        assert InterruptionHandler is not None

    def test_conversation_manager_exported(self):
        """Test ConversationManager is exported."""
        from voice_soundboard.llm import ConversationManager

        assert ConversationManager is not None

    def test_message_role_exported(self):
        """Test MessageRole is exported."""
        from voice_soundboard.llm import MessageRole

        assert MessageRole is not None

    def test_context_aware_speaker_exported(self):
        """Test ContextAwareSpeaker is exported."""
        from voice_soundboard.llm import ContextAwareSpeaker

        assert ContextAwareSpeaker is not None


# =============================================================================
# Module: cloning/__init__.py - Export Tests
# =============================================================================

class TestCloningModuleExports:
    """Tests for cloning module exports."""

    def test_voice_embedding_exported(self):
        """Test VoiceEmbedding is exported."""
        from voice_soundboard.cloning import VoiceEmbedding

        assert VoiceEmbedding is not None

    def test_voice_extractor_exported(self):
        """Test VoiceExtractor is exported."""
        from voice_soundboard.cloning import VoiceExtractor

        assert VoiceExtractor is not None

    def test_voice_library_exported(self):
        """Test VoiceLibrary is exported."""
        from voice_soundboard.cloning import VoiceLibrary

        assert VoiceLibrary is not None

    def test_voice_cloner_exported(self):
        """Test VoiceCloner is exported."""
        from voice_soundboard.cloning import VoiceCloner

        assert VoiceCloner is not None

    def test_cross_language_cloner_exported(self):
        """Test CrossLanguageCloner is exported."""
        from voice_soundboard.cloning import CrossLanguageCloner

        assert CrossLanguageCloner is not None

    def test_emotion_timbre_separator_exported(self):
        """Test EmotionTimbreSeparator is exported."""
        from voice_soundboard.cloning import EmotionTimbreSeparator

        assert EmotionTimbreSeparator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
