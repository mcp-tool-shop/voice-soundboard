"""
Tests for voice cloning module.

Tests cover:
- Voice embedding extraction
- Voice library management
- VoiceCloner operations
- Emotion-timbre separation
- Cross-language support
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    import voice_soundboard.cloning.library as library_module
    import voice_soundboard.server as server_module

    # Reset global library
    library_module._default_library = None

    # Reset global voice cloner in server
    server_module._voice_cloner = None

    yield

    # Clean up after test
    library_module._default_library = None
    server_module._voice_cloner = None


class TestVoiceEmbedding:
    """Tests for VoiceEmbedding dataclass."""

    def test_embedding_creation(self):
        """TEST-EMB01: Create embedding with basic fields."""
        from voice_soundboard.cloning import VoiceEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(
            embedding=embedding,
            embedding_dim=256,
            source_duration_seconds=5.0,
        )

        assert ve.embedding.shape == (256,)
        assert ve.embedding_dim == 256
        assert ve.source_duration_seconds == 5.0
        assert len(ve.embedding_id) == 16  # Auto-generated hash

    def test_embedding_similarity_identical(self):
        """TEST-EMB02: Identical embeddings have similarity 1.0."""
        from voice_soundboard.cloning import VoiceEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        ve1 = VoiceEmbedding(embedding=embedding)
        ve2 = VoiceEmbedding(embedding=embedding.copy())

        assert abs(ve1.similarity(ve2) - 1.0) < 0.001

    def test_embedding_similarity_orthogonal(self):
        """TEST-EMB03: Orthogonal embeddings have similarity ~0."""
        from voice_soundboard.cloning import VoiceEmbedding

        e1 = np.zeros(256, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(256, dtype=np.float32)
        e2[1] = 1.0

        ve1 = VoiceEmbedding(embedding=e1)
        ve2 = VoiceEmbedding(embedding=e2)

        assert abs(ve1.similarity(ve2)) < 0.001

    def test_embedding_similarity_opposite(self):
        """TEST-EMB04: Opposite embeddings have similarity -1.0."""
        from voice_soundboard.cloning import VoiceEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        ve1 = VoiceEmbedding(embedding=embedding)
        ve2 = VoiceEmbedding(embedding=-embedding)

        assert abs(ve1.similarity(ve2) + 1.0) < 0.001

    def test_embedding_to_dict(self):
        """TEST-EMB05: Serialize embedding to dictionary."""
        from voice_soundboard.cloning import VoiceEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(
            embedding=embedding,
            source_path="/audio/sample.wav",
            source_duration_seconds=5.0,
        )

        data = ve.to_dict()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 256
        assert data["source_path"] == "/audio/sample.wav"

    def test_embedding_from_dict(self):
        """TEST-EMB06: Deserialize embedding from dictionary."""
        from voice_soundboard.cloning import VoiceEmbedding

        original = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_duration_seconds=7.0,
        )

        data = original.to_dict()
        restored = VoiceEmbedding.from_dict(data)

        assert np.allclose(original.embedding, restored.embedding)
        assert restored.source_duration_seconds == 7.0

    def test_embedding_save_load_npz(self):
        """TEST-EMB07: Save and load embedding in NPZ format."""
        from voice_soundboard.cloning import VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embedding.npz"

            original = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=4.5,
                quality_score=0.85,
            )

            original.save(path)
            assert path.exists()

            restored = VoiceEmbedding.load(path)
            assert np.allclose(original.embedding, restored.embedding)
            assert restored.source_duration_seconds == 4.5
            assert restored.quality_score == 0.85

    def test_embedding_save_load_json(self):
        """TEST-EMB08: Save and load embedding in JSON format."""
        from voice_soundboard.cloning import VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embedding.json"

            original = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=6.0,
            )

            original.save(path)
            assert path.exists()

            restored = VoiceEmbedding.load(path)
            assert np.allclose(original.embedding, restored.embedding, atol=1e-5)


class TestVoiceExtractor:
    """Tests for VoiceExtractor."""

    def test_extractor_mock_backend(self):
        """TEST-EXT01: Extract with mock backend."""
        from voice_soundboard.cloning import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds

        embedding = extractor.extract(audio, sample_rate=16000)

        assert embedding.embedding.shape == (256,)
        assert embedding.source_duration_seconds == 5.0
        assert embedding.extractor_backend == "mock"

    def test_extractor_embedding_dim(self):
        """TEST-EXT02: Embedding dimension matches backend."""
        from voice_soundboard.cloning import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        assert extractor.embedding_dim == 256

    def test_extractor_extract_from_segments(self):
        """TEST-EXT03: Extract from multiple segments."""
        from voice_soundboard.cloning import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds

        embeddings = extractor.extract_from_segments(
            audio, segment_seconds=3.0, sample_rate=16000
        )

        assert len(embeddings) >= 3  # At least 3 segments

    def test_extractor_average_embeddings(self):
        """TEST-EXT04: Average multiple embeddings."""
        from voice_soundboard.cloning import VoiceExtractor, ExtractorBackend, VoiceEmbedding

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        embeddings = [
            VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_duration_seconds=3.0,
            )
            for _ in range(3)
        ]

        averaged = extractor.average_embeddings(embeddings)

        assert averaged.embedding.shape == (256,)
        assert averaged.source_duration_seconds == 9.0  # Sum of durations
        # Averaged embedding should be normalized
        norm = np.linalg.norm(averaged.embedding)
        assert abs(norm - 1.0) < 0.01

    def test_extract_embedding_convenience(self):
        """TEST-EXT05: Convenience function extract_embedding."""
        from voice_soundboard.cloning import extract_embedding

        audio = np.random.randn(16000 * 3).astype(np.float32)
        embedding = extract_embedding(audio, backend="mock")

        assert embedding.embedding.shape == (256,)


class TestVoiceProfile:
    """Tests for VoiceProfile dataclass."""

    def test_profile_creation(self):
        """TEST-PRO01: Create voice profile."""
        from voice_soundboard.cloning import VoiceProfile

        profile = VoiceProfile(
            voice_id="test_voice",
            name="Test Voice",
            description="A test voice",
            gender="male",
            language="en",
            tags=["narrator", "deep"],
        )

        assert profile.voice_id == "test_voice"
        assert profile.name == "Test Voice"
        assert profile.gender == "male"
        assert "narrator" in profile.tags

    def test_profile_record_usage(self):
        """TEST-PRO02: Record voice usage."""
        from voice_soundboard.cloning import VoiceProfile

        profile = VoiceProfile(voice_id="test", name="Test")
        assert profile.usage_count == 0

        profile.record_usage()
        assert profile.usage_count == 1
        assert profile.last_used_at is not None

    def test_profile_created_date(self):
        """TEST-PRO03: Created date formatting."""
        from voice_soundboard.cloning import VoiceProfile

        profile = VoiceProfile(voice_id="test", name="Test")
        date_str = profile.created_date

        assert "-" in date_str  # YYYY-MM-DD format
        assert ":" in date_str  # HH:MM

    def test_profile_to_dict(self):
        """TEST-PRO04: Serialize profile to dictionary."""
        from voice_soundboard.cloning import VoiceProfile, VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            embedding=embedding,
            tags=["tag1", "tag2"],
        )

        data = profile.to_dict()
        assert data["voice_id"] == "test"
        assert "embedding" in data
        assert data["tags"] == ["tag1", "tag2"]

    def test_profile_from_dict(self):
        """TEST-PRO05: Deserialize profile from dictionary."""
        from voice_soundboard.cloning import VoiceProfile

        original = VoiceProfile(
            voice_id="restored",
            name="Restored Voice",
            gender="female",
            consent_given=True,
        )

        data = original.to_dict()
        restored = VoiceProfile.from_dict(data)

        assert restored.voice_id == "restored"
        assert restored.gender == "female"
        assert restored.consent_given is True


class TestVoiceLibrary:
    """Tests for VoiceLibrary management."""

    def test_library_add_voice(self):
        """TEST-LIB01: Add voice to library."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            profile = library.add(
                voice_id="test_voice",
                name="Test Voice",
                embedding=embedding,
            )

            assert profile.voice_id == "test_voice"
            assert "test_voice" in library

    def test_library_get_voice(self):
        """TEST-LIB02: Get voice from library."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            library.add("voice1", "Voice One", embedding)

            profile = library.get("voice1")
            assert profile is not None
            assert profile.name == "Voice One"

            missing = library.get("nonexistent")
            assert missing is None

    def test_library_remove_voice(self):
        """TEST-LIB03: Remove voice from library."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            library.add("to_delete", "Delete Me", embedding)
            assert "to_delete" in library

            result = library.remove("to_delete")
            assert result is True
            assert "to_delete" not in library

    def test_library_update_voice(self):
        """TEST-LIB04: Update voice in library."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            library.add("updateable", "Original Name", embedding)

            updated = library.update("updateable", name="New Name", tags=["updated"])
            assert updated.name == "New Name"
            assert "updated" in updated.tags

    def test_library_search(self):
        """TEST-LIB05: Search voices in library."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            library.add("male1", "Male Voice 1", embedding, gender="male", tags=["deep"])
            library.add("female1", "Female Voice 1", embedding, gender="female", tags=["soft"])
            library.add("male2", "Male Voice 2", embedding, gender="male", tags=["narrator"])

            # Search by gender
            males = library.search(gender="male")
            assert len(males) == 2

            # Search by tags
            narrators = library.search(tags=["narrator"])
            assert len(narrators) == 1

            # Search by query
            voice1 = library.search(query="voice 1")
            assert len(voice1) == 2

    def test_library_list_all(self):
        """TEST-LIB06: List all voices."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            library.add("v1", "Voice 1", embedding)
            library.add("v2", "Voice 2", embedding)
            library.add("v3", "Voice 3", embedding)

            all_voices = library.list_all()
            assert len(all_voices) == 3

    def test_library_persistence(self):
        """TEST-LIB07: Library persists across instances."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create and add
            lib1 = VoiceLibrary(library_path=path)
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )
            lib1.add("persistent", "Persistent Voice", embedding)

            # New instance should find it
            lib2 = VoiceLibrary(library_path=path)
            profile = lib2.get("persistent")
            assert profile is not None
            assert profile.name == "Persistent Voice"

    def test_library_duplicate_id_error(self):
        """TEST-LIB08: Duplicate voice ID raises error."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32)
            )

            library.add("unique", "First", embedding)

            with pytest.raises(ValueError):
                library.add("unique", "Second", embedding)

    def test_library_find_similar(self):
        """TEST-LIB09: Find similar voices."""
        from voice_soundboard.cloning import VoiceLibrary, VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            base = np.random.randn(256).astype(np.float32)
            base = base / np.linalg.norm(base)

            # Add some voices with varying similarity
            library.add(
                "similar1", "Similar 1",
                VoiceEmbedding(embedding=base + np.random.randn(256).astype(np.float32) * 0.1)
            )
            library.add(
                "similar2", "Similar 2",
                VoiceEmbedding(embedding=base + np.random.randn(256).astype(np.float32) * 0.2)
            )
            library.add(
                "different", "Different",
                VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
            )

            query = VoiceEmbedding(embedding=base)
            similar = library.find_similar(query, top_k=2, min_similarity=0.5)

            assert len(similar) <= 2


class TestVoiceCloner:
    """Tests for VoiceCloner high-level API."""

    def test_cloner_clone_requires_consent(self):
        """TEST-CLO01: Clone requires consent."""
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))
            cloner = VoiceCloner(library=library)

            audio = np.random.randn(16000 * 5).astype(np.float32)

            result = cloner.clone(
                audio=audio,
                voice_id="test",
                consent_given=False,
            )

            assert not result.success
            assert "consent" in result.error.lower()

    def test_cloner_clone_success(self):
        """TEST-CLO02: Successful voice clone."""
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary
        import uuid

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique path to avoid conflicts
            unique_path = Path(tmpdir) / f"test_{uuid.uuid4().hex[:8]}"
            library = VoiceLibrary(library_path=unique_path)
            cloner = VoiceCloner(library=library)

            audio = np.random.randn(16000 * 5).astype(np.float32)
            voice_id = f"my_voice_{uuid.uuid4().hex[:8]}"

            result = cloner.clone(
                audio=audio,
                voice_id=voice_id,
                name="My Voice",
                consent_given=True,
                consent_notes="Self recording",
                tags=["personal"],
            )

            assert result.success, f"Clone failed: {result.error}"
            assert result.voice_id == voice_id
            assert result.profile is not None
            assert result.quality_score > 0

    def test_cloner_clone_duplicate_fails(self):
        """TEST-CLO03: Cloning duplicate ID fails."""
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))
            cloner = VoiceCloner(library=library)

            audio = np.random.randn(16000 * 5).astype(np.float32)

            # First clone
            cloner.clone(audio=audio, voice_id="dup", consent_given=True)

            # Second clone with same ID
            result = cloner.clone(audio=audio, voice_id="dup", consent_given=True)

            assert not result.success
            assert "exists" in result.error.lower()

    def test_cloner_clone_quick(self):
        """TEST-CLO04: Quick clone without saving."""
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))
            cloner = VoiceCloner(library=library)

            audio = np.random.randn(16000 * 5).astype(np.float32)

            result = cloner.clone_quick(audio, "Quick Voice")

            assert result.success
            assert result.embedding is not None
            # Quick clone should NOT save to library
            assert len(library) == 0

    def test_cloner_validate_audio(self):
        """TEST-CLO05: Validate audio for cloning."""
        from voice_soundboard.cloning import VoiceCloner

        cloner = VoiceCloner()

        # Valid audio (5 seconds)
        good_audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.validate_audio(good_audio)
        assert result["is_valid"]
        assert result["duration_seconds"] == 5.0

    def test_cloner_validate_audio_too_short(self):
        """TEST-CLO06: Validate rejects short audio."""
        from voice_soundboard.cloning import VoiceCloner, CloningConfig

        config = CloningConfig(min_audio_seconds=3.0)
        cloner = VoiceCloner(config=config)

        # Too short (0.5 seconds)
        short_audio = np.random.randn(16000 // 2).astype(np.float32)
        result = cloner.validate_audio(short_audio)

        assert not result["is_valid"]
        assert any("short" in issue.lower() for issue in result["issues"])

    def test_cloner_list_voices(self):
        """TEST-CLO07: List voices through cloner."""
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary
        import uuid

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique path to avoid conflicts with other tests
            unique_path = Path(tmpdir) / f"list_test_{uuid.uuid4().hex[:8]}"
            library = VoiceLibrary(library_path=unique_path)
            cloner = VoiceCloner(library=library)

            audio = np.random.randn(16000 * 5).astype(np.float32)
            v1 = f"list_v1_{uuid.uuid4().hex[:8]}"
            v2 = f"list_v2_{uuid.uuid4().hex[:8]}"
            cloner.clone(audio=audio, voice_id=v1, consent_given=True, gender="male")
            cloner.clone(audio=audio, voice_id=v2, consent_given=True, gender="female")

            all_voices = cloner.list_voices()
            assert len(all_voices) == 2

            males = cloner.list_voices(gender="male")
            assert len(males) == 1

    def test_cloner_delete_voice(self):
        """TEST-CLO08: Delete voice through cloner."""
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))
            cloner = VoiceCloner(library=library)

            audio = np.random.randn(16000 * 5).astype(np.float32)
            cloner.clone(audio=audio, voice_id="deleteme", consent_given=True)

            assert cloner.get_voice("deleteme") is not None

            deleted = cloner.delete_voice("deleteme")
            assert deleted
            assert cloner.get_voice("deleteme") is None


class TestEmotionTimbreSeparation:
    """Tests for emotion-timbre separation."""

    def test_separator_separate(self):
        """TEST-SEP01: Separate embedding into timbre and emotion."""
        from voice_soundboard.cloning import EmotionTimbreSeparator, VoiceEmbedding

        separator = EmotionTimbreSeparator()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        separated = separator.separate(embedding)

        assert separated.timbre.embedding.shape == (256,)
        assert separated.emotion.embedding.shape == (64,)  # Default emotion dim
        assert separated.original_embedding is not None

    def test_separator_recombine(self):
        """TEST-SEP02: Recombine timbre and emotion."""
        from voice_soundboard.cloning import EmotionTimbreSeparator, VoiceEmbedding

        separator = EmotionTimbreSeparator()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        separated = separator.separate(embedding)
        recombined = separated.recombine()

        assert recombined.shape == (256,)
        # Recombined should be normalized
        norm = np.linalg.norm(recombined)
        assert abs(norm - 1.0) < 0.01

    def test_separator_get_emotion_preset(self):
        """TEST-SEP03: Get emotion preset."""
        from voice_soundboard.cloning import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        happy = separator.get_emotion_preset("happy")
        assert happy.emotion_label == "happy"
        assert happy.valence > 0  # Happy is positive

        sad = separator.get_emotion_preset("sad")
        assert sad.emotion_label == "sad"
        assert sad.valence < 0  # Sad is negative

    def test_separator_transfer_emotion(self):
        """TEST-SEP04: Transfer emotion to voice."""
        from voice_soundboard.cloning import EmotionTimbreSeparator, VoiceEmbedding

        separator = EmotionTimbreSeparator()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        # Transfer "excited" emotion
        result = separator.transfer_emotion(embedding, "excited")

        assert result.shape == (256,)

    def test_separator_transfer_with_intensity(self):
        """TEST-SEP05: Transfer emotion with intensity - tests that shapes are correct."""
        from voice_soundboard.cloning import EmotionTimbreSeparator, VoiceEmbedding

        separator = EmotionTimbreSeparator()

        np.random.seed(42)
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        # Test that transfer works with various intensities
        low = separator.transfer_emotion(embedding, "happy", intensity=0.5)
        high = separator.transfer_emotion(embedding, "happy", intensity=1.5)

        # Both should produce valid 256-dim vectors
        assert low.shape == (256,)
        assert high.shape == (256,)

        # Both should be normalized (or close to it)
        assert 0.5 < np.linalg.norm(low) < 1.5
        assert 0.5 < np.linalg.norm(high) < 1.5

    def test_separator_list_presets(self):
        """TEST-SEP06: List emotion presets."""
        from voice_soundboard.cloning import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        presets = separator.list_emotion_presets()

        assert "neutral" in presets
        assert "happy" in presets
        assert "sad" in presets
        assert "angry" in presets

    def test_separator_blend_emotions(self):
        """TEST-SEP07: Blend multiple emotions."""
        from voice_soundboard.cloning import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        happy = separator.get_emotion_preset("happy")
        sad = separator.get_emotion_preset("sad")

        blended = separator.blend_emotions([
            (happy, 0.5),
            (sad, 0.5),
        ])

        # Blended should be between happy and sad
        assert blended.emotion_label == "blended"
        # Valence should be around 0 (happy positive + sad negative)
        if blended.valence is not None:
            assert abs(blended.valence) < 0.5

    def test_separate_voice_convenience(self):
        """TEST-SEP08: Convenience function separate_voice."""
        from voice_soundboard.cloning import separate_voice, VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        separated = separate_voice(embedding)
        assert separated.timbre is not None
        assert separated.emotion is not None

    def test_transfer_emotion_convenience(self):
        """TEST-SEP09: Convenience function transfer_emotion."""
        from voice_soundboard.cloning import transfer_emotion, VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        result = transfer_emotion(embedding, "calm")
        assert result.shape == (256,)


class TestCrossLanguage:
    """Tests for cross-language voice cloning."""

    def test_crosslang_list_languages(self):
        """TEST-XLN01: List supported languages."""
        from voice_soundboard.cloning import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        languages = cloner.list_supported_languages()

        assert len(languages) > 0
        # Check for common languages
        codes = [lang["code"] for lang in languages]
        assert "en" in codes
        assert "es" in codes
        assert "zh" in codes

    def test_crosslang_is_supported(self):
        """TEST-XLN02: Check language support."""
        from voice_soundboard.cloning import CrossLanguageCloner

        cloner = CrossLanguageCloner()

        assert cloner.is_language_supported("en")
        assert cloner.is_language_supported("zh")
        assert not cloner.is_language_supported("xyz")

    def test_crosslang_compatibility(self):
        """TEST-XLN03: Check language compatibility."""
        from voice_soundboard.cloning import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")

        # Same language should be highly compatible
        en_en = cloner.get_language_pair_compatibility("en", "en")
        assert en_en["compatible"]
        assert en_en["expected_quality"] == 1.0

        # Cross-language
        en_es = cloner.get_language_pair_compatibility("en", "es")
        assert en_es["compatible"]

    def test_crosslang_same_family_bonus(self):
        """TEST-XLN04: Same language family gets bonus."""
        from voice_soundboard.cloning import CrossLanguageCloner

        cloner = CrossLanguageCloner()

        # Romance languages
        es_fr = cloner.get_language_pair_compatibility("es", "fr")
        assert es_fr["same_language_family"]

        # Different families
        en_zh = cloner.get_language_pair_compatibility("en", "zh")
        assert not en_zh["same_language_family"]

    def test_crosslang_tonal_warning(self):
        """TEST-XLN05: Tonal language gets warning."""
        from voice_soundboard.cloning import CrossLanguageCloner

        cloner = CrossLanguageCloner()

        # English to Chinese (tonal)
        en_zh = cloner.get_language_pair_compatibility("en", "zh")
        assert any("tonal" in issue.lower() for issue in en_zh["phonetic_issues"])

    def test_crosslang_recommendations(self):
        """TEST-XLN06: Get recommendations for language pair."""
        from voice_soundboard.cloning import CrossLanguageCloner

        cloner = CrossLanguageCloner()

        en_ja = cloner.get_language_pair_compatibility("en", "ja")
        assert "recommendations" in en_ja
        assert len(en_ja["recommendations"]) > 0

    def test_crosslang_prepare_embedding(self):
        """TEST-XLN07: Prepare embedding for target language."""
        from voice_soundboard.cloning import CrossLanguageCloner, VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32)
        )

        prepared, metadata = cloner.prepare_embedding_for_language(embedding, "es")

        assert "source_language" in metadata
        assert "target_language" in metadata
        assert "recommended_speed_multiplier" in metadata

    def test_crosslang_estimate_quality(self):
        """TEST-XLN08: Estimate quality for target language."""
        from voice_soundboard.cloning import CrossLanguageCloner, VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")

        good_embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            quality_score=0.9,
            source_duration_seconds=10.0,
        )

        quality = cloner.estimate_quality(good_embedding, "es")
        assert 0 <= quality <= 1

    def test_detect_language(self):
        """TEST-XLN09: Detect language from text."""
        from voice_soundboard.cloning import detect_language

        # English (default for Latin)
        assert detect_language("Hello world") == "en"

        # Chinese
        assert detect_language("你好世界") == "zh"

        # Japanese
        assert detect_language("こんにちは") == "ja"

        # Korean
        assert detect_language("안녕하세요") == "ko"

    def test_supported_languages_constant(self):
        """TEST-XLN10: SUPPORTED_LANGUAGES constant exists."""
        from voice_soundboard.cloning import SUPPORTED_LANGUAGES

        assert "en" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES


class TestEmotionStyle:
    """Tests for EmotionStyle enum."""

    def test_emotion_style_values(self):
        """TEST-EST01: EmotionStyle has expected values."""
        from voice_soundboard.cloning import EmotionStyle

        assert EmotionStyle.NEUTRAL.value == "neutral"
        assert EmotionStyle.HAPPY.value == "happy"
        assert EmotionStyle.SAD.value == "sad"
        assert EmotionStyle.ANGRY.value == "angry"

    def test_emotion_style_all_present(self):
        """TEST-EST02: All expected styles present."""
        from voice_soundboard.cloning import EmotionStyle

        styles = [e.value for e in EmotionStyle]
        expected = ["neutral", "happy", "sad", "angry", "fearful",
                   "surprised", "disgusted", "calm", "excited", "tender"]

        for exp in expected:
            assert exp in styles


class TestTimbreEmbedding:
    """Tests for TimbreEmbedding dataclass."""

    def test_timbre_embedding_creation(self):
        """TEST-TIM01: Create timbre embedding."""
        from voice_soundboard.cloning import TimbreEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        te = TimbreEmbedding(embedding=embedding)

        assert te.embedding.shape == (256,)
        assert te.embedding_dim == 256

    def test_timbre_to_dict(self):
        """TEST-TIM02: Serialize timbre to dict."""
        from voice_soundboard.cloning import TimbreEmbedding

        te = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_voice_id="test123",
        )

        data = te.to_dict()
        assert "embedding" in data
        assert data["source_voice_id"] == "test123"

    def test_timbre_from_dict(self):
        """TEST-TIM03: Deserialize timbre from dict."""
        from voice_soundboard.cloning import TimbreEmbedding

        original = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            separation_quality=0.95,
        )

        restored = TimbreEmbedding.from_dict(original.to_dict())
        assert np.allclose(original.embedding, restored.embedding)
        assert restored.separation_quality == 0.95


class TestEmotionEmbedding:
    """Tests for EmotionEmbedding dataclass."""

    def test_emotion_embedding_creation(self):
        """TEST-EMO01: Create emotion embedding."""
        from voice_soundboard.cloning import EmotionEmbedding

        embedding = np.random.randn(64).astype(np.float32)
        ee = EmotionEmbedding(
            embedding=embedding,
            emotion_label="happy",
            valence=0.8,
            arousal=0.6,
        )

        assert ee.embedding.shape == (64,)
        assert ee.emotion_label == "happy"
        assert ee.valence == 0.8

    def test_emotion_to_dict(self):
        """TEST-EMO02: Serialize emotion to dict."""
        from voice_soundboard.cloning import EmotionEmbedding

        ee = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="excited",
            emotion_intensity=0.9,
        )

        data = ee.to_dict()
        assert data["emotion_label"] == "excited"
        assert data["emotion_intensity"] == 0.9


class TestCloningConfig:
    """Tests for CloningConfig."""

    def test_config_defaults(self):
        """TEST-CFG01: Config has sensible defaults."""
        from voice_soundboard.cloning import CloningConfig

        config = CloningConfig()

        assert config.min_audio_seconds == 1.0
        assert config.max_audio_seconds == 30.0
        assert config.optimal_audio_seconds == 5.0
        assert config.require_consent is True

    def test_config_custom(self):
        """TEST-CFG02: Config accepts custom values."""
        from voice_soundboard.cloning import CloningConfig

        config = CloningConfig(
            min_audio_seconds=2.0,
            require_consent=False,
            add_watermark=True,
        )

        assert config.min_audio_seconds == 2.0
        assert config.require_consent is False
        assert config.add_watermark is True


class TestCloningResult:
    """Tests for CloningResult."""

    def test_result_success(self):
        """TEST-RES01: Success result fields."""
        from voice_soundboard.cloning import CloningResult

        result = CloningResult(
            success=True,
            voice_id="test",
            quality_score=0.9,
            extraction_time=1.5,
        )

        assert result.success
        assert result.voice_id == "test"
        assert result.error is None

    def test_result_failure(self):
        """TEST-RES02: Failure result fields."""
        from voice_soundboard.cloning import CloningResult

        result = CloningResult(
            success=False,
            error="Audio too short",
        )

        assert not result.success
        assert result.error == "Audio too short"

    def test_result_warnings(self):
        """TEST-RES03: Result with warnings."""
        from voice_soundboard.cloning import CloningResult

        result = CloningResult(
            success=True,
            voice_id="test",
            warnings=["Low quality audio"],
            recommendations=["Use cleaner audio"],
        )

        assert result.success
        assert len(result.warnings) == 1
        assert len(result.recommendations) == 1


class TestExtractorBackend:
    """Tests for ExtractorBackend enum."""

    def test_backend_values(self):
        """TEST-BKD01: ExtractorBackend has expected values."""
        from voice_soundboard.cloning import ExtractorBackend

        assert ExtractorBackend.MOCK.value == "mock"
        assert ExtractorBackend.RESEMBLYZER.value == "resemblyzer"
        assert ExtractorBackend.SPEECHBRAIN.value == "speechbrain"
