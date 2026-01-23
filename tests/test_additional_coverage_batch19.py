"""
Batch 19: Additional Coverage Tests for Cloning Modules
- cloning/library.py: VoiceProfile, VoiceLibrary, voice management
- cloning/separation.py: EmotionTimbreSeparator, emotion transfer
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest


# ==============================================================================
# Tests for cloning/library.py
# ==============================================================================

class TestVoiceProfile:
    """Tests for VoiceProfile dataclass."""

    def test_default_values(self):
        """Test VoiceProfile default values."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(voice_id="test", name="Test Voice")
        assert profile.voice_id == "test"
        assert profile.name == "Test Voice"
        assert profile.description == ""
        assert profile.embedding is None
        assert profile.language == "en"
        assert profile.quality_rating == 1.0
        assert profile.usage_count == 0
        assert profile.consent_given is False

    def test_full_profile(self):
        """Test VoiceProfile with all fields."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(
            voice_id="voice1",
            name="Voice One",
            description="A test voice",
            gender="male",
            age_range="adult",
            accent="american",
            language="en",
            default_speed=1.2,
            default_emotion="happy",
            tags=["test", "sample"],
            consent_given=True,
            consent_notes="User provided consent",
        )
        assert profile.gender == "male"
        assert profile.age_range == "adult"
        assert profile.tags == ["test", "sample"]
        assert profile.consent_given is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            tags=["a", "b"],
        )
        data = profile.to_dict()
        assert data["voice_id"] == "test"
        assert data["name"] == "Test"
        assert data["tags"] == ["a", "b"]

    def test_to_dict_with_embedding(self):
        """Test serialization with embedding."""
        from voice_soundboard.cloning.library import VoiceProfile
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/path/to/audio.wav",
        )
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            embedding=embedding,
        )
        data = profile.to_dict()
        assert "embedding" in data
        assert isinstance(data["embedding"], dict)

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        from voice_soundboard.cloning.library import VoiceProfile

        data = {
            "voice_id": "test",
            "name": "Test Voice",
            "description": "A test",
            "tags": ["tag1"],
            "language": "en",
            "quality_rating": 0.9,
            "usage_count": 5,
            "consent_given": True,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        profile = VoiceProfile.from_dict(data)
        assert profile.voice_id == "test"
        assert profile.name == "Test Voice"
        assert profile.tags == ["tag1"]
        assert profile.usage_count == 5

    def test_from_dict_with_embedding(self):
        """Test deserialization with embedding."""
        from voice_soundboard.cloning.library import VoiceProfile

        data = {
            "voice_id": "test",
            "name": "Test",
            "embedding": {
                "embedding": np.random.randn(256).tolist(),
                "embedding_dim": 256,
                "source_path": "/path/audio.wav",
                "source_duration_seconds": 5.0,
            },
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        profile = VoiceProfile.from_dict(data)
        assert profile.embedding is not None
        assert profile.embedding.embedding_dim == 256

    def test_created_date_property(self):
        """Test created_date property."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            created_at=1700000000.0,
        )
        date_str = profile.created_date
        assert isinstance(date_str, str)
        assert "-" in date_str  # Date format

    def test_record_usage(self):
        """Test recording usage."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(voice_id="test", name="Test")
        assert profile.usage_count == 0
        assert profile.last_used_at is None

        profile.record_usage()
        assert profile.usage_count == 1
        assert profile.last_used_at is not None

        profile.record_usage()
        assert profile.usage_count == 2


class TestVoiceLibrary:
    """Tests for VoiceLibrary class."""

    @pytest.fixture
    def temp_library_dir(self):
        """Create a temporary library directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_initialization(self, temp_library_dir):
        """Test VoiceLibrary initialization."""
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(temp_library_dir)
        assert library.library_path == temp_library_dir
        assert not library._loaded

    def test_initialization_default_path(self):
        """Test VoiceLibrary with default path."""
        from voice_soundboard.cloning.library import VoiceLibrary, DEFAULT_LIBRARY_PATH

        library = VoiceLibrary()
        assert library.library_path == DEFAULT_LIBRARY_PATH

    def test_load_creates_directory(self, temp_library_dir):
        """Test load creates library directory."""
        from voice_soundboard.cloning.library import VoiceLibrary

        library_path = temp_library_dir / "new_library"
        library = VoiceLibrary(library_path)
        library.load()
        assert library_path.exists()

    def test_load_with_existing_index(self, temp_library_dir):
        """Test loading with existing index."""
        from voice_soundboard.cloning.library import VoiceLibrary

        # Create an index file
        index_path = temp_library_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump({"voices": []}, f)

        library = VoiceLibrary(temp_library_dir)
        library.load()
        assert library._loaded

    def test_load_with_corrupted_index(self, temp_library_dir):
        """Test loading with corrupted index."""
        from voice_soundboard.cloning.library import VoiceLibrary

        index_path = temp_library_dir / "index.json"
        with open(index_path, "w") as f:
            f.write("invalid json {{{")

        library = VoiceLibrary(temp_library_dir)
        library.load()  # Should handle gracefully
        assert library._loaded

    def test_save(self, temp_library_dir):
        """Test saving library."""
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(temp_library_dir)
        library.load()
        library.save()

        index_path = temp_library_dir / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            data = json.load(f)
        assert "voices" in data
        assert "version" in data

    def test_add_voice(self, temp_library_dir):
        """Test adding a voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
            source_duration_seconds=5.0,
        )

        profile = library.add(
            voice_id="test_voice",
            name="Test Voice",
            embedding=embedding,
            description="A test voice",
            tags=["test"],
        )

        assert profile.voice_id == "test_voice"
        assert profile.name == "Test Voice"
        assert "test_voice" in library

    def test_add_duplicate_raises(self, temp_library_dir):
        """Test adding duplicate voice raises error."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        library.add(voice_id="test", name="Test", embedding=embedding)

        with pytest.raises(ValueError, match="already exists"):
            library.add(voice_id="test", name="Test 2", embedding=embedding)

    def test_get(self, temp_library_dir):
        """Test getting a voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(voice_id="test", name="Test", embedding=embedding)

        profile = library.get("test")
        assert profile is not None
        assert profile.voice_id == "test"

        # Test non-existent
        assert library.get("nonexistent") is None

    def test_get_or_raise(self, temp_library_dir):
        """Test get_or_raise method."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(voice_id="test", name="Test", embedding=embedding)

        profile = library.get_or_raise("test")
        assert profile.voice_id == "test"

        with pytest.raises(KeyError):
            library.get_or_raise("nonexistent")

    def test_update(self, temp_library_dir):
        """Test updating a voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(voice_id="test", name="Test", embedding=embedding)

        updated = library.update("test", name="Updated Name", description="New desc")
        assert updated.name == "Updated Name"
        assert updated.description == "New desc"

    def test_update_nonexistent_raises(self, temp_library_dir):
        """Test updating non-existent voice raises error."""
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(temp_library_dir)
        library.load()

        with pytest.raises(KeyError):
            library.update("nonexistent", name="New Name")

    def test_remove(self, temp_library_dir):
        """Test removing a voice."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(voice_id="test", name="Test", embedding=embedding)

        assert "test" in library
        result = library.remove("test")
        assert result is True
        assert "test" not in library

    def test_remove_nonexistent(self, temp_library_dir):
        """Test removing non-existent voice."""
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(temp_library_dir)
        library.load()

        result = library.remove("nonexistent")
        assert result is False

    def test_list_all(self, temp_library_dir):
        """Test listing all voices."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        for i in range(3):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=f"/fake/path{i}.wav",
            )
            library.add(voice_id=f"voice{i}", name=f"Voice {i}", embedding=embedding)

        all_voices = library.list_all()
        assert len(all_voices) == 3

    def test_list_ids(self, temp_library_dir):
        """Test listing voice IDs."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        for i in range(3):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=f"/fake/path{i}.wav",
            )
            library.add(voice_id=f"voice{i}", name=f"Voice {i}", embedding=embedding)

        ids = library.list_ids()
        assert len(ids) == 3
        assert "voice0" in ids

    def test_search_by_query(self, temp_library_dir):
        """Test searching by text query."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(
            voice_id="alice",
            name="Alice Voice",
            description="A friendly female voice",
            embedding=embedding,
        )

        results = library.search(query="alice")
        assert len(results) == 1
        assert results[0].name == "Alice Voice"

        results = library.search(query="friendly")
        assert len(results) == 1

        results = library.search(query="nonexistent")
        assert len(results) == 0

    def test_search_by_tags(self, temp_library_dir):
        """Test searching by tags."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(
            voice_id="voice1",
            name="Voice 1",
            embedding=embedding,
            tags=["english", "female"],
        )

        results = library.search(tags=["english"])
        assert len(results) == 1

        results = library.search(tags=["spanish"])
        assert len(results) == 0

    def test_search_by_gender(self, temp_library_dir):
        """Test searching by gender."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(
            voice_id="voice1",
            name="Voice 1",
            embedding=embedding,
            gender="female",
        )

        results = library.search(gender="female")
        assert len(results) == 1

        results = library.search(gender="male")
        assert len(results) == 0

    def test_search_by_language(self, temp_library_dir):
        """Test searching by language."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(
            voice_id="voice1",
            name="Voice 1",
            embedding=embedding,
            language="fr",
        )

        results = library.search(language="fr")
        assert len(results) == 1

        results = library.search(language="en")
        assert len(results) == 0

    def test_search_by_quality(self, temp_library_dir):
        """Test searching by minimum quality."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        for i, quality in enumerate([0.5, 0.8, 0.9]):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=f"/fake/path{i}.wav",
            )
            library.add(
                voice_id=f"voice{i}",
                name=f"Voice {i}",
                embedding=embedding,
                quality_rating=quality,
            )

        results = library.search(min_quality=0.7)
        assert len(results) == 2

        results = library.search(min_quality=0.85)
        assert len(results) == 1

    def test_find_similar(self, temp_library_dir):
        """Test finding similar voices."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        # Add some voices
        base_embedding = np.random.randn(256).astype(np.float32)
        base_embedding /= np.linalg.norm(base_embedding)

        for i in range(3):
            # Create slightly different embeddings
            noise = np.random.randn(256).astype(np.float32) * 0.1
            emb_vector = base_embedding + noise
            emb_vector /= np.linalg.norm(emb_vector)

            embedding = VoiceEmbedding(
                embedding=emb_vector,
                source_path=f"/fake/path{i}.wav",
            )
            library.add(
                voice_id=f"voice{i}",
                name=f"Voice {i}",
                embedding=embedding,
            )

        # Search for similar
        search_embedding = VoiceEmbedding(
            embedding=base_embedding,
            source_path="/fake/search.wav",
        )

        results = library.find_similar(search_embedding, top_k=2, min_similarity=0.0)
        assert len(results) <= 2
        if len(results) > 0:
            assert results[0][1] >= results[-1][1]  # Sorted by similarity

    def test_len(self, temp_library_dir):
        """Test __len__ method."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(voice_id="test", name="Test", embedding=embedding)

        assert len(library) == 1

    def test_contains(self, temp_library_dir):
        """Test __contains__ method."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        library.add(voice_id="test", name="Test", embedding=embedding)

        assert "test" in library
        assert "nonexistent" not in library

    def test_iter(self, temp_library_dir):
        """Test __iter__ method."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)

        for i in range(3):
            embedding = VoiceEmbedding(
                embedding=np.random.randn(256).astype(np.float32),
                source_path=f"/fake/path{i}.wav",
            )
            library.add(voice_id=f"voice{i}", name=f"Voice {i}", embedding=embedding)

        count = 0
        for profile in library:
            count += 1
            assert profile.voice_id.startswith("voice")
        assert count == 3


class TestGetDefaultLibrary:
    """Tests for get_default_library function."""

    def test_returns_library(self):
        """Test get_default_library returns a library."""
        import voice_soundboard.cloning.library as lib_module

        # Reset global
        lib_module._default_library = None

        library = lib_module.get_default_library()
        assert library is not None
        assert isinstance(library, lib_module.VoiceLibrary)

    def test_returns_same_instance(self):
        """Test get_default_library returns same instance."""
        import voice_soundboard.cloning.library as lib_module

        lib_module._default_library = None

        lib1 = lib_module.get_default_library()
        lib2 = lib_module.get_default_library()
        assert lib1 is lib2


# ==============================================================================
# Tests for cloning/separation.py
# ==============================================================================

class TestEmotionStyle:
    """Tests for EmotionStyle enum."""

    def test_emotion_values(self):
        """Test EmotionStyle enum values."""
        from voice_soundboard.cloning.separation import EmotionStyle

        assert EmotionStyle.NEUTRAL.value == "neutral"
        assert EmotionStyle.HAPPY.value == "happy"
        assert EmotionStyle.SAD.value == "sad"
        assert EmotionStyle.ANGRY.value == "angry"

    def test_all_emotions(self):
        """Test all emotions are accessible."""
        from voice_soundboard.cloning.separation import EmotionStyle

        emotions = list(EmotionStyle)
        assert len(emotions) == 10


class TestTimbreEmbedding:
    """Tests for TimbreEmbedding dataclass."""

    def test_creation(self):
        """Test TimbreEmbedding creation."""
        from voice_soundboard.cloning.separation import TimbreEmbedding

        embedding = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
        )
        assert embedding.embedding_dim == 256
        assert embedding.separation_quality == 1.0

    def test_to_dict(self):
        """Test serialization."""
        from voice_soundboard.cloning.separation import TimbreEmbedding

        embedding = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_voice_id="voice1",
        )
        data = embedding.to_dict()
        assert "embedding" in data
        assert data["source_voice_id"] == "voice1"

    def test_from_dict(self):
        """Test deserialization."""
        from voice_soundboard.cloning.separation import TimbreEmbedding

        data = {
            "embedding": np.random.randn(256).tolist(),
            "embedding_dim": 256,
            "source_voice_id": "voice1",
            "source_embedding_id": "emb1",
            "separation_quality": 0.9,
        }
        embedding = TimbreEmbedding.from_dict(data)
        assert embedding.source_voice_id == "voice1"
        assert embedding.separation_quality == 0.9


class TestEmotionEmbedding:
    """Tests for EmotionEmbedding dataclass."""

    def test_creation(self):
        """Test EmotionEmbedding creation."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        embedding = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="happy",
        )
        assert embedding.embedding_dim == 64
        assert embedding.emotion_label == "happy"
        assert embedding.emotion_intensity == 1.0

    def test_with_vad(self):
        """Test EmotionEmbedding with VAD values."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        embedding = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            valence=0.8,
            arousal=0.6,
            dominance=0.7,
        )
        assert embedding.valence == 0.8
        assert embedding.arousal == 0.6
        assert embedding.dominance == 0.7

    def test_to_dict(self):
        """Test serialization."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        embedding = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="happy",
            valence=0.8,
        )
        data = embedding.to_dict()
        assert data["emotion_label"] == "happy"
        assert data["valence"] == 0.8

    def test_from_dict(self):
        """Test deserialization."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        data = {
            "embedding": np.random.randn(64).tolist(),
            "embedding_dim": 64,
            "emotion_label": "sad",
            "emotion_intensity": 0.8,
            "valence": -0.6,
            "arousal": 0.3,
            "dominance": 0.3,
            "source_path": None,
        }
        embedding = EmotionEmbedding.from_dict(data)
        assert embedding.emotion_label == "sad"
        assert embedding.valence == -0.6


class TestSeparatedVoice:
    """Tests for SeparatedVoice dataclass."""

    def test_creation(self):
        """Test SeparatedVoice creation."""
        from voice_soundboard.cloning.separation import (
            SeparatedVoice,
            TimbreEmbedding,
            EmotionEmbedding,
        )

        timbre = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
        )
        emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
        )
        separated = SeparatedVoice(timbre=timbre, emotion=emotion)
        assert separated.timbre is timbre
        assert separated.emotion is emotion
        assert separated.reconstruction_loss == 0.0

    def test_recombine(self):
        """Test recombine method."""
        from voice_soundboard.cloning.separation import (
            SeparatedVoice,
            TimbreEmbedding,
            EmotionEmbedding,
        )

        timbre = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
        )
        emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
        )
        separated = SeparatedVoice(timbre=timbre, emotion=emotion)

        combined = separated.recombine()
        assert combined.shape == (256,)
        # Should be normalized
        assert np.abs(np.linalg.norm(combined) - 1.0) < 0.01

    def test_with_emotion(self):
        """Test with_emotion method."""
        from voice_soundboard.cloning.separation import (
            SeparatedVoice,
            TimbreEmbedding,
            EmotionEmbedding,
        )

        timbre = TimbreEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
        )
        emotion1 = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="happy",
        )
        emotion2 = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="sad",
        )
        separated = SeparatedVoice(timbre=timbre, emotion=emotion1)

        combined = separated.with_emotion(emotion2)
        assert combined.shape == (256,)
        # Original emotion should be unchanged
        assert separated.emotion.emotion_label == "happy"


class TestEmotionTimbreSeparator:
    """Tests for EmotionTimbreSeparator class."""

    def test_initialization(self):
        """Test EmotionTimbreSeparator initialization."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        assert separator.timbre_dim == 256
        assert separator.emotion_dim == 64
        assert separator.device == "cpu"

    def test_initialization_custom(self):
        """Test with custom dimensions."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator(
            timbre_dim=512,
            emotion_dim=128,
        )
        assert separator.timbre_dim == 512
        assert separator.emotion_dim == 128

    def test_separate_with_embedding(self):
        """Test separating a VoiceEmbedding."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        separated = separator.separate(embedding)
        assert separated.timbre is not None
        assert separated.emotion is not None
        assert separated.original_embedding is embedding

    def test_separate_with_array(self):
        """Test separating a numpy array."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        vector = np.random.randn(256).astype(np.float32)
        separated = separator.separate(vector)
        assert separated.timbre is not None
        assert separated.emotion is not None
        assert separated.original_embedding is None

    def test_get_emotion_preset(self):
        """Test getting emotion preset."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        emotion = separator.get_emotion_preset("happy")
        assert emotion.emotion_label == "happy"
        assert emotion.valence == 0.8
        assert emotion.arousal == 0.6
        assert emotion.embedding.shape == (64,)

    def test_get_emotion_preset_unknown(self):
        """Test getting unknown emotion preset."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        with pytest.raises(ValueError, match="Unknown emotion"):
            separator.get_emotion_preset("unknown_emotion")

    def test_list_emotion_presets(self):
        """Test listing emotion presets."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        presets = separator.list_emotion_presets()
        assert "happy" in presets
        assert "sad" in presets
        assert len(presets) == 10

    def test_transfer_emotion_with_embedding(self):
        """Test emotion transfer with embedding."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        result = separator.transfer_emotion(voice, "happy")
        assert result.shape == (256,)

    def test_transfer_emotion_with_separated(self):
        """Test emotion transfer with separated voice."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        separated = separator.separate(voice)

        result = separator.transfer_emotion(separated, "sad")
        assert result.shape == (256,)

    def test_transfer_emotion_with_emotion_embedding(self):
        """Test emotion transfer with EmotionEmbedding."""
        from voice_soundboard.cloning.separation import (
            EmotionTimbreSeparator,
            EmotionEmbedding,
        )
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="custom",
        )

        result = separator.transfer_emotion(voice, emotion)
        assert result.shape == (256,)

    def test_transfer_emotion_from_voice(self):
        """Test emotion transfer from another voice embedding."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        emotion_source = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/emotion.wav",
        )

        result = separator.transfer_emotion(voice, emotion_source)
        assert result.shape == (256,)

    def test_transfer_emotion_with_intensity(self):
        """Test emotion transfer with intensity."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        result = separator.transfer_emotion(voice, "happy", intensity=0.5)
        assert result.shape == (256,)

    def test_extract_emotion_from_audio(self):
        """Test extracting emotion from audio file."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding, VoiceExtractor

        separator = EmotionTimbreSeparator()

        # Create a real VoiceEmbedding instead of a Mock
        mock_embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/audio.wav",
        )

        # Create a mock extractor and pass it directly
        mock_extractor = Mock(spec=VoiceExtractor)
        mock_extractor.extract.return_value = mock_embedding

        emotion = separator.extract_emotion_from_audio("/fake/audio.wav", extractor=mock_extractor)
        assert emotion is not None
        assert emotion.source_path == "/fake/audio.wav"

    def test_blend_emotions(self):
        """Test blending emotions."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        emotion1 = separator.get_emotion_preset("happy")
        emotion2 = separator.get_emotion_preset("sad")

        blended = separator.blend_emotions([
            (emotion1, 0.7),
            (emotion2, 0.3),
        ])

        assert blended.emotion_label == "blended"
        assert blended.embedding.shape == (64,)
        # VAD should be blended
        assert blended.valence is not None

    def test_blend_emotions_empty(self):
        """Test blending empty emotions list."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        blended = separator.blend_emotions([])
        assert blended.emotion_label == "neutral"

    def test_blend_emotions_zero_weights(self):
        """Test blending with zero weights."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        emotion1 = separator.get_emotion_preset("happy")
        emotion2 = separator.get_emotion_preset("sad")

        blended = separator.blend_emotions([
            (emotion1, 0),
            (emotion2, 0),
        ])
        # Should handle gracefully
        assert blended is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_separate_voice(self):
        """Test separate_voice function."""
        from voice_soundboard.cloning.separation import separate_voice
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        separated = separate_voice(embedding)
        assert separated.timbre is not None
        assert separated.emotion is not None

    def test_transfer_emotion_function(self):
        """Test transfer_emotion function."""
        from voice_soundboard.cloning.separation import transfer_emotion
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        result = transfer_emotion(voice, "happy", intensity=0.8)
        assert result.shape == (256,)


# ==============================================================================
# Additional Edge Case Tests
# ==============================================================================

class TestLibraryEdgeCases:
    """Edge case tests for library."""

    @pytest.fixture
    def temp_library_dir(self):
        """Create a temporary library directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_add_with_source_copy(self, temp_library_dir):
        """Test adding voice with source audio copy."""
        from voice_soundboard.cloning.library import VoiceLibrary
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        # Create a fake source file
        source_file = temp_library_dir / "source.wav"
        source_file.write_bytes(b"fake audio data")

        library = VoiceLibrary(temp_library_dir / "library")

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path=str(source_file),
        )

        profile = library.add(
            voice_id="test",
            name="Test",
            embedding=embedding,
            source_audio_path=str(source_file),
            copy_source=True,
        )

        # Source should be copied
        assert profile.source_audio_path is not None

    def test_find_similar_no_embeddings(self, temp_library_dir):
        """Test find_similar with profiles without embeddings."""
        from voice_soundboard.cloning.library import VoiceLibrary, VoiceProfile
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        library = VoiceLibrary(temp_library_dir)
        library.load()

        # Add profile directly without embedding
        library._profiles["no_embedding"] = VoiceProfile(
            voice_id="no_embedding",
            name="No Embedding",
            embedding=None,
        )

        search_embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/search.wav",
        )

        results = library.find_similar(search_embedding)
        # Should not include profile without embedding
        for profile, _ in results:
            assert profile.voice_id != "no_embedding"

    def test_numpy_type_conversion(self):
        """Test numpy type conversion in to_dict."""
        from voice_soundboard.cloning.library import VoiceProfile
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )
        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            embedding=embedding,
            quality_rating=np.float32(0.9),
        )

        data = profile.to_dict()
        # Should be JSON serializable
        json.dumps(data)


class TestSeparationEdgeCases:
    """Edge case tests for separation."""

    def test_extract_timbre_zero_norm(self):
        """Test extract timbre with zero norm vector."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        # Create a zero vector
        vector = np.zeros(256, dtype=np.float32)
        timbre = separator._extract_timbre(vector)
        # Should handle gracefully
        assert timbre.shape == (256,)

    def test_transfer_emotion_invalid_type(self):
        """Test transfer_emotion with invalid type."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            source_path="/fake/path.wav",
        )

        with pytest.raises(ValueError, match="Unknown emotion reference"):
            separator.transfer_emotion(voice, 12345)  # Invalid type
