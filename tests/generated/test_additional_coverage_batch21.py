"""
Additional coverage tests - Batch 21.

Tests for:
- cloning/extractor.py (VoiceEmbedding, VoiceExtractor, ExtractorBackend)
- dialogue/parser.py (DialogueParser, Speaker, DialogueLine, ParsedScript)
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil


# =============================================================================
# cloning/extractor.py tests
# =============================================================================


class TestExtractorBackend:
    """Tests for ExtractorBackend enum."""

    def test_backend_values(self):
        from voice_soundboard.cloning.extractor import ExtractorBackend

        assert ExtractorBackend.RESEMBLYZER.value == "resemblyzer"
        assert ExtractorBackend.SPEECHBRAIN.value == "speechbrain"
        assert ExtractorBackend.WESPEAKER.value == "wespeaker"
        assert ExtractorBackend.MOCK.value == "mock"


class TestVoiceEmbedding:
    """Tests for VoiceEmbedding dataclass."""

    def test_init_defaults(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
        )
        assert embedding.embedding_dim == 256
        assert embedding.source_path is None
        assert embedding.quality_score == 1.0
        assert embedding.snr_db == 20.0
        assert embedding.embedding_id != ""

    def test_embedding_id_generated(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = VoiceEmbedding(embedding=np.array([1, 2, 3], dtype=np.float32))
        emb2 = VoiceEmbedding(embedding=np.array([1, 2, 3], dtype=np.float32))
        emb3 = VoiceEmbedding(embedding=np.array([4, 5, 6], dtype=np.float32))

        # Same content should produce same ID
        assert emb1.embedding_id == emb2.embedding_id
        # Different content should produce different ID
        assert emb1.embedding_id != emb3.embedding_id

    def test_similarity_identical(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        vec = np.array([1, 0, 0], dtype=np.float32)
        emb1 = VoiceEmbedding(embedding=vec)
        emb2 = VoiceEmbedding(embedding=vec.copy())

        assert emb1.similarity(emb2) == pytest.approx(1.0)

    def test_similarity_orthogonal(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = VoiceEmbedding(embedding=np.array([1, 0, 0], dtype=np.float32))
        emb2 = VoiceEmbedding(embedding=np.array([0, 1, 0], dtype=np.float32))

        assert emb1.similarity(emb2) == pytest.approx(0.0)

    def test_similarity_opposite(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = VoiceEmbedding(embedding=np.array([1, 0, 0], dtype=np.float32))
        emb2 = VoiceEmbedding(embedding=np.array([-1, 0, 0], dtype=np.float32))

        assert emb1.similarity(emb2) == pytest.approx(-1.0)

    def test_similarity_dimension_mismatch(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = VoiceEmbedding(embedding=np.array([1, 0], dtype=np.float32))
        emb2 = VoiceEmbedding(embedding=np.array([1, 0, 0], dtype=np.float32))

        with pytest.raises(ValueError, match="dimension mismatch"):
            emb1.similarity(emb2)

    def test_similarity_zero_norm(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        emb1 = VoiceEmbedding(embedding=np.array([0, 0, 0], dtype=np.float32))
        emb2 = VoiceEmbedding(embedding=np.array([1, 0, 0], dtype=np.float32))

        assert emb1.similarity(emb2) == 0.0

    def test_to_dict(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            source_path="/path/audio.wav",
            quality_score=0.9,
        )

        data = embedding.to_dict()
        assert data["embedding"] == [1.0, 2.0, 3.0]
        assert data["source_path"] == "/path/audio.wav"
        assert data["quality_score"] == 0.9

    def test_to_dict_converts_numpy_types(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        embedding = VoiceEmbedding(
            embedding=np.array([1.0, 2.0], dtype=np.float32),
            quality_score=np.float32(0.9),
        )

        data = embedding.to_dict()
        # Should be Python float, not numpy
        assert isinstance(data["quality_score"], float)

    def test_from_dict(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        data = {
            "embedding": [1.0, 2.0, 3.0],
            "embedding_dim": 3,
            "source_path": "/path/audio.wav",
            "quality_score": 0.9,
        }

        embedding = VoiceEmbedding.from_dict(data)
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.source_path == "/path/audio.wav"
        assert embedding.quality_score == 0.9

    def test_save_npz(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            embedding = VoiceEmbedding(
                embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                source_path="/path/audio.wav",
            )

            path = Path(tmpdir) / "embedding.npz"
            result = embedding.save(path)

            assert result == path
            assert path.exists()

    def test_save_json(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            embedding = VoiceEmbedding(
                embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )

            path = Path(tmpdir) / "embedding.json"
            result = embedding.save(path)

            assert result == path
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert data["embedding"] == [1.0, 2.0, 3.0]

    def test_load_npz(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            original = VoiceEmbedding(
                embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                source_path="/path/audio.wav",
                quality_score=0.95,
            )

            path = Path(tmpdir) / "embedding.npz"
            original.save(path)

            loaded = VoiceEmbedding.load(path)
            assert np.allclose(loaded.embedding, original.embedding)
            assert loaded.source_path == original.source_path

    def test_load_json(self):
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        with tempfile.TemporaryDirectory() as tmpdir:
            original = VoiceEmbedding(
                embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                quality_score=0.9,
            )

            path = Path(tmpdir) / "embedding.json"
            original.save(path)

            loaded = VoiceEmbedding.load(path)
            assert np.allclose(loaded.embedding, original.embedding)
            assert loaded.quality_score == original.quality_score


class TestVoiceExtractor:
    """Tests for VoiceExtractor class."""

    def test_init_defaults(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor()
        assert extractor.backend == ExtractorBackend.MOCK
        assert extractor.device == "cpu"
        assert extractor._loaded is False

    def test_init_with_custom_backend(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(
            backend=ExtractorBackend.RESEMBLYZER,
            device="cuda",
        )
        assert extractor.backend == ExtractorBackend.RESEMBLYZER
        assert extractor.device == "cuda"

    def test_embedding_dim_property(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        mock_extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        assert mock_extractor.embedding_dim == 256

        speechbrain_extractor = VoiceExtractor(backend=ExtractorBackend.SPEECHBRAIN)
        assert speechbrain_extractor.embedding_dim == 192

    def test_load_mock(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        extractor.load()
        assert extractor._loaded is True

    def test_load_already_loaded(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        extractor.load()
        extractor.load()  # Should not raise
        assert extractor._loaded is True

    def test_unload(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, ExtractorBackend

        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        extractor.load()
        extractor.unload()
        assert extractor._loaded is False
        assert extractor._model is None

    def test_extract_from_array(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()
        audio = np.random.randn(16000).astype(np.float32)  # 1 second

        embedding = extractor.extract(audio, sample_rate=16000)
        assert embedding.embedding.shape == (256,)
        assert embedding.source_path is None
        assert embedding.source_duration_seconds == pytest.approx(1.0)

    def test_extract_normalizes_embedding(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()
        audio = np.random.randn(16000).astype(np.float32)

        embedding = extractor.extract(audio)
        # Mock extractor normalizes embeddings
        norm = np.linalg.norm(embedding.embedding)
        assert norm == pytest.approx(1.0, rel=0.01)

    def test_extract_from_file_not_found(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.wav")

    def test_extract_from_segments(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()
        # 10 seconds of audio
        audio = np.random.randn(160000).astype(np.float32)

        embeddings = extractor.extract_from_segments(
            audio, segment_seconds=3.0, sample_rate=16000
        )

        # Should have 3 segments (10/3 = 3.33, last one is > 1 second)
        assert len(embeddings) >= 2

    def test_extract_from_segments_skips_short(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()
        # 5.5 seconds of audio
        audio = np.random.randn(88000).astype(np.float32)

        embeddings = extractor.extract_from_segments(
            audio, segment_seconds=3.0, sample_rate=16000
        )

        # 5.5/3 = 1 full segment + 2.5 seconds (> 1s, so included)
        assert len(embeddings) == 2

    def test_average_embeddings(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor, VoiceEmbedding

        extractor = VoiceExtractor()

        emb1 = VoiceEmbedding(
            embedding=np.array([1, 0, 0, 0], dtype=np.float32),
            source_duration_seconds=2.0,
            quality_score=0.8,
        )
        emb2 = VoiceEmbedding(
            embedding=np.array([0, 1, 0, 0], dtype=np.float32),
            source_duration_seconds=3.0,
            quality_score=0.9,
        )

        averaged = extractor.average_embeddings([emb1, emb2])
        assert averaged.source_duration_seconds == 5.0
        assert averaged.quality_score == pytest.approx(0.85)

    def test_average_embeddings_empty(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()

        with pytest.raises(ValueError, match="No embeddings"):
            extractor.average_embeddings([])

    def test_preprocess_audio_normalize(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()

        audio = np.array([0.5, 1.0, -1.0, 0.5], dtype=np.float32)
        processed = extractor._preprocess_audio(audio, 16000, 16000)

        # Should be normalized to [-1, 1]
        assert np.max(np.abs(processed)) == pytest.approx(1.0)

    def test_estimate_quality(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()

        # Good quality audio
        good_audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
        quality, snr = extractor._estimate_quality(good_audio)
        assert 0 <= quality <= 1

        # Clipped audio should have lower quality
        clipped_audio = np.clip(good_audio * 2, -1, 1)
        clipped_quality, _ = extractor._estimate_quality(clipped_audio)
        assert clipped_quality <= quality

    def test_mock_extract_deterministic(self):
        from voice_soundboard.cloning.extractor import VoiceExtractor

        extractor = VoiceExtractor()

        audio = np.random.randn(16000).astype(np.float32)
        emb1 = extractor._mock_extract(audio)
        emb2 = extractor._mock_extract(audio)

        assert np.allclose(emb1, emb2)


class TestExtractEmbeddingFunction:
    """Tests for extract_embedding convenience function."""

    def test_extract_embedding_with_mock(self):
        from voice_soundboard.cloning.extractor import extract_embedding

        audio = np.random.randn(16000).astype(np.float32)
        embedding = extract_embedding(audio, backend="mock")

        assert embedding.embedding.shape == (256,)
        assert embedding.extractor_backend == "mock"

    def test_extract_embedding_with_enum(self):
        from voice_soundboard.cloning.extractor import (
            extract_embedding,
            ExtractorBackend,
        )

        audio = np.random.randn(16000).astype(np.float32)
        embedding = extract_embedding(audio, backend=ExtractorBackend.MOCK)

        assert embedding.extractor_backend == "mock"


# =============================================================================
# dialogue/parser.py tests
# =============================================================================


class TestStageDirectionType:
    """Tests for StageDirectionType enum."""

    def test_direction_types(self):
        from voice_soundboard.dialogue.parser import StageDirectionType

        assert StageDirectionType.EMOTION.value == "emotion"
        assert StageDirectionType.VOLUME.value == "volume"
        assert StageDirectionType.PACE.value == "pace"
        assert StageDirectionType.ACTION.value == "action"


class TestStageDirection:
    """Tests for StageDirection dataclass."""

    def test_volume_whispering(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="whispering")
        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier == 0.4
        assert direction.emotion_hint == "whisper"

    def test_volume_shouting(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="shouting")
        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier == 1.5
        assert direction.intensity == 0.8

    def test_volume_softly(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="softly")
        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier == 0.6

    def test_pace_slowly(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="slowly")
        assert direction.direction_type == StageDirectionType.PACE
        assert direction.speed_modifier == 0.7

    def test_pace_quickly(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="quickly")
        assert direction.direction_type == StageDirectionType.PACE
        assert direction.speed_modifier == 1.4

    def test_pace_hesitantly(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="hesitantly")
        assert direction.direction_type == StageDirectionType.PACE
        assert direction.emotion_hint == "nervous"

    def test_emotion_sadly(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="sadly")
        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "sad"

    def test_emotion_happily(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="happily")
        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "happy"

    def test_emotion_angrily(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="angrily")
        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "angry"

    def test_emotion_fearfully(self):
        from voice_soundboard.dialogue.parser import StageDirection

        direction = StageDirection(text="fearfully")
        assert direction.emotion_hint == "fearful"

    def test_emotion_sarcastically(self):
        from voice_soundboard.dialogue.parser import StageDirection

        direction = StageDirection(text="sarcastically")
        assert direction.emotion_hint == "sarcastic"

    def test_emotion_excitedly(self):
        from voice_soundboard.dialogue.parser import StageDirection

        direction = StageDirection(text="excitedly")
        assert direction.emotion_hint == "excited"
        assert direction.speed_modifier == 1.2

    def test_action_pause(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="pauses")
        assert direction.direction_type == StageDirectionType.ACTION

    def test_action_sighs(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="sighs")
        assert direction.direction_type == StageDirectionType.ACTION
        assert direction.emotion_hint == "sad"

    def test_action_laughs(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="laughing")
        assert direction.direction_type == StageDirectionType.ACTION
        assert direction.emotion_hint == "happy"

    def test_unknown_direction_defaults_to_emotion(self):
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="mysteriously")
        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "mysteriously"


class TestSpeaker:
    """Tests for Speaker dataclass."""

    def test_speaker_defaults(self):
        from voice_soundboard.dialogue.parser import Speaker

        speaker = Speaker(id="S1", name="narrator")
        assert speaker.voice is None
        assert speaker.gender_hint is None
        assert speaker.default_emotion is None

    def test_speaker_hash(self):
        from voice_soundboard.dialogue.parser import Speaker

        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S1", name="different_name")
        s3 = Speaker(id="S2", name="alice")

        assert hash(s1) == hash(s2)  # Same ID
        assert hash(s1) != hash(s3)  # Different ID

    def test_speaker_equality(self):
        from voice_soundboard.dialogue.parser import Speaker

        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S1", name="different")
        s3 = Speaker(id="S2", name="alice")

        assert s1 == s2  # Same ID
        assert s1 != s3  # Different ID
        assert s1 != "S1"  # Not a Speaker


class TestDialogueLine:
    """Tests for DialogueLine dataclass."""

    def test_dialogue_line_defaults(self):
        from voice_soundboard.dialogue.parser import DialogueLine, Speaker

        speaker = Speaker(id="S1", name="narrator")
        line = DialogueLine(
            speaker=speaker,
            text="Hello world",
            raw_text="[S1:narrator] Hello world",
        )

        assert line.emotion is None
        assert line.speed == 1.0
        assert line.pause_before_ms == 0
        assert line.pause_after_ms == 300

    def test_has_stage_directions(self):
        from voice_soundboard.dialogue.parser import (
            DialogueLine,
            Speaker,
            StageDirection,
        )

        speaker = Speaker(id="S1", name="narrator")

        line_without = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="Hello",
        )
        assert line_without.has_stage_directions() is False

        line_with = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(sadly) Hello",
            stage_directions=[StageDirection(text="sadly")],
        )
        assert line_with.has_stage_directions() is True

    def test_get_primary_emotion(self):
        from voice_soundboard.dialogue.parser import (
            DialogueLine,
            Speaker,
            StageDirection,
        )

        speaker = Speaker(id="S1", name="narrator")

        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(sadly) Hello",
            stage_directions=[StageDirection(text="sadly")],
        )
        assert line.get_primary_emotion() == "sad"

    def test_get_primary_emotion_fallback(self):
        from voice_soundboard.dialogue.parser import DialogueLine, Speaker

        speaker = Speaker(id="S1", name="narrator")

        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="Hello",
            emotion="happy",
        )
        assert line.get_primary_emotion() == "happy"


class TestParsedScript:
    """Tests for ParsedScript dataclass."""

    def test_get_speaker_lines(self):
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S2", name="bob")

        lines = [
            DialogueLine(speaker=s1, text="Hello", raw_text="Hello"),
            DialogueLine(speaker=s2, text="Hi", raw_text="Hi"),
            DialogueLine(speaker=s1, text="Bye", raw_text="Bye"),
        ]

        script = ParsedScript(lines=lines, speakers={"S1": s1, "S2": s2})

        alice_lines = script.get_speaker_lines("S1")
        assert len(alice_lines) == 2

    def test_get_speaker_names(self):
        from voice_soundboard.dialogue.parser import ParsedScript, Speaker

        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S2", name="bob")

        script = ParsedScript(lines=[], speakers={"S1": s1, "S2": s2})
        names = script.get_speaker_names()

        assert "alice" in names
        assert "bob" in names

    def test_speaker_count(self):
        from voice_soundboard.dialogue.parser import ParsedScript, Speaker

        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S2", name="bob")

        script = ParsedScript(lines=[], speakers={"S1": s1, "S2": s2})
        assert script.speaker_count() == 2

    def test_line_count(self):
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice")
        lines = [
            DialogueLine(speaker=s1, text="Hello", raw_text="Hello"),
            DialogueLine(speaker=s1, text="World", raw_text="World"),
        ]

        script = ParsedScript(lines=lines, speakers={"S1": s1})
        assert script.line_count() == 2


class TestDialogueParser:
    """Tests for DialogueParser class."""

    def test_parse_basic_script(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] The story begins.
            [S2:alice] Hello there!
        """)

        assert len(script.lines) == 2
        assert len(script.speakers) == 2
        assert script.lines[0].speaker.name == "narrator"
        assert script.lines[1].speaker.name == "alice"

    def test_parse_with_metadata(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            # title: My Story
            # author: Test
            [S1:narrator] Once upon a time...
        """)

        assert script.title == "My Story"
        assert script.metadata.get("author") == "Test"

    def test_parse_stage_directions(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] (whispering) Don't make a sound.
        """)

        line = script.lines[0]
        assert len(line.stage_directions) == 1
        assert line.stage_directions[0].text == "whispering"
        assert "Don't make a sound" in line.text

    def test_parse_continuation_line(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] First line.
            Second line continues.
        """)

        assert len(script.lines) == 2
        assert script.lines[0].speaker.name == "alice"
        assert script.lines[1].speaker.name == "alice"

    def test_parse_simple_format(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser(parse_simple_format=True)
        script = parser.parse("""
            ALICE: Hello there!
            BOB: Hi Alice!
        """)

        assert len(script.lines) == 2
        assert script.lines[0].speaker.name == "alice"
        assert script.lines[1].speaker.name == "bob"

    def test_parse_simple_format_disabled(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser(parse_simple_format=False)
        script = parser.parse("""
            ALICE: Hello there!
        """)

        # Without simple format, should create default narrator
        assert script.lines[0].speaker.name == "narrator"

    def test_parse_pause_between_speakers(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser(
            default_pause_between_speakers_ms=500,
            default_pause_same_speaker_ms=200,
        )
        script = parser.parse("""
            [S1:alice] Line one.
            [S2:bob] Line two.
            [S2:bob] Line three.
        """)

        assert script.lines[0].pause_before_ms == 0  # First line
        assert script.lines[1].pause_before_ms == 500  # Speaker change
        assert script.lines[2].pause_before_ms == 200  # Same speaker

    def test_parse_emotion_from_direction(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] (sadly) I'm so sad.
        """)

        assert script.lines[0].emotion == "sad"

    def test_parse_speed_modifier(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] (quickly) Hurry up!
        """)

        assert script.lines[0].speed > 1.0

    def test_infer_gender_female(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()

        assert parser._infer_gender("alice") == "female"
        assert parser._infer_gender("emma") == "female"
        assert parser._infer_gender("queen") == "female"

    def test_infer_gender_male(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()

        assert parser._infer_gender("bob") == "male"
        assert parser._infer_gender("king") == "male"
        assert parser._infer_gender("sir") == "male"

    def test_infer_gender_neutral(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()

        assert parser._infer_gender("narrator") is None
        assert parser._infer_gender("announcer") is None

    def test_infer_gender_suffix(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()

        assert parser._infer_gender("maria") == "female"
        assert parser._infer_gender("julie") == "female"

    def test_skip_empty_lines(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Line one.

            [S1:alice] Line two.
        """)

        assert len(script.lines) == 2

    def test_skip_metadata_lines(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            # This is a comment
            [S1:alice] Hello!
            # Another comment
        """)

        assert len(script.lines) == 1

    def test_speaker_tag_without_number(self):
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [speaker:alice] Hello!
        """)

        assert script.lines[0].speaker.name == "alice"


class TestParseDialogueFunction:
    """Tests for parse_dialogue convenience function."""

    def test_parse_dialogue(self):
        from voice_soundboard.dialogue.parser import parse_dialogue

        script = parse_dialogue("""
            [S1:narrator] The end.
        """)

        assert len(script.lines) == 1
        assert script.lines[0].speaker.name == "narrator"
