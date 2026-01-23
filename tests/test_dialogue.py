"""
Tests for Multi-Speaker Dialogue Module.

Tests cover:
- DialogueParser: script parsing, speaker detection, stage directions
- VoiceAssigner: voice auto-assignment, gender matching
- DialogueEngine: synthesis orchestration (mocked)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from voice_soundboard.dialogue.parser import (
    DialogueParser,
    DialogueLine,
    Speaker,
    StageDirection,
    StageDirectionType,
    ParsedScript,
    parse_dialogue,
)
from voice_soundboard.dialogue.voices import (
    VoiceAssigner,
    VoiceCharacteristics,
    VoiceGender,
    VoiceAge,
    VoiceStyle,
    auto_assign_voices,
    get_voice_for_gender,
    list_voices_by_gender,
    VOICE_CHARACTERISTICS,
)
from voice_soundboard.dialogue.engine import (
    DialogueEngine,
    DialogueResult,
    SpeakerTurn,
    synthesize_dialogue,
)


class TestDialogueParser:
    """Tests for DialogueParser class."""

    def test_parse_simple_script(self):
        """Test parsing a simple two-speaker script."""
        script = """
        [S1:narrator] The door opened slowly.
        [S2:alice] Hello? Is anyone there?
        """
        parser = DialogueParser()
        result = parser.parse(script)

        assert result.speaker_count() == 2
        assert result.line_count() == 2
        assert "narrator" in result.get_speaker_names()
        assert "alice" in result.get_speaker_names()

    def test_parse_speaker_tags_format(self):
        """Test various speaker tag formats."""
        script = """
        [S1:narrator] First line.
        [S2:Bob] Second line.
        [S3:mary_jane] Third line.
        """
        parser = DialogueParser()
        result = parser.parse(script)

        assert result.speaker_count() == 3
        names = result.get_speaker_names()
        assert "narrator" in names
        assert "bob" in names
        assert "mary_jane" in names

    def test_parse_continuation_lines(self):
        """Test that lines without speaker tags continue previous speaker."""
        script = """
        [S1:narrator] First line.
        And this continues the narrator.
        [S2:alice] Different speaker now.
        """
        parser = DialogueParser()
        result = parser.parse(script)

        assert result.line_count() == 3
        assert result.lines[0].speaker.name == "narrator"
        assert result.lines[1].speaker.name == "narrator"
        assert result.lines[2].speaker.name == "alice"

    def test_parse_stage_directions(self):
        """Test extraction of stage directions."""
        script = "[S1:bob] (whispering) Don't go in there."
        parser = DialogueParser()
        result = parser.parse(script)

        line = result.lines[0]
        assert line.has_stage_directions()
        assert len(line.stage_directions) == 1
        assert line.stage_directions[0].text == "whispering"

    def test_parse_multiple_stage_directions(self):
        """Test multiple stage directions in one line."""
        script = "[S1:alice] (nervously) (whispering) I'm scared."
        parser = DialogueParser()
        result = parser.parse(script)

        line = result.lines[0]
        assert len(line.stage_directions) == 2

    def test_stage_direction_cleaned_from_text(self):
        """Test that stage directions are removed from dialogue text."""
        script = "[S1:bob] (angrily) What do you mean?"
        parser = DialogueParser()
        result = parser.parse(script)

        assert "(angrily)" not in result.lines[0].text
        assert "What do you mean?" in result.lines[0].text

    def test_parse_simple_format(self):
        """Test parsing NAME: dialogue format."""
        script = """
        NARRATOR: The story begins.
        ALICE: Hello world!
        """
        parser = DialogueParser(parse_simple_format=True)
        result = parser.parse(script)

        assert result.speaker_count() == 2
        assert result.line_count() == 2

    def test_parse_metadata(self):
        """Test extraction of metadata comments."""
        script = """
        #title: My Story
        #author: John Doe
        [S1:narrator] Once upon a time...
        """
        parser = DialogueParser()
        result = parser.parse(script)

        assert result.title == "My Story"
        assert result.metadata.get("author") == "John Doe"

    def test_empty_script(self):
        """Test parsing empty script."""
        parser = DialogueParser()
        result = parser.parse("")

        assert result.line_count() == 0
        assert result.speaker_count() == 0

    def test_script_with_only_comments(self):
        """Test parsing script with only comments."""
        script = """
        #title: Empty Story
        # This is a comment
        """
        parser = DialogueParser()
        result = parser.parse(script)

        assert result.line_count() == 0
        assert result.title == "Empty Story"


class TestStageDirection:
    """Tests for StageDirection parsing and interpretation."""

    def test_whispering_direction(self):
        """Test whispering stage direction parameters."""
        direction = StageDirection(text="whispering")

        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier < 1.0
        assert direction.speed_modifier < 1.0
        assert direction.emotion_hint == "whisper"

    def test_shouting_direction(self):
        """Test shouting stage direction parameters."""
        direction = StageDirection(text="shouting")

        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier > 1.0
        assert direction.intensity > 0.5

    def test_slowly_direction(self):
        """Test slowly pace direction."""
        direction = StageDirection(text="slowly")

        assert direction.direction_type == StageDirectionType.PACE
        assert direction.speed_modifier < 1.0

    def test_quickly_direction(self):
        """Test quickly pace direction."""
        direction = StageDirection(text="quickly")

        assert direction.direction_type == StageDirectionType.PACE
        assert direction.speed_modifier > 1.0

    def test_emotion_directions(self):
        """Test emotion stage directions."""
        emotions = {
            "sadly": "sad",
            "happily": "happy",
            "angrily": "angry",
            "fearfully": "fearful",
        }

        for text, expected_hint in emotions.items():
            direction = StageDirection(text=text)
            assert direction.direction_type == StageDirectionType.EMOTION
            assert direction.emotion_hint == expected_hint

    def test_unknown_direction_defaults_to_emotion(self):
        """Test that unknown directions default to emotion type."""
        direction = StageDirection(text="mysteriously")

        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "mysteriously"


class TestSpeaker:
    """Tests for Speaker class."""

    def test_speaker_equality(self):
        """Test speaker equality based on ID."""
        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S1", name="alice")
        s3 = Speaker(id="S2", name="alice")

        assert s1 == s2
        assert s1 != s3

    def test_speaker_hash(self):
        """Test speaker hashing for use in sets/dicts."""
        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S1", name="bob")

        # Same ID = same hash
        assert hash(s1) == hash(s2)

        # Can use in set
        speakers = {s1}
        assert s1 in speakers


class TestDialogueLine:
    """Tests for DialogueLine class."""

    def test_has_stage_directions(self):
        """Test stage direction detection."""
        speaker = Speaker(id="S1", name="test")

        line_with = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(angrily) Hello",
            stage_directions=[StageDirection(text="angrily")],
        )
        line_without = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="Hello",
        )

        assert line_with.has_stage_directions()
        assert not line_without.has_stage_directions()

    def test_get_primary_emotion(self):
        """Test getting primary emotion from line."""
        speaker = Speaker(id="S1", name="test")

        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(sadly) Hello",
            stage_directions=[StageDirection(text="sadly")],
        )

        assert line.get_primary_emotion() == "sad"


class TestParsedScript:
    """Tests for ParsedScript class."""

    def test_get_speaker_lines(self):
        """Test getting lines for specific speaker."""
        speaker1 = Speaker(id="S1", name="alice")
        speaker2 = Speaker(id="S2", name="bob")

        lines = [
            DialogueLine(speaker=speaker1, text="Hi", raw_text="Hi"),
            DialogueLine(speaker=speaker2, text="Hello", raw_text="Hello"),
            DialogueLine(speaker=speaker1, text="How are you?", raw_text="How are you?"),
        ]

        script = ParsedScript(
            lines=lines,
            speakers={"S1": speaker1, "S2": speaker2}
        )

        alice_lines = script.get_speaker_lines("S1")
        assert len(alice_lines) == 2
        assert all(line.speaker.name == "alice" for line in alice_lines)


class TestVoiceCharacteristics:
    """Tests for voice characteristics."""

    def test_voice_characteristics_defined(self):
        """Test that voice characteristics are properly defined."""
        assert len(VOICE_CHARACTERISTICS) > 0

        # Check some expected voices exist
        expected = ["af_heart", "bm_george", "am_michael"]
        for voice_id in expected:
            assert voice_id in VOICE_CHARACTERISTICS

    def test_voice_has_required_fields(self):
        """Test that each voice has required characteristics."""
        for voice_id, chars in VOICE_CHARACTERISTICS.items():
            assert isinstance(chars.voice_id, str)
            assert isinstance(chars.gender, VoiceGender)
            assert isinstance(chars.age, VoiceAge)
            assert isinstance(chars.style, VoiceStyle)


class TestVoiceAssigner:
    """Tests for VoiceAssigner class."""

    def test_assign_voices_no_duplicates(self):
        """Test that voices are not duplicated when diversity preferred."""
        script = """
        [S1:alice] Hello.
        [S2:bob] Hi there.
        [S3:charlie] Hey everyone.
        """
        parser = DialogueParser()
        parsed = parser.parse(script)

        assigner = VoiceAssigner(prefer_diversity=True)
        assignments = assigner.assign_voices(parsed)

        # All speakers should get unique voices
        voices = list(assignments.values())
        assert len(voices) == len(set(voices))

    def test_assign_voices_respects_gender_hints(self):
        """Test that gender hints are respected."""
        script = "[S1:alice] Hello."
        parser = DialogueParser()
        parsed = parser.parse(script)

        # Alice should be inferred as female
        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(parsed)

        assigned_voice = assignments["alice"]
        voice_chars = VOICE_CHARACTERISTICS.get(assigned_voice)

        # Should get a female voice
        if voice_chars:
            assert voice_chars.gender == VoiceGender.FEMALE

    def test_assign_voices_with_overrides(self):
        """Test that manual overrides are respected."""
        script = """
        [S1:narrator] Hello.
        [S2:alice] Hi.
        """
        parser = DialogueParser()
        parsed = parser.parse(script)

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(
            parsed,
            voice_overrides={"narrator": "bm_george"}
        )

        assert assignments["narrator"] == "bm_george"

    def test_suggest_voices(self):
        """Test voice suggestions for a speaker."""
        speaker = Speaker(id="S1", name="alice", gender_hint="female")

        assigner = VoiceAssigner()
        suggestions = assigner.suggest_voices(speaker, count=3)

        assert len(suggestions) == 3
        # Suggestions are (voice_id, score) tuples
        assert all(isinstance(s, tuple) for s in suggestions)
        assert all(len(s) == 2 for s in suggestions)


class TestAutoAssignVoices:
    """Tests for auto_assign_voices convenience function."""

    def test_auto_assign_basic(self):
        """Test basic auto assignment."""
        script = """
        [S1:narrator] Hello.
        [S2:alice] Hi there.
        """
        parser = DialogueParser()
        parsed = parser.parse(script)

        assignments = auto_assign_voices(parsed)

        assert "narrator" in assignments
        assert "alice" in assignments
        assert all(isinstance(v, str) for v in assignments.values())


class TestGetVoiceForGender:
    """Tests for get_voice_for_gender utility."""

    def test_female_voice(self):
        """Test getting female voice."""
        voice = get_voice_for_gender("female")
        assert voice == "af_heart"

    def test_male_voice(self):
        """Test getting male voice."""
        voice = get_voice_for_gender("male")
        assert voice == "am_michael"

    def test_neutral_voice(self):
        """Test getting neutral/default voice."""
        voice = get_voice_for_gender("neutral")
        assert voice == "bm_george"


class TestListVoicesByGender:
    """Tests for list_voices_by_gender utility."""

    def test_list_female_voices(self):
        """Test listing female voices."""
        voices = list_voices_by_gender("female")

        assert len(voices) > 0
        for voice_id in voices:
            chars = VOICE_CHARACTERISTICS[voice_id]
            assert chars.gender == VoiceGender.FEMALE

    def test_list_male_voices(self):
        """Test listing male voices."""
        voices = list_voices_by_gender("male")

        assert len(voices) > 0
        for voice_id in voices:
            chars = VOICE_CHARACTERISTICS[voice_id]
            assert chars.gender == VoiceGender.MALE


class TestDialogueEngine:
    """Tests for DialogueEngine class."""

    def test_engine_initialization(self):
        """Test engine initializes with defaults."""
        engine = DialogueEngine()

        assert engine.default_pause_ms == 400
        assert engine.narrator_pause_ms == 600
        assert engine.sample_rate == 24000
        assert engine.parser is not None
        assert engine.voice_assigner is not None

    def test_preview_assignments(self):
        """Test previewing voice assignments."""
        script = """
        [S1:narrator] The story begins.
        [S2:hero] I will save the day!
        """

        engine = DialogueEngine()
        assignments = engine.preview_assignments(script)

        assert "narrator" in assignments
        assert "hero" in assignments

    def test_get_script_info(self):
        """Test getting script information."""
        script = """
        [S1:narrator] The door opened.
        [S2:alice] Hello?
        [S1:narrator] She walked in.
        """

        engine = DialogueEngine()
        info = engine.get_script_info(script)

        assert info["speaker_count"] == 2
        assert info["line_count"] == 3
        assert info["total_words"] > 0
        assert "narrator" in info["speaker_names"]
        assert "alice" in info["speaker_names"]
        assert info["speaker_lines"]["narrator"] == 2
        assert info["speaker_lines"]["alice"] == 1

    def test_generate_silence(self):
        """Test silence generation."""
        engine = DialogueEngine()
        silence = engine._generate_silence(1000)  # 1 second

        expected_samples = engine.sample_rate  # 24000 samples for 1s
        assert len(silence) == expected_samples
        assert np.all(silence == 0)


class TestSpeakerTurn:
    """Tests for SpeakerTurn dataclass."""

    def test_speaker_turn_creation(self):
        """Test creating a speaker turn."""
        turn = SpeakerTurn(
            speaker_name="alice",
            text="Hello world",
            voice_id="af_heart",
            duration_seconds=1.5,
            emotion="happy",
            speed=1.0,
            line_number=1,
        )

        assert turn.speaker_name == "alice"
        assert turn.duration_seconds == 1.5
        assert turn.emotion == "happy"


class TestDialogueResult:
    """Tests for DialogueResult dataclass."""

    def test_get_speaker_duration(self):
        """Test calculating speaker duration."""
        turns = [
            SpeakerTurn(speaker_name="alice", text="Hi", voice_id="af_heart", duration_seconds=0.5),
            SpeakerTurn(speaker_name="bob", text="Hello", voice_id="am_michael", duration_seconds=0.7),
            SpeakerTurn(speaker_name="alice", text="How are you?", voice_id="af_heart", duration_seconds=1.0),
        ]

        result = DialogueResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=2.2,
            turns=turns,
            speaker_count=2,
            line_count=3,
        )

        assert result.get_speaker_duration("alice") == 1.5
        assert result.get_speaker_duration("bob") == 0.7


class TestGenderInference:
    """Tests for gender inference from names."""

    def test_female_names(self):
        """Test common female names are detected."""
        parser = DialogueParser()

        female_names = ["alice", "emma", "sarah", "mary", "bella"]
        for name in female_names:
            speaker, _ = parser._parse_speaker_tag(f"[S1:{name}] Hello", {})
            assert speaker.gender_hint == "female", f"Expected {name} to be female"

    def test_male_names(self):
        """Test common male names are detected."""
        parser = DialogueParser()

        male_names = ["bob", "john", "michael", "david", "george"]
        for name in male_names:
            speaker, _ = parser._parse_speaker_tag(f"[S1:{name}] Hello", {})
            assert speaker.gender_hint == "male", f"Expected {name} to be male"

    def test_neutral_roles(self):
        """Test neutral roles have no gender hint."""
        parser = DialogueParser()

        neutral_roles = ["narrator", "announcer", "voice"]
        for role in neutral_roles:
            speaker, _ = parser._parse_speaker_tag(f"[S1:{role}] Hello", {})
            assert speaker.gender_hint is None, f"Expected {role} to be neutral"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_parse_dialogue_function(self):
        """Test parse_dialogue convenience function."""
        script = "[S1:narrator] Hello world."
        result = parse_dialogue(script)

        assert isinstance(result, ParsedScript)
        assert result.line_count() == 1


# Integration tests (require voice engine - skipped by default)
@pytest.mark.skip(reason="Integration test - requires TTS model loaded")
class TestDialogueEngineIntegration:
    """Integration tests for dialogue synthesis."""

    def test_full_synthesis(self, tmp_path):
        """Test full dialogue synthesis."""
        script = """
        [S1:narrator] Once upon a time.
        [S2:princess] Help me!
        [S3:knight] I will save you!
        """

        engine = DialogueEngine()
        result = engine.synthesize(
            script=script,
            output_path=tmp_path / "dialogue.wav",
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert result.speaker_count == 3
        assert result.line_count == 3

    def test_synthesis_with_voice_overrides(self, tmp_path):
        """Test synthesis with manual voice assignments."""
        script = """
        [S1:narrator] The story begins.
        [S2:hero] I am the hero!
        """

        engine = DialogueEngine()
        result = engine.synthesize(
            script=script,
            voices={
                "narrator": "bm_george",
                "hero": "am_michael",
            },
            output_path=tmp_path / "dialogue.wav",
        )

        assert result.voice_assignments["narrator"] == "bm_george"
        assert result.voice_assignments["hero"] == "am_michael"
