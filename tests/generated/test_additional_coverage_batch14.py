"""
Additional test coverage batch 14: dialogue/parser.py, dialogue/voices.py, dialogue/engine.py.

Tests for dialogue script parsing, voice auto-assignment, and dialogue synthesis engine.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np

from voice_soundboard.dialogue.parser import (
    StageDirectionType,
    StageDirection,
    Speaker,
    DialogueLine,
    ParsedScript,
    DialogueParser,
    parse_dialogue,
)
from voice_soundboard.dialogue.voices import (
    VoiceGender,
    VoiceAge,
    VoiceStyle,
    VoiceCharacteristics,
    VOICE_CHARACTERISTICS,
    ROLE_PREFERENCES,
    VoiceAssigner,
    auto_assign_voices,
    get_voice_for_gender,
    list_voices_by_gender,
)
from voice_soundboard.dialogue.engine import (
    SpeakerTurn,
    DialogueResult,
    DialogueEngine,
    synthesize_dialogue,
)


# =============================================================================
# StageDirectionType Enum Tests
# =============================================================================

class TestStageDirectionType:
    """Tests for StageDirectionType enum."""

    def test_emotion_value(self):
        """Test EMOTION value."""
        assert StageDirectionType.EMOTION.value == "emotion"

    def test_volume_value(self):
        """Test VOLUME value."""
        assert StageDirectionType.VOLUME.value == "volume"

    def test_pace_value(self):
        """Test PACE value."""
        assert StageDirectionType.PACE.value == "pace"

    def test_action_value(self):
        """Test ACTION value."""
        assert StageDirectionType.ACTION.value == "action"


# =============================================================================
# StageDirection Tests
# =============================================================================

class TestStageDirection:
    """Tests for StageDirection dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        sd = StageDirection(text="sadly")
        assert sd.text == "sadly"
        assert sd.direction_type == StageDirectionType.EMOTION
        assert sd.emotion_hint == "sad"

    def test_whispering(self):
        """Test whispering direction."""
        sd = StageDirection(text="whispering")
        assert sd.direction_type == StageDirectionType.VOLUME
        assert sd.volume_modifier < 1.0
        assert sd.speed_modifier < 1.0

    def test_shouting(self):
        """Test shouting direction."""
        sd = StageDirection(text="shouting")
        assert sd.direction_type == StageDirectionType.VOLUME
        assert sd.volume_modifier > 1.0

    def test_slowly(self):
        """Test slowly direction."""
        sd = StageDirection(text="slowly")
        assert sd.direction_type == StageDirectionType.PACE
        assert sd.speed_modifier < 1.0

    def test_quickly(self):
        """Test quickly direction."""
        sd = StageDirection(text="quickly")
        assert sd.direction_type == StageDirectionType.PACE
        assert sd.speed_modifier > 1.0

    def test_happily(self):
        """Test happily direction."""
        sd = StageDirection(text="happily")
        assert sd.direction_type == StageDirectionType.EMOTION
        assert sd.emotion_hint == "happy"

    def test_angrily(self):
        """Test angrily direction."""
        sd = StageDirection(text="angrily")
        assert sd.direction_type == StageDirectionType.EMOTION
        assert sd.emotion_hint == "angry"

    def test_pauses(self):
        """Test pauses direction."""
        sd = StageDirection(text="pauses")
        assert sd.direction_type == StageDirectionType.ACTION

    def test_laughing(self):
        """Test laughing direction."""
        sd = StageDirection(text="laughing")
        assert sd.direction_type == StageDirectionType.ACTION
        assert sd.emotion_hint == "happy"

    def test_unknown_becomes_emotion(self):
        """Test unknown direction becomes emotion type."""
        sd = StageDirection(text="mysteriously")
        assert sd.direction_type == StageDirectionType.EMOTION
        assert sd.emotion_hint == "mysteriously"


# =============================================================================
# Speaker Tests
# =============================================================================

class TestSpeaker:
    """Tests for Speaker dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        speaker = Speaker(id="S1", name="narrator")
        assert speaker.id == "S1"
        assert speaker.name == "narrator"
        assert speaker.voice is None

    def test_creation_with_all_fields(self):
        """Test creation with all fields."""
        speaker = Speaker(
            id="S1",
            name="alice",
            voice="af_heart",
            default_emotion="happy",
            gender_hint="female",
            age_hint="adult",
            accent_hint="british",
        )
        assert speaker.voice == "af_heart"
        assert speaker.gender_hint == "female"

    def test_hash(self):
        """Test __hash__ method."""
        s1 = Speaker(id="S1", name="test")
        s2 = Speaker(id="S1", name="different")
        assert hash(s1) == hash(s2)  # Same ID

    def test_eq_same_id(self):
        """Test __eq__ with same ID."""
        s1 = Speaker(id="S1", name="test")
        s2 = Speaker(id="S1", name="different")
        assert s1 == s2

    def test_eq_different_id(self):
        """Test __eq__ with different ID."""
        s1 = Speaker(id="S1", name="test")
        s2 = Speaker(id="S2", name="test")
        assert s1 != s2

    def test_eq_non_speaker(self):
        """Test __eq__ with non-Speaker."""
        speaker = Speaker(id="S1", name="test")
        assert speaker != "S1"


# =============================================================================
# DialogueLine Tests
# =============================================================================

class TestDialogueLine:
    """Tests for DialogueLine dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(
            speaker=speaker,
            text="Hello world",
            raw_text="[S1:test] Hello world",
        )
        assert line.text == "Hello world"
        assert line.speed == 1.0

    def test_has_stage_directions_false(self):
        """Test has_stage_directions returns False."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="Hello",
            stage_directions=[],
        )
        assert line.has_stage_directions() is False

    def test_has_stage_directions_true(self):
        """Test has_stage_directions returns True."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(sadly) Hello",
            stage_directions=[StageDirection(text="sadly")],
        )
        assert line.has_stage_directions() is True

    def test_get_primary_emotion_from_direction(self):
        """Test get_primary_emotion from stage direction."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(happily) Hello",
            stage_directions=[StageDirection(text="happily")],
        )
        assert line.get_primary_emotion() == "happy"

    def test_get_primary_emotion_from_line_emotion(self):
        """Test get_primary_emotion from line emotion."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="Hello",
            stage_directions=[],
            emotion="sad",
        )
        assert line.get_primary_emotion() == "sad"


# =============================================================================
# ParsedScript Tests
# =============================================================================

class TestParsedScript:
    """Tests for ParsedScript dataclass."""

    def test_creation(self):
        """Test basic creation."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(speaker=speaker, text="Hello", raw_text="Hello")
        script = ParsedScript(
            lines=[line],
            speakers={"S1": speaker},
        )
        assert len(script.lines) == 1

    def test_get_speaker_lines(self):
        """Test get_speaker_lines method."""
        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S2", name="bob")
        lines = [
            DialogueLine(speaker=s1, text="Hi", raw_text="Hi"),
            DialogueLine(speaker=s2, text="Hey", raw_text="Hey"),
            DialogueLine(speaker=s1, text="Bye", raw_text="Bye"),
        ]
        script = ParsedScript(lines=lines, speakers={"S1": s1, "S2": s2})

        alice_lines = script.get_speaker_lines("S1")
        assert len(alice_lines) == 2

    def test_get_speaker_names(self):
        """Test get_speaker_names method."""
        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S2", name="bob")
        script = ParsedScript(lines=[], speakers={"S1": s1, "S2": s2})

        names = script.get_speaker_names()
        assert "alice" in names
        assert "bob" in names

    def test_speaker_count(self):
        """Test speaker_count method."""
        s1 = Speaker(id="S1", name="alice")
        s2 = Speaker(id="S2", name="bob")
        script = ParsedScript(lines=[], speakers={"S1": s1, "S2": s2})
        assert script.speaker_count() == 2

    def test_line_count(self):
        """Test line_count method."""
        speaker = Speaker(id="S1", name="test")
        lines = [
            DialogueLine(speaker=speaker, text="One", raw_text="One"),
            DialogueLine(speaker=speaker, text="Two", raw_text="Two"),
        ]
        script = ParsedScript(lines=lines, speakers={"S1": speaker})
        assert script.line_count() == 2


# =============================================================================
# DialogueParser Tests
# =============================================================================

class TestDialogueParser:
    """Tests for DialogueParser class."""

    def test_init_default(self):
        """Test default initialization."""
        parser = DialogueParser()
        assert parser.default_pause_between_speakers == 400
        assert parser.default_pause_same_speaker == 200
        assert parser.parse_simple_format is True

    def test_init_custom(self):
        """Test custom initialization."""
        parser = DialogueParser(
            default_pause_between_speakers_ms=500,
            default_pause_same_speaker_ms=100,
            parse_simple_format=False,
        )
        assert parser.default_pause_between_speakers == 500

    def test_parse_empty(self):
        """Test parsing empty script."""
        parser = DialogueParser()
        script = parser.parse("")
        assert len(script.lines) == 0

    def test_parse_single_line(self):
        """Test parsing single line."""
        parser = DialogueParser()
        script = parser.parse("[S1:narrator] Hello world")
        assert len(script.lines) == 1
        assert script.lines[0].text == "Hello world"
        assert script.lines[0].speaker.name == "narrator"

    def test_parse_multiple_speakers(self):
        """Test parsing multiple speakers."""
        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello Bob!
            [S2:bob] Hey Alice!
        """)
        assert script.speaker_count() == 2
        assert len(script.lines) == 2

    def test_parse_stage_directions(self):
        """Test parsing stage directions."""
        parser = DialogueParser()
        script = parser.parse("[S1:alice] (whispering) Be quiet!")
        assert len(script.lines) == 1
        assert len(script.lines[0].stage_directions) == 1
        assert script.lines[0].stage_directions[0].text == "whispering"

    def test_parse_continuation(self):
        """Test continuation without speaker tag."""
        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] First line.
            Second line continues.
        """)
        assert len(script.lines) == 2
        assert script.lines[1].speaker.name == "alice"

    def test_parse_default_narrator(self):
        """Test default narrator for lines without speaker."""
        parser = DialogueParser()
        script = parser.parse("Just some text")
        assert script.lines[0].speaker.name == "narrator"

    def test_parse_simple_format(self):
        """Test parsing NAME: dialogue format."""
        parser = DialogueParser(parse_simple_format=True)
        script = parser.parse("ALICE: Hello there!")
        assert len(script.lines) == 1
        assert script.lines[0].speaker.name == "alice"

    def test_parse_metadata(self):
        """Test parsing metadata."""
        parser = DialogueParser()
        script = parser.parse("""
            #title: My Script
            #author: Test
            [S1:narrator] Hello
        """)
        assert script.title == "My Script"
        assert script.metadata.get("author") == "Test"

    def test_parse_pauses(self):
        """Test pause calculation."""
        parser = DialogueParser(
            default_pause_between_speakers_ms=400,
            default_pause_same_speaker_ms=200,
        )
        script = parser.parse("""
            [S1:alice] First line.
            [S2:bob] Second line.
            [S2:bob] Third line.
        """)
        # First line has no pause before
        assert script.lines[0].pause_before_ms == 0
        # Second line (different speaker) has 400ms
        assert script.lines[1].pause_before_ms == 400
        # Third line (same speaker) has 200ms
        assert script.lines[2].pause_before_ms == 200

    def test_infer_gender_female(self):
        """Test gender inference for female names."""
        parser = DialogueParser()
        assert parser._infer_gender("alice") == "female"
        assert parser._infer_gender("emma") == "female"
        assert parser._infer_gender("princess") == "female"

    def test_infer_gender_male(self):
        """Test gender inference for male names."""
        parser = DialogueParser()
        assert parser._infer_gender("bob") == "male"
        assert parser._infer_gender("king") == "male"
        assert parser._infer_gender("michael") == "male"

    def test_infer_gender_neutral(self):
        """Test gender inference for neutral roles."""
        parser = DialogueParser()
        assert parser._infer_gender("narrator") is None


# =============================================================================
# parse_dialogue Function Tests
# =============================================================================

class TestParseDialogueFunction:
    """Tests for parse_dialogue convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        script = parse_dialogue("[S1:test] Hello world")
        assert isinstance(script, ParsedScript)
        assert len(script.lines) == 1


# =============================================================================
# VoiceGender, VoiceAge, VoiceStyle Enum Tests
# =============================================================================

class TestVoiceEnums:
    """Tests for voice enum types."""

    def test_voice_gender_values(self):
        """Test VoiceGender values."""
        assert VoiceGender.MALE.value == "male"
        assert VoiceGender.FEMALE.value == "female"
        assert VoiceGender.NEUTRAL.value == "neutral"

    def test_voice_age_values(self):
        """Test VoiceAge values."""
        assert VoiceAge.YOUNG.value == "young"
        assert VoiceAge.ADULT.value == "adult"
        assert VoiceAge.ELDERLY.value == "elderly"

    def test_voice_style_values(self):
        """Test VoiceStyle values."""
        assert VoiceStyle.NEUTRAL.value == "neutral"
        assert VoiceStyle.WARM.value == "warm"
        assert VoiceStyle.AUTHORITATIVE.value == "authoritative"


# =============================================================================
# VoiceCharacteristics Tests
# =============================================================================

class TestVoiceCharacteristics:
    """Tests for VoiceCharacteristics dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        vc = VoiceCharacteristics(
            voice_id="test_voice",
            gender=VoiceGender.FEMALE,
        )
        assert vc.voice_id == "test_voice"
        assert vc.gender == VoiceGender.FEMALE
        assert vc.age == VoiceAge.ADULT

    def test_creation_all_fields(self):
        """Test creation with all fields."""
        vc = VoiceCharacteristics(
            voice_id="test",
            gender=VoiceGender.MALE,
            age=VoiceAge.ELDERLY,
            style=VoiceStyle.AUTHORITATIVE,
            accent="british",
            quality_score=1.5,
            good_for_narrator=True,
        )
        assert vc.good_for_narrator is True
        assert vc.quality_score == 1.5


# =============================================================================
# VOICE_CHARACTERISTICS Tests
# =============================================================================

class TestVoiceCharacteristicsDict:
    """Tests for VOICE_CHARACTERISTICS dictionary."""

    def test_contains_expected_voices(self):
        """Test expected voices exist."""
        assert "af_heart" in VOICE_CHARACTERISTICS
        assert "am_michael" in VOICE_CHARACTERISTICS
        assert "bm_george" in VOICE_CHARACTERISTICS

    def test_all_are_voice_characteristics(self):
        """Test all values are VoiceCharacteristics."""
        for voice_id, chars in VOICE_CHARACTERISTICS.items():
            assert isinstance(chars, VoiceCharacteristics)
            assert chars.voice_id == voice_id


# =============================================================================
# ROLE_PREFERENCES Tests
# =============================================================================

class TestRolePreferences:
    """Tests for ROLE_PREFERENCES dictionary."""

    def test_narrator_preferences(self):
        """Test narrator role preferences."""
        assert "narrator" in ROLE_PREFERENCES
        prefs = ROLE_PREFERENCES["narrator"]
        assert prefs.get("use_good_for_narrator") is True

    def test_protagonist_preferences(self):
        """Test protagonist role preferences."""
        assert "protagonist" in ROLE_PREFERENCES

    def test_villain_preferences(self):
        """Test villain role preferences."""
        assert "villain" in ROLE_PREFERENCES


# =============================================================================
# VoiceAssigner Tests
# =============================================================================

class TestVoiceAssigner:
    """Tests for VoiceAssigner class."""

    def test_init_default(self):
        """Test default initialization."""
        assigner = VoiceAssigner()
        assert assigner.prefer_quality is True
        assert assigner.prefer_diversity is True

    def test_init_custom(self):
        """Test custom initialization."""
        assigner = VoiceAssigner(
            prefer_quality=False,
            prefer_diversity=False,
        )
        assert assigner.prefer_quality is False

    def test_reset(self):
        """Test reset method."""
        assigner = VoiceAssigner()
        assigner._assigned.add("test")
        assigner.reset()
        assert len(assigner._assigned) == 0

    def test_assign_voices_basic(self):
        """Test basic voice assignment."""
        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="alice", gender_hint="female")
        line = DialogueLine(speaker=speaker, text="Hello", raw_text="Hello")
        script = ParsedScript(lines=[line], speakers={"S1": speaker})

        assignments = assigner.assign_voices(script)
        assert "alice" in assignments
        # Should assign a female voice
        voice_id = assignments["alice"]
        assert voice_id in VOICE_CHARACTERISTICS

    def test_assign_voices_with_override(self):
        """Test voice assignment with manual override."""
        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="alice")
        line = DialogueLine(speaker=speaker, text="Hello", raw_text="Hello")
        script = ParsedScript(lines=[line], speakers={"S1": speaker})

        assignments = assigner.assign_voices(
            script,
            voice_overrides={"alice": "bm_george"}
        )
        assert assignments["alice"] == "bm_george"

    def test_assign_voices_diversity(self):
        """Test voice diversity preference."""
        assigner = VoiceAssigner(prefer_diversity=True)
        s1 = Speaker(id="S1", name="alice", gender_hint="female")
        s2 = Speaker(id="S2", name="emma", gender_hint="female")
        lines = [
            DialogueLine(speaker=s1, text="Hi", raw_text="Hi"),
            DialogueLine(speaker=s2, text="Hey", raw_text="Hey"),
        ]
        script = ParsedScript(lines=lines, speakers={"S1": s1, "S2": s2})

        assignments = assigner.assign_voices(script)
        # Should assign different voices if possible
        assert assignments["alice"] != assignments["emma"] or len(list_voices_by_gender("female")) <= 1

    def test_suggest_voices(self):
        """Test suggest_voices method."""
        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="alice", gender_hint="female")

        suggestions = assigner.suggest_voices(speaker, count=3)
        assert len(suggestions) <= 3
        assert all(isinstance(s, tuple) for s in suggestions)

    def test_calculate_match_score_gender_match(self):
        """Test match score with gender match."""
        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="test", gender_hint="female")
        voice = VoiceCharacteristics(
            voice_id="test",
            gender=VoiceGender.FEMALE,
        )
        score = assigner._calculate_match_score(speaker, voice)
        assert score > 1.0  # Should get bonus

    def test_calculate_match_score_gender_mismatch(self):
        """Test match score with gender mismatch."""
        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="test", gender_hint="female")
        voice = VoiceCharacteristics(
            voice_id="test",
            gender=VoiceGender.MALE,
        )
        score = assigner._calculate_match_score(speaker, voice)
        assert score < 1.0  # Should get penalty


# =============================================================================
# auto_assign_voices Function Tests
# =============================================================================

class TestAutoAssignVoices:
    """Tests for auto_assign_voices convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        speaker = Speaker(id="S1", name="test")
        line = DialogueLine(speaker=speaker, text="Hello", raw_text="Hello")
        script = ParsedScript(lines=[line], speakers={"S1": speaker})

        assignments = auto_assign_voices(script)
        assert "test" in assignments


# =============================================================================
# get_voice_for_gender Function Tests
# =============================================================================

class TestGetVoiceForGender:
    """Tests for get_voice_for_gender function."""

    def test_female(self):
        """Test getting female voice."""
        voice = get_voice_for_gender("female")
        assert voice == "af_heart"

    def test_male(self):
        """Test getting male voice."""
        voice = get_voice_for_gender("male")
        assert voice == "am_michael"

    def test_other(self):
        """Test getting other gender voice."""
        voice = get_voice_for_gender("neutral")
        assert voice == "bm_george"


# =============================================================================
# list_voices_by_gender Function Tests
# =============================================================================

class TestListVoicesByGender:
    """Tests for list_voices_by_gender function."""

    def test_female_voices(self):
        """Test listing female voices."""
        voices = list_voices_by_gender("female")
        assert len(voices) > 0
        for voice_id in voices:
            assert VOICE_CHARACTERISTICS[voice_id].gender == VoiceGender.FEMALE

    def test_male_voices(self):
        """Test listing male voices."""
        voices = list_voices_by_gender("male")
        assert len(voices) > 0
        for voice_id in voices:
            assert VOICE_CHARACTERISTICS[voice_id].gender == VoiceGender.MALE


# =============================================================================
# SpeakerTurn Tests
# =============================================================================

class TestSpeakerTurn:
    """Tests for SpeakerTurn dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        turn = SpeakerTurn(
            speaker_name="alice",
            text="Hello",
            voice_id="af_heart",
        )
        assert turn.speaker_name == "alice"
        assert turn.duration_seconds == 0.0

    def test_creation_with_audio(self):
        """Test creation with audio."""
        audio = np.random.randn(24000).astype(np.float32)
        turn = SpeakerTurn(
            speaker_name="alice",
            text="Hello",
            voice_id="af_heart",
            audio_samples=audio,
            duration_seconds=1.0,
        )
        assert turn.audio_samples is not None
        assert turn.duration_seconds == 1.0


# =============================================================================
# DialogueResult Tests
# =============================================================================

class TestDialogueResult:
    """Tests for DialogueResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = DialogueResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=10.5,
            turns=[],
            speaker_count=2,
            line_count=5,
        )
        assert result.speaker_count == 2
        assert result.line_count == 5

    def test_get_speaker_duration(self):
        """Test get_speaker_duration method."""
        turns = [
            SpeakerTurn(speaker_name="alice", text="Hi", voice_id="af", duration_seconds=1.0),
            SpeakerTurn(speaker_name="bob", text="Hey", voice_id="am", duration_seconds=0.5),
            SpeakerTurn(speaker_name="alice", text="Bye", voice_id="af", duration_seconds=1.5),
        ]
        result = DialogueResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=3.0,
            turns=turns,
            speaker_count=2,
            line_count=3,
        )
        assert result.get_speaker_duration("alice") == 2.5
        assert result.get_speaker_duration("bob") == 0.5


# =============================================================================
# DialogueEngine Tests
# =============================================================================

class TestDialogueEngine:
    """Tests for DialogueEngine class."""

    def test_init_default(self):
        """Test default initialization."""
        engine = DialogueEngine()
        assert engine.default_pause_ms == 400
        assert engine.narrator_pause_ms == 600
        assert engine.sample_rate == 24000

    def test_init_custom(self):
        """Test custom initialization."""
        engine = DialogueEngine(
            default_pause_ms=500,
            narrator_pause_ms=800,
            sample_rate=22050,
        )
        assert engine.default_pause_ms == 500
        assert engine.sample_rate == 22050

    def test_generate_silence(self):
        """Test silence generation."""
        engine = DialogueEngine(sample_rate=24000)
        silence = engine._generate_silence(1000)  # 1 second
        assert len(silence) == 24000

    def test_preview_assignments(self):
        """Test preview_assignments method."""
        engine = DialogueEngine()
        script = "[S1:alice] Hello"
        assignments = engine.preview_assignments(script)
        assert "alice" in assignments

    def test_get_script_info(self):
        """Test get_script_info method."""
        engine = DialogueEngine()
        script = """
            #title: Test Script
            [S1:alice] Hello Bob!
            [S2:bob] Hey Alice!
        """
        info = engine.get_script_info(script)
        assert info["speaker_count"] == 2
        assert info["line_count"] == 2
        assert info["title"] == "Test Script"
        assert "total_words" in info
        assert "estimated_duration_seconds" in info

    @patch.object(DialogueEngine, '_get_engine')
    def test_synthesize_with_mock_engine(self, mock_get_engine):
        """Test synthesize with mocked voice engine."""
        # Mock the voice engine
        mock_engine = MagicMock()
        mock_engine.speak_raw.return_value = (
            np.zeros(24000, dtype=np.float32),
            24000
        )
        mock_get_engine.return_value = mock_engine

        engine = DialogueEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dialogue.wav"
            result = engine.synthesize(
                "[S1:test] Hello world",
                output_path=output_path,
            )

            assert result.audio_path.exists()
            assert result.speaker_count == 1
            assert len(result.turns) == 1

    @patch.object(DialogueEngine, '_get_engine')
    def test_synthesize_streaming_with_mock(self, mock_get_engine):
        """Test synthesize_streaming with mocked engine."""
        mock_engine = MagicMock()
        mock_engine.speak_raw.return_value = (
            np.zeros(24000, dtype=np.float32),
            24000
        )
        mock_get_engine.return_value = mock_engine

        engine = DialogueEngine()
        script = "[S1:test] Hello"

        turns = list(engine.synthesize_streaming(script))
        assert len(turns) == 1
        assert turns[0].speaker_name == "test"
