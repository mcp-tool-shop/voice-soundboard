"""
Additional coverage tests - Batch 42: Dialogue System Coverage.

Comprehensive tests for:
- voice_soundboard/dialogue/parser.py
- voice_soundboard/dialogue/voices.py
- voice_soundboard/dialogue/engine.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# StageDirection Tests
# =============================================================================

class TestStageDirection:
    """Tests for StageDirection dataclass."""

    def test_stage_direction_whispering(self):
        """Test whispering stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="whispering")
        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier < 1.0
        assert direction.emotion_hint == "whisper"

    def test_stage_direction_shouting(self):
        """Test shouting stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="shouting")
        assert direction.direction_type == StageDirectionType.VOLUME
        assert direction.volume_modifier > 1.0

    def test_stage_direction_slowly(self):
        """Test slowly stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="slowly")
        assert direction.direction_type == StageDirectionType.PACE
        assert direction.speed_modifier < 1.0

    def test_stage_direction_quickly(self):
        """Test quickly stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="quickly")
        assert direction.direction_type == StageDirectionType.PACE
        assert direction.speed_modifier > 1.0

    def test_stage_direction_sadly(self):
        """Test sadly stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="sadly")
        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "sad"

    def test_stage_direction_happily(self):
        """Test happily stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="happily")
        assert direction.emotion_hint == "happy"

    def test_stage_direction_angrily(self):
        """Test angrily stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="angrily")
        assert direction.emotion_hint == "angry"

    def test_stage_direction_pauses(self):
        """Test pauses stage direction."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="pauses")
        assert direction.direction_type == StageDirectionType.ACTION

    def test_stage_direction_unknown(self):
        """Test unknown stage direction defaults to emotion."""
        from voice_soundboard.dialogue.parser import StageDirection, StageDirectionType

        direction = StageDirection(text="mysteriously")
        assert direction.direction_type == StageDirectionType.EMOTION
        assert direction.emotion_hint == "mysteriously"


# =============================================================================
# Speaker Tests
# =============================================================================

class TestSpeaker:
    """Tests for Speaker dataclass."""

    def test_speaker_creation(self):
        """Test creating a Speaker."""
        from voice_soundboard.dialogue.parser import Speaker

        speaker = Speaker(id="S1", name="narrator")
        assert speaker.id == "S1"
        assert speaker.name == "narrator"

    def test_speaker_with_characteristics(self):
        """Test Speaker with characteristic hints."""
        from voice_soundboard.dialogue.parser import Speaker

        speaker = Speaker(
            id="S2",
            name="alice",
            gender_hint="female",
            age_hint="young",
        )
        assert speaker.gender_hint == "female"
        assert speaker.age_hint == "young"

    def test_speaker_equality(self):
        """Test Speaker equality based on ID."""
        from voice_soundboard.dialogue.parser import Speaker

        s1 = Speaker(id="S1", name="narrator")
        s2 = Speaker(id="S1", name="different")
        s3 = Speaker(id="S2", name="narrator")

        assert s1 == s2  # Same ID
        assert s1 != s3  # Different ID

    def test_speaker_hash(self):
        """Test Speaker hashing."""
        from voice_soundboard.dialogue.parser import Speaker

        s1 = Speaker(id="S1", name="narrator")
        s2 = Speaker(id="S1", name="narrator")

        # Same ID should hash the same
        assert hash(s1) == hash(s2)


# =============================================================================
# DialogueLine Tests
# =============================================================================

class TestDialogueLine:
    """Tests for DialogueLine dataclass."""

    def test_dialogue_line_creation(self):
        """Test creating a DialogueLine."""
        from voice_soundboard.dialogue.parser import DialogueLine, Speaker

        speaker = Speaker(id="S1", name="narrator")
        line = DialogueLine(
            speaker=speaker,
            text="The door creaked open.",
            raw_text="[S1:narrator] The door creaked open.",
            line_number=1,
        )
        assert line.text == "The door creaked open."
        assert line.speaker.name == "narrator"

    def test_dialogue_line_has_stage_directions(self):
        """Test has_stage_directions method."""
        from voice_soundboard.dialogue.parser import DialogueLine, Speaker, StageDirection

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
            raw_text="(whispered) Hello",
            stage_directions=[StageDirection(text="whispered")],
        )
        assert line_with.has_stage_directions() is True

    def test_dialogue_line_get_primary_emotion(self):
        """Test get_primary_emotion method."""
        from voice_soundboard.dialogue.parser import DialogueLine, Speaker, StageDirection

        speaker = Speaker(id="S1", name="narrator")

        # With stage direction
        line = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="(sadly) Hello",
            stage_directions=[StageDirection(text="sadly")],
        )
        assert line.get_primary_emotion() == "sad"

        # Without stage direction but with emotion attribute
        line2 = DialogueLine(
            speaker=speaker,
            text="Hello",
            raw_text="Hello",
            emotion="happy",
        )
        assert line2.get_primary_emotion() == "happy"


# =============================================================================
# DialogueParser Tests
# =============================================================================

class TestDialogueParser:
    """Tests for DialogueParser class."""

    def test_parser_basic_script(self):
        """Test parsing a basic script."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] The room was dark.
            [S2:alice] Hello? Is anyone there?
        """)

        assert script.line_count() == 2
        assert script.speaker_count() == 2
        assert "narrator" in script.get_speaker_names()
        assert "alice" in script.get_speaker_names()

    def test_parser_stage_directions(self):
        """Test parsing stage directions."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] (whispering) Don't make a sound.
        """)

        assert len(script.lines) == 1
        assert script.lines[0].has_stage_directions()
        assert script.lines[0].stage_directions[0].text == "whispering"

    def test_parser_multiple_stage_directions(self):
        """Test parsing multiple stage directions."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:bob] (loudly) (angrily) Get out of here!
        """)

        assert len(script.lines[0].stage_directions) == 2

    def test_parser_continuation_lines(self):
        """Test continuation without new speaker tag."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] The sun was setting.
            The sky turned orange.
        """)

        assert script.line_count() == 2
        # Both lines should be from narrator
        assert script.lines[0].speaker.name == "narrator"
        assert script.lines[1].speaker.name == "narrator"

    def test_parser_simple_format(self):
        """Test parsing NAME: dialogue format."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser(parse_simple_format=True)
        script = parser.parse("""
            NARRATOR: The story begins.
            ALICE: Where are we?
        """)

        assert script.line_count() == 2
        assert "narrator" in script.get_speaker_names()

    def test_parser_metadata_extraction(self):
        """Test extracting metadata from script."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            # title: My Story
            # author: Test Author
            [S1:narrator] Once upon a time...
        """)

        assert script.title == "My Story"
        assert script.metadata.get("author") == "Test Author"

    def test_parser_empty_lines(self):
        """Test that empty lines are skipped."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] First line.

            [S1:narrator] Second line.
        """)

        assert script.line_count() == 2

    def test_parser_gender_inference_female(self):
        """Test gender inference for female names."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello!
            [S2:emma] Hi!
        """)

        alice = script.speakers["S1"]
        assert alice.gender_hint == "female"

    def test_parser_gender_inference_male(self):
        """Test gender inference for male names."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:bob] Hello!
            [S2:john] Hi!
        """)

        bob = script.speakers["S1"]
        assert bob.gender_hint == "male"

    def test_parser_neutral_roles(self):
        """Test neutral roles like narrator."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] The story begins.
        """)

        narrator = script.speakers["S1"]
        # Narrator should be gender neutral
        assert narrator.gender_hint is None


# =============================================================================
# ParsedScript Tests
# =============================================================================

class TestParsedScript:
    """Tests for ParsedScript class."""

    def test_script_get_speaker_lines(self):
        """Test getting lines for specific speaker."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] First alice line.
            [S2:bob] First bob line.
            [S1:alice] Second alice line.
        """)

        alice_lines = script.get_speaker_lines("S1")
        assert len(alice_lines) == 2

    def test_script_speaker_count(self):
        """Test speaker_count method."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello.
            [S2:bob] Hi.
            [S3:charlie] Hey.
        """)

        assert script.speaker_count() == 3

    def test_script_line_count(self):
        """Test line_count method."""
        from voice_soundboard.dialogue.parser import DialogueParser

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] Line one.
            [S1:narrator] Line two.
            [S1:narrator] Line three.
        """)

        assert script.line_count() == 3


# =============================================================================
# parse_dialogue Function Tests
# =============================================================================

class TestParseDialogueFunction:
    """Tests for parse_dialogue convenience function."""

    def test_parse_dialogue(self):
        """Test parse_dialogue function."""
        from voice_soundboard.dialogue.parser import parse_dialogue

        script = parse_dialogue("""
            [S1:narrator] Hello world.
        """)

        assert script.line_count() == 1


# =============================================================================
# VoiceCharacteristics Tests
# =============================================================================

class TestVoiceCharacteristics:
    """Tests for VoiceCharacteristics dataclass."""

    def test_voice_characteristics_creation(self):
        """Test creating VoiceCharacteristics."""
        from voice_soundboard.dialogue.voices import (
            VoiceCharacteristics, VoiceGender, VoiceAge, VoiceStyle
        )

        chars = VoiceCharacteristics(
            voice_id="af_bella",
            gender=VoiceGender.FEMALE,
            age=VoiceAge.ADULT,
            style=VoiceStyle.WARM,
        )
        assert chars.voice_id == "af_bella"
        assert chars.gender == VoiceGender.FEMALE


# =============================================================================
# VoiceAssigner Tests
# =============================================================================

class TestVoiceAssigner:
    """Tests for VoiceAssigner class."""

    def test_assigner_creation(self):
        """Test creating a VoiceAssigner."""
        from voice_soundboard.dialogue.voices import VoiceAssigner

        assigner = VoiceAssigner()
        assert assigner is not None

    def test_assigner_assign_voices(self):
        """Test assigning voices to speakers."""
        from voice_soundboard.dialogue.parser import DialogueParser
        from voice_soundboard.dialogue.voices import VoiceAssigner

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello!
            [S2:bob] Hi!
        """)

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(script)

        assert "alice" in assignments
        assert "bob" in assignments
        assert assignments["alice"] != assignments["bob"]  # Should be different

    def test_assigner_with_overrides(self):
        """Test voice assignment with manual overrides."""
        from voice_soundboard.dialogue.parser import DialogueParser
        from voice_soundboard.dialogue.voices import VoiceAssigner

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello!
        """)

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(
            script,
            voice_overrides={"alice": "bf_emma"}
        )

        assert assignments["alice"] == "bf_emma"

    def test_assigner_gender_matching(self):
        """Test that gender hints are respected."""
        from voice_soundboard.dialogue.parser import DialogueParser
        from voice_soundboard.dialogue.voices import VoiceAssigner

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello!
        """)

        # Alice should get female gender hint
        assert script.speakers["S1"].gender_hint == "female"

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(script)

        # Should be assigned a female voice (af_*)
        assert assignments["alice"].startswith("af") or assignments["alice"].startswith("bf")

    def test_assigner_diversity(self):
        """Test that voice diversity is maintained."""
        from voice_soundboard.dialogue.parser import DialogueParser
        from voice_soundboard.dialogue.voices import VoiceAssigner

        parser = DialogueParser()
        script = parser.parse("""
            [S1:alice] Hello!
            [S2:emma] Hi!
            [S3:sarah] Hey!
        """)

        assigner = VoiceAssigner(prefer_diversity=True)
        assignments = assigner.assign_voices(script)

        # All three should ideally have different voices
        voices = list(assignments.values())
        # At least some diversity
        assert len(set(voices)) >= 2

    def test_assigner_suggest_voices(self):
        """Test voice suggestion method."""
        from voice_soundboard.dialogue.parser import Speaker
        from voice_soundboard.dialogue.voices import VoiceAssigner

        speaker = Speaker(id="S1", name="alice", gender_hint="female")

        assigner = VoiceAssigner()
        suggestions = assigner.suggest_voices(speaker, count=3)

        assert len(suggestions) <= 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in suggestions)

    def test_assigner_reset(self):
        """Test reset method."""
        from voice_soundboard.dialogue.parser import DialogueParser
        from voice_soundboard.dialogue.voices import VoiceAssigner

        parser = DialogueParser()
        script = parser.parse("[S1:alice] Hello!")

        assigner = VoiceAssigner()
        assigner.assign_voices(script)

        # Assigned set should not be empty
        assert len(assigner._assigned) > 0

        assigner.reset()
        assert len(assigner._assigned) == 0


# =============================================================================
# Auto-Assign and Helper Function Tests
# =============================================================================

class TestVoiceHelperFunctions:
    """Tests for voice helper functions."""

    def test_auto_assign_voices(self):
        """Test auto_assign_voices function."""
        from voice_soundboard.dialogue.parser import DialogueParser
        from voice_soundboard.dialogue.voices import auto_assign_voices

        parser = DialogueParser()
        script = parser.parse("""
            [S1:narrator] The story.
            [S2:hero] I will save the day!
        """)

        assignments = auto_assign_voices(script)
        assert len(assignments) == 2

    def test_get_voice_for_gender(self):
        """Test get_voice_for_gender function."""
        from voice_soundboard.dialogue.voices import get_voice_for_gender

        female_voice = get_voice_for_gender("female")
        assert "f" in female_voice  # Female voices have 'f' in ID

        male_voice = get_voice_for_gender("male")
        assert "m" in male_voice  # Male voices have 'm' in ID

    def test_list_voices_by_gender(self):
        """Test list_voices_by_gender function."""
        from voice_soundboard.dialogue.voices import list_voices_by_gender

        female_voices = list_voices_by_gender("female")
        assert len(female_voices) > 0
        # All should be female voices
        for v in female_voices:
            assert "f" in v

        male_voices = list_voices_by_gender("male")
        assert len(male_voices) > 0


# =============================================================================
# Voice Enums Tests
# =============================================================================

class TestVoiceEnums:
    """Tests for voice-related enums."""

    def test_voice_gender_enum(self):
        """Test VoiceGender enum."""
        from voice_soundboard.dialogue.voices import VoiceGender

        assert VoiceGender.MALE.value == "male"
        assert VoiceGender.FEMALE.value == "female"
        assert VoiceGender.NEUTRAL.value == "neutral"

    def test_voice_age_enum(self):
        """Test VoiceAge enum."""
        from voice_soundboard.dialogue.voices import VoiceAge

        assert VoiceAge.YOUNG.value == "young"
        assert VoiceAge.ADULT.value == "adult"
        assert VoiceAge.ELDERLY.value == "elderly"

    def test_voice_style_enum(self):
        """Test VoiceStyle enum."""
        from voice_soundboard.dialogue.voices import VoiceStyle

        assert VoiceStyle.NEUTRAL.value == "neutral"
        assert VoiceStyle.WARM.value == "warm"
        assert VoiceStyle.AUTHORITATIVE.value == "authoritative"


# =============================================================================
# StageDirectionType Enum Tests
# =============================================================================

class TestStageDirectionType:
    """Tests for StageDirectionType enum."""

    def test_stage_direction_types(self):
        """Test all stage direction types exist."""
        from voice_soundboard.dialogue.parser import StageDirectionType

        assert StageDirectionType.EMOTION.value == "emotion"
        assert StageDirectionType.VOLUME.value == "volume"
        assert StageDirectionType.PACE.value == "pace"
        assert StageDirectionType.ACTION.value == "action"


# =============================================================================
# Role Preferences Tests
# =============================================================================

class TestRolePreferences:
    """Tests for role-based voice preferences."""

    def test_narrator_preferences(self):
        """Test narrator role preferences."""
        from voice_soundboard.dialogue.voices import ROLE_PREFERENCES

        assert "narrator" in ROLE_PREFERENCES
        prefs = ROLE_PREFERENCES["narrator"]
        assert prefs.get("use_good_for_narrator") is True

    def test_protagonist_preferences(self):
        """Test protagonist role preferences."""
        from voice_soundboard.dialogue.voices import ROLE_PREFERENCES

        assert "protagonist" in ROLE_PREFERENCES
        prefs = ROLE_PREFERENCES["protagonist"]
        assert prefs.get("use_good_for_protagonist") is True

    def test_villain_preferences(self):
        """Test villain role preferences."""
        from voice_soundboard.dialogue.voices import ROLE_PREFERENCES

        assert "villain" in ROLE_PREFERENCES
        prefs = ROLE_PREFERENCES["villain"]
        assert prefs.get("use_good_for_villain") is True


# =============================================================================
# Voice Characteristics Database Tests
# =============================================================================

class TestVoiceCharacteristicsDatabase:
    """Tests for VOICE_CHARACTERISTICS database."""

    def test_characteristics_exist(self):
        """Test that voice characteristics database exists."""
        from voice_soundboard.dialogue.voices import VOICE_CHARACTERISTICS

        assert len(VOICE_CHARACTERISTICS) > 0

    def test_common_voices_present(self):
        """Test common voices are in database."""
        from voice_soundboard.dialogue.voices import VOICE_CHARACTERISTICS

        common_voices = ["af_bella", "am_michael", "bm_george"]
        for voice in common_voices:
            assert voice in VOICE_CHARACTERISTICS

    def test_voice_characteristics_complete(self):
        """Test voice characteristics have required fields."""
        from voice_soundboard.dialogue.voices import VOICE_CHARACTERISTICS, VoiceGender

        for voice_id, chars in VOICE_CHARACTERISTICS.items():
            assert chars.voice_id == voice_id
            assert isinstance(chars.gender, VoiceGender)
            assert 0 <= chars.quality_score <= 2.0
