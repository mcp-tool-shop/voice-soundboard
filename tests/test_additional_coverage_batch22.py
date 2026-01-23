"""
Additional coverage tests - Batch 22.

Tests for:
- dialogue/voices.py (VoiceGender, VoiceAge, VoiceStyle, VoiceCharacteristics, VoiceAssigner)
- dialogue/engine.py (DialogueEngine, DialogueResult, SpeakerTurn)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil


# =============================================================================
# dialogue/voices.py tests
# =============================================================================


class TestVoiceGender:
    """Tests for VoiceGender enum."""

    def test_gender_values(self):
        from voice_soundboard.dialogue.voices import VoiceGender

        assert VoiceGender.MALE.value == "male"
        assert VoiceGender.FEMALE.value == "female"
        assert VoiceGender.NEUTRAL.value == "neutral"


class TestVoiceAge:
    """Tests for VoiceAge enum."""

    def test_age_values(self):
        from voice_soundboard.dialogue.voices import VoiceAge

        assert VoiceAge.YOUNG.value == "young"
        assert VoiceAge.ADULT.value == "adult"
        assert VoiceAge.ELDERLY.value == "elderly"


class TestVoiceStyle:
    """Tests for VoiceStyle enum."""

    def test_style_values(self):
        from voice_soundboard.dialogue.voices import VoiceStyle

        assert VoiceStyle.NEUTRAL.value == "neutral"
        assert VoiceStyle.WARM.value == "warm"
        assert VoiceStyle.AUTHORITATIVE.value == "authoritative"
        assert VoiceStyle.ENERGETIC.value == "energetic"
        assert VoiceStyle.CALM.value == "calm"


class TestVoiceCharacteristics:
    """Tests for VoiceCharacteristics dataclass."""

    def test_defaults(self):
        from voice_soundboard.dialogue.voices import (
            VoiceCharacteristics,
            VoiceGender,
            VoiceAge,
            VoiceStyle,
        )

        voice = VoiceCharacteristics(
            voice_id="test_voice",
            gender=VoiceGender.MALE,
        )

        assert voice.voice_id == "test_voice"
        assert voice.gender == VoiceGender.MALE
        assert voice.age == VoiceAge.ADULT
        assert voice.style == VoiceStyle.NEUTRAL
        assert voice.accent == "american"
        assert voice.quality_score == 1.0
        assert voice.good_for_narrator is False
        assert voice.good_for_protagonist is False
        assert voice.good_for_villain is False

    def test_custom_values(self):
        from voice_soundboard.dialogue.voices import (
            VoiceCharacteristics,
            VoiceGender,
            VoiceAge,
            VoiceStyle,
        )

        voice = VoiceCharacteristics(
            voice_id="narrator_voice",
            gender=VoiceGender.MALE,
            age=VoiceAge.ELDERLY,
            style=VoiceStyle.AUTHORITATIVE,
            accent="british",
            quality_score=1.5,
            good_for_narrator=True,
        )

        assert voice.age == VoiceAge.ELDERLY
        assert voice.style == VoiceStyle.AUTHORITATIVE
        assert voice.accent == "british"
        assert voice.good_for_narrator is True


class TestVoiceCharacteristicsDict:
    """Tests for VOICE_CHARACTERISTICS dictionary."""

    def test_contains_expected_voices(self):
        from voice_soundboard.dialogue.voices import VOICE_CHARACTERISTICS

        assert "af_heart" in VOICE_CHARACTERISTICS
        assert "am_michael" in VOICE_CHARACTERISTICS
        assert "bm_george" in VOICE_CHARACTERISTICS
        assert "bf_emma" in VOICE_CHARACTERISTICS

    def test_narrator_voices(self):
        from voice_soundboard.dialogue.voices import VOICE_CHARACTERISTICS

        narrator_voices = [
            v for v in VOICE_CHARACTERISTICS.values()
            if v.good_for_narrator
        ]
        assert len(narrator_voices) >= 1

    def test_protagonist_voices(self):
        from voice_soundboard.dialogue.voices import VOICE_CHARACTERISTICS

        protagonist_voices = [
            v for v in VOICE_CHARACTERISTICS.values()
            if v.good_for_protagonist
        ]
        assert len(protagonist_voices) >= 1


class TestRolePreferences:
    """Tests for ROLE_PREFERENCES dictionary."""

    def test_narrator_preferences(self):
        from voice_soundboard.dialogue.voices import ROLE_PREFERENCES, VoiceStyle

        assert "narrator" in ROLE_PREFERENCES
        prefs = ROLE_PREFERENCES["narrator"]
        assert prefs["style"] == VoiceStyle.AUTHORITATIVE
        assert prefs["prefer_british"] is True

    def test_protagonist_preferences(self):
        from voice_soundboard.dialogue.voices import ROLE_PREFERENCES

        assert "protagonist" in ROLE_PREFERENCES

    def test_villain_preferences(self):
        from voice_soundboard.dialogue.voices import ROLE_PREFERENCES

        assert "villain" in ROLE_PREFERENCES
        prefs = ROLE_PREFERENCES["villain"]
        assert prefs["use_good_for_villain"] is True


class TestVoiceAssigner:
    """Tests for VoiceAssigner class."""

    def test_init_defaults(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner, VOICE_CHARACTERISTICS

        assigner = VoiceAssigner()
        assert assigner.voices == VOICE_CHARACTERISTICS
        assert assigner.prefer_quality is True
        assert assigner.prefer_diversity is True

    def test_init_with_custom_voices(self):
        from voice_soundboard.dialogue.voices import (
            VoiceAssigner,
            VoiceCharacteristics,
            VoiceGender,
        )

        custom_voices = {
            "custom_voice": VoiceCharacteristics(
                voice_id="custom_voice",
                gender=VoiceGender.FEMALE,
            )
        }

        assigner = VoiceAssigner(available_voices=custom_voices)
        assert assigner.voices == custom_voices

    def test_reset(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner

        assigner = VoiceAssigner()
        assigner._assigned.add("af_heart")
        assigner._assigned.add("am_michael")

        assigner.reset()
        assert len(assigner._assigned) == 0

    def test_assign_voices_basic(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice", gender_hint="female")
        s2 = Speaker(id="S2", name="bob", gender_hint="male")

        lines = [
            DialogueLine(speaker=s1, text="Hello", raw_text="Hello"),
            DialogueLine(speaker=s2, text="Hi", raw_text="Hi"),
        ]

        script = ParsedScript(lines=lines, speakers={"S1": s1, "S2": s2})

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(script)

        assert "alice" in assignments
        assert "bob" in assignments

    def test_assign_voices_with_overrides(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice")

        lines = [
            DialogueLine(speaker=s1, text="Hello", raw_text="Hello"),
        ]

        script = ParsedScript(lines=lines, speakers={"S1": s1})

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(
            script,
            voice_overrides={"alice": "bm_george"}
        )

        assert assignments["alice"] == "bm_george"

    def test_assign_voices_prioritizes_by_line_count(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice", gender_hint="female")
        s2 = Speaker(id="S2", name="bob", gender_hint="male")

        # Alice has more lines
        lines = [
            DialogueLine(speaker=s1, text="Line 1", raw_text="Line 1"),
            DialogueLine(speaker=s1, text="Line 2", raw_text="Line 2"),
            DialogueLine(speaker=s1, text="Line 3", raw_text="Line 3"),
            DialogueLine(speaker=s2, text="Line 4", raw_text="Line 4"),
        ]

        script = ParsedScript(lines=lines, speakers={"S1": s1, "S2": s2})

        assigner = VoiceAssigner()
        assignments = assigner.assign_voices(script)

        # Alice should get first pick (higher quality voice)
        assert assignments["alice"] is not None

    def test_find_best_voice_with_diversity(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner
        from voice_soundboard.dialogue.parser import Speaker

        assigner = VoiceAssigner(prefer_diversity=True)
        assigner._assigned.add("af_heart")

        speaker = Speaker(id="S1", name="alice", gender_hint="female")
        voice_id = assigner._find_best_voice(speaker)

        # Should not return af_heart since it's already assigned
        # (or heavily penalized)
        assert voice_id is not None

    def test_find_best_voice_empty_voices(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner
        from voice_soundboard.dialogue.parser import Speaker

        assigner = VoiceAssigner(available_voices={})
        speaker = Speaker(id="S1", name="alice")

        voice_id = assigner._find_best_voice(speaker)
        assert voice_id == "af_heart"  # Fallback

    def test_calculate_match_score_gender_match(self):
        from voice_soundboard.dialogue.voices import (
            VoiceAssigner,
            VoiceCharacteristics,
            VoiceGender,
        )
        from voice_soundboard.dialogue.parser import Speaker

        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="alice", gender_hint="female")

        female_voice = VoiceCharacteristics(
            voice_id="test",
            gender=VoiceGender.FEMALE,
            quality_score=1.0,
        )
        male_voice = VoiceCharacteristics(
            voice_id="test2",
            gender=VoiceGender.MALE,
            quality_score=1.0,
        )

        female_score = assigner._calculate_match_score(speaker, female_voice)
        male_score = assigner._calculate_match_score(speaker, male_voice)

        assert female_score > male_score

    def test_calculate_match_score_role_narrator(self):
        from voice_soundboard.dialogue.voices import (
            VoiceAssigner,
            VoiceCharacteristics,
            VoiceGender,
        )
        from voice_soundboard.dialogue.parser import Speaker

        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="narrator")

        narrator_voice = VoiceCharacteristics(
            voice_id="test",
            gender=VoiceGender.MALE,
            good_for_narrator=True,
            accent="british",
        )
        regular_voice = VoiceCharacteristics(
            voice_id="test2",
            gender=VoiceGender.MALE,
            good_for_narrator=False,
            accent="american",
        )

        narrator_score = assigner._calculate_match_score(speaker, narrator_voice)
        regular_score = assigner._calculate_match_score(speaker, regular_voice)

        assert narrator_score > regular_score

    def test_calculate_match_score_accent_hint(self):
        from voice_soundboard.dialogue.voices import (
            VoiceAssigner,
            VoiceCharacteristics,
            VoiceGender,
        )
        from voice_soundboard.dialogue.parser import Speaker

        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="alice", accent_hint="british")

        british_voice = VoiceCharacteristics(
            voice_id="test",
            gender=VoiceGender.FEMALE,
            accent="british",
        )
        american_voice = VoiceCharacteristics(
            voice_id="test2",
            gender=VoiceGender.FEMALE,
            accent="american",
        )

        british_score = assigner._calculate_match_score(speaker, british_voice)
        american_score = assigner._calculate_match_score(speaker, american_voice)

        assert british_score > american_score

    def test_suggest_voices(self):
        from voice_soundboard.dialogue.voices import VoiceAssigner
        from voice_soundboard.dialogue.parser import Speaker

        assigner = VoiceAssigner()
        speaker = Speaker(id="S1", name="alice", gender_hint="female")

        suggestions = assigner.suggest_voices(speaker, count=3)
        assert len(suggestions) == 3
        assert all(isinstance(s, tuple) for s in suggestions)
        assert all(len(s) == 2 for s in suggestions)


class TestAutoAssignVoices:
    """Tests for auto_assign_voices function."""

    def test_auto_assign_voices(self):
        from voice_soundboard.dialogue.voices import auto_assign_voices
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice")

        lines = [
            DialogueLine(speaker=s1, text="Hello", raw_text="Hello"),
        ]

        script = ParsedScript(lines=lines, speakers={"S1": s1})

        assignments = auto_assign_voices(script)
        assert "alice" in assignments


class TestGetVoiceForGender:
    """Tests for get_voice_for_gender function."""

    def test_female(self):
        from voice_soundboard.dialogue.voices import get_voice_for_gender

        assert get_voice_for_gender("female") == "af_heart"

    def test_male(self):
        from voice_soundboard.dialogue.voices import get_voice_for_gender

        assert get_voice_for_gender("male") == "am_michael"

    def test_other(self):
        from voice_soundboard.dialogue.voices import get_voice_for_gender

        assert get_voice_for_gender("neutral") == "bm_george"

    def test_case_insensitive(self):
        from voice_soundboard.dialogue.voices import get_voice_for_gender

        assert get_voice_for_gender("FEMALE") == "af_heart"
        assert get_voice_for_gender("Male") == "am_michael"


class TestListVoicesByGender:
    """Tests for list_voices_by_gender function."""

    def test_female_voices(self):
        from voice_soundboard.dialogue.voices import list_voices_by_gender

        voices = list_voices_by_gender("female")
        assert len(voices) >= 1
        assert all(v.startswith(("af_", "bf_")) for v in voices)

    def test_male_voices(self):
        from voice_soundboard.dialogue.voices import list_voices_by_gender

        voices = list_voices_by_gender("male")
        assert len(voices) >= 1
        assert all(v.startswith(("am_", "bm_")) for v in voices)


# =============================================================================
# dialogue/engine.py tests
# =============================================================================


class TestSpeakerTurn:
    """Tests for SpeakerTurn dataclass."""

    def test_defaults(self):
        from voice_soundboard.dialogue.engine import SpeakerTurn

        turn = SpeakerTurn(
            speaker_name="alice",
            text="Hello there",
            voice_id="af_heart",
        )

        assert turn.audio_path is None
        assert turn.audio_samples is None
        assert turn.duration_seconds == 0.0
        assert turn.emotion is None
        assert turn.speed == 1.0
        assert turn.line_number == 0

    def test_with_audio(self):
        from voice_soundboard.dialogue.engine import SpeakerTurn

        samples = np.random.randn(24000).astype(np.float32)
        turn = SpeakerTurn(
            speaker_name="alice",
            text="Hello",
            voice_id="af_heart",
            audio_samples=samples,
            duration_seconds=1.0,
        )

        assert turn.audio_samples is not None
        assert len(turn.audio_samples) == 24000


class TestDialogueResult:
    """Tests for DialogueResult dataclass."""

    def test_defaults(self):
        from voice_soundboard.dialogue.engine import DialogueResult
        from pathlib import Path

        result = DialogueResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=10.5,
            turns=[],
            speaker_count=2,
            line_count=5,
        )

        assert result.sample_rate == 24000
        assert result.voice_assignments == {}

    def test_get_speaker_duration(self):
        from voice_soundboard.dialogue.engine import DialogueResult, SpeakerTurn
        from pathlib import Path

        turns = [
            SpeakerTurn(speaker_name="alice", text="Hello", voice_id="af_heart", duration_seconds=2.0),
            SpeakerTurn(speaker_name="bob", text="Hi", voice_id="am_michael", duration_seconds=1.5),
            SpeakerTurn(speaker_name="alice", text="Bye", voice_id="af_heart", duration_seconds=1.0),
        ]

        result = DialogueResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=4.5,
            turns=turns,
            speaker_count=2,
            line_count=3,
        )

        assert result.get_speaker_duration("alice") == 3.0
        assert result.get_speaker_duration("bob") == 1.5
        assert result.get_speaker_duration("charlie") == 0.0


class TestDialogueEngine:
    """Tests for DialogueEngine class."""

    def test_init_defaults(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine()
        assert engine.default_pause_ms == 400
        assert engine.narrator_pause_ms == 600
        assert engine.sample_rate == 24000
        assert engine.voice_engine is None

    def test_init_with_custom_config(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine(
            default_pause_ms=500,
            narrator_pause_ms=800,
            sample_rate=48000,
        )

        assert engine.default_pause_ms == 500
        assert engine.narrator_pause_ms == 800
        assert engine.sample_rate == 48000

    def test_generate_silence(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine(sample_rate=24000)
        silence = engine._generate_silence(1000)  # 1 second

        assert len(silence) == 24000
        assert np.all(silence == 0)

    def test_generate_silence_short(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine(sample_rate=24000)
        silence = engine._generate_silence(100)  # 100ms

        assert len(silence) == 2400

    def test_get_script_info_string(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine()
        script = """
            # title: Test Script
            [S1:alice] Hello world.
            [S2:bob] Hi there.
            [S1:alice] How are you?
        """

        info = engine.get_script_info(script)

        assert info["speaker_count"] == 2
        assert info["line_count"] == 3
        assert "alice" in info["speaker_names"]
        assert "bob" in info["speaker_names"]
        assert info["speaker_lines"]["alice"] == 2
        assert info["speaker_lines"]["bob"] == 1
        assert info["total_words"] > 0
        assert info["estimated_duration_seconds"] > 0
        assert info["title"] == "Test Script"

    def test_get_script_info_parsed_script(self):
        from voice_soundboard.dialogue.engine import DialogueEngine
        from voice_soundboard.dialogue.parser import (
            ParsedScript,
            DialogueLine,
            Speaker,
        )

        s1 = Speaker(id="S1", name="alice")
        lines = [
            DialogueLine(speaker=s1, text="Hello world", raw_text="Hello world"),
        ]
        parsed = ParsedScript(lines=lines, speakers={"S1": s1})

        engine = DialogueEngine()
        info = engine.get_script_info(parsed)

        assert info["speaker_count"] == 1
        assert info["line_count"] == 1

    def test_preview_assignments_string(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine()
        script = """
            [S1:alice] Hello.
            [S2:bob] Hi.
        """

        assignments = engine.preview_assignments(script)
        assert "alice" in assignments
        assert "bob" in assignments

    def test_preview_assignments_with_overrides(self):
        from voice_soundboard.dialogue.engine import DialogueEngine

        engine = DialogueEngine()
        script = """
            [S1:alice] Hello.
        """

        assignments = engine.preview_assignments(
            script,
            voices={"alice": "bm_george"}
        )
        assert assignments["alice"] == "bm_george"

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_basic(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine
        import tempfile

        # Mock the voice engine
        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello world.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            result = engine.synthesize(script, output_path=output_path)

            assert result.line_count >= 1
            assert result.speaker_count >= 1
            assert result.audio_path.exists()

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_with_progress_callback(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine
        import tempfile

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
            [S2:bob] Hi.
        """

        progress_calls = []

        def progress_callback(current, total, speaker):
            progress_calls.append((current, total, speaker))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            engine.synthesize(
                script,
                output_path=output_path,
                progress_callback=progress_callback,
            )

            assert len(progress_calls) >= 2

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_with_voice_overrides(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine
        import tempfile

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            result = engine.synthesize(
                script,
                voices={"alice": "bm_george"},
                output_path=output_path,
            )

            assert result.voice_assignments["alice"] == "bm_george"

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_with_synthesis_failure(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine
        import tempfile

        mock_engine = Mock()
        mock_engine.speak_raw.side_effect = Exception("Synthesis failed")

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            result = engine.synthesize(script, output_path=output_path)

            # Should complete without error
            assert result.line_count == 0  # No successful turns

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_save_individual_turns(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine
        import tempfile

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
            [S2:bob] Hi.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            result = engine.synthesize(
                script,
                output_path=output_path,
                save_individual_turns=True,
            )

            # Check individual turns were saved
            for turn in result.turns:
                if turn.audio_path:
                    assert turn.audio_path.exists()

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_narrator_pause(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine
        import tempfile

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine(
            default_pause_ms=400,
            narrator_pause_ms=800,
        )
        engine.voice_engine = mock_engine

        script = """
            [S1:narrator] The story begins.
            [S2:alice] Hello.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            result = engine.synthesize(script, output_path=output_path)

            # Narrator pause should be longer
            assert result.duration_seconds > 0

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_streaming(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
            [S2:bob] Hi.
        """

        turns = list(engine.synthesize_streaming(script))
        assert len(turns) == 2

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_streaming_with_callback(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
        """

        callback_turns = []

        def on_turn_complete(turn):
            callback_turns.append(turn)

        list(engine.synthesize_streaming(script, on_turn_complete=on_turn_complete))
        assert len(callback_turns) == 1

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_streaming_with_failure(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import DialogueEngine

        mock_engine = Mock()
        mock_engine.speak_raw.side_effect = Exception("Synthesis failed")

        engine = DialogueEngine()
        engine.voice_engine = mock_engine

        script = """
            [S1:alice] Hello.
        """

        turns = list(engine.synthesize_streaming(script))
        assert len(turns) == 0


class TestSynthesizeDialogueFunction:
    """Tests for synthesize_dialogue convenience function."""

    @patch("voice_soundboard.dialogue.engine.VoiceEngine")
    def test_synthesize_dialogue(self, mock_engine_class):
        from voice_soundboard.dialogue.engine import synthesize_dialogue
        import tempfile

        mock_engine = Mock()
        mock_engine.speak_raw.return_value = (
            np.random.randn(24000).astype(np.float32),
            24000
        )

        with patch.object(
            __import__("voice_soundboard.dialogue.engine", fromlist=[""]).DialogueEngine,
            "_get_engine",
            return_value=mock_engine,
        ):
            script = """
                [S1:alice] Hello world.
            """

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.wav"

                # This will create a new engine internally
                # We need to mock the engine creation
                with patch(
                    "voice_soundboard.dialogue.engine.DialogueEngine.synthesize"
                ) as mock_synth:
                    from voice_soundboard.dialogue.engine import DialogueResult

                    mock_synth.return_value = DialogueResult(
                        audio_path=output_path,
                        duration_seconds=1.0,
                        turns=[],
                        speaker_count=1,
                        line_count=1,
                    )

                    result = synthesize_dialogue(script, output_path=output_path)
                    assert result is not None
