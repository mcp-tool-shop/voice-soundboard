"""
Smoke tests: verify the package imports and core objects exist.

These run in <5 seconds with zero model downloads.
Run with: pytest tests/smoke/ -v
"""

import pytest


class TestPackageImports:
    """Verify the public API is importable."""

    def test_import_voice_engine(self):
        from voice_soundboard import VoiceEngine
        assert VoiceEngine is not None

    def test_import_speech_result(self):
        from voice_soundboard import SpeechResult
        assert SpeechResult is not None

    def test_import_config(self):
        from voice_soundboard import Config
        assert Config is not None

    def test_import_quick_speak(self):
        from voice_soundboard import quick_speak
        assert callable(quick_speak)

    def test_import_play_audio(self):
        from voice_soundboard import play_audio
        assert callable(play_audio)

    def test_import_stop_playback(self):
        from voice_soundboard import stop_playback
        assert callable(stop_playback)

    def test_import_effects(self):
        from voice_soundboard import get_effect, play_effect, list_effects
        assert callable(get_effect)
        assert callable(play_effect)
        assert callable(list_effects)

    def test_import_emotions(self):
        from voice_soundboard import get_emotion_params, list_emotions, EMOTIONS
        assert callable(get_emotion_params)
        assert callable(list_emotions)
        assert isinstance(EMOTIONS, dict)

    def test_import_presets(self):
        from voice_soundboard import VOICE_PRESETS
        assert isinstance(VOICE_PRESETS, dict)

    def test_import_voices(self):
        from voice_soundboard import KOKORO_VOICES
        assert isinstance(KOKORO_VOICES, dict)


class TestSubpackageImports:
    """Verify advanced subpackages are importable."""

    def test_import_engines(self):
        from voice_soundboard.engines import TTSEngine, KokoroEngine
        assert TTSEngine is not None
        assert KokoroEngine is not None

    def test_import_dialogue(self):
        from voice_soundboard.dialogue import DialogueParser, DialogueEngine
        assert DialogueParser is not None
        assert DialogueEngine is not None

    def test_import_emotion(self):
        from voice_soundboard.emotion import blend_emotions, EmotionCurve
        assert callable(blend_emotions)
        assert EmotionCurve is not None

    def test_import_cloning(self):
        from voice_soundboard.cloning import VoiceCloner, VoiceLibrary
        assert VoiceCloner is not None
        assert VoiceLibrary is not None

    def test_import_presets_catalog(self):
        from voice_soundboard.presets import PresetCatalog, get_catalog
        assert PresetCatalog is not None
        assert callable(get_catalog)


class TestCoreDataStructures:
    """Verify core data structures work without models."""

    def test_config_defaults(self):
        from unittest.mock import patch
        with patch('pathlib.Path.mkdir'):
            from voice_soundboard import Config
            config = Config()
            assert config.default_voice == "af_bella"
            assert config.default_speed == 1.0
            assert config.sample_rate == 24000

    def test_config_to_dict(self):
        from unittest.mock import patch
        with patch('pathlib.Path.mkdir'):
            from voice_soundboard import Config
            config = Config()
            d = config.to_dict()
            assert "output_dir" in d
            assert "device" in d
            assert "default_voice" in d

    def test_voice_presets_have_required_keys(self):
        from voice_soundboard import VOICE_PRESETS
        for name, preset in VOICE_PRESETS.items():
            assert "voice" in preset, f"Preset '{name}' missing 'voice'"
            assert "description" in preset, f"Preset '{name}' missing 'description'"

    def test_kokoro_voices_have_required_keys(self):
        from voice_soundboard import KOKORO_VOICES
        for voice_id, info in KOKORO_VOICES.items():
            assert "name" in info, f"Voice '{voice_id}' missing 'name'"
            assert "gender" in info, f"Voice '{voice_id}' missing 'gender'"
            assert "accent" in info, f"Voice '{voice_id}' missing 'accent'"

    def test_emotions_list_nonempty(self):
        from voice_soundboard import list_emotions
        emotions = list_emotions()
        assert len(emotions) > 10
        assert "happy" in emotions
        assert "calm" in emotions

    def test_effects_list_nonempty(self):
        from voice_soundboard import list_effects
        effects = list_effects()
        assert len(effects) > 0


class TestCLI:
    """Verify CLI parser works."""

    def test_cli_parser_builds(self):
        from voice_soundboard.cli import build_parser
        parser = build_parser()
        assert parser is not None

    def test_cli_parser_accepts_speak(self):
        from voice_soundboard.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["speak", "hello"])
        assert args.command == "speak"
        assert args.text == "hello"

    def test_cli_parser_accepts_options(self):
        from voice_soundboard.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "speak", "hello",
            "--voice", "bm_george",
            "--preset", "narrator",
            "--emotion", "calm",
            "--speed", "0.9",
            "-o", "test_output",
        ])
        assert args.voice == "bm_george"
        assert args.preset == "narrator"
        assert args.emotion == "calm"
        assert args.speed == 0.9
        assert args.output == "test_output"
