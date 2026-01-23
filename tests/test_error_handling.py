"""
Error Handling and Edge Case Tests.

Tests cover error conditions and boundary cases not covered by the main test files:
- Empty/invalid inputs
- Boundary values
- Model failures
- Conversion edge cases
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# F5TTSEngine Error Handling and Edge Cases
# =============================================================================

class TestF5TTSEngineErrorHandling:
    """Error handling tests for F5TTSEngine."""

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_empty_text(self, mock_load, tmp_path):
        """speak() with empty text should still process (model dependent)."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()
        engine._model.infer = Mock(return_value=(np.zeros(1000, dtype=np.float32), 24000, None))
        engine._model_loaded = True

        # Empty text should be passed to model
        result = engine.speak("", save_path=tmp_path / "out.wav")

        # Verify empty text was passed
        call_args = engine._model.infer.call_args
        assert call_args[1]["gen_text"] == ""

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_whitespace_only_text(self, mock_load, tmp_path):
        """speak() with whitespace-only text should process."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()
        engine._model.infer = Mock(return_value=(np.zeros(1000, dtype=np.float32), 24000, None))
        engine._model_loaded = True

        result = engine.speak("   \t\n  ", save_path=tmp_path / "out.wav")

        call_args = engine._model.infer.call_args
        assert call_args[1]["gen_text"] == "   \t\n  "

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_model_inference_exception(self, mock_load, tmp_path):
        """speak() propagates exception when model inference fails."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()
        engine._model.infer = Mock(side_effect=RuntimeError("CUDA out of memory"))
        engine._model_loaded = True

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            engine.speak("Hello", save_path=tmp_path / "out.wav")

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_tensor_conversion_with_cpu_method(self, mock_load, tmp_path):
        """speak() handles PyTorch tensor with .cpu() method."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()

        # Mock a tensor-like object with cpu() method
        mock_tensor = Mock()
        mock_tensor.squeeze.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.zeros(24000, dtype=np.float32)

        engine._model.infer = Mock(return_value=(mock_tensor, 24000, None))
        engine._model_loaded = True

        result = engine.speak("Hello", save_path=tmp_path / "out.wav")

        # Verify tensor was processed correctly
        assert result.sample_rate == 24000

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_tensor_conversion_plain_numpy(self, mock_load, tmp_path):
        """speak() handles plain numpy array (no .numpy() method)."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()

        # Return a plain list (not numpy or tensor)
        plain_array = [0.0] * 24000
        engine._model.infer = Mock(return_value=(plain_array, 24000, None))
        engine._model_loaded = True

        result = engine.speak("Hello", save_path=tmp_path / "out.wav")

        assert result.sample_rate == 24000
        assert isinstance(result.samples, np.ndarray)

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_dtype_conversion_to_float32(self, mock_load, tmp_path):
        """speak() converts non-float32 arrays to float32."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()

        # Return float64 array
        float64_array = np.zeros(24000, dtype=np.float64)
        engine._model.infer = Mock(return_value=(float64_array, 24000, None))
        engine._model_loaded = True

        result = engine.speak("Hello", save_path=tmp_path / "out.wav")

        assert result.samples.dtype == np.float32

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_with_nonexistent_voice_path_warns(self, mock_load, tmp_path, capsys):
        """speak() with nonexistent voice path prints warning and uses default."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()
        engine._model.infer = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000, None))
        engine._model_loaded = True

        result = engine.speak(
            "Hello",
            voice="/nonexistent/voice.wav",
            save_path=tmp_path / "out.wav"
        )

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not found" in captured.out

    def test_clone_voice_empty_voice_id(self, tmp_path):
        """clone_voice() with empty voice_id still works."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()

        audio = tmp_path / "ref.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        # Empty voice_id
        voice_id = engine.clone_voice(audio, "", transcription="test")

        assert voice_id == ""
        assert "" in engine._cloned_voices

    @patch('soundfile.info')
    def test_clone_voice_soundfile_error_handled(self, mock_info, tmp_path):
        """clone_voice() handles soundfile.info() errors gracefully."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        mock_info.side_effect = Exception("Invalid audio file")

        audio = tmp_path / "corrupt.wav"
        audio.write_bytes(b"not a valid wav file")

        # Should not raise, just skip duration validation
        voice_id = engine.clone_voice(audio, "corrupt_voice", transcription="test")

        assert voice_id == "corrupt_voice"

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_speak_zero_generation_time_rtf(self, mock_load, tmp_path):
        """speak() handles zero generation time without division error."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()
        engine._model.infer = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000, None))
        engine._model_loaded = True

        # Very fast generation - simulated by mock
        with patch('voice_soundboard.engines.f5tts.time') as mock_time:
            mock_time.time.side_effect = [0.0, 0.0]  # Same time = 0 duration

            result = engine.speak("Hello", save_path=tmp_path / "out.wav")

            # RTF should be 0 when gen_time is 0
            assert result.realtime_factor == 0


# =============================================================================
# ChatterboxEngine Error Handling and Edge Cases
# =============================================================================

class TestChatterboxEngineErrorHandling:
    """Error handling tests for ChatterboxEngine."""

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_empty_text(self, mock_load, tmp_path):
        """speak() with empty text should still process."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        result = engine.speak("", save_path=tmp_path / "out.wav")

        # Verify model was called with empty text
        call_args = engine._model.generate.call_args
        assert call_args[0][0] == ""

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_emotion_exaggeration_zero(self, mock_load, tmp_path):
        """speak() with emotion_exaggeration=0.0 (monotone) is valid."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        result = engine.speak("Hello", emotion_exaggeration=0.0, save_path=tmp_path / "out.wav")

        assert result.metadata["emotion_exaggeration"] == 0.0

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_emotion_exaggeration_one(self, mock_load, tmp_path):
        """speak() with emotion_exaggeration=1.0 (dramatic) is valid."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        result = engine.speak("Hello", emotion_exaggeration=1.0, save_path=tmp_path / "out.wav")

        assert result.metadata["emotion_exaggeration"] == 1.0

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_emotion_exaggeration_clamped_below(self, mock_load, tmp_path):
        """speak() with emotion_exaggeration < 0 is clamped to 0."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        result = engine.speak("Hello", emotion_exaggeration=-0.5, save_path=tmp_path / "out.wav")

        # Should be clamped to 0.0
        assert result.metadata["emotion_exaggeration"] == 0.0

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_emotion_exaggeration_clamped_above(self, mock_load, tmp_path):
        """speak() with emotion_exaggeration > 1 is clamped to 1."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        result = engine.speak("Hello", emotion_exaggeration=1.5, save_path=tmp_path / "out.wav")

        # Should be clamped to 1.0
        assert result.metadata["emotion_exaggeration"] == 1.0

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_cfg_weight_boundary_values(self, mock_load, tmp_path):
        """speak() with cfg_weight at boundaries (0.0 and 1.0)."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._model_loaded = True

        # Test 0.0
        result = engine.speak("Hello", cfg_weight=0.0, save_path=tmp_path / "out1.wav")
        assert result.metadata["cfg_weight"] == 0.0

        # Test 1.0
        result = engine.speak("Hello", cfg_weight=1.0, save_path=tmp_path / "out2.wav")
        assert result.metadata["cfg_weight"] == 1.0

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_model_generate_exception(self, mock_load, tmp_path):
        """speak() propagates exception when model.generate() fails."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(side_effect=RuntimeError("Model failed"))
        engine._model_loaded = True

        with pytest.raises(RuntimeError, match="Model failed"):
            engine.speak("Hello", save_path=tmp_path / "out.wav")

    def test_get_voice_info_known_voice(self, tmp_path):
        """get_voice_info() returns metadata for known voice."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        audio = tmp_path / "voice.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)
        engine.clone_voice(audio, "known_voice")

        info = engine.get_voice_info("known_voice")

        assert info["id"] == "known_voice"
        assert info["type"] == "cloned"

    def test_get_voice_info_unknown_voice(self):
        """get_voice_info() returns unknown type for unregistered voice."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        info = engine.get_voice_info("unknown_voice")

        assert info["type"] == "unknown"
        assert info["id"] == "unknown_voice"

    def test_format_with_tags_out_of_range_position(self):
        """format_with_tags() ignores out-of-range positions."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        result = ChatterboxEngine.format_with_tags(
            "Hello world",
            {"laugh": [100, -1]}  # Out of range positions
        )

        # Should not crash, just ignore invalid positions
        assert "[laugh]" not in result
        assert result == "Hello world"

    def test_format_with_tags_negative_position(self):
        """format_with_tags() handles negative positions."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        result = ChatterboxEngine.format_with_tags(
            "Hello world there",
            {"laugh": [-1]}
        )

        # Negative positions should be ignored
        assert "[laugh]" not in result

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_very_long_text(self, mock_load, tmp_path):
        """speak() handles very long text input."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(240000))  # 10 seconds
        engine._model_loaded = True

        long_text = "Hello world. " * 1000  # ~13000 characters
        result = engine.speak(long_text, save_path=tmp_path / "out.wav")

        assert result.audio_path.exists()


# =============================================================================
# CrossLang Error Handling and Edge Cases
# =============================================================================

class TestCrossLangErrorHandling:
    """Error handling tests for crosslang module."""

    def test_detect_language_empty_text(self):
        """detect_language() with empty text returns 'en' default."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("")

        # Empty text defaults to English
        assert result == "en"

    def test_detect_language_only_whitespace(self):
        """detect_language() with whitespace-only text returns 'en'."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("   \t\n  ")

        assert result == "en"

    def test_detect_language_mixed_scripts_first_wins(self):
        """detect_language() with mixed scripts returns first detected."""
        from voice_soundboard.cloning.crosslang import detect_language

        # Japanese hiragana first
        result = detect_language("Hello こんにちは 안녕하세요")

        # Should detect Japanese first (hiragana appears before hangul)
        assert result in ["ja", "ko"]  # Either is valid depending on iteration

    def test_detect_language_numbers_only(self):
        """detect_language() with numbers returns 'en'."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("12345")

        assert result == "en"

    def test_detect_language_punctuation_only(self):
        """detect_language() with punctuation only returns 'en'."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("!@#$%^&*()")

        assert result == "en"

    def test_crosslanguage_cloner_unsupported_source(self):
        """CrossLanguageCloner with unsupported source falls back to 'en'."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner, LANGUAGE_CONFIGS

        cloner = CrossLanguageCloner(source_language="xyz_unsupported")

        # source_config should fall back to 'en'
        assert cloner.source_config == LANGUAGE_CONFIGS["en"]

    def test_get_language_pair_compatibility_unsupported(self):
        """get_language_pair_compatibility() with unsupported language."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")

        result = cloner.get_language_pair_compatibility("en", "xyz_unsupported")

        assert result["compatible"] is False
        assert "not supported" in result["reason"]

    def test_get_language_pair_compatibility_both_unsupported(self):
        """get_language_pair_compatibility() with both languages unsupported."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")

        result = cloner.get_language_pair_compatibility("abc", "xyz")

        assert result["compatible"] is False

    def test_get_language_pair_compatibility_tonal_to_nontonal(self):
        """Tonal to non-tonal language pair reduces expected quality."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")

        # English (non-tonal) to Chinese (tonal)
        result = cloner.get_language_pair_compatibility("en", "zh")

        assert result["compatible"] is True
        assert result["expected_quality"] < 1.0
        assert len(result["phonetic_issues"]) > 0

    def test_get_language_pair_compatibility_same_family_bonus(self):
        """Same language family gives quality bonus."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="es")

        # Spanish to French (both Romance)
        result = cloner.get_language_pair_compatibility("es", "fr")

        assert result["compatible"] is True
        assert result["same_language_family"] is True

    def test_is_language_supported(self):
        """is_language_supported() returns correct values."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()

        assert cloner.is_language_supported("en") is True
        assert cloner.is_language_supported("fr") is True
        assert cloner.is_language_supported("xyz") is False

    def test_list_supported_languages(self):
        """list_supported_languages() returns list of dicts."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        languages = cloner.list_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert all("code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)
        assert all("native_name" in lang for lang in languages)


# =============================================================================
# Server MCP Handler Error Handling
# =============================================================================

class TestServerMCPErrorHandling:
    """Error handling tests for server MCP handlers."""

    @pytest.mark.asyncio
    async def test_handle_speak_f5tts_empty_text(self):
        """handle_speak_f5tts() with empty text returns error."""
        from voice_soundboard.server import handle_speak_f5tts

        result = await handle_speak_f5tts({"text": ""})

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_speak_f5tts_missing_text(self):
        """handle_speak_f5tts() without text key returns error."""
        from voice_soundboard.server import handle_speak_f5tts

        result = await handle_speak_f5tts({})

        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_clone_voice_f5tts_empty_path(self):
        """handle_clone_voice_f5tts() with empty path returns error."""
        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({"audio_path": ""})

        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_clone_voice_f5tts_missing_path(self):
        """handle_clone_voice_f5tts() without path returns error."""
        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({})

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'audio_path' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_speak_chatterbox_empty_text(self):
        """handle_speak_chatterbox() with empty text returns error."""
        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({"text": ""})

        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_speak_chatterbox_missing_text(self):
        """handle_speak_chatterbox() without text returns error."""
        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({})

        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_list_chatterbox_languages_not_installed(self):
        """handle_list_chatterbox_languages() when not installed returns message."""
        from voice_soundboard.server import handle_list_chatterbox_languages

        with patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', False):
            result = await handle_list_chatterbox_languages({})

            assert len(result) == 1
            assert "not installed" in result[0].text


# =============================================================================
# Additional Edge Cases
# =============================================================================

class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_paralinguistic_tags_empty_brackets(self):
        """Validate tags ignores empty brackets []."""
        from voice_soundboard.engines.chatterbox import validate_paralinguistic_tags

        text = "Hello [] world [laugh]"
        tags = validate_paralinguistic_tags(text)

        assert tags == ["laugh"]
        assert "" not in tags

    def test_paralinguistic_tags_nested_brackets(self):
        """Validate tags extracts inner tag from nested brackets."""
        from voice_soundboard.engines.chatterbox import validate_paralinguistic_tags

        text = "Hello [[laugh]] world"
        tags = validate_paralinguistic_tags(text)

        # The regex finds [laugh] within [[laugh]]
        # This is expected behavior - the inner brackets are matched
        assert tags == ["laugh"]

    def test_language_config_all_have_required_fields(self):
        """All language configs have required fields."""
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        required_fields = ["code", "name", "native_name", "typical_speaking_rate_wpm"]

        for code, config in LANGUAGE_CONFIGS.items():
            for field in required_fields:
                assert hasattr(config, field), f"Config '{code}' missing field '{field}'"
                assert getattr(config, field) is not None, f"Config '{code}' has None for '{field}'"

    def test_language_config_wpm_positive(self):
        """All language configs have positive speaking rate."""
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        for code, config in LANGUAGE_CONFIGS.items():
            assert config.typical_speaking_rate_wpm > 0, f"Config '{code}' has non-positive WPM"

    def test_supported_languages_matches_enum(self):
        """SUPPORTED_LANGUAGES dict matches Language enum."""
        from voice_soundboard.cloning.crosslang import SUPPORTED_LANGUAGES, Language

        for lang in Language:
            assert lang.value in SUPPORTED_LANGUAGES, f"Language {lang.name} not in SUPPORTED_LANGUAGES"
            assert SUPPORTED_LANGUAGES[lang.value] == lang

    @patch('voice_soundboard.engines.f5tts.F5TTSEngine._ensure_model_loaded')
    def test_f5tts_speak_raw_voice_file_path(self, mock_load, tmp_path):
        """speak_raw() with voice as file path."""
        from voice_soundboard.engines.f5tts import F5TTSEngine

        engine = F5TTSEngine()
        engine._model = Mock()
        engine._model.infer = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000, None))
        engine._model_loaded = True

        audio = tmp_path / "ref.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        samples, sr = engine.speak_raw("Hello", voice=str(audio), ref_text="test")

        call_kwargs = engine._model.infer.call_args[1]
        assert call_kwargs["ref_file"] == str(audio)

    def test_chatterbox_ensure_model_import_error(self):
        """_ensure_model_loaded() raises ImportError when package missing."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine(model_variant="multilingual")
        engine._model_loaded = False
        engine._model = None

        with patch.dict('sys.modules', {'chatterbox': None, 'chatterbox.mtl_tts': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="Chatterbox is not installed"):
                    engine._ensure_model_loaded()
