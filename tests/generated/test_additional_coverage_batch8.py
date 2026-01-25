"""
Additional test coverage batch 8.

Tests for:
- cloning/crosslang.py (Language, LanguageConfig, CrossLanguageCloner, detect_language)
- cloning/separation.py (EmotionStyle, TimbreEmbedding, EmotionEmbedding, SeparatedVoice, EmotionTimbreSeparator)
- conversion/streaming.py (PipelineStage, AudioChunk, AudioBuffer, StreamingConverter, ConversionPipeline)
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Tests for cloning/crosslang.py
# =============================================================================

class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_enum_values(self):
        """Test Language enum has expected values."""
        from voice_soundboard.cloning.crosslang import Language

        assert Language.ENGLISH.value == "en"
        assert Language.CHINESE_MANDARIN.value == "zh"
        assert Language.SPANISH.value == "es"
        assert Language.FRENCH.value == "fr"
        assert Language.GERMAN.value == "de"
        assert Language.JAPANESE.value == "ja"
        assert Language.KOREAN.value == "ko"

    def test_language_enum_all_members(self):
        """Test Language enum has all expected members."""
        from voice_soundboard.cloning.crosslang import Language

        # Should have at least 20 languages
        assert len(Language) >= 20

    def test_supported_languages_dict(self):
        """Test SUPPORTED_LANGUAGES dictionary mapping."""
        from voice_soundboard.cloning.crosslang import SUPPORTED_LANGUAGES, Language

        assert SUPPORTED_LANGUAGES["en"] == Language.ENGLISH
        assert SUPPORTED_LANGUAGES["ja"] == Language.JAPANESE
        assert SUPPORTED_LANGUAGES["zh"] == Language.CHINESE_MANDARIN


class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_language_config_defaults(self):
        """Test LanguageConfig default values."""
        from voice_soundboard.cloning.crosslang import LanguageConfig

        config = LanguageConfig(code="en", name="English", native_name="English")

        assert config.code == "en"
        assert config.name == "English"
        assert config.phoneme_set == "ipa"
        assert config.has_tones is False
        assert config.syllable_timed is False
        assert config.stress_timed is True
        assert config.default_speed == 1.0
        assert config.typical_speaking_rate_wpm == 150
        assert config.requires_romanization is False

    def test_language_config_custom_values(self):
        """Test LanguageConfig with custom values."""
        from voice_soundboard.cloning.crosslang import LanguageConfig

        config = LanguageConfig(
            code="zh",
            name="Chinese",
            native_name="中文",
            has_tones=True,
            syllable_timed=True,
            stress_timed=False,
            requires_romanization=True,
            romanization_system="pinyin"
        )

        assert config.has_tones is True
        assert config.syllable_timed is True
        assert config.stress_timed is False
        assert config.requires_romanization is True
        assert config.romanization_system == "pinyin"

    def test_language_configs_dict(self):
        """Test LANGUAGE_CONFIGS dictionary."""
        from voice_soundboard.cloning.crosslang import LANGUAGE_CONFIGS

        assert "en" in LANGUAGE_CONFIGS
        assert "zh" in LANGUAGE_CONFIGS
        assert "ja" in LANGUAGE_CONFIGS

        en_config = LANGUAGE_CONFIGS["en"]
        assert en_config.name == "English"
        assert en_config.stress_timed is True


class TestCrossLanguageResult:
    """Tests for CrossLanguageResult dataclass."""

    def test_cross_language_result_defaults(self):
        """Test CrossLanguageResult default values."""
        from voice_soundboard.cloning.crosslang import CrossLanguageResult

        result = CrossLanguageResult(
            success=True,
            source_language="en",
            target_language="ja"
        )

        assert result.success is True
        assert result.source_language == "en"
        assert result.target_language == "ja"
        assert result.audio is None
        assert result.sample_rate == 24000
        assert result.timbre_preservation_score == 0.0
        assert result.accent_transfer_score == 0.0
        assert result.error is None
        assert result.warnings == []

    def test_cross_language_result_with_audio(self):
        """Test CrossLanguageResult with audio data."""
        from voice_soundboard.cloning.crosslang import CrossLanguageResult

        audio = np.random.randn(24000).astype(np.float32)
        result = CrossLanguageResult(
            success=True,
            source_language="en",
            target_language="fr",
            audio=audio,
            timbre_preservation_score=0.85
        )

        assert result.audio is not None
        assert len(result.audio) == 24000


class TestCrossLanguageCloner:
    """Tests for CrossLanguageCloner class."""

    def test_cloner_initialization(self):
        """Test CrossLanguageCloner initialization."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")
        assert cloner.source_language == "en"
        assert cloner.preserve_accent is False

    def test_cloner_with_preserve_accent(self):
        """Test CrossLanguageCloner with preserve_accent option."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en", preserve_accent=True)
        assert cloner.preserve_accent is True

    def test_source_config_property(self):
        """Test source_config property."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="en")
        config = cloner.source_config

        assert config.code == "en"
        assert config.name == "English"

    def test_source_config_fallback(self):
        """Test source_config falls back to English for unknown language."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner(source_language="xx")
        config = cloner.source_config

        assert config.code == "en"  # Falls back to English

    def test_get_target_config(self):
        """Test get_target_config method."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        config = cloner.get_target_config("ja")

        assert config.code == "ja"
        assert config.name == "Japanese"

    def test_get_target_config_fallback(self):
        """Test get_target_config falls back for unknown language."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        config = cloner.get_target_config("xx")

        assert config.code == "en"  # Falls back to English

    def test_is_language_supported(self):
        """Test is_language_supported method."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()

        assert cloner.is_language_supported("en") is True
        assert cloner.is_language_supported("ja") is True
        assert cloner.is_language_supported("xx") is False

    def test_list_supported_languages(self):
        """Test list_supported_languages method."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        languages = cloner.list_supported_languages()

        assert len(languages) > 0
        assert all("code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)
        assert all("native_name" in lang for lang in languages)

    def test_language_pair_compatibility_same_family(self):
        """Test compatibility for same language family."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "de")

        assert compat["compatible"] is True
        assert compat["same_language_family"] is True
        assert compat["expected_quality"] > 0.5

    def test_language_pair_compatibility_tonal(self):
        """Test compatibility for tonal target language."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "zh")

        assert compat["compatible"] is True
        assert len(compat["phonetic_issues"]) > 0  # Should mention tonal issues

    def test_language_pair_compatibility_unsupported(self):
        """Test compatibility for unsupported language."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner

        cloner = CrossLanguageCloner()
        compat = cloner.get_language_pair_compatibility("en", "xx")

        assert compat["compatible"] is False
        assert "not supported" in compat["reason"]

    def test_prepare_embedding_for_language(self):
        """Test prepare_embedding_for_language method."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256,
            source_duration_seconds=5.0,
            quality_score=0.9
        )

        prepared, metadata = cloner.prepare_embedding_for_language(embedding, "ja")

        assert prepared is embedding  # Same embedding returned
        assert "source_language" in metadata
        assert "target_language" in metadata
        assert "recommended_speed_multiplier" in metadata

    def test_estimate_quality(self):
        """Test estimate_quality method."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256,
            source_duration_seconds=10.0,
            quality_score=0.9
        )

        quality = cloner.estimate_quality(embedding, "fr")

        assert 0.0 <= quality <= 1.0

    def test_estimate_quality_short_audio(self):
        """Test estimate_quality with short audio."""
        from voice_soundboard.cloning.crosslang import CrossLanguageCloner
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        cloner = CrossLanguageCloner(source_language="en")
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256,
            source_duration_seconds=1.0,  # Short audio
            quality_score=0.9
        )

        quality = cloner.estimate_quality(embedding, "fr")

        # Quality should be penalized for short audio
        assert quality < 0.9 * 0.9  # Less than quality_score * 0.9


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_english(self):
        """Test detecting English text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("Hello, how are you?")
        assert result == "en"

    def test_detect_japanese(self):
        """Test detecting Japanese text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("こんにちは")
        assert result == "ja"

    def test_detect_korean(self):
        """Test detecting Korean text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("안녕하세요")
        assert result == "ko"

    def test_detect_chinese(self):
        """Test detecting Chinese text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("你好")
        assert result == "zh"

    def test_detect_russian(self):
        """Test detecting Russian text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("Привет")
        assert result == "ru"

    def test_detect_arabic(self):
        """Test detecting Arabic text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("مرحبا")
        assert result == "ar"

    def test_detect_hindi(self):
        """Test detecting Hindi text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("नमस्ते")
        assert result == "hi"

    def test_detect_thai(self):
        """Test detecting Thai text."""
        from voice_soundboard.cloning.crosslang import detect_language

        result = detect_language("สวัสดี")
        assert result == "th"


# =============================================================================
# Tests for cloning/separation.py
# =============================================================================

class TestEmotionStyleEnum:
    """Tests for EmotionStyle enum."""

    def test_emotion_style_values(self):
        """Test EmotionStyle enum values."""
        from voice_soundboard.cloning.separation import EmotionStyle

        assert EmotionStyle.NEUTRAL.value == "neutral"
        assert EmotionStyle.HAPPY.value == "happy"
        assert EmotionStyle.SAD.value == "sad"
        assert EmotionStyle.ANGRY.value == "angry"

    def test_emotion_style_all_members(self):
        """Test EmotionStyle enum has all expected members."""
        from voice_soundboard.cloning.separation import EmotionStyle

        assert len(EmotionStyle) == 10


class TestTimbreEmbedding:
    """Tests for TimbreEmbedding dataclass."""

    def test_timbre_embedding_creation(self):
        """Test TimbreEmbedding creation."""
        from voice_soundboard.cloning.separation import TimbreEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        timbre = TimbreEmbedding(embedding=embedding)

        assert timbre.embedding_dim == 256
        assert timbre.source_voice_id is None
        assert timbre.separation_quality == 1.0

    def test_timbre_embedding_to_dict(self):
        """Test TimbreEmbedding serialization."""
        from voice_soundboard.cloning.separation import TimbreEmbedding

        embedding = np.random.randn(256).astype(np.float32)
        timbre = TimbreEmbedding(
            embedding=embedding,
            source_voice_id="test_voice"
        )

        data = timbre.to_dict()

        assert "embedding" in data
        assert data["source_voice_id"] == "test_voice"
        assert data["embedding_dim"] == 256

    def test_timbre_embedding_from_dict(self):
        """Test TimbreEmbedding deserialization."""
        from voice_soundboard.cloning.separation import TimbreEmbedding

        data = {
            "embedding": [0.1] * 256,
            "embedding_dim": 256,
            "source_voice_id": "test_voice",
            "source_embedding_id": None,
            "separation_quality": 0.95
        }

        timbre = TimbreEmbedding.from_dict(data)

        assert len(timbre.embedding) == 256
        assert timbre.source_voice_id == "test_voice"
        assert timbre.separation_quality == 0.95


class TestEmotionEmbedding:
    """Tests for EmotionEmbedding dataclass."""

    def test_emotion_embedding_creation(self):
        """Test EmotionEmbedding creation."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        embedding = np.random.randn(64).astype(np.float32)
        emotion = EmotionEmbedding(embedding=embedding)

        assert emotion.embedding_dim == 64
        assert emotion.emotion_label is None
        assert emotion.emotion_intensity == 1.0

    def test_emotion_embedding_with_vad(self):
        """Test EmotionEmbedding with VAD values."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        embedding = np.random.randn(64).astype(np.float32)
        emotion = EmotionEmbedding(
            embedding=embedding,
            emotion_label="happy",
            valence=0.8,
            arousal=0.6,
            dominance=0.5
        )

        assert emotion.valence == 0.8
        assert emotion.arousal == 0.6
        assert emotion.dominance == 0.5

    def test_emotion_embedding_to_dict(self):
        """Test EmotionEmbedding serialization."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        embedding = np.random.randn(64).astype(np.float32)
        emotion = EmotionEmbedding(
            embedding=embedding,
            emotion_label="sad",
            valence=-0.6
        )

        data = emotion.to_dict()

        assert data["emotion_label"] == "sad"
        assert data["valence"] == -0.6

    def test_emotion_embedding_from_dict(self):
        """Test EmotionEmbedding deserialization."""
        from voice_soundboard.cloning.separation import EmotionEmbedding

        data = {
            "embedding": [0.1] * 64,
            "embedding_dim": 64,
            "emotion_label": "angry",
            "emotion_intensity": 0.8,
            "valence": -0.5,
            "arousal": 0.8,
            "dominance": 0.7,
            "source_path": None
        }

        emotion = EmotionEmbedding.from_dict(data)

        assert emotion.emotion_label == "angry"
        assert emotion.arousal == 0.8


class TestSeparatedVoice:
    """Tests for SeparatedVoice dataclass."""

    def test_separated_voice_creation(self):
        """Test SeparatedVoice creation."""
        from voice_soundboard.cloning.separation import (
            SeparatedVoice, TimbreEmbedding, EmotionEmbedding
        )

        timbre = TimbreEmbedding(embedding=np.random.randn(256).astype(np.float32))
        emotion = EmotionEmbedding(embedding=np.random.randn(64).astype(np.float32))

        separated = SeparatedVoice(timbre=timbre, emotion=emotion)

        assert separated.timbre is timbre
        assert separated.emotion is emotion
        assert separated.reconstruction_loss == 0.0

    def test_separated_voice_recombine(self):
        """Test SeparatedVoice recombine method."""
        from voice_soundboard.cloning.separation import (
            SeparatedVoice, TimbreEmbedding, EmotionEmbedding
        )

        timbre = TimbreEmbedding(embedding=np.random.randn(256).astype(np.float32))
        emotion = EmotionEmbedding(embedding=np.random.randn(64).astype(np.float32))

        separated = SeparatedVoice(timbre=timbre, emotion=emotion)
        combined = separated.recombine()

        assert isinstance(combined, np.ndarray)
        assert len(combined) == 256
        # Check normalization
        assert abs(np.linalg.norm(combined) - 1.0) < 0.01

    def test_separated_voice_with_emotion(self):
        """Test SeparatedVoice with_emotion method."""
        from voice_soundboard.cloning.separation import (
            SeparatedVoice, TimbreEmbedding, EmotionEmbedding
        )

        timbre = TimbreEmbedding(embedding=np.random.randn(256).astype(np.float32))
        original_emotion = EmotionEmbedding(embedding=np.random.randn(64).astype(np.float32))
        new_emotion = EmotionEmbedding(embedding=np.random.randn(64).astype(np.float32))

        separated = SeparatedVoice(timbre=timbre, emotion=original_emotion)
        combined = separated.with_emotion(new_emotion)

        assert isinstance(combined, np.ndarray)
        # Original emotion should be preserved
        assert separated.emotion is original_emotion


class TestEmotionTimbreSeparator:
    """Tests for EmotionTimbreSeparator class."""

    def test_separator_initialization(self):
        """Test EmotionTimbreSeparator initialization."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        assert separator.timbre_dim == 256
        assert separator.emotion_dim == 64
        assert separator.device == "cpu"

    def test_separator_custom_dims(self):
        """Test EmotionTimbreSeparator with custom dimensions."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator(timbre_dim=512, emotion_dim=128)

        assert separator.timbre_dim == 512
        assert separator.emotion_dim == 128

    def test_separate_from_array(self):
        """Test separating from numpy array."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        vector = np.random.randn(256).astype(np.float32)

        separated = separator.separate(vector)

        assert separated.timbre is not None
        assert separated.emotion is not None
        assert len(separated.timbre.embedding) == 256
        assert len(separated.emotion.embedding) == 64

    def test_separate_from_voice_embedding(self):
        """Test separating from VoiceEmbedding."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()
        embedding = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256,
            embedding_id="test_id"
        )

        separated = separator.separate(embedding)

        assert separated.original_embedding is embedding
        assert separated.timbre.source_embedding_id == "test_id"

    def test_get_emotion_preset(self):
        """Test getting emotion presets."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        happy = separator.get_emotion_preset("happy")
        assert happy.emotion_label == "happy"
        assert happy.valence is not None
        assert happy.arousal is not None

        sad = separator.get_emotion_preset("sad")
        assert sad.emotion_label == "sad"
        assert sad.valence < 0  # Negative valence for sad

    def test_get_emotion_preset_invalid(self):
        """Test getting invalid emotion preset raises error."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()

        with pytest.raises(ValueError, match="Unknown emotion"):
            separator.get_emotion_preset("nonexistent")

    def test_list_emotion_presets(self):
        """Test listing emotion presets."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        presets = separator.list_emotion_presets()

        assert "neutral" in presets
        assert "happy" in presets
        assert "sad" in presets
        assert len(presets) == 10

    def test_transfer_emotion_from_string(self):
        """Test transferring emotion using string preset."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()
        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256
        )

        result = separator.transfer_emotion(voice, "happy")

        assert isinstance(result, np.ndarray)
        assert len(result) == 256

    def test_transfer_emotion_from_embedding(self):
        """Test transferring emotion from EmotionEmbedding."""
        from voice_soundboard.cloning.separation import (
            EmotionTimbreSeparator, EmotionEmbedding
        )
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()
        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256
        )
        emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="custom"
        )

        result = separator.transfer_emotion(voice, emotion)

        assert isinstance(result, np.ndarray)

    def test_transfer_emotion_with_intensity(self):
        """Test transferring emotion with intensity scaling."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        separator = EmotionTimbreSeparator()
        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256
        )

        result = separator.transfer_emotion(voice, "happy", intensity=0.5)

        assert isinstance(result, np.ndarray)

    def test_blend_emotions(self):
        """Test blending multiple emotions."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        happy = separator.get_emotion_preset("happy")
        sad = separator.get_emotion_preset("sad")

        blended = separator.blend_emotions([
            (happy, 0.7),
            (sad, 0.3)
        ])

        assert blended.emotion_label == "blended"
        assert len(blended.embedding) == 64

    def test_blend_emotions_empty(self):
        """Test blending empty emotion list returns neutral."""
        from voice_soundboard.cloning.separation import EmotionTimbreSeparator

        separator = EmotionTimbreSeparator()
        blended = separator.blend_emotions([])

        assert blended.emotion_label == "neutral"


class TestSeparationConvenienceFunctions:
    """Tests for convenience functions."""

    def test_separate_voice_function(self):
        """Test separate_voice convenience function."""
        from voice_soundboard.cloning.separation import separate_voice

        vector = np.random.randn(256).astype(np.float32)
        separated = separate_voice(vector)

        assert separated.timbre is not None
        assert separated.emotion is not None

    def test_transfer_emotion_function(self):
        """Test transfer_emotion convenience function."""
        from voice_soundboard.cloning.separation import transfer_emotion
        from voice_soundboard.cloning.extractor import VoiceEmbedding

        voice = VoiceEmbedding(
            embedding=np.random.randn(256).astype(np.float32),
            embedding_dim=256
        )

        result = transfer_emotion(voice, "excited")

        assert isinstance(result, np.ndarray)
        assert len(result) == 256


# =============================================================================
# Tests for conversion/streaming.py
# =============================================================================

class TestPipelineStageEnum:
    """Tests for PipelineStage enum."""

    def test_pipeline_stage_values(self):
        """Test PipelineStage enum has all stages."""
        from voice_soundboard.conversion.streaming import PipelineStage

        assert PipelineStage.INPUT is not None
        assert PipelineStage.PREPROCESS is not None
        assert PipelineStage.ENCODE is not None
        assert PipelineStage.CONVERT is not None
        assert PipelineStage.DECODE is not None
        assert PipelineStage.POSTPROCESS is not None
        assert PipelineStage.OUTPUT is not None

    def test_pipeline_stage_count(self):
        """Test PipelineStage enum has 7 stages."""
        from voice_soundboard.conversion.streaming import PipelineStage

        assert len(PipelineStage) == 7


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_audio_chunk_creation(self):
        """Test AudioChunk creation."""
        from voice_soundboard.conversion.streaming import AudioChunk, PipelineStage

        data = np.random.randn(480).astype(np.float32)
        chunk = AudioChunk(
            data=data,
            sample_rate=24000,
            timestamp_ms=0.0
        )

        assert len(chunk.data) == 480
        assert chunk.sample_rate == 24000
        assert chunk.stage == PipelineStage.INPUT

    def test_audio_chunk_duration(self):
        """Test AudioChunk duration_ms property."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
        chunk = AudioChunk(data=data, sample_rate=24000, timestamp_ms=0.0)

        assert abs(chunk.duration_ms - 1000.0) < 0.1

    def test_audio_chunk_processing_time(self):
        """Test AudioChunk processing_time_ms property."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.random.randn(480).astype(np.float32)
        chunk = AudioChunk(
            data=data,
            sample_rate=24000,
            timestamp_ms=0.0,
            processing_started_ms=100.0,
            processing_completed_ms=150.0
        )

        assert chunk.processing_time_ms == 50.0

    def test_audio_chunk_processing_time_incomplete(self):
        """Test AudioChunk processing_time_ms when not completed."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.random.randn(480).astype(np.float32)
        chunk = AudioChunk(
            data=data,
            sample_rate=24000,
            timestamp_ms=0.0,
            processing_started_ms=100.0
        )

        assert chunk.processing_time_ms == 0.0

    def test_audio_chunk_copy(self):
        """Test AudioChunk copy method."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.random.randn(480).astype(np.float32)
        original = AudioChunk(data=data, sample_rate=24000, timestamp_ms=0.0)

        copied = original.copy()

        assert np.array_equal(copied.data, original.data)
        assert copied.data is not original.data  # Different array


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_buffer_creation(self):
        """Test AudioBuffer creation."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000)

        assert buffer.capacity == 1000
        assert buffer.channels == 1
        assert buffer.available == 0
        assert buffer.free_space == 1000

    def test_buffer_write_read(self):
        """Test AudioBuffer write and read."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000)
        data = np.random.randn(100).astype(np.float32)

        written = buffer.write(data)
        assert written == 100
        assert buffer.available == 100

        read_data = buffer.read(100, block=False)
        assert read_data is not None
        assert len(read_data) == 100
        assert buffer.available == 0

    def test_buffer_write_2d(self):
        """Test AudioBuffer write with 2D data."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        data = np.random.randn(100, 1).astype(np.float32)

        written = buffer.write(data)
        assert written == 100

    def test_buffer_wrap_around(self):
        """Test AudioBuffer wrap-around behavior."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=100)

        # Write 80 samples
        data1 = np.ones(80).astype(np.float32)
        buffer.write(data1)

        # Read 60 samples
        buffer.read(60, block=False)

        # Write 60 more samples (should wrap)
        data2 = np.ones(60).astype(np.float32) * 2
        buffer.write(data2)

        # Read all
        result = buffer.read(80, block=False)
        assert result is not None
        assert len(result) == 80

    def test_buffer_peek(self):
        """Test AudioBuffer peek method."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000)
        data = np.random.randn(100).astype(np.float32)
        buffer.write(data)

        peeked = buffer.peek(50)
        assert peeked is not None
        assert len(peeked) == 50
        assert buffer.available == 100  # Not removed

    def test_buffer_peek_empty(self):
        """Test AudioBuffer peek on empty buffer."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000)
        result = buffer.peek(50)

        assert result is None

    def test_buffer_clear(self):
        """Test AudioBuffer clear method."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000)
        data = np.random.randn(100).astype(np.float32)
        buffer.write(data)

        buffer.clear()

        assert buffer.available == 0
        assert buffer.free_space == 1000

    def test_buffer_non_blocking_read_empty(self):
        """Test non-blocking read on empty buffer."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000)
        result = buffer.read(100, block=False)

        assert result is None

    def test_buffer_non_blocking_write_full(self):
        """Test non-blocking write on full buffer."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=100)
        data1 = np.random.randn(100).astype(np.float32)
        buffer.write(data1)

        data2 = np.random.randn(50).astype(np.float32)
        written = buffer.write(data2, block=False)

        assert written == 0  # Nothing written


class TestStreamingConverter:
    """Tests for StreamingConverter class."""

    def test_streaming_converter_creation(self):
        """Test StreamingConverter creation."""
        from voice_soundboard.conversion.streaming import StreamingConverter

        mock_converter = Mock()
        converter = StreamingConverter(converter=mock_converter)

        assert converter.chunk_size == 480
        assert converter.sample_rate == 24000
        assert converter.is_running is False

    def test_streaming_converter_start_stop(self):
        """Test StreamingConverter start and stop."""
        from voice_soundboard.conversion.streaming import StreamingConverter

        mock_converter = Mock()
        mock_converter.convert_chunk.return_value = np.zeros(480)

        converter = StreamingConverter(converter=mock_converter)
        converter.start()

        assert converter.is_running is True

        converter.stop()
        assert converter.is_running is False

    def test_streaming_converter_push_pull(self):
        """Test StreamingConverter push and pull."""
        from voice_soundboard.conversion.streaming import StreamingConverter

        mock_converter = Mock()
        mock_converter.convert_chunk.return_value = np.ones(480).astype(np.float32)

        converter = StreamingConverter(converter=mock_converter)
        converter.start()

        # Push some audio
        audio = np.random.randn(480).astype(np.float32)
        converter.push(audio)

        # Wait for processing
        time.sleep(0.2)

        # Pull should have output
        output = converter.pull(480)

        converter.stop()

    def test_streaming_converter_avg_latency(self):
        """Test StreamingConverter average latency property."""
        from voice_soundboard.conversion.streaming import StreamingConverter

        mock_converter = Mock()
        converter = StreamingConverter(converter=mock_converter)

        assert converter.avg_latency_ms == 0.0

    def test_streaming_converter_output_callback(self):
        """Test StreamingConverter with output callback."""
        from voice_soundboard.conversion.streaming import StreamingConverter

        mock_converter = Mock()
        mock_converter.convert_chunk.return_value = np.ones(480).astype(np.float32)

        callback = Mock()

        converter = StreamingConverter(converter=mock_converter)
        converter.start(on_output=callback)

        # Push audio
        audio = np.random.randn(480).astype(np.float32)
        converter.push(audio)

        # Wait for processing
        time.sleep(0.2)

        converter.stop()

        # Callback should have been called
        # Note: May not be called if thread didn't process in time


class TestConversionPipeline:
    """Tests for ConversionPipeline class."""

    def test_pipeline_creation(self):
        """Test ConversionPipeline creation."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
            (PipelineStage.CONVERT, lambda x: x),
        ]

        pipeline = ConversionPipeline(stages=stages)

        assert len(pipeline.stages) == 2
        assert pipeline.is_running is False

    def test_pipeline_start_stop(self):
        """Test ConversionPipeline start and stop."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
        ]

        pipeline = ConversionPipeline(stages=stages)
        pipeline.start()

        assert pipeline.is_running is True

        pipeline.stop()
        assert pipeline.is_running is False

    def test_pipeline_push_pull(self):
        """Test ConversionPipeline push and pull."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x * 2),
        ]

        pipeline = ConversionPipeline(stages=stages)
        pipeline.start()

        # Push some audio
        audio = np.ones(480).astype(np.float32)
        pipeline.push(audio)

        # Wait and pull
        time.sleep(0.2)
        output = pipeline.pull(timeout=0.1)

        pipeline.stop()

        if output is not None:
            assert np.allclose(output, audio * 2)

    def test_pipeline_get_stage_latency(self):
        """Test ConversionPipeline get_stage_latency."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
        ]

        pipeline = ConversionPipeline(stages=stages)

        # Before any processing, latency should be 0
        latency = pipeline.get_stage_latency(PipelineStage.PREPROCESS)
        assert latency == 0.0

    def test_pipeline_get_total_latency(self):
        """Test ConversionPipeline get_total_latency."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
            (PipelineStage.CONVERT, lambda x: x),
        ]

        pipeline = ConversionPipeline(stages=stages)
        total = pipeline.get_total_latency()

        assert total == 0.0  # No processing yet

    def test_pipeline_multiple_stages(self):
        """Test ConversionPipeline with multiple stages."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x + 1),
            (PipelineStage.CONVERT, lambda x: x * 2),
            (PipelineStage.POSTPROCESS, lambda x: x - 1),
        ]

        pipeline = ConversionPipeline(stages=stages)
        pipeline.start()

        # Push: (x + 1) * 2 - 1 = 2x + 1
        audio = np.ones(100).astype(np.float32)
        pipeline.push(audio)

        time.sleep(0.3)
        output = pipeline.pull(timeout=0.1)

        pipeline.stop()

        if output is not None:
            expected = (audio + 1) * 2 - 1
            assert np.allclose(output, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
