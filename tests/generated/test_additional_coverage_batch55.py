"""
Test Additional Coverage Batch 55: Vocology Naturalness Tests

Tests for:
- FillerType enum
- BreathType enum
- NaturalnessConfig dataclass
- TextAnnotation dataclass
- Helper functions (_is_sentence_end, _is_clause_boundary, etc.)
- NaturalSpeechProcessor class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== FillerType Enum Tests ==============

class TestFillerTypeEnum:
    """Tests for FillerType enum."""

    def test_filler_type_um(self):
        """Test FillerType.UM value."""
        from voice_soundboard.vocology.naturalness import FillerType
        assert FillerType.UM.value == "um"

    def test_filler_type_uh(self):
        """Test FillerType.UH value."""
        from voice_soundboard.vocology.naturalness import FillerType
        assert FillerType.UH.value == "uh"

    def test_filler_type_er(self):
        """Test FillerType.ER value."""
        from voice_soundboard.vocology.naturalness import FillerType
        assert FillerType.ER.value == "er"

    def test_filler_type_ah(self):
        """Test FillerType.AH value."""
        from voice_soundboard.vocology.naturalness import FillerType
        assert FillerType.AH.value == "ah"


# ============== BreathType Enum Tests ==============

class TestBreathTypeEnum:
    """Tests for BreathType enum."""

    def test_breath_type_inhale(self):
        """Test BreathType.INHALE value."""
        from voice_soundboard.vocology.naturalness import BreathType
        assert BreathType.INHALE.value == "inhale"

    def test_breath_type_exhale(self):
        """Test BreathType.EXHALE value."""
        from voice_soundboard.vocology.naturalness import BreathType
        assert BreathType.EXHALE.value == "exhale"

    def test_breath_type_catch_breath(self):
        """Test BreathType.CATCH_BREATH value."""
        from voice_soundboard.vocology.naturalness import BreathType
        assert BreathType.CATCH_BREATH.value == "catch"


# ============== NaturalnessConfig Tests ==============

class TestNaturalnessConfig:
    """Tests for NaturalnessConfig dataclass."""

    def test_naturalness_config_defaults(self):
        """Test NaturalnessConfig default values."""
        from voice_soundboard.vocology.naturalness import NaturalnessConfig
        config = NaturalnessConfig()
        assert config.filler_rate == 0.025
        assert config.um_probability == 0.6
        assert config.creaky_probability == 0.3

    def test_naturalness_config_custom_values(self):
        """Test NaturalnessConfig with custom values."""
        from voice_soundboard.vocology.naturalness import NaturalnessConfig
        config = NaturalnessConfig(
            filler_rate=0.05,
            um_probability=0.8,
            creaky_probability=0.5
        )
        assert config.filler_rate == 0.05
        assert config.um_probability == 0.8

    def test_naturalness_config_breath_settings(self):
        """Test NaturalnessConfig breath settings."""
        from voice_soundboard.vocology.naturalness import NaturalnessConfig
        config = NaturalnessConfig()
        assert config.breath_min_phrase_length == 8
        assert config.breath_at_clause_boundary == 0.4
        assert config.breath_duration_min_ms == 100
        assert config.breath_duration_max_ms == 250

    def test_naturalness_config_timing_settings(self):
        """Test NaturalnessConfig timing settings."""
        from voice_soundboard.vocology.naturalness import NaturalnessConfig
        config = NaturalnessConfig()
        assert config.timing_jitter_ms == 8.0
        assert config.timing_delay_bias == 0.6

    def test_naturalness_config_pitch_drift_settings(self):
        """Test NaturalnessConfig pitch drift settings."""
        from voice_soundboard.vocology.naturalness import NaturalnessConfig
        config = NaturalnessConfig()
        assert config.pitch_drift_cents == 6.0
        assert config.pitch_drift_rate_hz == 0.5


# ============== TextAnnotation Tests ==============

class TestTextAnnotation:
    """Tests for TextAnnotation dataclass."""

    def test_text_annotation_creation(self):
        """Test TextAnnotation basic creation."""
        from voice_soundboard.vocology.naturalness import TextAnnotation
        annotation = TextAnnotation(
            text="hello",
            start_idx=0,
            end_idx=5
        )
        assert annotation.text == "hello"
        assert annotation.start_idx == 0
        assert annotation.end_idx == 5

    def test_text_annotation_defaults(self):
        """Test TextAnnotation default marker values."""
        from voice_soundboard.vocology.naturalness import TextAnnotation
        annotation = TextAnnotation(text="word", start_idx=0, end_idx=4)
        assert annotation.insert_filler is None
        assert annotation.insert_breath is None
        assert annotation.apply_creaky is False
        assert annotation.timing_offset_ms == 0.0

    def test_text_annotation_with_filler(self):
        """Test TextAnnotation with filler marker."""
        from voice_soundboard.vocology.naturalness import TextAnnotation, FillerType
        annotation = TextAnnotation(
            text="word",
            start_idx=0,
            end_idx=4,
            insert_filler=FillerType.UM
        )
        assert annotation.insert_filler == FillerType.UM


# ============== Helper Functions Tests ==============

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_sentence_end_period(self):
        """Test _is_sentence_end with period."""
        from voice_soundboard.vocology.naturalness import _is_sentence_end
        assert _is_sentence_end("Hello.", 5) is True

    def test_is_sentence_end_question(self):
        """Test _is_sentence_end with question mark."""
        from voice_soundboard.vocology.naturalness import _is_sentence_end
        assert _is_sentence_end("Hello?", 5) is True

    def test_is_sentence_end_exclamation(self):
        """Test _is_sentence_end with exclamation mark."""
        from voice_soundboard.vocology.naturalness import _is_sentence_end
        assert _is_sentence_end("Hello!", 5) is True

    def test_is_sentence_end_not_end(self):
        """Test _is_sentence_end at non-end position."""
        from voice_soundboard.vocology.naturalness import _is_sentence_end
        assert _is_sentence_end("Hello world.", 5) is False

    def test_is_clause_boundary_comma(self):
        """Test _is_clause_boundary with comma."""
        from voice_soundboard.vocology.naturalness import _is_clause_boundary
        assert _is_clause_boundary("Hello, world", 5) is True

    def test_is_clause_boundary_semicolon(self):
        """Test _is_clause_boundary with semicolon."""
        from voice_soundboard.vocology.naturalness import _is_clause_boundary
        assert _is_clause_boundary("Hello; world", 5) is True

    def test_is_clause_boundary_not_boundary(self):
        """Test _is_clause_boundary at non-boundary."""
        from voice_soundboard.vocology.naturalness import _is_clause_boundary
        assert _is_clause_boundary("Hello world", 5) is False

    def test_is_question_true(self):
        """Test _is_question with question sentence."""
        from voice_soundboard.vocology.naturalness import _is_question
        assert _is_question("Is this a question?") is True

    def test_is_question_false(self):
        """Test _is_question with non-question sentence."""
        from voice_soundboard.vocology.naturalness import _is_question
        assert _is_question("This is a statement.") is False

    def test_count_syllables_simple(self):
        """Test _count_syllables with simple words."""
        from voice_soundboard.vocology.naturalness import _count_syllables
        assert _count_syllables("hello") == 2
        assert _count_syllables("world") == 1

    def test_count_syllables_complex(self):
        """Test _count_syllables with complex words."""
        from voice_soundboard.vocology.naturalness import _count_syllables
        assert _count_syllables("beautiful") >= 3
        assert _count_syllables("communication") >= 4

    def test_get_word_tokens(self):
        """Test _get_word_tokens function."""
        from voice_soundboard.vocology.naturalness import _get_word_tokens
        tokens = _get_word_tokens("Hello world!")
        assert len(tokens) == 2
        assert tokens[0][0] == "Hello"
        assert tokens[1][0] == "world"


# ============== NaturalSpeechProcessor Tests ==============

class TestNaturalSpeechProcessor:
    """Tests for NaturalSpeechProcessor class."""

    def test_natural_speech_processor_init(self):
        """Test NaturalSpeechProcessor initialization."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        assert processor.config is not None

    def test_natural_speech_processor_init_with_config(self):
        """Test NaturalSpeechProcessor with custom config."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor, NaturalnessConfig
        config = NaturalnessConfig(filler_rate=0.05)
        processor = NaturalSpeechProcessor(config=config)
        assert processor.config.filler_rate == 0.05

    def test_natural_speech_processor_seed(self):
        """Test NaturalSpeechProcessor.seed method."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        processor.seed(42)
        # Should not raise

    def test_natural_speech_processor_looks_like_noun_function_word(self):
        """Test NaturalSpeechProcessor._looks_like_noun with function word."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        assert processor._looks_like_noun("the") is False
        assert processor._looks_like_noun("and") is False

    def test_natural_speech_processor_looks_like_noun_capitalized(self):
        """Test NaturalSpeechProcessor._looks_like_noun with capitalized word."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        assert processor._looks_like_noun("John") is True

    def test_natural_speech_processor_looks_like_noun_ending(self):
        """Test NaturalSpeechProcessor._looks_like_noun with noun ending."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        assert processor._looks_like_noun("information") is True
        assert processor._looks_like_noun("happiness") is True

    def test_natural_speech_processor_choose_filler_type(self):
        """Test NaturalSpeechProcessor._choose_filler_type method."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor, NaturalnessConfig, FillerType
        processor = NaturalSpeechProcessor()
        processor.seed(42)  # For reproducibility

        config = NaturalnessConfig()
        filler = processor._choose_filler_type("word", config)
        assert filler in [FillerType.UM, FillerType.UH]

    def test_natural_speech_processor_adjust_for_style_formal(self):
        """Test NaturalSpeechProcessor._adjust_for_style with formal style."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        adjusted = processor._adjust_for_style("formal")
        # Formal should have fewer fillers
        assert adjusted.filler_rate < processor.config.filler_rate

    def test_natural_speech_processor_adjust_for_style_casual(self):
        """Test NaturalSpeechProcessor._adjust_for_style with casual style."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        adjusted = processor._adjust_for_style("casual")
        # Casual should have more fillers
        assert adjusted.filler_rate > processor.config.filler_rate

    def test_natural_speech_processor_process(self):
        """Test NaturalSpeechProcessor.process method."""
        from voice_soundboard.vocology.naturalness import NaturalSpeechProcessor
        processor = NaturalSpeechProcessor()
        processor.seed(42)

        result = processor.process("Hello world, how are you today?")
        assert result.original_text == "Hello world, how are you today?"
        assert len(result.annotations) > 0


# ============== FUNCTION_WORDS Constant Tests ==============

class TestFunctionWordsConstant:
    """Tests for FUNCTION_WORDS constant."""

    def test_function_words_contains_common_words(self):
        """Test FUNCTION_WORDS contains common function words."""
        from voice_soundboard.vocology.naturalness import FUNCTION_WORDS
        assert "the" in FUNCTION_WORDS
        assert "a" in FUNCTION_WORDS
        assert "is" in FUNCTION_WORDS
        assert "to" in FUNCTION_WORDS

    def test_function_words_contains_pronouns(self):
        """Test FUNCTION_WORDS contains pronouns."""
        from voice_soundboard.vocology.naturalness import FUNCTION_WORDS
        assert "i" in FUNCTION_WORDS
        assert "you" in FUNCTION_WORDS
        assert "he" in FUNCTION_WORDS
        assert "she" in FUNCTION_WORDS
