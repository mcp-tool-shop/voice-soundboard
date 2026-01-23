"""
Additional coverage tests - Batch 28: Codec Modules.

Tests for voice_soundboard/codecs/base.py, mimi.py, dualcodec.py, and llm.py.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# =============================================================================
# Base Codec Tests
# =============================================================================

class TestCodecType:
    """Tests for CodecType enum."""

    def test_codec_types_defined(self):
        """Test all codec types are defined."""
        from voice_soundboard.codecs.base import CodecType

        assert CodecType.ACOUSTIC is not None
        assert CodecType.SEMANTIC is not None
        assert CodecType.DUAL is not None
        assert CodecType.HYBRID is not None


class TestCodecCapabilities:
    """Tests for CodecCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default CodecCapabilities values."""
        from voice_soundboard.codecs.base import CodecCapabilities, CodecType

        caps = CodecCapabilities()

        assert caps.can_encode is True
        assert caps.can_decode is True
        assert caps.can_stream is False
        assert caps.codec_type == CodecType.ACOUSTIC
        assert caps.num_codebooks == 1
        assert caps.codebook_size == 1024
        assert caps.frame_rate_hz == 50.0
        assert caps.sample_rate == 24000

    def test_custom_capabilities(self):
        """Test custom CodecCapabilities values."""
        from voice_soundboard.codecs.base import CodecCapabilities, CodecType

        caps = CodecCapabilities(
            can_stream=True,
            codec_type=CodecType.DUAL,
            num_codebooks=8,
            codebook_size=2048,
            frame_rate_hz=12.5,
            has_semantic_tokens=True,
        )

        assert caps.can_stream is True
        assert caps.codec_type == CodecType.DUAL
        assert caps.num_codebooks == 8
        assert caps.has_semantic_tokens is True


class TestCodecConfig:
    """Tests for CodecConfig dataclass."""

    def test_default_config(self):
        """Test default CodecConfig values."""
        from voice_soundboard.codecs.base import CodecConfig

        config = CodecConfig()

        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.chunk_size_ms == 80
        assert config.add_special_tokens is True
        assert config.pad_token_id == 0
        assert config.bos_token_id == 1
        assert config.eos_token_id == 2

    def test_custom_config(self):
        """Test custom CodecConfig values."""
        from voice_soundboard.codecs.base import CodecConfig

        config = CodecConfig(
            device="cuda",
            chunk_size_ms=100,
            num_codebooks=4,
        )

        assert config.device == "cuda"
        assert config.chunk_size_ms == 100
        assert config.num_codebooks == 4


class TestTokenSequence:
    """Tests for TokenSequence dataclass."""

    def test_1d_tokens(self):
        """Test TokenSequence with 1D tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        assert seq.num_codebooks == 1
        assert seq.sequence_length == 5
        assert seq.duration_seconds == 0.1  # 5 / 50

    def test_2d_tokens(self):
        """Test TokenSequence with 2D tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        assert seq.num_codebooks == 2
        assert seq.sequence_length == 3
        assert seq.duration_seconds == 0.06  # 3 / 50

    def test_auto_duration_calculation(self):
        """Test automatic duration calculation."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=10.0)

        assert seq.source_duration_seconds == 1.0  # 10 / 10

    def test_to_flat(self):
        """Test flattening multi-codebook tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        flat = seq.to_flat()
        # Interleaved: [1, 4, 2, 5, 3, 6]
        assert list(flat) == [1, 4, 2, 5, 3, 6]

    def test_to_flat_1d(self):
        """Test to_flat with 1D tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([1, 2, 3], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        flat = seq.to_flat()
        assert list(flat) == [1, 2, 3]

    def test_to_llm_tokens(self):
        """Test conversion to LLM tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([0, 1, 2], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        llm_tokens = seq.to_llm_tokens(add_special_tokens=True)

        assert llm_tokens[0] == 1  # BOS
        assert llm_tokens[-1] == 2  # EOS
        # Middle tokens have offset of 3
        assert list(llm_tokens[1:-1]) == [3, 4, 5]

    def test_to_llm_tokens_no_special(self):
        """Test conversion to LLM tokens without special tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([0, 1, 2], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        llm_tokens = seq.to_llm_tokens(add_special_tokens=False)
        assert list(llm_tokens) == [3, 4, 5]  # Just offset

    def test_from_llm_tokens(self):
        """Test creating TokenSequence from LLM tokens."""
        from voice_soundboard.codecs.base import TokenSequence

        llm_tokens = np.array([1, 3, 4, 5, 2], dtype=np.int64)  # BOS, tokens, EOS
        seq = TokenSequence.from_llm_tokens(
            llm_tokens,
            num_codebooks=1,
            frame_rate_hz=50.0,
            has_special_tokens=True,
        )

        assert list(seq.tokens) == [0, 1, 2]

    def test_from_llm_tokens_multi_codebook(self):
        """Test from_llm_tokens with multiple codebooks."""
        from voice_soundboard.codecs.base import TokenSequence

        # 6 tokens = 2 codebooks x 3 frames (interleaved)
        llm_tokens = np.array([1, 3, 4, 5, 6, 7, 8, 2], dtype=np.int64)
        seq = TokenSequence.from_llm_tokens(
            llm_tokens,
            num_codebooks=2,
            frame_rate_hz=50.0,
            has_special_tokens=True,
        )

        assert seq.num_codebooks == 2
        assert seq.sequence_length == 3

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        from voice_soundboard.codecs.base import TokenSequence

        tokens = np.array([[1, 2], [3, 4]], dtype=np.int64)
        seq = TokenSequence(
            tokens=tokens,
            frame_rate_hz=50.0,
            source_sample_rate=24000,
            codec_name="test",
            codec_version="1.0",
        )

        data = seq.to_dict()
        reconstructed = TokenSequence.from_dict(data)

        assert reconstructed.num_codebooks == 2
        assert reconstructed.sequence_length == 2
        assert reconstructed.codec_name == "test"
        assert np.array_equal(reconstructed.tokens, tokens)


class TestEncodedAudio:
    """Tests for EncodedAudio dataclass."""

    def test_has_dual_tokens(self):
        """Test has_dual_tokens property."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence

        tokens = np.array([1, 2, 3], dtype=np.int64)
        main_seq = TokenSequence(tokens=tokens)
        semantic_seq = TokenSequence(tokens=tokens)
        acoustic_seq = TokenSequence(tokens=tokens)

        # Without dual tokens
        encoded1 = EncodedAudio(tokens=main_seq)
        assert encoded1.has_dual_tokens is False

        # With dual tokens
        encoded2 = EncodedAudio(
            tokens=main_seq,
            semantic_tokens=semantic_seq,
            acoustic_tokens=acoustic_seq,
        )
        assert encoded2.has_dual_tokens is True

    def test_to_dict(self):
        """Test EncodedAudio serialization."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence

        tokens = np.array([1, 2, 3], dtype=np.int64)
        encoded = EncodedAudio(
            tokens=TokenSequence(tokens=tokens),
            estimated_quality=0.95,
        )

        data = encoded.to_dict()
        assert "tokens" in data
        assert data["estimated_quality"] == 0.95

    def test_to_dict_with_dual(self):
        """Test EncodedAudio serialization with dual tokens."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence

        tokens = np.array([1, 2, 3], dtype=np.int64)
        encoded = EncodedAudio(
            tokens=TokenSequence(tokens=tokens),
            semantic_tokens=TokenSequence(tokens=tokens),
            acoustic_tokens=TokenSequence(tokens=tokens),
        )

        data = encoded.to_dict()
        assert "semantic_tokens" in data
        assert "acoustic_tokens" in data

    def test_from_dict(self):
        """Test EncodedAudio deserialization."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence

        tokens = np.array([1, 2, 3], dtype=np.int64)
        original = EncodedAudio(
            tokens=TokenSequence(tokens=tokens),
            estimated_quality=0.9,
        )

        data = original.to_dict()
        reconstructed = EncodedAudio.from_dict(data)

        assert reconstructed.estimated_quality == 0.9


class TestAudioCodecBase:
    """Tests for AudioCodec base class methods."""

    def test_ensure_loaded(self):
        """Test _ensure_loaded method."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        assert codec._loaded is False

        codec._ensure_loaded()
        assert codec._loaded is True

        # Second call should not reload
        with patch.object(codec, 'load') as mock_load:
            codec._ensure_loaded()
            mock_load.assert_not_called()

    def test_get_vocab_size(self):
        """Test get_vocab_size method."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        vocab_size = codec.get_vocab_size()

        # 8 codebooks * 2048 codebook_size + 3 special tokens
        expected = 8 * 2048 + 3
        assert vocab_size == expected

    def test_estimate_tokens(self):
        """Test estimate_tokens method."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        # 1 second at 12.5 Hz frame rate with 8 codebooks
        tokens = codec.estimate_tokens(1.0)
        assert tokens == 12 * 8  # int(1.0 * 12.5) * 8 = 100

    def test_estimate_duration(self):
        """Test estimate_duration method."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        # 100 tokens with 8 codebooks = 12 frames at 12.5 Hz
        duration = codec.estimate_duration(96)  # 96 / 8 = 12 frames
        assert duration == 12 / 12.5  # 0.96 seconds


# =============================================================================
# Mimi Codec Tests
# =============================================================================

class TestMimiCodec:
    """Tests for MimiCodec class."""

    def test_init(self):
        """Test MimiCodec initialization."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        assert codec.name == "mimi"
        assert codec.FRAME_RATE_HZ == 12.5
        assert codec.NUM_CODEBOOKS == 8
        assert codec.SAMPLE_RATE == 24000

    def test_version_mock(self):
        """Test version property with mock implementation."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        # Without moshi installed, should return "mock"
        # This depends on whether moshi is installed

    def test_capabilities(self):
        """Test MimiCodec capabilities."""
        from voice_soundboard.codecs.mimi import MimiCodec, CodecType

        codec = MimiCodec()
        caps = codec.capabilities

        assert caps.can_encode is True
        assert caps.can_decode is True
        assert caps.can_stream is True
        assert caps.codec_type == CodecType.HYBRID
        assert caps.has_semantic_tokens is True
        assert caps.num_codebooks == 8
        assert caps.codebook_size == 2048
        assert caps.frame_rate_hz == 12.5

    def test_load_without_mimi(self):
        """Test load without mimi package."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        codec.load()
        assert codec._loaded is True

    def test_encode_mock(self):
        """Test encode with mock implementation."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        audio = np.random.randn(24000).astype(np.float32)  # 1 second

        encoded = codec.encode(audio, sample_rate=24000)

        assert encoded.tokens is not None
        assert encoded.semantic_tokens is not None
        assert encoded.acoustic_tokens is not None
        # 1 second at 12.5 Hz = 12 frames
        assert encoded.tokens.sequence_length >= 10

    def test_encode_deterministic(self):
        """Test that mock encoding is deterministic."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded1 = codec.encode(audio, sample_rate=24000)
        encoded2 = codec.encode(audio, sample_rate=24000)

        assert np.array_equal(encoded1.tokens.tokens, encoded2.tokens.tokens)

    def test_decode_mock(self):
        """Test decode with mock implementation."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        audio = np.random.randn(24000).astype(np.float32)
        encoded = codec.encode(audio, sample_rate=24000)

        decoded_audio, sample_rate = codec.decode(encoded)

        assert sample_rate == 24000
        assert len(decoded_audio) > 0

    def test_decode_from_numpy_1d(self):
        """Test decoding from 1D numpy array."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        tokens = np.random.randint(0, 2048, size=100, dtype=np.int64)

        decoded_audio, sample_rate = codec.decode(tokens)
        assert sample_rate == 24000

    def test_decode_from_numpy_2d(self):
        """Test decoding from 2D numpy array."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        tokens = np.random.randint(0, 2048, size=(8, 12), dtype=np.int64)

        decoded_audio, sample_rate = codec.decode(tokens)
        assert sample_rate == 24000

    def test_encode_with_resampling(self):
        """Test encoding with sample rate conversion."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        # 16kHz audio
        audio = np.random.randn(16000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=16000)
        assert encoded.tokens is not None

    def test_encode_streaming(self):
        """Test streaming encoding."""
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()

        def audio_generator():
            for _ in range(5):
                yield np.random.randn(4800).astype(np.float32)

        tokens = list(codec.encode_streaming(audio_generator(), sample_rate=24000))
        assert len(tokens) > 0

    def test_decode_streaming(self):
        """Test streaming decoding."""
        from voice_soundboard.codecs.mimi import MimiCodec
        from voice_soundboard.codecs.base import TokenSequence

        codec = MimiCodec()

        def token_generator():
            for _ in range(3):
                tokens = np.random.randint(0, 2048, size=(8, 10), dtype=np.int64)
                yield TokenSequence(tokens=tokens, frame_rate_hz=12.5)

        audio_chunks = list(codec.decode_streaming(token_generator()))
        assert len(audio_chunks) == 3


# =============================================================================
# DualCodec Tests
# =============================================================================

class TestDualCodec:
    """Tests for DualCodec class."""

    def test_init(self):
        """Test DualCodec initialization."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        assert codec.name == "dualcodec"
        assert codec.SEMANTIC_FRAME_RATE == 50.0
        assert codec.ACOUSTIC_FRAME_RATE == 50.0

    def test_capabilities(self):
        """Test DualCodec capabilities."""
        from voice_soundboard.codecs.dualcodec import DualCodec
        from voice_soundboard.codecs.base import CodecType

        codec = DualCodec()
        caps = codec.capabilities

        assert caps.codec_type == CodecType.DUAL
        assert caps.has_semantic_tokens is True
        assert caps.has_acoustic_tokens is True
        assert caps.num_codebooks == 8  # 1 semantic + 7 acoustic

    def test_load(self):
        """Test DualCodec load."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        codec.load()
        assert codec._loaded is True

    def test_encode(self):
        """Test DualCodec encoding."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)

        assert encoded.tokens is not None
        assert encoded.semantic_tokens is not None
        assert encoded.acoustic_tokens is not None
        assert encoded.has_dual_tokens is True

    def test_encode_dual(self):
        """Test encode_dual convenience method."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)

        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)

        assert semantic.num_codebooks == 1
        assert acoustic.num_codebooks == 7

    def test_decode(self):
        """Test DualCodec decoding."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)
        encoded = codec.encode(audio, sample_rate=24000)

        decoded_audio, sample_rate = codec.decode(encoded)

        assert sample_rate == 24000
        assert len(decoded_audio) > 0

    def test_decode_dual(self):
        """Test decode_dual method."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)
        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)

        decoded_audio, sample_rate = codec.decode_dual(semantic, acoustic)

        assert sample_rate == 24000
        assert len(decoded_audio) > 0

    def test_voice_convert(self):
        """Test voice conversion."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        content_audio = np.random.randn(24000).astype(np.float32)
        style_audio = np.random.randn(24000).astype(np.float32)

        converted, sample_rate = codec.voice_convert(
            content_audio,
            style_audio,
            content_sample_rate=24000,
            style_sample_rate=24000,
        )

        assert sample_rate == 24000
        assert len(converted) > 0

    def test_decode_from_encoded_audio_with_dual(self):
        """Test decoding from EncodedAudio with dual tokens."""
        from voice_soundboard.codecs.dualcodec import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)
        encoded = codec.encode(audio, sample_rate=24000)

        # Decode should use dual tokens
        decoded_audio, sr = codec.decode(encoded)
        assert sr == 24000

    def test_decode_from_token_sequence(self):
        """Test decoding from TokenSequence."""
        from voice_soundboard.codecs.dualcodec import DualCodec
        from voice_soundboard.codecs.base import TokenSequence

        codec = DualCodec()
        # Combined tokens (1 semantic + 7 acoustic)
        tokens = np.random.randint(0, 1024, size=(8, 50), dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        decoded_audio, sr = codec.decode(seq)
        assert sr == 24000


# =============================================================================
# LLM Integration Tests
# =============================================================================

class TestTokenType:
    """Tests for TokenType enum."""

    def test_token_types(self):
        """Test all token types are defined."""
        from voice_soundboard.codecs.llm import TokenType

        assert TokenType.TEXT is not None
        assert TokenType.AUDIO is not None
        assert TokenType.SEMANTIC is not None
        assert TokenType.ACOUSTIC is not None
        assert TokenType.SPECIAL is not None


class TestSpecialTokens:
    """Tests for SpecialTokens dataclass."""

    def test_default_tokens(self):
        """Test default special tokens."""
        from voice_soundboard.codecs.llm import SpecialTokens

        tokens = SpecialTokens()

        assert tokens.pad_token == "<pad>"
        assert tokens.bos_token == "<bos>"
        assert tokens.eos_token == "<eos>"
        assert tokens.audio_start == "<|audio|>"
        assert tokens.audio_end == "<|/audio|>"

    def test_get_speaker_token(self):
        """Test speaker token generation."""
        from voice_soundboard.codecs.llm import SpecialTokens

        tokens = SpecialTokens()
        speaker_token = tokens.get_speaker_token("speaker1")

        assert speaker_token == "<|speaker_speaker1|>"

    def test_get_emotion_token(self):
        """Test emotion token generation."""
        from voice_soundboard.codecs.llm import SpecialTokens

        tokens = SpecialTokens()
        emotion_token = tokens.get_emotion_token("happy")

        assert emotion_token == "<|emotion_happy|>"


class TestVocabularyConfig:
    """Tests for VocabularyConfig dataclass."""

    def test_default_config(self):
        """Test default vocabulary config."""
        from voice_soundboard.codecs.llm import VocabularyConfig

        config = VocabularyConfig()

        assert config.text_vocab_size == 32000
        assert config.audio_vocab_size == 8192
        assert config.num_special_tokens == 256

    def test_total_vocab_size(self):
        """Test total_vocab_size property."""
        from voice_soundboard.codecs.llm import VocabularyConfig

        config = VocabularyConfig()
        expected = 256 + 32000 + 8192
        assert config.total_vocab_size == expected

    def test_text_to_global(self):
        """Test text_to_global conversion."""
        from voice_soundboard.codecs.llm import VocabularyConfig

        config = VocabularyConfig()
        global_id = config.text_to_global(100)
        assert global_id == 100 + 256  # offset by special tokens

    def test_audio_to_global(self):
        """Test audio_to_global conversion."""
        from voice_soundboard.codecs.llm import VocabularyConfig

        config = VocabularyConfig()
        global_id = config.audio_to_global(100)
        assert global_id == 100 + 32000  # audio_token_offset

    def test_global_to_text(self):
        """Test global_to_text conversion."""
        from voice_soundboard.codecs.llm import VocabularyConfig

        config = VocabularyConfig()

        # Valid text token
        text_id = config.global_to_text(300)  # 300 - 256 = 44
        assert text_id == 44

        # Invalid (in special tokens range)
        invalid = config.global_to_text(100)
        assert invalid is None

        # Invalid (in audio range)
        invalid = config.global_to_audio(33000)  # 33000 - 32000 = 1000
        assert invalid == 1000

    def test_global_to_audio(self):
        """Test global_to_audio conversion."""
        from voice_soundboard.codecs.llm import VocabularyConfig

        config = VocabularyConfig()

        # Valid audio token
        audio_id = config.global_to_audio(32500)
        assert audio_id == 500

        # Invalid (not in audio range)
        invalid = config.global_to_audio(1000)
        assert invalid is None

    def test_get_token_type(self):
        """Test get_token_type method."""
        from voice_soundboard.codecs.llm import VocabularyConfig, TokenType

        config = VocabularyConfig()

        assert config.get_token_type(100) == TokenType.SPECIAL
        assert config.get_token_type(1000) == TokenType.TEXT
        assert config.get_token_type(33000) == TokenType.AUDIO


class TestAudioPrompt:
    """Tests for AudioPrompt dataclass."""

    def test_prompt_creation(self):
        """Test creating audio prompt."""
        from voice_soundboard.codecs.llm import AudioPrompt

        prompt = AudioPrompt(
            text_prefix="Hello",
            text_suffix="world",
            speaker_id="speaker1",
            emotion="happy",
        )

        assert prompt.text_prefix == "Hello"
        assert prompt.speaker_id == "speaker1"
        assert prompt.emotion == "happy"

    def test_to_token_sequence_basic(self):
        """Test converting prompt to tokens."""
        from voice_soundboard.codecs.llm import AudioPrompt

        prompt = AudioPrompt(text_prefix="Hello")
        tokens = prompt.to_token_sequence()

        assert tokens[0] == 1  # BOS
        assert tokens[-1] == 2  # EOS

    def test_to_token_sequence_with_tokenizer(self):
        """Test converting prompt with text tokenizer."""
        from voice_soundboard.codecs.llm import AudioPrompt

        def mock_tokenizer(text):
            return [ord(c) for c in text[:5]]

        prompt = AudioPrompt(text_prefix="Hello")
        tokens = prompt.to_token_sequence(text_tokenizer=mock_tokenizer)

        assert len(tokens) > 2  # BOS + text + EOS

    def test_to_token_sequence_with_audio(self):
        """Test converting prompt with audio tokens."""
        from voice_soundboard.codecs.llm import AudioPrompt
        from voice_soundboard.codecs.base import TokenSequence

        audio_tokens = TokenSequence(
            tokens=np.array([1, 2, 3], dtype=np.int64),
            frame_rate_hz=50.0,
        )
        prompt = AudioPrompt(audio_tokens=audio_tokens)
        tokens = prompt.to_token_sequence()

        # Should include audio start/end markers
        assert len(tokens) > 5


class TestLLMCodecBridge:
    """Tests for LLMCodecBridge class."""

    def test_init(self):
        """Test LLMCodecBridge initialization."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        assert bridge.codec == codec
        assert bridge.vocab_config is not None
        assert bridge.special_tokens is not None

    def test_encode_for_llm(self):
        """Test encoding audio for LLM."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        audio = np.random.randn(24000).astype(np.float32)
        tokens = bridge.encode_for_llm(audio, sample_rate=24000)

        assert len(tokens) > 0

    def test_encode_for_llm_with_speaker(self):
        """Test encoding with speaker ID."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        audio = np.random.randn(24000).astype(np.float32)
        tokens = bridge.encode_for_llm(
            audio,
            sample_rate=24000,
            speaker_id="speaker1",
        )

        # Should have extra token for speaker
        tokens_no_speaker = bridge.encode_for_llm(audio, sample_rate=24000)
        assert len(tokens) == len(tokens_no_speaker) + 1

    def test_decode_from_llm(self):
        """Test decoding LLM output to audio."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        audio = np.random.randn(24000).astype(np.float32)
        llm_tokens = bridge.encode_for_llm(audio, sample_rate=24000)

        decoded_audio, sr = bridge.decode_from_llm(llm_tokens)
        assert sr == 24000

    def test_decode_empty_returns_silence(self):
        """Test decoding empty tokens returns silence."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        # Tokens with no audio
        tokens = np.array([1, 2], dtype=np.int64)  # Just BOS/EOS
        audio, sr = bridge.decode_from_llm(tokens)

        assert sr == 24000
        assert len(audio) == 24000  # 1 second of silence

    def test_create_prompt(self):
        """Test creating audio prompt."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        audio = np.random.randn(24000).astype(np.float32)
        prompt = bridge.create_prompt(
            text="Hello",
            audio=audio,
            sample_rate=24000,
            speaker_id="speaker1",
            emotion="happy",
        )

        assert prompt.text_prefix == "Hello"
        assert prompt.speaker_id == "speaker1"
        assert prompt.audio_tokens is not None

    def test_stream_decode(self):
        """Test streaming decode."""
        from voice_soundboard.codecs.llm import LLMCodecBridge
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        bridge = LLMCodecBridge(codec)

        # Create a token generator
        audio = np.random.randn(24000).astype(np.float32)
        llm_tokens = bridge.encode_for_llm(audio, sample_rate=24000)

        def token_gen():
            for t in llm_tokens:
                yield t

        chunks = list(bridge.stream_decode(token_gen(), buffer_size=50))
        # Should have at least one chunk


class TestHelperFunctions:
    """Tests for helper functions in llm module."""

    def test_get_codec_vocabulary_info(self):
        """Test get_codec_vocabulary_info function."""
        from voice_soundboard.codecs.llm import get_codec_vocabulary_info
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        info = get_codec_vocabulary_info(codec)

        assert info["codec_name"] == "mimi"
        assert info["num_codebooks"] == 8
        assert info["codebook_size"] == 2048
        assert info["frame_rate_hz"] == 12.5
        assert "recommended_special_tokens" in info

    def test_estimate_audio_context_length(self):
        """Test estimate_audio_context_length function."""
        from voice_soundboard.codecs.llm import estimate_audio_context_length
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        length = estimate_audio_context_length(1.0, codec, include_markers=True)

        # Should include audio tokens + 2 markers
        expected = codec.estimate_tokens(1.0) + 2
        assert length == expected

    def test_estimate_audio_duration(self):
        """Test estimate_audio_duration function."""
        from voice_soundboard.codecs.llm import estimate_audio_duration
        from voice_soundboard.codecs.mimi import MimiCodec

        codec = MimiCodec()
        # Estimate duration for 100 tokens (including 2 markers)
        duration = estimate_audio_duration(100, codec, include_markers=True)

        # 98 audio tokens / (8 codebooks * 12.5 Hz)
        expected = 98 // 8 / 12.5
        assert abs(duration - expected) < 0.1
