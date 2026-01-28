"""
Test Additional Coverage Batch 48: Codecs & Integration Tests

Tests for:
- CodecType enum
- CodecCapabilities dataclass
- CodecConfig dataclass
- TokenSequence class
- EncodedAudio dataclass
- AudioCodec abstract base class
- MockCodec implementation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path


# ============== CodecType Enum Tests ==============

class TestCodecTypeEnum:
    """Tests for CodecType enum."""

    def test_codec_type_acoustic(self):
        """Test CodecType.ACOUSTIC exists."""
        from voice_soundboard.codecs.base import CodecType
        assert CodecType.ACOUSTIC is not None

    def test_codec_type_semantic(self):
        """Test CodecType.SEMANTIC exists."""
        from voice_soundboard.codecs.base import CodecType
        assert CodecType.SEMANTIC is not None

    def test_codec_type_dual(self):
        """Test CodecType.DUAL exists."""
        from voice_soundboard.codecs.base import CodecType
        assert CodecType.DUAL is not None

    def test_codec_type_hybrid(self):
        """Test CodecType.HYBRID exists."""
        from voice_soundboard.codecs.base import CodecType
        assert CodecType.HYBRID is not None


# ============== CodecCapabilities Tests ==============

class TestCodecCapabilities:
    """Tests for CodecCapabilities dataclass."""

    def test_capabilities_default_values(self):
        """Test CodecCapabilities default values."""
        from voice_soundboard.codecs.base import CodecCapabilities
        caps = CodecCapabilities()
        assert caps.can_encode is True
        assert caps.can_decode is True
        assert caps.can_stream is False
        assert caps.num_codebooks == 1

    def test_capabilities_with_streaming(self):
        """Test CodecCapabilities with streaming enabled."""
        from voice_soundboard.codecs.base import CodecCapabilities
        caps = CodecCapabilities(can_stream=True)
        assert caps.can_stream is True

    def test_capabilities_frame_rate(self):
        """Test CodecCapabilities frame rate."""
        from voice_soundboard.codecs.base import CodecCapabilities
        caps = CodecCapabilities(frame_rate_hz=75.0)
        assert caps.frame_rate_hz == 75.0

    def test_capabilities_bitrate_range(self):
        """Test CodecCapabilities bitrate settings."""
        from voice_soundboard.codecs.base import CodecCapabilities
        caps = CodecCapabilities(
            min_bitrate_kbps=1.5,
            max_bitrate_kbps=24.0
        )
        assert caps.min_bitrate_kbps == 1.5
        assert caps.max_bitrate_kbps == 24.0


# ============== CodecConfig Tests ==============

class TestCodecConfig:
    """Tests for CodecConfig dataclass."""

    def test_config_default_values(self):
        """Test CodecConfig default values."""
        from voice_soundboard.codecs.base import CodecConfig
        config = CodecConfig()
        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.target_bandwidth_kbps == 6.0

    def test_config_with_device(self):
        """Test CodecConfig with custom device."""
        from voice_soundboard.codecs.base import CodecConfig
        config = CodecConfig(device="cuda")
        assert config.device == "cuda"

    def test_config_special_tokens(self):
        """Test CodecConfig special token settings."""
        from voice_soundboard.codecs.base import CodecConfig
        config = CodecConfig()
        assert config.add_special_tokens is True
        assert config.bos_token_id == 1
        assert config.eos_token_id == 2


# ============== TokenSequence Tests ==============

class TestTokenSequence:
    """Tests for TokenSequence class."""

    def test_token_sequence_1d_creation(self):
        """Test TokenSequence with 1D tokens."""
        from voice_soundboard.codecs.base import TokenSequence
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)
        assert seq.sequence_length == 5
        assert seq.num_codebooks == 1

    def test_token_sequence_2d_creation(self):
        """Test TokenSequence with 2D tokens."""
        from voice_soundboard.codecs.base import TokenSequence
        tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)
        assert seq.num_codebooks == 2
        assert seq.sequence_length == 3

    def test_token_sequence_duration_property(self):
        """Test TokenSequence duration calculation."""
        from voice_soundboard.codecs.base import TokenSequence
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)
        assert seq.duration_seconds == 0.1  # 5 frames / 50 Hz

    def test_token_sequence_to_flat(self):
        """Test TokenSequence.to_flat method."""
        from voice_soundboard.codecs.base import TokenSequence
        tokens = np.array([[1, 2], [3, 4]], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)
        flat = seq.to_flat()
        # Interleaved: [1, 3, 2, 4]
        assert len(flat) == 4

    def test_token_sequence_to_llm_tokens(self):
        """Test TokenSequence.to_llm_tokens with special tokens."""
        from voice_soundboard.codecs.base import TokenSequence
        tokens = np.array([10, 20, 30], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)
        llm_tokens = seq.to_llm_tokens(add_special_tokens=True)
        assert llm_tokens[0] == 1  # BOS
        assert llm_tokens[-1] == 2  # EOS

    def test_token_sequence_from_llm_tokens(self):
        """Test TokenSequence.from_llm_tokens class method."""
        from voice_soundboard.codecs.base import TokenSequence
        llm_tokens = np.array([1, 13, 23, 33, 2], dtype=np.int64)  # BOS, tokens+offset, EOS
        seq = TokenSequence.from_llm_tokens(llm_tokens, offset=3)
        assert len(seq.tokens) == 3

    def test_token_sequence_to_dict(self):
        """Test TokenSequence.to_dict serialization."""
        from voice_soundboard.codecs.base import TokenSequence
        tokens = np.array([1, 2, 3], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, codec_name="test")
        d = seq.to_dict()
        assert "tokens" in d
        assert d["codec_name"] == "test"

    def test_token_sequence_from_dict(self):
        """Test TokenSequence.from_dict deserialization."""
        from voice_soundboard.codecs.base import TokenSequence
        data = {
            "tokens": [1, 2, 3],
            "num_codebooks": 1,
            "sequence_length": 3,
            "frame_rate_hz": 50.0,
            "source_duration_seconds": 0.06,
            "source_sample_rate": 24000,
            "codec_name": "test",
            "codec_version": "1.0"
        }
        seq = TokenSequence.from_dict(data)
        assert seq.codec_name == "test"


# ============== EncodedAudio Tests ==============

class TestEncodedAudio:
    """Tests for EncodedAudio dataclass."""

    def test_encoded_audio_basic(self):
        """Test EncodedAudio basic creation."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence
        tokens = np.array([1, 2, 3], dtype=np.int64)
        token_seq = TokenSequence(tokens=tokens)
        encoded = EncodedAudio(tokens=token_seq)
        assert encoded.tokens is not None
        assert encoded.estimated_quality == 1.0

    def test_encoded_audio_has_dual_tokens_false(self):
        """Test EncodedAudio.has_dual_tokens when no dual tokens."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence
        tokens = np.array([1, 2, 3], dtype=np.int64)
        token_seq = TokenSequence(tokens=tokens)
        encoded = EncodedAudio(tokens=token_seq)
        assert encoded.has_dual_tokens is False

    def test_encoded_audio_has_dual_tokens_true(self):
        """Test EncodedAudio.has_dual_tokens when dual tokens present."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence
        tokens = np.array([1, 2, 3], dtype=np.int64)
        token_seq = TokenSequence(tokens=tokens)
        semantic = TokenSequence(tokens=np.array([1, 2], dtype=np.int64))
        acoustic = TokenSequence(tokens=np.array([3, 4], dtype=np.int64))
        encoded = EncodedAudio(
            tokens=token_seq,
            semantic_tokens=semantic,
            acoustic_tokens=acoustic
        )
        assert encoded.has_dual_tokens is True

    def test_encoded_audio_to_dict(self):
        """Test EncodedAudio.to_dict serialization."""
        from voice_soundboard.codecs.base import EncodedAudio, TokenSequence
        tokens = np.array([1, 2, 3], dtype=np.int64)
        token_seq = TokenSequence(tokens=tokens)
        encoded = EncodedAudio(tokens=token_seq, estimated_quality=0.9)
        d = encoded.to_dict()
        assert "tokens" in d
        assert d["estimated_quality"] == 0.9


# ============== MockCodec Tests ==============

class TestMockCodec:
    """Tests for MockCodec implementation."""

    def test_mock_codec_name(self):
        """Test MockCodec name property."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        assert codec.name == "mock"

    def test_mock_codec_version(self):
        """Test MockCodec version property."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        assert codec.version == "1.0.0"

    def test_mock_codec_capabilities(self):
        """Test MockCodec capabilities."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        caps = codec.capabilities
        assert caps.can_encode is True
        assert caps.can_decode is True
        assert caps.can_stream is True

    def test_mock_codec_load(self):
        """Test MockCodec.load method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        codec.load()
        assert codec._loaded is True

    def test_mock_codec_encode_array(self):
        """Test MockCodec.encode with numpy array."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)
        encoded = codec.encode(audio, sample_rate=24000)
        assert encoded.tokens is not None
        assert encoded.semantic_tokens is not None
        assert encoded.acoustic_tokens is not None

    def test_mock_codec_decode(self):
        """Test MockCodec.decode method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)
        encoded = codec.encode(audio, sample_rate=24000)
        decoded_audio, sr = codec.decode(encoded)
        assert len(decoded_audio) > 0
        assert sr == 24000

    def test_mock_codec_encode_dual(self):
        """Test MockCodec.encode_dual method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)
        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)
        assert semantic is not None
        assert acoustic is not None

    def test_mock_codec_decode_dual(self):
        """Test MockCodec.decode_dual method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)
        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)
        decoded_audio, sr = codec.decode_dual(semantic, acoustic)
        assert len(decoded_audio) > 0

    def test_mock_codec_deterministic(self):
        """Test MockCodec produces deterministic output."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec()
        audio = np.ones(24000, dtype=np.float32) * 0.5
        encoded1 = codec.encode(audio, sample_rate=24000)
        encoded2 = codec.encode(audio, sample_rate=24000)
        np.testing.assert_array_equal(encoded1.tokens.tokens, encoded2.tokens.tokens)

    def test_mock_codec_get_vocab_size(self):
        """Test MockCodec.get_vocab_size method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec(codebook_size=1024, num_codebooks=8)
        vocab_size = codec.get_vocab_size()
        assert vocab_size == (1024 * 8) + 3  # + special tokens

    def test_mock_codec_estimate_tokens(self):
        """Test MockCodec.estimate_tokens method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)
        tokens = codec.estimate_tokens(duration_seconds=1.0)
        assert tokens == 50 * 8  # 50 frames * 8 codebooks

    def test_mock_codec_estimate_duration(self):
        """Test MockCodec.estimate_duration method."""
        from voice_soundboard.codecs.mock import MockCodec
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)
        duration = codec.estimate_duration(num_tokens=400)
        assert duration == 1.0  # 400 tokens / 8 codebooks / 50 Hz
