"""
Tests for Neural Audio Codecs Module (codecs/).

Tests cover:
- CodecType enum
- CodecCapabilities dataclass
- CodecConfig dataclass
- TokenSequence dataclass
- EncodedAudio dataclass
- AudioCodec abstract base class
- MockCodec implementation
- SpecialTokens dataclass
- VocabularyConfig dataclass
- AudioPrompt dataclass
- LLMCodecBridge class
- Utility functions
"""

import pytest
import numpy as np
from pathlib import Path
from dataclasses import fields
from unittest.mock import MagicMock, patch

from voice_soundboard.codecs import (
    AudioCodec,
    CodecConfig,
    CodecCapabilities,
    CodecType,
    EncodedAudio,
    TokenSequence,
    MockCodec,
    LLMCodecBridge,
    AudioPrompt,
    VocabularyConfig,
    SpecialTokens,
    TokenType,
    get_codec_vocabulary_info,
    estimate_audio_context_length,
    estimate_audio_duration,
)


class TestCodecType:
    """Tests for CodecType enum."""

    def test_codec_types_exist(self):
        """Test all codec types are defined."""
        assert CodecType.ACOUSTIC
        assert CodecType.SEMANTIC
        assert CodecType.DUAL
        assert CodecType.HYBRID

    def test_codec_types_unique(self):
        """Test codec types have unique values."""
        types = [CodecType.ACOUSTIC, CodecType.SEMANTIC, CodecType.DUAL, CodecType.HYBRID]
        values = [t.value for t in types]
        assert len(values) == len(set(values))


class TestCodecCapabilities:
    """Tests for CodecCapabilities dataclass."""

    def test_default_values(self):
        """Test CodecCapabilities has sensible defaults."""
        caps = CodecCapabilities()

        assert caps.can_encode is True
        assert caps.can_decode is True
        assert caps.can_stream is False
        assert caps.codec_type == CodecType.ACOUSTIC
        assert caps.num_codebooks == 1
        assert caps.codebook_size == 1024
        assert caps.frame_rate_hz == 50.0
        assert caps.sample_rate == 24000
        assert caps.channels == 1

    def test_custom_values(self):
        """Test CodecCapabilities with custom values."""
        caps = CodecCapabilities(
            can_stream=True,
            codec_type=CodecType.HYBRID,
            num_codebooks=8,
            codebook_size=2048,
            frame_rate_hz=12.5,
            sample_rate=44100,
        )

        assert caps.can_stream is True
        assert caps.codec_type == CodecType.HYBRID
        assert caps.num_codebooks == 8
        assert caps.codebook_size == 2048


class TestCodecConfig:
    """Tests for CodecConfig dataclass."""

    def test_default_values(self):
        """Test CodecConfig has sensible defaults."""
        config = CodecConfig()

        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.target_bandwidth_kbps == 6.0
        assert config.chunk_size_ms == 80
        assert config.add_special_tokens is True
        assert config.pad_token_id == 0
        assert config.bos_token_id == 1
        assert config.eos_token_id == 2

    def test_custom_config(self):
        """Test CodecConfig with custom values."""
        config = CodecConfig(
            device="cuda",
            target_bandwidth_kbps=12.0,
            num_codebooks=4,
        )

        assert config.device == "cuda"
        assert config.target_bandwidth_kbps == 12.0
        assert config.num_codebooks == 4


class TestTokenSequence:
    """Tests for TokenSequence dataclass."""

    def test_1d_tokens(self):
        """Test TokenSequence with 1D tokens."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        assert seq.sequence_length == 5
        assert seq.num_codebooks == 1

    def test_2d_tokens(self):
        """Test TokenSequence with 2D tokens (multi-codebook)."""
        tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        assert seq.sequence_length == 3
        assert seq.num_codebooks == 2

    def test_duration_calculation(self):
        """Test duration is calculated from frame rate."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        seq = TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        assert seq.duration_seconds == 0.1  # 5 frames / 50 Hz

    def test_to_flat(self):
        """Test flattening multi-codebook tokens."""
        tokens = np.array([[1, 2], [3, 4]], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        flat = seq.to_flat()
        # Interleaved: [1, 3, 2, 4]
        assert len(flat) == 4

    def test_to_llm_tokens_with_special(self):
        """Test converting to LLM tokens with special tokens."""
        tokens = np.array([10, 20, 30], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        llm_tokens = seq.to_llm_tokens(
            add_special_tokens=True,
            bos_token_id=1,
            eos_token_id=2,
            offset=3,
        )

        assert llm_tokens[0] == 1  # BOS
        assert llm_tokens[-1] == 2  # EOS
        # Middle tokens should be offset by 3
        assert llm_tokens[1] == 10 + 3

    def test_to_llm_tokens_without_special(self):
        """Test converting to LLM tokens without special tokens."""
        tokens = np.array([10, 20, 30], dtype=np.int64)
        seq = TokenSequence(tokens=tokens)

        llm_tokens = seq.to_llm_tokens(add_special_tokens=False, offset=0)

        assert len(llm_tokens) == 3
        np.testing.assert_array_equal(llm_tokens, tokens)

    def test_from_llm_tokens(self):
        """Test creating from LLM tokens."""
        llm_tokens = np.array([1, 13, 23, 33, 2], dtype=np.int64)  # BOS, data, EOS

        seq = TokenSequence.from_llm_tokens(
            llm_tokens,
            num_codebooks=1,
            frame_rate_hz=50.0,
            has_special_tokens=True,
            offset=3,
        )

        assert seq.sequence_length == 3
        # Tokens should have offset removed: [10, 20, 30]
        np.testing.assert_array_equal(seq.tokens, [10, 20, 30])

    def test_to_dict_and_back(self):
        """Test serialization to dict and back."""
        tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        seq = TokenSequence(
            tokens=tokens,
            frame_rate_hz=50.0,
            codec_name="test",
            codec_version="1.0",
        )

        data = seq.to_dict()
        restored = TokenSequence.from_dict(data)

        assert restored.num_codebooks == seq.num_codebooks
        assert restored.sequence_length == seq.sequence_length
        assert restored.codec_name == "test"


class TestEncodedAudio:
    """Tests for EncodedAudio dataclass."""

    def test_basic_encoded_audio(self):
        """Test basic EncodedAudio creation."""
        tokens = TokenSequence(tokens=np.array([1, 2, 3]))
        encoded = EncodedAudio(tokens=tokens)

        assert encoded.tokens is not None
        assert encoded.semantic_tokens is None
        assert encoded.acoustic_tokens is None
        assert encoded.estimated_quality == 1.0

    def test_has_dual_tokens(self):
        """Test has_dual_tokens property."""
        tokens = TokenSequence(tokens=np.array([1, 2, 3]))
        semantic = TokenSequence(tokens=np.array([1, 2]))
        acoustic = TokenSequence(tokens=np.array([3, 4]))

        # Without dual
        encoded1 = EncodedAudio(tokens=tokens)
        assert encoded1.has_dual_tokens is False

        # With dual
        encoded2 = EncodedAudio(
            tokens=tokens,
            semantic_tokens=semantic,
            acoustic_tokens=acoustic,
        )
        assert encoded2.has_dual_tokens is True

    def test_to_dict_and_back(self):
        """Test serialization."""
        tokens = TokenSequence(tokens=np.array([1, 2, 3]))
        encoded = EncodedAudio(tokens=tokens, estimated_quality=0.95)

        data = encoded.to_dict()
        restored = EncodedAudio.from_dict(data)

        assert restored.estimated_quality == 0.95


class TestMockCodec:
    """Tests for MockCodec implementation."""

    def test_init_defaults(self):
        """Test MockCodec initialization."""
        codec = MockCodec()

        assert codec.name == "mock"
        assert codec.version == "1.0.0"
        assert codec._loaded is False

    def test_init_custom(self):
        """Test MockCodec with custom parameters."""
        codec = MockCodec(
            frame_rate_hz=12.5,
            num_codebooks=4,
            codebook_size=2048,
        )

        assert codec._frame_rate_hz == 12.5
        assert codec._num_codebooks == 4
        assert codec._codebook_size == 2048

    def test_capabilities(self):
        """Test MockCodec capabilities."""
        codec = MockCodec()
        caps = codec.capabilities

        assert caps.can_encode is True
        assert caps.can_decode is True
        assert caps.can_stream is True
        assert caps.codec_type == CodecType.HYBRID
        assert caps.has_semantic_tokens is True
        assert caps.has_acoustic_tokens is True

    def test_load(self):
        """Test loading mock codec."""
        codec = MockCodec()
        codec.load()

        assert codec._loaded is True

    def test_encode_array(self):
        """Test encoding numpy array."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)

        assert isinstance(encoded, EncodedAudio)
        assert encoded.tokens is not None
        assert encoded.semantic_tokens is not None
        assert encoded.acoustic_tokens is not None

    def test_encode_deterministic(self):
        """Test encoding is deterministic for same input."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded1 = codec.encode(audio, sample_rate=24000)
        encoded2 = codec.encode(audio, sample_rate=24000)

        np.testing.assert_array_equal(
            encoded1.tokens.tokens,
            encoded2.tokens.tokens,
        )

    def test_decode(self):
        """Test decoding tokens to audio."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)
        decoded, sr = codec.decode(encoded)

        assert isinstance(decoded, np.ndarray)
        assert sr == 24000
        assert len(decoded) > 0

    def test_encode_decode_roundtrip(self):
        """Test encode-decode produces audio."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)
        decoded, sr = codec.decode(encoded)

        assert sr == 24000
        # Decoded should have similar duration
        expected_samples = int(encoded.tokens.source_duration_seconds * sr)
        assert abs(len(decoded) - expected_samples) < sr  # Within 1 second

    def test_encode_dual(self):
        """Test dual encoding."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)

        assert semantic.num_codebooks == 2
        assert acoustic.num_codebooks == 6  # 8 - 2

    def test_decode_dual(self):
        """Test dual decoding."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)
        decoded, sr = codec.decode_dual(semantic, acoustic)

        assert sr == 24000
        assert len(decoded) > 0


class TestSpecialTokens:
    """Tests for SpecialTokens dataclass."""

    def test_default_tokens(self):
        """Test default special tokens."""
        tokens = SpecialTokens()

        assert tokens.pad_token == "<pad>"
        assert tokens.bos_token == "<bos>"
        assert tokens.eos_token == "<eos>"
        assert tokens.audio_start == "<|audio|>"
        assert tokens.audio_end == "<|/audio|>"

    def test_get_speaker_token(self):
        """Test speaker token generation."""
        tokens = SpecialTokens()

        speaker_tok = tokens.get_speaker_token("alice")
        assert "alice" in speaker_tok
        assert speaker_tok == "<|speaker_alice|>"

    def test_get_emotion_token(self):
        """Test emotion token generation."""
        tokens = SpecialTokens()

        emotion_tok = tokens.get_emotion_token("happy")
        assert "happy" in emotion_tok
        assert emotion_tok == "<|emotion_happy|>"


class TestVocabularyConfig:
    """Tests for VocabularyConfig dataclass."""

    def test_default_values(self):
        """Test default vocabulary config."""
        config = VocabularyConfig()

        assert config.text_vocab_size == 32000
        assert config.audio_vocab_size == 8192
        assert config.num_special_tokens == 256

    def test_total_vocab_size(self):
        """Test total vocabulary size calculation."""
        config = VocabularyConfig(
            text_vocab_size=32000,
            audio_vocab_size=8192,
            num_special_tokens=256,
        )

        expected = 256 + 32000 + 8192
        assert config.total_vocab_size == expected

    def test_text_to_global(self):
        """Test text to global ID conversion."""
        config = VocabularyConfig(num_special_tokens=256)

        global_id = config.text_to_global(100)
        assert global_id == 100 + 256

    def test_audio_to_global(self):
        """Test audio to global ID conversion."""
        config = VocabularyConfig(audio_token_offset=32000)

        global_id = config.audio_to_global(100)
        assert global_id == 100 + 32000

    def test_global_to_text(self):
        """Test global to text ID conversion."""
        config = VocabularyConfig(
            num_special_tokens=256,
            text_vocab_size=32000,
        )

        # Valid text token
        text_id = config.global_to_text(256 + 100)
        assert text_id == 100

        # Invalid (special token)
        assert config.global_to_text(0) is None

    def test_global_to_audio(self):
        """Test global to audio ID conversion."""
        config = VocabularyConfig(
            audio_token_offset=32000,
            audio_vocab_size=8192,
        )

        # Valid audio token
        audio_id = config.global_to_audio(32000 + 100)
        assert audio_id == 100

        # Invalid (text token)
        assert config.global_to_audio(1000) is None

    def test_get_token_type(self):
        """Test token type detection."""
        config = VocabularyConfig(
            num_special_tokens=256,
            text_vocab_size=32000,
            audio_token_offset=32256,  # 256 + 32000
        )

        assert config.get_token_type(0) == TokenType.SPECIAL
        assert config.get_token_type(1000) == TokenType.TEXT
        assert config.get_token_type(35000) == TokenType.AUDIO


class TestAudioPrompt:
    """Tests for AudioPrompt dataclass."""

    def test_basic_prompt(self):
        """Test basic audio prompt."""
        prompt = AudioPrompt(text_prefix="Hello")

        assert prompt.text_prefix == "Hello"
        assert prompt.audio_tokens is None

    def test_prompt_with_audio(self):
        """Test prompt with audio tokens."""
        tokens = TokenSequence(tokens=np.array([1, 2, 3]))
        prompt = AudioPrompt(
            text_prefix="Say: ",
            audio_tokens=tokens,
            text_suffix=" End.",
        )

        assert prompt.audio_tokens is not None
        assert prompt.text_suffix == " End."

    def test_prompt_with_speaker(self):
        """Test prompt with speaker ID."""
        prompt = AudioPrompt(
            text_prefix="Hello",
            speaker_id="alice",
        )

        assert prompt.speaker_id == "alice"

    def test_to_token_sequence(self):
        """Test converting prompt to tokens."""
        prompt = AudioPrompt(text_prefix="Hello")

        tokens = prompt.to_token_sequence()

        assert isinstance(tokens, np.ndarray)
        assert tokens[0] == 1  # BOS
        assert tokens[-1] == 2  # EOS


class TestLLMCodecBridge:
    """Tests for LLMCodecBridge class."""

    @pytest.fixture
    def bridge(self):
        """Create bridge with mock codec."""
        codec = MockCodec()
        return LLMCodecBridge(codec)

    def test_init(self, bridge):
        """Test bridge initialization."""
        assert bridge.codec is not None
        assert bridge.vocab_config is not None
        assert bridge.special_tokens is not None

    def test_encode_for_llm(self, bridge):
        """Test encoding audio for LLM."""
        audio = np.random.randn(24000).astype(np.float32)

        tokens = bridge.encode_for_llm(audio, sample_rate=24000)

        assert isinstance(tokens, np.ndarray)
        assert len(tokens) > 0

    def test_encode_with_speaker(self, bridge):
        """Test encoding with speaker ID."""
        audio = np.random.randn(24000).astype(np.float32)

        tokens = bridge.encode_for_llm(
            audio,
            sample_rate=24000,
            speaker_id="alice",
        )

        assert len(tokens) > 0

    def test_decode_from_llm(self, bridge):
        """Test decoding LLM output to audio."""
        audio = np.random.randn(24000).astype(np.float32)

        # Encode first
        encoded_tokens = bridge.encode_for_llm(audio, sample_rate=24000)

        # Then decode
        decoded, sr = bridge.decode_from_llm(encoded_tokens)

        assert isinstance(decoded, np.ndarray)
        assert sr > 0

    def test_decode_empty_returns_silence(self, bridge):
        """Test decoding empty tokens returns silence."""
        tokens = np.array([], dtype=np.int64)

        audio, sr = bridge.decode_from_llm(tokens)

        assert sr == 24000
        # Should return some silence

    def test_create_prompt(self, bridge):
        """Test creating audio prompt."""
        audio = np.random.randn(24000).astype(np.float32)

        prompt = bridge.create_prompt(
            text="Hello",
            audio=audio,
            sample_rate=24000,
            speaker_id="alice",
        )

        assert isinstance(prompt, AudioPrompt)
        assert prompt.text_prefix == "Hello"
        assert prompt.speaker_id == "alice"
        assert prompt.audio_tokens is not None


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_codec_vocabulary_info(self):
        """Test getting codec vocabulary info."""
        codec = MockCodec()

        info = get_codec_vocabulary_info(codec)

        assert info["codec_name"] == "mock"
        assert info["codec_version"] == "1.0.0"
        assert "num_codebooks" in info
        assert "codebook_size" in info
        assert "tokens_per_second" in info

    def test_estimate_audio_context_length(self):
        """Test estimating context length for audio."""
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)

        # 1 second of audio
        length = estimate_audio_context_length(1.0, codec)

        # 50 frames * 8 codebooks + 2 markers
        expected = 50 * 8 + 2
        assert length == expected

    def test_estimate_audio_context_without_markers(self):
        """Test estimating context length without markers."""
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)

        length = estimate_audio_context_length(
            1.0, codec, include_markers=False
        )

        expected = 50 * 8
        assert length == expected

    def test_estimate_audio_duration(self):
        """Test estimating duration from context length."""
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)

        # 402 tokens = 50 frames * 8 codebooks + 2 markers
        duration = estimate_audio_duration(402, codec)

        # Should be close to 1 second
        assert abs(duration - 1.0) < 0.1

    def test_estimate_duration_without_markers(self):
        """Test estimating duration without markers."""
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)

        duration = estimate_audio_duration(
            400, codec, include_markers=False
        )

        assert abs(duration - 1.0) < 0.1


class TestCodecAbstractMethods:
    """Tests for AudioCodec abstract base class behavior."""

    def test_ensure_loaded(self):
        """Test _ensure_loaded triggers load."""
        codec = MockCodec()
        assert codec._loaded is False

        codec._ensure_loaded()

        assert codec._loaded is True

    def test_get_vocab_size(self):
        """Test vocabulary size calculation."""
        codec = MockCodec(codebook_size=1024, num_codebooks=8)

        vocab_size = codec.get_vocab_size()

        # 1024 * 8 + 3 (special tokens)
        assert vocab_size == 1024 * 8 + 3

    def test_estimate_tokens(self):
        """Test token count estimation."""
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)

        tokens = codec.estimate_tokens(2.0)  # 2 seconds

        # 50 Hz * 2s * 8 codebooks
        assert tokens == 50 * 2 * 8

    def test_estimate_duration_from_tokens(self):
        """Test duration estimation from token count."""
        codec = MockCodec(frame_rate_hz=50.0, num_codebooks=8)

        duration = codec.estimate_duration(800)  # 800 tokens

        # 800 / 8 = 100 frames, 100 / 50 = 2 seconds
        assert duration == 2.0

    def test_to_llm_tokens(self):
        """Test full audio to LLM tokens conversion."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        tokens = codec.to_llm_tokens(audio, sample_rate=24000)

        assert isinstance(tokens, np.ndarray)
        assert len(tokens) > 0

    def test_from_llm_tokens(self):
        """Test LLM tokens to audio conversion."""
        codec = MockCodec()
        audio = np.random.randn(24000).astype(np.float32)

        # Encode to LLM tokens
        llm_tokens = codec.to_llm_tokens(audio, sample_rate=24000)

        # Decode back
        decoded, sr = codec.from_llm_tokens(llm_tokens)

        assert sr == 24000
        assert len(decoded) > 0


class TestStreamingEncodeDecode:
    """Tests for streaming encode/decode."""

    def test_encode_streaming(self):
        """Test streaming encoding."""
        codec = MockCodec()

        def audio_generator():
            for _ in range(3):
                yield np.random.randn(4800).astype(np.float32)

        chunks = list(codec.encode_streaming(audio_generator(), sample_rate=24000))

        assert len(chunks) == 3
        assert all(isinstance(c, TokenSequence) for c in chunks)

    def test_decode_streaming(self):
        """Test streaming decoding."""
        codec = MockCodec()

        # Create some token sequences
        def token_generator():
            for _ in range(3):
                tokens = np.random.randint(0, 1024, size=(8, 10), dtype=np.int64)
                yield TokenSequence(tokens=tokens, frame_rate_hz=50.0)

        chunks = list(codec.decode_streaming(token_generator()))

        assert len(chunks) == 3
        assert all(isinstance(c[0], np.ndarray) for c in chunks)


class TestMimiCodec:
    """Tests for MimiCodec wrapper."""

    def test_init(self):
        """Test MimiCodec initialization."""
        from voice_soundboard.codecs import MimiCodec, MIMI_AVAILABLE

        codec = MimiCodec()

        assert codec.name == "mimi"
        # Version depends on whether moshi is installed
        assert codec.version in ["mock", "1.0.0"] or codec.version.startswith("0.")

    def test_specifications(self):
        """Test Mimi codec specifications."""
        from voice_soundboard.codecs import MimiCodec

        codec = MimiCodec()
        caps = codec.capabilities

        # Mimi's known specifications
        assert caps.frame_rate_hz == 12.5
        assert caps.num_codebooks == 8
        assert caps.codebook_size == 2048
        assert caps.sample_rate == 24000
        assert caps.can_stream is True

    def test_encode(self):
        """Test Mimi encoding."""
        from voice_soundboard.codecs import MimiCodec

        codec = MimiCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)

        assert isinstance(encoded, EncodedAudio)
        assert encoded.tokens.tokens.shape[0] == 8  # 8 codebooks
        # Frame count should match 12.5 Hz
        expected_frames = int(1.0 * 12.5)
        assert abs(encoded.tokens.tokens.shape[1] - expected_frames) <= 1

    def test_decode(self):
        """Test Mimi decoding."""
        from voice_soundboard.codecs import MimiCodec

        codec = MimiCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)
        decoded, sr = codec.decode(encoded)

        assert sr == 24000
        assert isinstance(decoded, np.ndarray)
        assert len(decoded) > 0

    def test_semantic_acoustic_split(self):
        """Test semantic/acoustic token separation in Mimi."""
        from voice_soundboard.codecs import MimiCodec

        codec = MimiCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)

        # First codebook is semantic
        assert encoded.semantic_tokens is not None
        assert encoded.semantic_tokens.num_codebooks == 1

        # Remaining 7 codebooks are acoustic
        assert encoded.acoustic_tokens is not None
        assert encoded.acoustic_tokens.num_codebooks == 7

    def test_streaming_encode(self):
        """Test Mimi streaming encode."""
        from voice_soundboard.codecs import MimiCodec

        codec = MimiCodec()

        def audio_generator():
            for _ in range(5):
                yield np.random.randn(4800).astype(np.float32)  # 200ms chunks

        chunks = list(codec.encode_streaming(audio_generator(), sample_rate=24000))

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, TokenSequence)

    def test_low_token_rate(self):
        """Test that Mimi has low token rate (12.5 Hz)."""
        from voice_soundboard.codecs import MimiCodec

        codec = MimiCodec()
        # 2 seconds of audio
        audio = np.random.randn(48000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)

        # Should have ~25 frames for 2 seconds at 12.5 Hz
        expected_frames = int(2.0 * 12.5)
        actual_frames = encoded.tokens.tokens.shape[1]
        assert abs(actual_frames - expected_frames) <= 1


class TestDualCodec:
    """Tests for DualCodec wrapper."""

    def test_init(self):
        """Test DualCodec initialization."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()

        assert codec.name == "dualcodec"
        # Version depends on whether dualcodec lib is installed
        assert codec.version in ["mock", "1.0.0"] or codec.version.startswith("0.")

    def test_capabilities(self):
        """Test DualCodec capabilities."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()
        caps = codec.capabilities

        assert caps.has_semantic_tokens is True
        assert caps.has_acoustic_tokens is True
        assert caps.can_stream is True
        assert caps.num_codebooks == 8  # 1 semantic + 7 acoustic

    def test_encode(self):
        """Test DualCodec encoding."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)

        assert isinstance(encoded, EncodedAudio)
        assert encoded.tokens is not None
        assert encoded.semantic_tokens is not None
        assert encoded.acoustic_tokens is not None

    def test_decode(self):
        """Test DualCodec decoding."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)

        encoded = codec.encode(audio, sample_rate=24000)
        decoded, sr = codec.decode(encoded)

        assert sr == codec.capabilities.sample_rate
        assert isinstance(decoded, np.ndarray)
        assert len(decoded) > 0

    def test_encode_dual(self):
        """Test dual-stream encoding."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)

        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)

        assert semantic.num_codebooks == 1
        assert acoustic.num_codebooks == 7

    def test_decode_dual(self):
        """Test dual-stream decoding."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()
        audio = np.random.randn(24000).astype(np.float32)

        semantic, acoustic = codec.encode_dual(audio, sample_rate=24000)
        decoded, sr = codec.decode_dual(semantic, acoustic)

        assert isinstance(decoded, np.ndarray)
        assert len(decoded) > 0

    def test_voice_convert(self):
        """Test voice conversion."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()

        # Content audio (what we want to say)
        content = np.random.randn(24000).astype(np.float32)

        # Style audio (how we want to sound)
        style = np.random.randn(12000).astype(np.float32)

        converted, sr = codec.voice_convert(
            content_audio=content,
            style_audio=style,
            content_sample_rate=24000,
            style_sample_rate=24000,
        )

        assert sr == codec.capabilities.sample_rate
        assert isinstance(converted, np.ndarray)
        # Output length should be similar to content length
        content_duration = len(content) / 24000
        converted_duration = len(converted) / sr
        assert abs(content_duration - converted_duration) < 0.3

    def test_voice_convert_different_lengths(self):
        """Test voice conversion with different length inputs."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()

        # Long content
        content = np.random.randn(48000).astype(np.float32)  # 2 seconds

        # Short style reference
        style = np.random.randn(6000).astype(np.float32)  # 0.25 seconds

        converted, sr = codec.voice_convert(
            content_audio=content,
            style_audio=style,
            content_sample_rate=24000,
            style_sample_rate=24000,
        )

        assert len(converted) > 0

    def test_streaming_encode(self):
        """Test DualCodec streaming encode."""
        from voice_soundboard.codecs import DualCodec

        codec = DualCodec()

        def audio_generator():
            for _ in range(3):
                yield np.random.randn(4800).astype(np.float32)

        chunks = list(codec.encode_streaming(audio_generator(), sample_rate=24000))

        assert len(chunks) > 0


class TestCodecIntegration:
    """Integration tests for codec system."""

    def test_all_codecs_load(self):
        """Test all codec implementations load successfully."""
        from voice_soundboard.codecs import MockCodec, MimiCodec, DualCodec

        codecs = [MockCodec(), MimiCodec(), DualCodec()]

        for codec in codecs:
            codec.load()
            assert codec._loaded, f"{codec.name} should be loaded"

    def test_codec_roundtrip_all(self):
        """Test encode-decode roundtrip for all codecs."""
        from voice_soundboard.codecs import MockCodec, MimiCodec, DualCodec

        audio = np.random.randn(24000).astype(np.float32)

        for codec in [MockCodec(), MimiCodec(), DualCodec()]:
            encoded = codec.encode(audio, sample_rate=24000)
            decoded, sr = codec.decode(encoded)

            assert sr > 0, f"{codec.name} should return valid sample rate"
            assert len(decoded) > 0, f"{codec.name} should produce audio"

    def test_llm_bridge_all_codecs(self):
        """Test LLMCodecBridge with all codecs."""
        from voice_soundboard.codecs import MockCodec, MimiCodec, DualCodec

        audio = np.random.randn(24000).astype(np.float32)

        for codec in [MockCodec(), MimiCodec(), DualCodec()]:
            bridge = LLMCodecBridge(codec)

            # Encode for LLM
            llm_tokens = bridge.encode_for_llm(audio, sample_rate=24000)
            assert len(llm_tokens) > 0, f"{codec.name} should produce LLM tokens"

            # Decode back
            decoded, sr = bridge.decode_from_llm(llm_tokens)
            assert sr > 0, f"{codec.name} bridge should return valid sample rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
