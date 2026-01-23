"""
Audio Codec Base Classes.

Abstract interface for neural audio codecs that convert
audio to/from discrete tokens for LLM integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Iterator, Tuple
from enum import Enum, auto

import numpy as np


class CodecType(Enum):
    """Types of audio codecs."""

    ACOUSTIC = auto()      # Pure acoustic tokens (e.g., EnCodec)
    SEMANTIC = auto()      # Semantic/content tokens (e.g., HuBERT)
    DUAL = auto()          # Both semantic and acoustic (e.g., DualCodec)
    HYBRID = auto()        # Combined approach (e.g., Mimi)


@dataclass
class CodecCapabilities:
    """Describes what a codec can do."""

    # Core capabilities
    can_encode: bool = True
    can_decode: bool = True
    can_stream: bool = False

    # Token types
    codec_type: CodecType = CodecType.ACOUSTIC
    has_semantic_tokens: bool = False
    has_acoustic_tokens: bool = True

    # Quantization
    num_codebooks: int = 1
    codebook_size: int = 1024

    # Frame rate
    frame_rate_hz: float = 50.0  # Tokens per second

    # Audio specs
    sample_rate: int = 24000
    channels: int = 1

    # Quality tiers
    supports_variable_bitrate: bool = False
    min_bitrate_kbps: float = 1.5
    max_bitrate_kbps: float = 24.0


@dataclass
class CodecConfig:
    """Configuration for audio codec."""

    # Model settings
    device: str = "cpu"
    dtype: str = "float32"

    # Quality settings
    target_bandwidth_kbps: float = 6.0

    # Streaming settings
    chunk_size_ms: int = 80  # Encode/decode chunk size
    overlap_ms: int = 10     # Overlap for smooth transitions

    # Quantization
    num_codebooks: Optional[int] = None  # None = use codec default

    # LLM integration
    add_special_tokens: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TokenSequence:
    """
    A sequence of audio tokens.

    Can represent single or multi-codebook tokens.
    """

    # Token data
    tokens: np.ndarray  # Shape: (num_codebooks, seq_len) or (seq_len,)

    # Metadata
    num_codebooks: int = 1
    sequence_length: int = 0
    frame_rate_hz: float = 50.0

    # Source info
    source_duration_seconds: float = 0.0
    source_sample_rate: int = 24000

    # Codec info
    codec_name: str = ""
    codec_version: str = ""

    def __post_init__(self):
        """Compute derived fields."""
        if self.tokens.ndim == 1:
            self.sequence_length = len(self.tokens)
            self.num_codebooks = 1
        else:
            self.num_codebooks, self.sequence_length = self.tokens.shape

        if self.source_duration_seconds == 0 and self.sequence_length > 0:
            self.source_duration_seconds = self.sequence_length / self.frame_rate_hz

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds based on frame rate."""
        return self.sequence_length / self.frame_rate_hz

    def to_flat(self) -> np.ndarray:
        """Flatten multi-codebook tokens to single sequence."""
        if self.tokens.ndim == 1:
            return self.tokens
        # Interleave codebooks
        return self.tokens.T.flatten()

    def to_llm_tokens(
        self,
        add_special_tokens: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        offset: int = 3,  # Offset for special tokens
    ) -> np.ndarray:
        """
        Convert to LLM-compatible token sequence.

        Args:
            add_special_tokens: Add BOS/EOS tokens
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            offset: Offset to add to audio tokens (to avoid collision with special tokens)

        Returns:
            Token sequence ready for LLM input
        """
        flat = self.to_flat() + offset

        if add_special_tokens:
            return np.concatenate([
                [bos_token_id],
                flat,
                [eos_token_id],
            ])
        return flat

    @classmethod
    def from_llm_tokens(
        cls,
        tokens: np.ndarray,
        num_codebooks: int = 1,
        frame_rate_hz: float = 50.0,
        has_special_tokens: bool = True,
        offset: int = 3,
    ) -> "TokenSequence":
        """
        Create from LLM-generated tokens.

        Args:
            tokens: LLM output tokens
            num_codebooks: Number of codebooks used
            frame_rate_hz: Frame rate of the codec
            has_special_tokens: Whether tokens include BOS/EOS
            offset: Offset that was added to audio tokens

        Returns:
            TokenSequence ready for decoding
        """
        if has_special_tokens:
            # Remove BOS/EOS
            tokens = tokens[1:-1]

        # Remove offset
        tokens = tokens - offset
        tokens = np.maximum(tokens, 0)  # Clamp to valid range

        if num_codebooks > 1:
            # De-interleave
            seq_len = len(tokens) // num_codebooks
            tokens = tokens[:seq_len * num_codebooks].reshape(seq_len, num_codebooks).T

        return cls(
            tokens=tokens.astype(np.int64),
            num_codebooks=num_codebooks,
            frame_rate_hz=frame_rate_hz,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tokens": self.tokens.tolist(),
            "num_codebooks": self.num_codebooks,
            "sequence_length": self.sequence_length,
            "frame_rate_hz": self.frame_rate_hz,
            "source_duration_seconds": self.source_duration_seconds,
            "source_sample_rate": self.source_sample_rate,
            "codec_name": self.codec_name,
            "codec_version": self.codec_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenSequence":
        """Deserialize from dictionary."""
        data = data.copy()
        data["tokens"] = np.array(data["tokens"], dtype=np.int64)
        return cls(**data)


@dataclass
class EncodedAudio:
    """
    Encoded audio with optional semantic/acoustic separation.

    For codecs that support dual-stream encoding.
    """

    # Primary token sequence
    tokens: TokenSequence

    # Optional semantic tokens (for DualCodec-style)
    semantic_tokens: Optional[TokenSequence] = None

    # Optional acoustic tokens (for DualCodec-style)
    acoustic_tokens: Optional[TokenSequence] = None

    # Latent representation (for some codecs)
    latents: Optional[np.ndarray] = None

    # Quality metrics
    estimated_quality: float = 1.0  # 0-1

    @property
    def has_dual_tokens(self) -> bool:
        """Check if both semantic and acoustic tokens available."""
        return self.semantic_tokens is not None and self.acoustic_tokens is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "tokens": self.tokens.to_dict(),
            "estimated_quality": self.estimated_quality,
        }
        if self.semantic_tokens:
            result["semantic_tokens"] = self.semantic_tokens.to_dict()
        if self.acoustic_tokens:
            result["acoustic_tokens"] = self.acoustic_tokens.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncodedAudio":
        """Deserialize from dictionary."""
        return cls(
            tokens=TokenSequence.from_dict(data["tokens"]),
            semantic_tokens=TokenSequence.from_dict(data["semantic_tokens"])
            if data.get("semantic_tokens") else None,
            acoustic_tokens=TokenSequence.from_dict(data["acoustic_tokens"])
            if data.get("acoustic_tokens") else None,
            estimated_quality=data.get("estimated_quality", 1.0),
        )


class AudioCodec(ABC):
    """
    Abstract base class for neural audio codecs.

    Provides interface for encoding audio to tokens and decoding
    tokens back to audio, suitable for LLM integration.
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        """
        Initialize the codec.

        Args:
            config: Codec configuration
        """
        self.config = config or CodecConfig()
        self._model = None
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Codec name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Codec version."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> CodecCapabilities:
        """Codec capabilities."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the codec model."""
        pass

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def encode(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> EncodedAudio:
        """
        Encode audio to tokens.

        Args:
            audio: Audio array or path to audio file
            sample_rate: Sample rate (if audio is array)

        Returns:
            EncodedAudio with token sequences
        """
        pass

    @abstractmethod
    def decode(
        self,
        encoded: Union[EncodedAudio, TokenSequence, np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        """
        Decode tokens to audio.

        Args:
            encoded: Encoded audio, token sequence, or raw tokens

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass

    def encode_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TokenSequence]:
        """
        Encode audio stream to token stream.

        Args:
            audio_stream: Iterator of audio chunks
            sample_rate: Sample rate of audio

        Yields:
            Token sequences for each chunk
        """
        if not self.capabilities.can_stream:
            raise NotImplementedError(f"{self.name} does not support streaming")

        # Default implementation: buffer and encode
        for chunk in audio_stream:
            encoded = self.encode(chunk, sample_rate)
            yield encoded.tokens

    def decode_streaming(
        self,
        token_stream: Iterator[TokenSequence],
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Decode token stream to audio stream.

        Args:
            token_stream: Iterator of token sequences

        Yields:
            Tuple of (audio_chunk, sample_rate)
        """
        if not self.capabilities.can_stream:
            raise NotImplementedError(f"{self.name} does not support streaming")

        # Default implementation: decode each sequence
        for tokens in token_stream:
            yield self.decode(tokens)

    def to_llm_tokens(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert audio to LLM-ready tokens.

        Args:
            audio: Audio array or path
            sample_rate: Sample rate if audio is array

        Returns:
            Token array suitable for LLM input
        """
        encoded = self.encode(audio, sample_rate)
        return encoded.tokens.to_llm_tokens(
            add_special_tokens=self.config.add_special_tokens,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
        )

    def from_llm_tokens(
        self,
        tokens: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert LLM-generated tokens to audio.

        Args:
            tokens: LLM output tokens

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        token_seq = TokenSequence.from_llm_tokens(
            tokens,
            num_codebooks=self.capabilities.num_codebooks,
            frame_rate_hz=self.capabilities.frame_rate_hz,
            has_special_tokens=self.config.add_special_tokens,
        )
        return self.decode(token_seq)

    def get_vocab_size(self) -> int:
        """
        Get total vocabulary size for LLM.

        Returns:
            Vocabulary size including special tokens
        """
        # Base vocab = codebook_size * num_codebooks
        # Plus special tokens (pad, bos, eos)
        caps = self.capabilities
        base_vocab = caps.codebook_size * caps.num_codebooks
        return base_vocab + 3  # +3 for special tokens

    def estimate_tokens(self, duration_seconds: float) -> int:
        """
        Estimate number of tokens for audio duration.

        Args:
            duration_seconds: Audio duration

        Returns:
            Estimated token count
        """
        caps = self.capabilities
        frames = int(duration_seconds * caps.frame_rate_hz)
        return frames * caps.num_codebooks

    def estimate_duration(self, num_tokens: int) -> float:
        """
        Estimate audio duration from token count.

        Args:
            num_tokens: Number of tokens

        Returns:
            Estimated duration in seconds
        """
        caps = self.capabilities
        frames = num_tokens // caps.num_codebooks
        return frames / caps.frame_rate_hz
