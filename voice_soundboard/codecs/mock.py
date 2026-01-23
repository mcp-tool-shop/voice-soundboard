"""
Mock Audio Codec.

A mock implementation for testing without heavy dependencies.
Simulates encoding/decoding behavior with random tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Iterator

import numpy as np

from voice_soundboard.codecs.base import (
    AudioCodec,
    CodecConfig,
    CodecCapabilities,
    CodecType,
    EncodedAudio,
    TokenSequence,
)


class MockCodec(AudioCodec):
    """
    Mock codec for testing.

    Generates deterministic pseudo-random tokens based on audio content.
    Useful for testing LLM integration without loading heavy models.
    """

    def __init__(
        self,
        config: Optional[CodecConfig] = None,
        frame_rate_hz: float = 50.0,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        sample_rate: int = 24000,
    ):
        """
        Initialize mock codec.

        Args:
            config: Codec configuration
            frame_rate_hz: Simulated frame rate
            num_codebooks: Number of codebooks to simulate
            codebook_size: Size of each codebook
            sample_rate: Sample rate for audio
        """
        super().__init__(config)
        self._frame_rate_hz = frame_rate_hz
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._sample_rate = sample_rate

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            can_encode=True,
            can_decode=True,
            can_stream=True,
            codec_type=CodecType.HYBRID,
            has_semantic_tokens=True,
            has_acoustic_tokens=True,
            num_codebooks=self._num_codebooks,
            codebook_size=self._codebook_size,
            frame_rate_hz=self._frame_rate_hz,
            sample_rate=self._sample_rate,
            channels=1,
            supports_variable_bitrate=False,
            min_bitrate_kbps=1.5,
            max_bitrate_kbps=12.0,
        )

    def load(self) -> None:
        """Mock load - nothing to load."""
        self._loaded = True

    def encode(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> EncodedAudio:
        """
        Encode audio to mock tokens.

        Generates deterministic tokens based on audio hash.
        """
        self._ensure_loaded()

        # Handle file path
        if isinstance(audio, (str, Path)):
            # Simulate loading - use path hash
            path_hash = hashlib.md5(str(audio).encode()).hexdigest()
            seed = int(path_hash[:8], 16)
            duration_seconds = 5.0  # Assume 5 seconds for file
            sample_rate = self._sample_rate
        else:
            # Use audio content for seed
            audio = np.asarray(audio)
            content_hash = hashlib.md5(audio.tobytes()[:1000]).hexdigest()
            seed = int(content_hash[:8], 16)
            sample_rate = sample_rate or self._sample_rate
            duration_seconds = len(audio) / sample_rate

        # Calculate sequence length
        seq_len = int(duration_seconds * self._frame_rate_hz)

        # Generate deterministic tokens
        rng = np.random.Generator(np.random.PCG64(seed))
        tokens = rng.integers(
            0, self._codebook_size,
            size=(self._num_codebooks, seq_len),
            dtype=np.int64,
        )

        token_seq = TokenSequence(
            tokens=tokens,
            num_codebooks=self._num_codebooks,
            frame_rate_hz=self._frame_rate_hz,
            source_duration_seconds=duration_seconds,
            source_sample_rate=sample_rate,
            codec_name=self.name,
            codec_version=self.version,
        )

        # Generate semantic/acoustic tokens for dual-codec simulation
        semantic_tokens = TokenSequence(
            tokens=tokens[:2].copy(),  # First 2 codebooks = semantic
            num_codebooks=2,
            frame_rate_hz=self._frame_rate_hz,
            source_duration_seconds=duration_seconds,
            codec_name=self.name,
        )

        acoustic_tokens = TokenSequence(
            tokens=tokens[2:].copy(),  # Rest = acoustic
            num_codebooks=self._num_codebooks - 2,
            frame_rate_hz=self._frame_rate_hz,
            source_duration_seconds=duration_seconds,
            codec_name=self.name,
        )

        return EncodedAudio(
            tokens=token_seq,
            semantic_tokens=semantic_tokens,
            acoustic_tokens=acoustic_tokens,
            estimated_quality=0.95,
        )

    def decode(
        self,
        encoded: Union[EncodedAudio, TokenSequence, np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        """
        Decode mock tokens to audio.

        Generates deterministic audio based on token hash.
        """
        self._ensure_loaded()

        # Extract tokens
        if isinstance(encoded, EncodedAudio):
            tokens = encoded.tokens.tokens
            duration = encoded.tokens.source_duration_seconds
        elif isinstance(encoded, TokenSequence):
            tokens = encoded.tokens
            duration = encoded.source_duration_seconds
        else:
            tokens = np.asarray(encoded)
            duration = len(tokens.flatten()) / (
                self._frame_rate_hz * self._num_codebooks
            )

        # Generate deterministic audio from tokens
        token_hash = hashlib.md5(tokens.tobytes()).hexdigest()
        seed = int(token_hash[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))

        # Generate audio samples
        num_samples = int(duration * self._sample_rate)
        audio = rng.standard_normal(num_samples).astype(np.float32) * 0.1

        # Add some structure based on tokens (simulate tonal content)
        if tokens.ndim > 1:
            first_codebook = tokens[0]
        else:
            first_codebook = tokens

        # Create simple frequency modulation
        samples_per_frame = self._sample_rate // int(self._frame_rate_hz)
        for i, tok in enumerate(first_codebook[:len(audio) // samples_per_frame]):
            freq = 100 + (tok % 500)  # Frequency based on token
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio))
            t = np.arange(end - start) / self._sample_rate
            audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * t)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio, self._sample_rate

    def encode_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TokenSequence]:
        """Stream encoding."""
        for chunk in audio_stream:
            encoded = self.encode(chunk, sample_rate)
            yield encoded.tokens

    def decode_streaming(
        self,
        token_stream: Iterator[TokenSequence],
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Stream decoding."""
        for tokens in token_stream:
            yield self.decode(tokens)

    def encode_dual(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> Tuple[TokenSequence, TokenSequence]:
        """
        Encode to separate semantic and acoustic tokens.

        Args:
            audio: Audio to encode
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (semantic_tokens, acoustic_tokens)
        """
        encoded = self.encode(audio, sample_rate)
        return encoded.semantic_tokens, encoded.acoustic_tokens

    def decode_dual(
        self,
        semantic: TokenSequence,
        acoustic: TokenSequence,
    ) -> Tuple[np.ndarray, int]:
        """
        Decode from separate semantic and acoustic tokens.

        Args:
            semantic: Semantic token sequence
            acoustic: Acoustic token sequence

        Returns:
            Tuple of (audio, sample_rate)
        """
        # Combine tokens
        combined = np.vstack([semantic.tokens, acoustic.tokens])

        token_seq = TokenSequence(
            tokens=combined,
            num_codebooks=semantic.num_codebooks + acoustic.num_codebooks,
            frame_rate_hz=semantic.frame_rate_hz,
            source_duration_seconds=semantic.source_duration_seconds,
        )

        return self.decode(token_seq)
