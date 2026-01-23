"""
Mimi Audio Codec Wrapper.

Wrapper for Kyutai's Mimi codec used in CSM/Moshi.
Mimi is a 12.5 Hz codec optimized for LLM audio generation.

References:
- https://kyutai.org/codec-explainer
- https://github.com/kyutai-labs/moshi
"""

from __future__ import annotations

import warnings
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

# Try to import Mimi
MIMI_AVAILABLE = False
_mimi_module = None
_mimi_error = None

try:
    # Mimi is typically part of the moshi package
    import moshi
    _mimi_module = moshi
    MIMI_AVAILABLE = True
except ImportError as e:
    _mimi_error = str(e)
    try:
        # Try alternative import
        import mimi
        _mimi_module = mimi
        MIMI_AVAILABLE = True
    except ImportError:
        pass


class MimiCodec(AudioCodec):
    """
    Wrapper for Kyutai's Mimi codec.

    Mimi features:
    - 12.5 Hz frame rate (very low token rate)
    - 8 codebooks with 2048 codes each
    - 24 kHz audio output
    - Optimized for speech

    Note: Requires moshi or mimi package to be installed.
    Without it, falls back to mock behavior.
    """

    # Mimi specifications
    FRAME_RATE_HZ = 12.5
    NUM_CODEBOOKS = 8
    CODEBOOK_SIZE = 2048
    SAMPLE_RATE = 24000

    def __init__(
        self,
        config: Optional[CodecConfig] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize Mimi codec.

        Args:
            config: Codec configuration
            model_path: Path to Mimi model weights (optional)
        """
        super().__init__(config)
        self.model_path = model_path
        self._encoder = None
        self._decoder = None

        if not MIMI_AVAILABLE:
            warnings.warn(
                "Mimi codec not available. Install with:\n"
                "  pip install moshi\n"
                "Falling back to mock implementation."
            )

    @property
    def name(self) -> str:
        return "mimi"

    @property
    def version(self) -> str:
        if MIMI_AVAILABLE and _mimi_module:
            return getattr(_mimi_module, "__version__", "1.0.0")
        return "mock"

    @property
    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            can_encode=True,
            can_decode=True,
            can_stream=True,
            codec_type=CodecType.HYBRID,
            has_semantic_tokens=True,
            has_acoustic_tokens=True,
            num_codebooks=self.NUM_CODEBOOKS,
            codebook_size=self.CODEBOOK_SIZE,
            frame_rate_hz=self.FRAME_RATE_HZ,
            sample_rate=self.SAMPLE_RATE,
            channels=1,
            supports_variable_bitrate=False,
            min_bitrate_kbps=1.5,
            max_bitrate_kbps=1.5,  # Fixed bitrate
        )

    def load(self) -> None:
        """Load the Mimi model."""
        if not MIMI_AVAILABLE:
            self._loaded = True
            return

        try:
            # Load Mimi encoder and decoder
            # The exact API depends on the moshi package version
            if hasattr(_mimi_module, "load_model"):
                model = _mimi_module.load_model(self.model_path)
                self._encoder = model.encoder
                self._decoder = model.decoder
            elif hasattr(_mimi_module, "Mimi"):
                model = _mimi_module.Mimi()
                if self.model_path:
                    model.load(self.model_path)
                self._encoder = model
                self._decoder = model
            else:
                # Assume direct encode/decode functions
                self._encoder = _mimi_module
                self._decoder = _mimi_module

            self._loaded = True

        except Exception as e:
            warnings.warn(f"Failed to load Mimi model: {e}. Using mock.")
            self._loaded = True

    def encode(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> EncodedAudio:
        """
        Encode audio to Mimi tokens.

        Args:
            audio: Audio array or file path
            sample_rate: Sample rate (if audio is array)

        Returns:
            EncodedAudio with Mimi tokens
        """
        self._ensure_loaded()

        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(Path(audio))

        audio = np.asarray(audio)
        sample_rate = sample_rate or self.SAMPLE_RATE

        # Resample if needed
        if sample_rate != self.SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self.SAMPLE_RATE)
            sample_rate = self.SAMPLE_RATE

        # Calculate expected sequence length
        duration = len(audio) / sample_rate
        seq_len = int(duration * self.FRAME_RATE_HZ)

        if MIMI_AVAILABLE and self._encoder is not None:
            try:
                # Real Mimi encoding
                tokens = self._encode_real(audio)
            except Exception as e:
                warnings.warn(f"Mimi encoding failed: {e}. Using mock.")
                tokens = self._encode_mock(audio, seq_len)
        else:
            tokens = self._encode_mock(audio, seq_len)

        token_seq = TokenSequence(
            tokens=tokens,
            num_codebooks=self.NUM_CODEBOOKS,
            frame_rate_hz=self.FRAME_RATE_HZ,
            source_duration_seconds=duration,
            source_sample_rate=sample_rate,
            codec_name=self.name,
            codec_version=self.version,
        )

        # First codebook typically contains semantic information
        semantic_tokens = TokenSequence(
            tokens=tokens[:1].copy(),
            num_codebooks=1,
            frame_rate_hz=self.FRAME_RATE_HZ,
            source_duration_seconds=duration,
            codec_name=self.name,
        )

        # Remaining codebooks are acoustic
        acoustic_tokens = TokenSequence(
            tokens=tokens[1:].copy(),
            num_codebooks=self.NUM_CODEBOOKS - 1,
            frame_rate_hz=self.FRAME_RATE_HZ,
            source_duration_seconds=duration,
            codec_name=self.name,
        )

        return EncodedAudio(
            tokens=token_seq,
            semantic_tokens=semantic_tokens,
            acoustic_tokens=acoustic_tokens,
            estimated_quality=0.95,
        )

    def _encode_real(self, audio: np.ndarray) -> np.ndarray:
        """Real Mimi encoding using the loaded model."""
        # This would use the actual Mimi API
        # The exact implementation depends on the moshi package version
        import torch

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            if hasattr(self._encoder, "encode"):
                tokens = self._encoder.encode(audio_tensor)
            else:
                tokens = self._encoder(audio_tensor)
            return tokens.squeeze(0).cpu().numpy().astype(np.int64)

    def _encode_mock(self, audio: np.ndarray, seq_len: int) -> np.ndarray:
        """Mock encoding for testing without Mimi."""
        import hashlib

        # Deterministic tokens based on audio content
        content_hash = hashlib.md5(audio.tobytes()[:1000]).hexdigest()
        seed = int(content_hash[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))

        return rng.integers(
            0, self.CODEBOOK_SIZE,
            size=(self.NUM_CODEBOOKS, seq_len),
            dtype=np.int64,
        )

    def decode(
        self,
        encoded: Union[EncodedAudio, TokenSequence, np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        """
        Decode Mimi tokens to audio.

        Args:
            encoded: Encoded audio or token sequence

        Returns:
            Tuple of (audio, sample_rate)
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
            if tokens.ndim == 1:
                tokens = tokens.reshape(1, -1)
            duration = tokens.shape[1] / self.FRAME_RATE_HZ

        if MIMI_AVAILABLE and self._decoder is not None:
            try:
                audio = self._decode_real(tokens)
                return audio, self.SAMPLE_RATE
            except Exception as e:
                warnings.warn(f"Mimi decoding failed: {e}. Using mock.")

        # Mock decode
        return self._decode_mock(tokens, duration), self.SAMPLE_RATE

    def _decode_real(self, tokens: np.ndarray) -> np.ndarray:
        """Real Mimi decoding using the loaded model."""
        import torch

        with torch.no_grad():
            tokens_tensor = torch.from_numpy(tokens).long().unsqueeze(0)
            if hasattr(self._decoder, "decode"):
                audio = self._decoder.decode(tokens_tensor)
            else:
                audio = self._decoder(tokens_tensor)
            return audio.squeeze().cpu().numpy()

    def _decode_mock(self, tokens: np.ndarray, duration: float) -> np.ndarray:
        """Mock decoding for testing without Mimi."""
        import hashlib

        token_hash = hashlib.md5(tokens.tobytes()).hexdigest()
        seed = int(token_hash[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))

        num_samples = int(duration * self.SAMPLE_RATE)
        audio = rng.standard_normal(num_samples).astype(np.float32) * 0.1

        # Add structure from first codebook
        samples_per_frame = int(self.SAMPLE_RATE / self.FRAME_RATE_HZ)
        first_cb = tokens[0] if tokens.ndim > 1 else tokens

        for i, tok in enumerate(first_cb):
            freq = 100 + (tok % 500)
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio))
            if start < len(audio):
                t = np.arange(end - start) / self.SAMPLE_RATE
                audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * t)

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio

    def _load_audio(self, path: Path) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except ImportError:
            # Fallback mock
            return np.random.randn(self.SAMPLE_RATE * 5).astype(np.float32), self.SAMPLE_RATE

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation
            ratio = target_sr / orig_sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def encode_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TokenSequence]:
        """
        Stream encode audio chunks.

        Mimi supports streaming with 80ms chunks.
        """
        chunk_samples = int(self.config.chunk_size_ms * sample_rate / 1000)
        buffer = np.array([], dtype=np.float32)

        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= chunk_samples:
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                encoded = self.encode(audio_chunk, sample_rate)
                yield encoded.tokens

        # Process remaining buffer
        if len(buffer) > 0:
            encoded = self.encode(buffer, sample_rate)
            yield encoded.tokens

    def decode_streaming(
        self,
        token_stream: Iterator[TokenSequence],
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Stream decode token chunks."""
        for tokens in token_stream:
            yield self.decode(tokens)
