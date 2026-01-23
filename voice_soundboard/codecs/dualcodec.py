"""
DualCodec Audio Codec Wrapper.

Wrapper for DualCodec which provides semantic-acoustic separation.
DualCodec uses dual-stream encoding for better TTS quality.

References:
- https://dualcodec.github.io/
- Semantic-enhanced audio codec for LLM integration
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

# Try to import DualCodec
DUALCODEC_AVAILABLE = False
_dualcodec_module = None
_dualcodec_error = None

try:
    import dualcodec
    _dualcodec_module = dualcodec
    DUALCODEC_AVAILABLE = True
except ImportError as e:
    _dualcodec_error = str(e)


class DualCodec(AudioCodec):
    """
    Wrapper for DualCodec - semantic-acoustic dual-stream codec.

    DualCodec features:
    - Semantic stream: Content/linguistic information (HuBERT-like)
    - Acoustic stream: Speaker timbre and prosody
    - 50 Hz frame rate for semantic, 50 Hz for acoustic
    - Enables voice conversion by swapping acoustic streams
    - Better TTS quality than single-stream codecs

    Note: Requires dualcodec package to be installed.
    Without it, falls back to mock behavior.
    """

    # DualCodec specifications
    SEMANTIC_FRAME_RATE = 50.0
    ACOUSTIC_FRAME_RATE = 50.0
    SEMANTIC_CODEBOOKS = 1
    ACOUSTIC_CODEBOOKS = 7
    SEMANTIC_VOCAB_SIZE = 1024
    ACOUSTIC_VOCAB_SIZE = 1024
    SAMPLE_RATE = 24000

    def __init__(
        self,
        config: Optional[CodecConfig] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize DualCodec.

        Args:
            config: Codec configuration
            model_path: Path to DualCodec model weights (optional)
        """
        super().__init__(config)
        self.model_path = model_path
        self._semantic_encoder = None
        self._acoustic_encoder = None
        self._decoder = None

        if not DUALCODEC_AVAILABLE:
            warnings.warn(
                "DualCodec not available. Install with:\n"
                "  pip install dualcodec\n"
                "Falling back to mock implementation."
            )

    @property
    def name(self) -> str:
        return "dualcodec"

    @property
    def version(self) -> str:
        if DUALCODEC_AVAILABLE and _dualcodec_module:
            return getattr(_dualcodec_module, "__version__", "1.0.0")
        return "mock"

    @property
    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            can_encode=True,
            can_decode=True,
            can_stream=True,
            codec_type=CodecType.DUAL,
            has_semantic_tokens=True,
            has_acoustic_tokens=True,
            num_codebooks=self.SEMANTIC_CODEBOOKS + self.ACOUSTIC_CODEBOOKS,
            codebook_size=max(self.SEMANTIC_VOCAB_SIZE, self.ACOUSTIC_VOCAB_SIZE),
            frame_rate_hz=self.SEMANTIC_FRAME_RATE,  # Primary frame rate
            sample_rate=self.SAMPLE_RATE,
            channels=1,
            supports_variable_bitrate=True,
            min_bitrate_kbps=1.5,
            max_bitrate_kbps=12.0,
        )

    def load(self) -> None:
        """Load the DualCodec model."""
        if not DUALCODEC_AVAILABLE:
            self._loaded = True
            return

        try:
            # Load DualCodec components
            if hasattr(_dualcodec_module, "load_model"):
                model = _dualcodec_module.load_model(self.model_path)
                self._semantic_encoder = model.semantic_encoder
                self._acoustic_encoder = model.acoustic_encoder
                self._decoder = model.decoder
            elif hasattr(_dualcodec_module, "DualCodec"):
                model = _dualcodec_module.DualCodec()
                if self.model_path:
                    model.load(self.model_path)
                self._semantic_encoder = model
                self._acoustic_encoder = model
                self._decoder = model
            else:
                self._semantic_encoder = _dualcodec_module
                self._acoustic_encoder = _dualcodec_module
                self._decoder = _dualcodec_module

            self._loaded = True

        except Exception as e:
            warnings.warn(f"Failed to load DualCodec model: {e}. Using mock.")
            self._loaded = True

    def encode(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> EncodedAudio:
        """
        Encode audio to DualCodec tokens.

        Returns both combined tokens and separated semantic/acoustic tokens.
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

        duration = len(audio) / sample_rate

        if DUALCODEC_AVAILABLE and self._semantic_encoder is not None:
            try:
                semantic_tokens, acoustic_tokens = self._encode_real(audio)
            except Exception as e:
                warnings.warn(f"DualCodec encoding failed: {e}. Using mock.")
                semantic_tokens, acoustic_tokens = self._encode_mock(audio, duration)
        else:
            semantic_tokens, acoustic_tokens = self._encode_mock(audio, duration)

        # Create token sequences
        semantic_seq = TokenSequence(
            tokens=semantic_tokens,
            num_codebooks=self.SEMANTIC_CODEBOOKS,
            frame_rate_hz=self.SEMANTIC_FRAME_RATE,
            source_duration_seconds=duration,
            source_sample_rate=sample_rate,
            codec_name=self.name,
            codec_version=self.version,
        )

        acoustic_seq = TokenSequence(
            tokens=acoustic_tokens,
            num_codebooks=self.ACOUSTIC_CODEBOOKS,
            frame_rate_hz=self.ACOUSTIC_FRAME_RATE,
            source_duration_seconds=duration,
            source_sample_rate=sample_rate,
            codec_name=self.name,
            codec_version=self.version,
        )

        # Combined tokens (semantic first, then acoustic)
        combined = np.vstack([semantic_tokens, acoustic_tokens])

        combined_seq = TokenSequence(
            tokens=combined,
            num_codebooks=self.SEMANTIC_CODEBOOKS + self.ACOUSTIC_CODEBOOKS,
            frame_rate_hz=self.SEMANTIC_FRAME_RATE,
            source_duration_seconds=duration,
            source_sample_rate=sample_rate,
            codec_name=self.name,
            codec_version=self.version,
        )

        return EncodedAudio(
            tokens=combined_seq,
            semantic_tokens=semantic_seq,
            acoustic_tokens=acoustic_seq,
            estimated_quality=0.95,
        )

    def _encode_real(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Real DualCodec encoding."""
        import torch

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            # Encode semantic
            if hasattr(self._semantic_encoder, "encode_semantic"):
                semantic = self._semantic_encoder.encode_semantic(audio_tensor)
            else:
                semantic = self._semantic_encoder.encode(audio_tensor, mode="semantic")

            # Encode acoustic
            if hasattr(self._acoustic_encoder, "encode_acoustic"):
                acoustic = self._acoustic_encoder.encode_acoustic(audio_tensor)
            else:
                acoustic = self._acoustic_encoder.encode(audio_tensor, mode="acoustic")

            return (
                semantic.squeeze(0).cpu().numpy().astype(np.int64),
                acoustic.squeeze(0).cpu().numpy().astype(np.int64),
            )

    def _encode_mock(
        self,
        audio: np.ndarray,
        duration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mock encoding for testing."""
        import hashlib

        content_hash = hashlib.md5(audio.tobytes()[:1000]).hexdigest()
        seed = int(content_hash[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))

        semantic_len = int(duration * self.SEMANTIC_FRAME_RATE)
        acoustic_len = int(duration * self.ACOUSTIC_FRAME_RATE)

        semantic = rng.integers(
            0, self.SEMANTIC_VOCAB_SIZE,
            size=(self.SEMANTIC_CODEBOOKS, semantic_len),
            dtype=np.int64,
        )

        acoustic = rng.integers(
            0, self.ACOUSTIC_VOCAB_SIZE,
            size=(self.ACOUSTIC_CODEBOOKS, acoustic_len),
            dtype=np.int64,
        )

        return semantic, acoustic

    def decode(
        self,
        encoded: Union[EncodedAudio, TokenSequence, np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        """
        Decode DualCodec tokens to audio.

        Can decode from combined tokens or EncodedAudio with separate streams.
        """
        self._ensure_loaded()

        # Handle different input types
        if isinstance(encoded, EncodedAudio):
            if encoded.has_dual_tokens:
                return self.decode_dual(
                    encoded.semantic_tokens,
                    encoded.acoustic_tokens,
                )
            tokens = encoded.tokens.tokens
            duration = encoded.tokens.source_duration_seconds
        elif isinstance(encoded, TokenSequence):
            tokens = encoded.tokens
            duration = encoded.source_duration_seconds
        else:
            tokens = np.asarray(encoded)
            if tokens.ndim == 1:
                tokens = tokens.reshape(1, -1)
            duration = tokens.shape[1] / self.SEMANTIC_FRAME_RATE

        # Split into semantic and acoustic
        semantic = tokens[:self.SEMANTIC_CODEBOOKS]
        acoustic = tokens[self.SEMANTIC_CODEBOOKS:]

        semantic_seq = TokenSequence(
            tokens=semantic,
            num_codebooks=self.SEMANTIC_CODEBOOKS,
            frame_rate_hz=self.SEMANTIC_FRAME_RATE,
        )

        acoustic_seq = TokenSequence(
            tokens=acoustic,
            num_codebooks=self.ACOUSTIC_CODEBOOKS,
            frame_rate_hz=self.ACOUSTIC_FRAME_RATE,
        )

        return self.decode_dual(semantic_seq, acoustic_seq)

    def decode_dual(
        self,
        semantic: TokenSequence,
        acoustic: TokenSequence,
    ) -> Tuple[np.ndarray, int]:
        """
        Decode from separate semantic and acoustic tokens.

        This is the key feature of DualCodec - allows voice conversion
        by combining semantic tokens from one source with acoustic
        tokens from another.

        Args:
            semantic: Semantic token sequence (content)
            acoustic: Acoustic token sequence (timbre/prosody)

        Returns:
            Tuple of (audio, sample_rate)
        """
        self._ensure_loaded()

        duration = max(
            semantic.source_duration_seconds,
            acoustic.source_duration_seconds,
        )
        if duration == 0:
            duration = semantic.sequence_length / self.SEMANTIC_FRAME_RATE

        if DUALCODEC_AVAILABLE and self._decoder is not None:
            try:
                audio = self._decode_real(semantic.tokens, acoustic.tokens)
                return audio, self.SAMPLE_RATE
            except Exception as e:
                warnings.warn(f"DualCodec decoding failed: {e}. Using mock.")

        return self._decode_mock(semantic.tokens, acoustic.tokens, duration), self.SAMPLE_RATE

    def _decode_real(
        self,
        semantic: np.ndarray,
        acoustic: np.ndarray,
    ) -> np.ndarray:
        """Real DualCodec decoding."""
        import torch

        with torch.no_grad():
            semantic_tensor = torch.from_numpy(semantic).long().unsqueeze(0)
            acoustic_tensor = torch.from_numpy(acoustic).long().unsqueeze(0)

            if hasattr(self._decoder, "decode_dual"):
                audio = self._decoder.decode_dual(semantic_tensor, acoustic_tensor)
            else:
                audio = self._decoder.decode(
                    semantic=semantic_tensor,
                    acoustic=acoustic_tensor,
                )

            return audio.squeeze().cpu().numpy()

    def _decode_mock(
        self,
        semantic: np.ndarray,
        acoustic: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Mock decoding for testing."""
        import hashlib

        # Combine both streams for deterministic output
        combined_hash = hashlib.md5(
            semantic.tobytes() + acoustic.tobytes()
        ).hexdigest()
        seed = int(combined_hash[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))

        num_samples = int(duration * self.SAMPLE_RATE)
        audio = rng.standard_normal(num_samples).astype(np.float32) * 0.1

        # Use semantic for pitch contour
        samples_per_semantic = int(self.SAMPLE_RATE / self.SEMANTIC_FRAME_RATE)
        semantic_flat = semantic.flatten()

        for i, tok in enumerate(semantic_flat):
            freq = 100 + (tok % 400)  # Pitch from semantic
            start = i * samples_per_semantic
            end = min(start + samples_per_semantic, len(audio))
            if start < len(audio):
                t = np.arange(end - start) / self.SAMPLE_RATE
                audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * t)

        # Use acoustic for amplitude envelope
        samples_per_acoustic = int(self.SAMPLE_RATE / self.ACOUSTIC_FRAME_RATE)
        acoustic_flat = acoustic[0] if acoustic.ndim > 1 else acoustic

        for i, tok in enumerate(acoustic_flat):
            amp = 0.5 + (tok % 512) / 1024  # Amplitude from acoustic
            start = i * samples_per_acoustic
            end = min(start + samples_per_acoustic, len(audio))
            if start < len(audio):
                audio[start:end] *= amp

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio

    def encode_dual(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: Optional[int] = None,
    ) -> Tuple[TokenSequence, TokenSequence]:
        """
        Encode to separate semantic and acoustic tokens.

        Convenience method for voice conversion workflows.

        Args:
            audio: Audio to encode
            sample_rate: Sample rate if audio is array

        Returns:
            Tuple of (semantic_tokens, acoustic_tokens)
        """
        encoded = self.encode(audio, sample_rate)
        return encoded.semantic_tokens, encoded.acoustic_tokens

    def voice_convert(
        self,
        content_audio: Union[np.ndarray, Path, str],
        style_audio: Union[np.ndarray, Path, str],
        content_sample_rate: Optional[int] = None,
        style_sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert voice by combining content from one audio with style from another.

        This is the main use case for DualCodec:
        - Take semantic (content) from content_audio
        - Take acoustic (style/timbre) from style_audio
        - Generate new audio with content in the style voice

        Args:
            content_audio: Audio providing the content/words
            style_audio: Audio providing the voice/style
            content_sample_rate: Sample rate for content audio
            style_sample_rate: Sample rate for style audio

        Returns:
            Tuple of (converted_audio, sample_rate)
        """
        # Encode both
        content_encoded = self.encode(content_audio, content_sample_rate)
        style_encoded = self.encode(style_audio, style_sample_rate)

        # Combine semantic from content with acoustic from style
        return self.decode_dual(
            content_encoded.semantic_tokens,
            style_encoded.acoustic_tokens,
        )

    def _load_audio(self, path: Path) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except ImportError:
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
            ratio = target_sr / orig_sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def encode_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TokenSequence]:
        """Stream encode audio chunks."""
        chunk_samples = int(self.config.chunk_size_ms * sample_rate / 1000)
        buffer = np.array([], dtype=np.float32)

        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= chunk_samples:
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                encoded = self.encode(audio_chunk, sample_rate)
                yield encoded.tokens

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
