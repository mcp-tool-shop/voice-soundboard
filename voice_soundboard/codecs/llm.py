"""
LLM Integration Utilities for Audio Codecs.

Provides utilities for integrating audio codecs with LLMs:
- Token vocabulary management
- Prompt construction for audio LLMs
- Streaming audio generation
- Multi-modal token handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union, Iterator, Tuple, Callable
from enum import Enum, auto

import numpy as np

from voice_soundboard.codecs.base import (
    AudioCodec,
    TokenSequence,
    EncodedAudio,
    CodecCapabilities,
)


class TokenType(Enum):
    """Types of tokens in multi-modal vocabulary."""

    TEXT = auto()
    AUDIO = auto()
    SEMANTIC = auto()
    ACOUSTIC = auto()
    SPECIAL = auto()


@dataclass
class SpecialTokens:
    """Special tokens for LLM audio integration."""

    # Standard special tokens
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    # Audio-specific tokens
    audio_start: str = "<|audio|>"
    audio_end: str = "<|/audio|>"
    speech_start: str = "<|speech|>"
    speech_end: str = "<|/speech|>"

    # Semantic/acoustic separation
    semantic_start: str = "<|semantic|>"
    semantic_end: str = "<|/semantic|>"
    acoustic_start: str = "<|acoustic|>"
    acoustic_end: str = "<|/acoustic|>"

    # Speaker tokens
    speaker_prefix: str = "<|speaker_"
    speaker_suffix: str = "|>"

    # Emotion tokens (optional)
    emotion_prefix: str = "<|emotion_"
    emotion_suffix: str = "|>"

    def get_speaker_token(self, speaker_id: Union[int, str]) -> str:
        """Get speaker token for given ID."""
        return f"{self.speaker_prefix}{speaker_id}{self.speaker_suffix}"

    def get_emotion_token(self, emotion: str) -> str:
        """Get emotion token for given emotion."""
        return f"{self.emotion_prefix}{emotion}{self.emotion_suffix}"


@dataclass
class VocabularyConfig:
    """Configuration for multi-modal vocabulary."""

    # Text vocabulary (from tokenizer)
    text_vocab_size: int = 32000

    # Audio vocabulary
    audio_vocab_size: int = 8192  # Typically codebook_size * num_codebooks
    audio_token_offset: int = 32000  # Where audio tokens start

    # Special tokens
    num_special_tokens: int = 256
    special_token_offset: int = 0

    # Semantic/acoustic split (for DualCodec)
    semantic_vocab_size: int = 1024
    acoustic_vocab_size: int = 7168  # Rest of audio vocab

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size."""
        return (
            self.num_special_tokens +
            self.text_vocab_size +
            self.audio_vocab_size
        )

    def text_to_global(self, token_id: int) -> int:
        """Convert text token ID to global vocabulary ID."""
        return token_id + self.num_special_tokens

    def audio_to_global(self, token_id: int) -> int:
        """Convert audio token ID to global vocabulary ID."""
        return token_id + self.audio_token_offset

    def global_to_text(self, global_id: int) -> Optional[int]:
        """Convert global ID to text token ID, if applicable."""
        local = global_id - self.num_special_tokens
        if 0 <= local < self.text_vocab_size:
            return local
        return None

    def global_to_audio(self, global_id: int) -> Optional[int]:
        """Convert global ID to audio token ID, if applicable."""
        local = global_id - self.audio_token_offset
        if 0 <= local < self.audio_vocab_size:
            return local
        return None

    def get_token_type(self, global_id: int) -> TokenType:
        """Get the type of a token from its global ID."""
        if global_id < self.num_special_tokens:
            return TokenType.SPECIAL
        if global_id < self.audio_token_offset:
            return TokenType.TEXT
        return TokenType.AUDIO


@dataclass
class AudioPrompt:
    """
    A prompt that can contain both text and audio.

    Used for constructing inputs to audio-capable LLMs.
    """

    # Prompt components
    text_prefix: str = ""
    audio_tokens: Optional[TokenSequence] = None
    text_suffix: str = ""

    # Metadata
    speaker_id: Optional[str] = None
    emotion: Optional[str] = None

    # Configuration
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    vocab_config: VocabularyConfig = field(default_factory=VocabularyConfig)

    def to_token_sequence(
        self,
        text_tokenizer: Optional[Callable[[str], List[int]]] = None,
    ) -> np.ndarray:
        """
        Convert prompt to token sequence for LLM input.

        Args:
            text_tokenizer: Function to tokenize text (optional)

        Returns:
            Token array ready for LLM
        """
        tokens = []

        # Add BOS
        tokens.append(1)  # BOS token ID

        # Add speaker token if specified
        if self.speaker_id:
            speaker_tok = self.special_tokens.get_speaker_token(self.speaker_id)
            tokens.append(hash(speaker_tok) % 256)  # Simple hash to ID

        # Add emotion token if specified
        if self.emotion:
            emotion_tok = self.special_tokens.get_emotion_token(self.emotion)
            tokens.append(hash(emotion_tok) % 256)

        # Tokenize text prefix
        if self.text_prefix and text_tokenizer:
            text_tokens = text_tokenizer(self.text_prefix)
            tokens.extend([
                self.vocab_config.text_to_global(t)
                for t in text_tokens
            ])

        # Add audio tokens
        if self.audio_tokens is not None:
            # Audio start marker
            tokens.append(hash(self.special_tokens.audio_start) % 256)

            # Convert audio tokens to global vocabulary
            audio_flat = self.audio_tokens.to_llm_tokens(add_special_tokens=False)
            tokens.extend([
                self.vocab_config.audio_to_global(t)
                for t in audio_flat
            ])

            # Audio end marker
            tokens.append(hash(self.special_tokens.audio_end) % 256)

        # Tokenize text suffix
        if self.text_suffix and text_tokenizer:
            text_tokens = text_tokenizer(self.text_suffix)
            tokens.extend([
                self.vocab_config.text_to_global(t)
                for t in text_tokens
            ])

        # Add EOS
        tokens.append(2)  # EOS token ID

        return np.array(tokens, dtype=np.int64)


class LLMCodecBridge:
    """
    Bridge between audio codecs and LLMs.

    Handles:
    - Encoding audio to LLM-compatible tokens
    - Decoding LLM output tokens to audio
    - Streaming generation
    - Multi-speaker handling
    """

    def __init__(
        self,
        codec: AudioCodec,
        vocab_config: Optional[VocabularyConfig] = None,
        special_tokens: Optional[SpecialTokens] = None,
    ):
        """
        Initialize the bridge.

        Args:
            codec: Audio codec to use
            vocab_config: Vocabulary configuration
            special_tokens: Special token definitions
        """
        self.codec = codec
        self.vocab_config = vocab_config or VocabularyConfig(
            audio_vocab_size=codec.get_vocab_size(),
        )
        self.special_tokens = special_tokens or SpecialTokens()

    def encode_for_llm(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: Optional[int] = None,
        speaker_id: Optional[str] = None,
        include_markers: bool = True,
    ) -> np.ndarray:
        """
        Encode audio for LLM input.

        Args:
            audio: Audio array or file path
            sample_rate: Sample rate if audio is array
            speaker_id: Optional speaker identifier
            include_markers: Include audio start/end markers

        Returns:
            Token array for LLM input
        """
        # Encode with codec
        encoded = self.codec.encode(audio, sample_rate)

        # Convert to LLM tokens
        audio_tokens = encoded.tokens.to_llm_tokens(add_special_tokens=False)

        # Convert to global vocabulary
        global_tokens = [
            self.vocab_config.audio_to_global(t)
            for t in audio_tokens
        ]

        result = []

        # Add speaker token
        if speaker_id:
            speaker_tok = self.special_tokens.get_speaker_token(speaker_id)
            result.append(hash(speaker_tok) % self.vocab_config.num_special_tokens)

        # Add markers
        if include_markers:
            result.append(
                hash(self.special_tokens.audio_start) %
                self.vocab_config.num_special_tokens
            )

        result.extend(global_tokens)

        if include_markers:
            result.append(
                hash(self.special_tokens.audio_end) %
                self.vocab_config.num_special_tokens
            )

        return np.array(result, dtype=np.int64)

    def decode_from_llm(
        self,
        tokens: np.ndarray,
        remove_markers: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Decode LLM output tokens to audio.

        Args:
            tokens: LLM output token array
            remove_markers: Remove audio markers from tokens

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Filter to only audio tokens
        audio_tokens = []

        in_audio = not remove_markers
        audio_start_hash = hash(self.special_tokens.audio_start) % self.vocab_config.num_special_tokens
        audio_end_hash = hash(self.special_tokens.audio_end) % self.vocab_config.num_special_tokens

        for tok in tokens:
            if remove_markers:
                if tok == audio_start_hash:
                    in_audio = True
                    continue
                if tok == audio_end_hash:
                    in_audio = False
                    continue

            if in_audio:
                # Convert from global to local audio token
                local_tok = self.vocab_config.global_to_audio(tok)
                if local_tok is not None:
                    audio_tokens.append(local_tok)

        if not audio_tokens:
            # Return silence
            caps = self.codec.capabilities
            return np.zeros(caps.sample_rate, dtype=np.float32), caps.sample_rate

        # Create token sequence
        audio_array = np.array(audio_tokens, dtype=np.int64)
        token_seq = TokenSequence.from_llm_tokens(
            audio_array,
            num_codebooks=self.codec.capabilities.num_codebooks,
            frame_rate_hz=self.codec.capabilities.frame_rate_hz,
            has_special_tokens=False,
            offset=0,
        )

        return self.codec.decode(token_seq)

    def create_prompt(
        self,
        text: str = "",
        audio: Optional[Union[np.ndarray, str]] = None,
        sample_rate: Optional[int] = None,
        speaker_id: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> AudioPrompt:
        """
        Create a prompt with optional text and audio.

        Args:
            text: Text prompt
            audio: Optional audio to include
            sample_rate: Sample rate if audio is array
            speaker_id: Optional speaker identifier
            emotion: Optional emotion tag

        Returns:
            AudioPrompt object
        """
        audio_tokens = None
        if audio is not None:
            encoded = self.codec.encode(audio, sample_rate)
            audio_tokens = encoded.tokens

        return AudioPrompt(
            text_prefix=text,
            audio_tokens=audio_tokens,
            speaker_id=speaker_id,
            emotion=emotion,
            special_tokens=self.special_tokens,
            vocab_config=self.vocab_config,
        )

    def stream_decode(
        self,
        token_generator: Iterator[int],
        on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None,
        buffer_size: int = 100,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Stream decode tokens as they're generated.

        Args:
            token_generator: Iterator yielding tokens one at a time
            on_audio_chunk: Optional callback for each audio chunk
            buffer_size: Number of tokens to buffer before decoding

        Yields:
            Tuple of (audio_chunk, sample_rate)
        """
        buffer = []
        in_audio = False

        audio_start_hash = hash(self.special_tokens.audio_start) % self.vocab_config.num_special_tokens
        audio_end_hash = hash(self.special_tokens.audio_end) % self.vocab_config.num_special_tokens

        for token in token_generator:
            if token == audio_start_hash:
                in_audio = True
                buffer = []
                continue

            if token == audio_end_hash:
                in_audio = False
                if buffer:
                    audio, sr = self._decode_buffer(buffer)
                    if on_audio_chunk:
                        on_audio_chunk(audio, sr)
                    yield audio, sr
                buffer = []
                continue

            if in_audio:
                local_tok = self.vocab_config.global_to_audio(token)
                if local_tok is not None:
                    buffer.append(local_tok)

                    if len(buffer) >= buffer_size:
                        audio, sr = self._decode_buffer(buffer)
                        if on_audio_chunk:
                            on_audio_chunk(audio, sr)
                        yield audio, sr
                        buffer = []

        # Flush remaining buffer
        if buffer:
            audio, sr = self._decode_buffer(buffer)
            if on_audio_chunk:
                on_audio_chunk(audio, sr)
            yield audio, sr

    def _decode_buffer(self, buffer: List[int]) -> Tuple[np.ndarray, int]:
        """Decode a buffer of audio tokens."""
        audio_array = np.array(buffer, dtype=np.int64)
        token_seq = TokenSequence.from_llm_tokens(
            audio_array,
            num_codebooks=self.codec.capabilities.num_codebooks,
            frame_rate_hz=self.codec.capabilities.frame_rate_hz,
            has_special_tokens=False,
            offset=0,
        )
        return self.codec.decode(token_seq)


def get_codec_vocabulary_info(codec: AudioCodec) -> Dict[str, Any]:
    """
    Get vocabulary information for a codec.

    Useful for configuring LLM tokenizers.

    Args:
        codec: Audio codec

    Returns:
        Dictionary with vocabulary info
    """
    caps = codec.capabilities

    return {
        "codec_name": codec.name,
        "codec_version": codec.version,
        "num_codebooks": caps.num_codebooks,
        "codebook_size": caps.codebook_size,
        "total_audio_vocab_size": caps.codebook_size * caps.num_codebooks,
        "frame_rate_hz": caps.frame_rate_hz,
        "sample_rate": caps.sample_rate,
        "tokens_per_second": caps.frame_rate_hz * caps.num_codebooks,
        "has_semantic_tokens": caps.has_semantic_tokens,
        "has_acoustic_tokens": caps.has_acoustic_tokens,
        "recommended_special_tokens": [
            "<pad>", "<bos>", "<eos>", "<unk>",
            "<|audio|>", "<|/audio|>",
            "<|speech|>", "<|/speech|>",
        ],
    }


def estimate_audio_context_length(
    duration_seconds: float,
    codec: AudioCodec,
    include_markers: bool = True,
) -> int:
    """
    Estimate context length needed for audio of given duration.

    Args:
        duration_seconds: Audio duration in seconds
        codec: Audio codec to use
        include_markers: Include start/end markers

    Returns:
        Estimated token count
    """
    base_tokens = codec.estimate_tokens(duration_seconds)

    # Add markers
    if include_markers:
        base_tokens += 2  # audio_start, audio_end

    return base_tokens


def estimate_audio_duration(
    context_length: int,
    codec: AudioCodec,
    include_markers: bool = True,
) -> float:
    """
    Estimate audio duration from context length.

    Args:
        context_length: Number of tokens
        codec: Audio codec to use
        include_markers: Account for start/end markers

    Returns:
        Estimated duration in seconds
    """
    tokens = context_length
    if include_markers:
        tokens -= 2

    tokens = max(0, tokens)
    return codec.estimate_duration(tokens)
