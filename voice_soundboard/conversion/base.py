"""
Voice Conversion Base Classes.

Provides the abstract interface and core data structures
for real-time voice conversion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, List
import threading
import time

import numpy as np


class LatencyMode(Enum):
    """Latency/quality trade-off modes."""

    ULTRA_LOW = auto()    # ~60ms, may sacrifice quality
    LOW = auto()          # ~100ms, good balance
    BALANCED = auto()     # ~150ms, better quality
    HIGH_QUALITY = auto() # ~300ms, best quality


class ConversionState(Enum):
    """State of the voice converter."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class ConversionConfig:
    """Configuration for voice conversion."""

    # Latency settings
    latency_mode: LatencyMode = LatencyMode.BALANCED
    target_latency_ms: Optional[float] = None  # Override latency mode

    # Audio settings
    sample_rate: int = 24000
    channels: int = 1
    chunk_size_ms: float = 20.0  # Processing chunk size

    # Buffer settings
    input_buffer_ms: float = 100.0
    output_buffer_ms: float = 100.0

    # Quality settings
    preserve_pitch: bool = True
    preserve_formants: bool = True
    smoothing_factor: float = 0.1  # Cross-fade smoothing

    # Voice settings
    pitch_shift_semitones: float = 0.0
    formant_shift_ratio: float = 1.0

    # Processing
    use_gpu: bool = True
    device: str = "cuda"  # or "cpu"

    def get_latency_ms(self) -> float:
        """Get target latency in milliseconds."""
        if self.target_latency_ms is not None:
            return self.target_latency_ms

        latency_map = {
            LatencyMode.ULTRA_LOW: 60.0,
            LatencyMode.LOW: 100.0,
            LatencyMode.BALANCED: 150.0,
            LatencyMode.HIGH_QUALITY: 300.0,
        }
        return latency_map[self.latency_mode]

    def get_chunk_samples(self) -> int:
        """Get chunk size in samples."""
        return int(self.sample_rate * self.chunk_size_ms / 1000)

    def get_buffer_samples(self, buffer_ms: float) -> int:
        """Get buffer size in samples."""
        return int(self.sample_rate * buffer_ms / 1000)


@dataclass
class ConversionResult:
    """Result of a voice conversion operation."""

    # Audio data
    audio: np.ndarray
    sample_rate: int

    # Timing
    input_duration_ms: float
    output_duration_ms: float
    processing_time_ms: float
    latency_ms: float

    # Quality metrics
    similarity_score: float = 0.0  # 0-1, how similar to target voice
    naturalness_score: float = 0.0  # 0-1, how natural sounding

    # Metadata
    source_voice: Optional[str] = None
    target_voice: Optional[str] = None
    config: Optional[ConversionConfig] = None

    @property
    def realtime_factor(self) -> float:
        """Ratio of processing time to audio duration."""
        if self.input_duration_ms > 0:
            return self.processing_time_ms / self.input_duration_ms
        return 0.0

    @property
    def is_realtime(self) -> bool:
        """Whether conversion is fast enough for real-time."""
        return self.realtime_factor < 1.0


@dataclass
class ConversionStats:
    """Statistics for conversion session."""

    # Counts
    chunks_processed: int = 0
    chunks_dropped: int = 0
    buffer_underruns: int = 0
    buffer_overruns: int = 0

    # Timing
    total_input_ms: float = 0.0
    total_output_ms: float = 0.0
    total_processing_ms: float = 0.0

    # Latency tracking
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0
    _latency_count: int = 0

    # Quality
    avg_similarity: float = 0.0
    _similarity_sum: float = 0.0

    def update_latency(self, latency_ms: float) -> None:
        """Update latency statistics."""
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self._latency_sum += latency_ms
        self._latency_count += 1
        self.avg_latency_ms = self._latency_sum / self._latency_count

    def update_similarity(self, score: float) -> None:
        """Update similarity statistics."""
        self._similarity_sum += score
        if self.chunks_processed > 0:
            self.avg_similarity = self._similarity_sum / self.chunks_processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_dropped": self.chunks_dropped,
            "buffer_underruns": self.buffer_underruns,
            "buffer_overruns": self.buffer_overruns,
            "total_input_ms": self.total_input_ms,
            "total_output_ms": self.total_output_ms,
            "total_processing_ms": self.total_processing_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_similarity": self.avg_similarity,
            "realtime_factor": self.total_processing_ms / self.total_input_ms if self.total_input_ms > 0 else 0.0,
        }


class VoiceConverter(ABC):
    """
    Abstract base class for voice conversion.

    Provides the interface for converting voice characteristics
    while preserving linguistic content.
    """

    def __init__(
        self,
        config: Optional[ConversionConfig] = None,
        name: str = "voice_converter",
    ):
        self.config = config or ConversionConfig()
        self.name = name
        self._state = ConversionState.IDLE
        self._stats = ConversionStats()
        self._lock = threading.Lock()

        # Voice embeddings
        self._source_embedding: Optional[np.ndarray] = None
        self._target_embedding: Optional[np.ndarray] = None
        self._target_voice_id: Optional[str] = None

    @property
    def state(self) -> ConversionState:
        """Current converter state."""
        return self._state

    @property
    def stats(self) -> ConversionStats:
        """Conversion statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Whether converter is actively processing."""
        return self._state == ConversionState.RUNNING

    @abstractmethod
    def load(self) -> None:
        """Load the conversion model."""
        pass

    @abstractmethod
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_voice: Optional[Union[str, np.ndarray]] = None,
    ) -> ConversionResult:
        """
        Convert a single audio chunk.

        Args:
            audio: Input audio as numpy array
            sample_rate: Sample rate of input audio
            target_voice: Target voice ID or embedding

        Returns:
            ConversionResult with converted audio
        """
        pass

    @abstractmethod
    def convert_chunk(
        self,
        chunk: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a single chunk in streaming mode.

        Args:
            chunk: Audio chunk (assumes sample_rate from config)

        Returns:
            Converted audio chunk
        """
        pass

    def set_target_voice(
        self,
        voice: Union[str, np.ndarray, Path],
    ) -> None:
        """
        Set the target voice for conversion.

        Args:
            voice: Voice ID, embedding array, or path to reference audio
        """
        if isinstance(voice, str):
            # Voice ID - look up in library
            self._target_voice_id = voice
            self._target_embedding = self._load_voice_embedding(voice)
        elif isinstance(voice, Path):
            # Path to audio - extract embedding
            self._target_embedding = self._extract_embedding(voice)
            self._target_voice_id = voice.stem
        else:
            # Direct embedding
            self._target_embedding = voice
            self._target_voice_id = "custom"

    def _load_voice_embedding(self, voice_id: str) -> np.ndarray:
        """Load voice embedding from library."""
        # Try to load from voice library
        try:
            from voice_soundboard.cloning import get_default_library
            library = get_default_library()
            profile = library.get(voice_id)  # Use .get() method
            if profile and profile.embedding:
                return profile.embedding.vector
        except (ImportError, Exception):
            pass

        # Return random embedding as fallback (for mock)
        return np.random.randn(256).astype(np.float32)

    def _extract_embedding(self, audio_path: Path) -> np.ndarray:
        """Extract voice embedding from audio file."""
        try:
            from voice_soundboard.cloning import extract_embedding
            import soundfile as sf

            audio, sr = sf.read(audio_path)
            embedding = extract_embedding(audio, sr)
            return embedding.vector
        except ImportError:
            pass

        # Return random embedding as fallback
        return np.random.randn(256).astype(np.float32)

    def reset_stats(self) -> None:
        """Reset conversion statistics."""
        self._stats = ConversionStats()


class MockVoiceConverter(VoiceConverter):
    """
    Mock voice converter for testing.

    Simulates voice conversion with configurable latency.
    """

    def __init__(
        self,
        config: Optional[ConversionConfig] = None,
        simulate_latency: bool = True,
    ):
        super().__init__(config, name="mock")
        self._simulate_latency = simulate_latency
        self._loaded = False

    def load(self) -> None:
        """Load mock model."""
        self._loaded = True
        self._state = ConversionState.IDLE

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_voice: Optional[Union[str, np.ndarray]] = None,
    ) -> ConversionResult:
        """Convert audio with mock transformation."""
        if not self._loaded:
            self.load()

        start_time = time.perf_counter()

        # Set target voice if provided
        if target_voice is not None:
            self.set_target_voice(target_voice)

        # Simulate processing latency based on mode
        if self._simulate_latency:
            latency_ms = self.config.get_latency_ms()
            time.sleep(latency_ms / 1000 * 0.1)  # 10% of target latency

        # Mock conversion: apply slight frequency shift
        converted = self._mock_convert(audio, sample_rate)

        processing_time = (time.perf_counter() - start_time) * 1000
        input_duration = len(audio) / sample_rate * 1000

        # Update stats
        self._stats.chunks_processed += 1
        self._stats.total_input_ms += input_duration
        self._stats.total_output_ms += len(converted) / sample_rate * 1000
        self._stats.total_processing_ms += processing_time
        self._stats.update_latency(processing_time)

        return ConversionResult(
            audio=converted,
            sample_rate=sample_rate,
            input_duration_ms=input_duration,
            output_duration_ms=len(converted) / sample_rate * 1000,
            processing_time_ms=processing_time,
            latency_ms=processing_time,
            similarity_score=0.85,
            naturalness_score=0.90,
            target_voice=self._target_voice_id,
            config=self.config,
        )

    def convert_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Convert a streaming chunk."""
        if not self._loaded:
            self.load()

        return self._mock_convert(chunk, self.config.sample_rate)

    def _mock_convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply mock voice conversion."""
        # Simple mock: apply subtle modifications
        # In real implementation, this would use a neural network

        # Apply pitch shift simulation (just amplitude modulation for mock)
        if self.config.pitch_shift_semitones != 0:
            factor = 2 ** (self.config.pitch_shift_semitones / 12)
            # Simple resampling simulation
            if len(audio) > 10:
                indices = np.linspace(0, len(audio) - 1, int(len(audio) / factor))
                indices = np.clip(indices.astype(int), 0, len(audio) - 1)
                audio = audio[indices]

        # Apply formant shift simulation
        if self.config.formant_shift_ratio != 1.0:
            # Mock: just scale amplitude slightly
            audio = audio * (0.9 + 0.1 * self.config.formant_shift_ratio)

        # Add very subtle "conversion" effect
        if self._target_embedding is not None:
            # Use embedding to create deterministic but subtle modification
            seed = int(np.abs(self._target_embedding[:4]).sum() * 1000) % 10000
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 0.001, len(audio)).astype(np.float32)
            audio = audio + noise

        # Ensure output is float32
        return audio.astype(np.float32)
