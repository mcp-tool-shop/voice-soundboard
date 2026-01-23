"""
Streaming Voice Conversion Pipeline.

Provides low-latency streaming infrastructure with circular buffers
and pipelined processing stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Callable, Iterator, Any, Tuple
import threading
import queue
import time

import numpy as np


class PipelineStage(Enum):
    """Stages in the conversion pipeline."""

    INPUT = auto()      # Raw input from device
    PREPROCESS = auto() # Preprocessing (resampling, normalization)
    ENCODE = auto()     # Feature extraction / encoding
    CONVERT = auto()    # Voice conversion
    DECODE = auto()     # Audio reconstruction
    POSTPROCESS = auto() # Post-processing (denoising, smoothing)
    OUTPUT = auto()     # Ready for playback


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""

    data: np.ndarray
    sample_rate: int
    timestamp_ms: float
    stage: PipelineStage = PipelineStage.INPUT

    # Processing metadata
    processing_started_ms: float = 0.0
    processing_completed_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Duration of chunk in milliseconds."""
        return len(self.data) / self.sample_rate * 1000

    @property
    def processing_time_ms(self) -> float:
        """Time spent processing this chunk."""
        if self.processing_completed_ms > 0:
            return self.processing_completed_ms - self.processing_started_ms
        return 0.0

    def copy(self) -> "AudioChunk":
        """Create a copy of this chunk."""
        return AudioChunk(
            data=self.data.copy(),
            sample_rate=self.sample_rate,
            timestamp_ms=self.timestamp_ms,
            stage=self.stage,
            processing_started_ms=self.processing_started_ms,
            processing_completed_ms=self.processing_completed_ms,
        )


class AudioBuffer:
    """
    Thread-safe circular buffer for audio samples.

    Optimized for low-latency streaming with minimal allocations.
    """

    def __init__(
        self,
        capacity_samples: int,
        channels: int = 1,
        dtype: np.dtype = np.float32,
    ):
        self.capacity = capacity_samples
        self.channels = channels
        self.dtype = dtype

        # Circular buffer
        self._buffer = np.zeros((capacity_samples, channels), dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    @property
    def available(self) -> int:
        """Number of samples available for reading."""
        with self._lock:
            return self._count

    @property
    def free_space(self) -> int:
        """Number of samples that can be written."""
        with self._lock:
            return self.capacity - self._count

    def write(
        self,
        data: np.ndarray,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> int:
        """
        Write samples to buffer.

        Args:
            data: Audio samples to write
            block: Whether to wait if buffer is full
            timeout: Maximum wait time in seconds

        Returns:
            Number of samples written
        """
        # Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        samples_to_write = len(data)

        with self._not_full:
            # Wait for space if blocking
            while self._count + samples_to_write > self.capacity:
                if not block:
                    # Non-blocking: write what we can
                    samples_to_write = self.capacity - self._count
                    if samples_to_write <= 0:
                        return 0
                    break
                if not self._not_full.wait(timeout):
                    return 0  # Timeout

            # Write samples (may wrap around)
            data = data[:samples_to_write]

            end_pos = self._write_pos + samples_to_write

            if end_pos <= self.capacity:
                # No wrap
                self._buffer[self._write_pos:end_pos] = data
            else:
                # Wrap around
                first_part = self.capacity - self._write_pos
                self._buffer[self._write_pos:] = data[:first_part]
                self._buffer[:end_pos - self.capacity] = data[first_part:]

            self._write_pos = end_pos % self.capacity
            self._count += samples_to_write

            self._not_empty.notify_all()

            return samples_to_write

    def read(
        self,
        num_samples: int,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Read samples from buffer.

        Args:
            num_samples: Number of samples to read
            block: Whether to wait if buffer is empty
            timeout: Maximum wait time in seconds

        Returns:
            Audio samples or None if timeout
        """
        with self._not_empty:
            # Wait for data if blocking
            while self._count < num_samples:
                if not block:
                    if self._count == 0:
                        return None
                    num_samples = self._count
                    break
                if not self._not_empty.wait(timeout):
                    return None  # Timeout

            # Read samples (may wrap around)
            end_pos = self._read_pos + num_samples

            if end_pos <= self.capacity:
                # No wrap
                data = self._buffer[self._read_pos:end_pos].copy()
            else:
                # Wrap around
                first_part = self.capacity - self._read_pos
                data = np.concatenate([
                    self._buffer[self._read_pos:],
                    self._buffer[:end_pos - self.capacity]
                ])

            self._read_pos = end_pos % self.capacity
            self._count -= num_samples

            self._not_full.notify_all()

            # Return as 1D if mono
            if self.channels == 1:
                return data.flatten()
            return data

    def peek(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples without removing them from buffer."""
        with self._lock:
            if self._count < num_samples:
                num_samples = self._count
            if num_samples == 0:
                return None

            end_pos = self._read_pos + num_samples

            if end_pos <= self.capacity:
                data = self._buffer[self._read_pos:end_pos].copy()
            else:
                first_part = self.capacity - self._read_pos
                data = np.concatenate([
                    self._buffer[self._read_pos:],
                    self._buffer[:end_pos - self.capacity]
                ])

            if self.channels == 1:
                return data.flatten()
            return data

    def clear(self) -> None:
        """Clear all data from buffer."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._count = 0
            self._not_full.notify_all()


class StreamingConverter:
    """
    Streaming voice conversion with pipelined processing.

    Uses double-buffering and threading for smooth conversion.
    """

    def __init__(
        self,
        converter: Any,  # VoiceConverter
        chunk_size: int = 480,  # 20ms at 24kHz
        buffer_chunks: int = 5,
        sample_rate: int = 24000,
    ):
        self.converter = converter
        self.chunk_size = chunk_size
        self.buffer_chunks = buffer_chunks
        self.sample_rate = sample_rate

        # Buffers
        buffer_size = chunk_size * buffer_chunks
        self._input_buffer = AudioBuffer(buffer_size)
        self._output_buffer = AudioBuffer(buffer_size)

        # Processing thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Statistics
        self._chunks_processed = 0
        self._total_latency_ms = 0.0

        # Callback
        self._on_output: Optional[Callable[[np.ndarray], None]] = None

    def start(
        self,
        on_output: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        Start the streaming converter.

        Args:
            on_output: Callback for output audio chunks
        """
        if self._running:
            return

        self._on_output = on_output
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the streaming converter."""
        self._running = False
        if self._thread is not None:
            # Unblock any waiting reads
            self._input_buffer.clear()
            self._thread.join(timeout=1.0)
            self._thread = None

    def push(self, audio: np.ndarray) -> None:
        """
        Push audio into the input buffer.

        Args:
            audio: Audio samples to process
        """
        self._input_buffer.write(audio, block=False)

    def pull(self, num_samples: int) -> Optional[np.ndarray]:
        """
        Pull converted audio from output buffer.

        Args:
            num_samples: Number of samples to read

        Returns:
            Converted audio or None if not enough available
        """
        return self._output_buffer.read(num_samples, block=False)

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            # Read a chunk from input
            chunk = self._input_buffer.read(
                self.chunk_size,
                block=True,
                timeout=0.1,
            )

            if chunk is None:
                continue

            # Process the chunk
            start_time = time.perf_counter()
            converted = self.converter.convert_chunk(chunk)
            processing_time = (time.perf_counter() - start_time) * 1000

            # Update stats
            self._chunks_processed += 1
            self._total_latency_ms += processing_time

            # Write to output buffer
            self._output_buffer.write(converted, block=False)

            # Call output callback
            if self._on_output is not None:
                self._on_output(converted)

    @property
    def is_running(self) -> bool:
        """Whether converter is running."""
        return self._running

    @property
    def input_available(self) -> int:
        """Samples available in input buffer."""
        return self._input_buffer.available

    @property
    def output_available(self) -> int:
        """Samples available in output buffer."""
        return self._output_buffer.available

    @property
    def avg_latency_ms(self) -> float:
        """Average processing latency per chunk."""
        if self._chunks_processed > 0:
            return self._total_latency_ms / self._chunks_processed
        return 0.0


class ConversionPipeline:
    """
    Multi-stage voice conversion pipeline.

    Supports parallel processing of different stages.
    """

    def __init__(
        self,
        stages: List[Tuple[PipelineStage, Callable[[np.ndarray], np.ndarray]]],
        chunk_size: int = 480,
        sample_rate: int = 24000,
    ):
        self.stages = stages
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        # Queues between stages
        self._queues: List[queue.Queue] = [
            queue.Queue(maxsize=10) for _ in range(len(stages) + 1)
        ]

        # Processing threads
        self._threads: List[threading.Thread] = []
        self._running = False

        # Statistics per stage
        self._stage_times: Dict[PipelineStage, List[float]] = {
            stage: [] for stage, _ in stages
        }

    def start(self) -> None:
        """Start all pipeline stages."""
        if self._running:
            return

        self._running = True

        # Create a thread for each stage
        for i, (stage, processor) in enumerate(self.stages):
            thread = threading.Thread(
                target=self._stage_loop,
                args=(i, stage, processor),
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def stop(self) -> None:
        """Stop all pipeline stages."""
        self._running = False

        # Clear queues to unblock threads
        for q in self._queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Wait for threads
        for thread in self._threads:
            thread.join(timeout=0.5)
        self._threads.clear()

    def push(self, audio: np.ndarray) -> None:
        """Push audio into the pipeline."""
        try:
            self._queues[0].put_nowait(audio)
        except queue.Full:
            pass  # Drop if queue is full

    def pull(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Pull processed audio from the pipeline."""
        try:
            return self._queues[-1].get(timeout=timeout)
        except queue.Empty:
            return None

    def _stage_loop(
        self,
        stage_idx: int,
        stage: PipelineStage,
        processor: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Processing loop for a single stage."""
        input_queue = self._queues[stage_idx]
        output_queue = self._queues[stage_idx + 1]

        while self._running:
            try:
                audio = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Process
            start_time = time.perf_counter()
            result = processor(audio)
            processing_time = (time.perf_counter() - start_time) * 1000

            # Record timing
            self._stage_times[stage].append(processing_time)
            if len(self._stage_times[stage]) > 100:
                self._stage_times[stage] = self._stage_times[stage][-100:]

            # Pass to next stage
            try:
                output_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop if next stage is full

    def get_stage_latency(self, stage: PipelineStage) -> float:
        """Get average latency for a specific stage."""
        times = self._stage_times.get(stage, [])
        if times:
            return sum(times) / len(times)
        return 0.0

    def get_total_latency(self) -> float:
        """Get total pipeline latency."""
        return sum(self.get_stage_latency(stage) for stage, _ in self.stages)

    @property
    def is_running(self) -> bool:
        """Whether pipeline is running."""
        return self._running


# Import for type hints
from typing import Dict
