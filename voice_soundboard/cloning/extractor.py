"""
Voice Embedding Extraction.

Extracts speaker embeddings from audio samples for voice cloning.
Supports multiple extraction backends (speaker verification models).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum

import numpy as np


class ExtractorBackend(Enum):
    """Available embedding extraction backends."""

    RESEMBLYZER = "resemblyzer"  # d-vector based
    SPEECHBRAIN = "speechbrain"  # ECAPA-TDNN
    WESPEAKER = "wespeaker"  # ResNet-based
    MOCK = "mock"  # For testing


@dataclass
class VoiceEmbedding:
    """
    Voice embedding representing a speaker's vocal characteristics.

    Contains the embedding vector plus metadata about extraction.
    """

    # Core embedding data
    embedding: np.ndarray  # Shape: (embedding_dim,) typically 256 or 512
    embedding_dim: int = 256

    # Audio source info
    source_path: Optional[str] = None
    source_duration_seconds: float = 0.0
    source_sample_rate: int = 16000

    # Extraction metadata
    extractor_backend: str = "mock"
    extraction_time: float = 0.0
    created_at: float = field(default_factory=time.time)

    # Quality metrics
    quality_score: float = 1.0  # 0-1, based on audio quality
    snr_db: float = 20.0  # Signal-to-noise ratio estimate

    # Speaker characteristics (estimated)
    estimated_gender: Optional[str] = None  # "male", "female", None
    estimated_age_range: Optional[str] = None  # "young", "adult", "senior"
    language_detected: Optional[str] = None

    # Unique identifier
    embedding_id: str = ""

    def __post_init__(self):
        """Generate embedding ID if not provided."""
        if not self.embedding_id:
            # Hash based on embedding values for consistency
            hash_input = self.embedding.tobytes()
            self.embedding_id = hashlib.sha256(hash_input).hexdigest()[:16]

    def similarity(self, other: "VoiceEmbedding") -> float:
        """
        Compute cosine similarity with another embedding.

        Args:
            other: Another VoiceEmbedding to compare

        Returns:
            Similarity score from -1 (opposite) to 1 (identical)
        """
        if self.embedding.shape != other.embedding.shape:
            raise ValueError(
                f"Embedding dimension mismatch: {self.embedding.shape} vs {other.embedding.shape}"
            )

        dot_product = np.dot(self.embedding, other.embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other.embedding)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize embedding to dictionary (embedding as list)."""
        data = asdict(self)
        data["embedding"] = self.embedding.tolist()

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            return obj

        return convert_numpy(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceEmbedding":
        """Deserialize embedding from dictionary."""
        data = data.copy()
        data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        return cls(**data)

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save embedding to file.

        Args:
            path: Output path (.npz for numpy, .json for JSON)

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".npz":
            # NumPy format (more efficient for embeddings)
            np.savez_compressed(
                path,
                embedding=self.embedding,
                metadata=json.dumps(
                    {k: v for k, v in self.to_dict().items() if k != "embedding"}
                ),
            )
        else:
            # JSON format (human readable)
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VoiceEmbedding":
        """
        Load embedding from file.

        Args:
            path: Path to saved embedding (.npz or .json)

        Returns:
            Loaded VoiceEmbedding instance
        """
        path = Path(path)

        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            embedding = data["embedding"]
            metadata = json.loads(str(data["metadata"]))
            metadata["embedding"] = embedding
            return cls.from_dict(metadata)
        else:
            with open(path) as f:
                return cls.from_dict(json.load(f))


class VoiceExtractor:
    """
    Extract voice embeddings from audio samples.

    Supports multiple backends for embedding extraction:
    - resemblyzer: d-vector based (default)
    - speechbrain: ECAPA-TDNN
    - wespeaker: ResNet-based
    """

    def __init__(
        self,
        backend: ExtractorBackend = ExtractorBackend.MOCK,
        device: str = "cpu",
        model_path: Optional[Path] = None,
    ):
        """
        Initialize the voice extractor.

        Args:
            backend: Which extraction backend to use
            device: Device for inference ("cpu", "cuda", "cuda:0")
            model_path: Optional custom model path
        """
        self.backend = backend
        self.device = device
        self.model_path = model_path
        self._model = None
        self._loaded = False

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for current backend."""
        dims = {
            ExtractorBackend.RESEMBLYZER: 256,
            ExtractorBackend.SPEECHBRAIN: 192,
            ExtractorBackend.WESPEAKER: 256,
            ExtractorBackend.MOCK: 256,
        }
        return dims.get(self.backend, 256)

    def load(self) -> None:
        """Load the extraction model."""
        if self._loaded:
            return

        if self.backend == ExtractorBackend.MOCK:
            self._loaded = True
            return

        if self.backend == ExtractorBackend.RESEMBLYZER:
            try:
                from resemblyzer import VoiceEncoder

                self._model = VoiceEncoder(device=self.device)
                self._loaded = True
            except ImportError:
                raise ImportError(
                    "resemblyzer not installed. Install with: pip install resemblyzer"
                )

        elif self.backend == ExtractorBackend.SPEECHBRAIN:
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device},
                )
                self._loaded = True
            except ImportError:
                raise ImportError(
                    "speechbrain not installed. Install with: pip install speechbrain"
                )

        elif self.backend == ExtractorBackend.WESPEAKER:
            try:
                import wespeaker

                self._model = wespeaker.load_model("english")
                self._loaded = True
            except ImportError:
                raise ImportError(
                    "wespeaker not installed. Install with: pip install wespeaker"
                )

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._loaded = False

    def extract(
        self,
        audio: Union[Path, str, np.ndarray],
        sample_rate: int = 16000,
    ) -> VoiceEmbedding:
        """
        Extract voice embedding from audio.

        Args:
            audio: Audio file path or numpy array of samples
            sample_rate: Sample rate if audio is numpy array

        Returns:
            VoiceEmbedding with extracted embedding
        """
        self.load()
        start_time = time.time()

        # Load audio if path provided
        source_path = None
        if isinstance(audio, (str, Path)):
            source_path = str(audio)
            audio, sample_rate = self._load_audio(audio)

        # Normalize audio
        audio = self._preprocess_audio(audio, sample_rate)

        # Extract embedding based on backend
        if self.backend == ExtractorBackend.MOCK:
            embedding = self._mock_extract(audio)
        elif self.backend == ExtractorBackend.RESEMBLYZER:
            embedding = self._resemblyzer_extract(audio)
        elif self.backend == ExtractorBackend.SPEECHBRAIN:
            embedding = self._speechbrain_extract(audio, sample_rate)
        elif self.backend == ExtractorBackend.WESPEAKER:
            embedding = self._wespeaker_extract(audio, sample_rate)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        extraction_time = time.time() - start_time

        # Estimate audio quality
        quality_score, snr_db = self._estimate_quality(audio)

        return VoiceEmbedding(
            embedding=embedding,
            embedding_dim=len(embedding),
            source_path=source_path,
            source_duration_seconds=len(audio) / sample_rate,
            source_sample_rate=sample_rate,
            extractor_backend=self.backend.value,
            extraction_time=extraction_time,
            quality_score=quality_score,
            snr_db=snr_db,
        )

    def extract_from_segments(
        self,
        audio: Union[Path, str, np.ndarray],
        segment_seconds: float = 3.0,
        sample_rate: int = 16000,
    ) -> List[VoiceEmbedding]:
        """
        Extract multiple embeddings from segments of audio.

        Useful for longer audio files where speaker may vary.

        Args:
            audio: Audio file path or numpy array
            segment_seconds: Length of each segment
            sample_rate: Sample rate if audio is numpy array

        Returns:
            List of VoiceEmbeddings, one per segment
        """
        # Load audio if path
        source_path = None
        if isinstance(audio, (str, Path)):
            source_path = str(audio)
            audio, sample_rate = self._load_audio(audio)

        segment_samples = int(segment_seconds * sample_rate)
        embeddings = []

        for i in range(0, len(audio), segment_samples):
            segment = audio[i : i + segment_samples]
            if len(segment) < sample_rate:  # Skip segments shorter than 1 second
                continue

            emb = self.extract(segment, sample_rate)
            emb.source_path = source_path
            embeddings.append(emb)

        return embeddings

    def average_embeddings(
        self, embeddings: List[VoiceEmbedding]
    ) -> VoiceEmbedding:
        """
        Average multiple embeddings into one.

        Args:
            embeddings: List of embeddings to average

        Returns:
            Averaged VoiceEmbedding
        """
        if not embeddings:
            raise ValueError("No embeddings to average")

        # Stack and average
        stacked = np.stack([e.embedding for e in embeddings])
        averaged = np.mean(stacked, axis=0)

        # Normalize to unit length
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        # Use metadata from first embedding as base
        base = embeddings[0]
        return VoiceEmbedding(
            embedding=averaged,
            embedding_dim=len(averaged),
            source_path=base.source_path,
            source_duration_seconds=sum(e.source_duration_seconds for e in embeddings),
            source_sample_rate=base.source_sample_rate,
            extractor_backend=base.extractor_backend,
            quality_score=np.mean([e.quality_score for e in embeddings]),
            snr_db=np.mean([e.snr_db for e in embeddings]),
        )

    def _load_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples + sample rate."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            import soundfile as sf

            audio, sr = sf.read(path)
        except ImportError:
            try:
                from scipy.io import wavfile

                sr, audio = wavfile.read(path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
            except Exception:
                raise ImportError(
                    "soundfile or scipy required for audio loading. "
                    "Install with: pip install soundfile"
                )

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return audio.astype(np.float32), sr

    def _preprocess_audio(
        self, audio: np.ndarray, sample_rate: int, target_sr: int = 16000
    ) -> np.ndarray:
        """Preprocess audio: resample, normalize."""
        # Resample if needed
        if sample_rate != target_sr:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            except ImportError:
                # Simple resampling fallback
                ratio = target_sr / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
                audio = audio[indices]

        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return audio

    def _estimate_quality(self, audio: np.ndarray) -> Tuple[float, float]:
        """Estimate audio quality and SNR."""
        # Simple energy-based quality estimation
        rms = np.sqrt(np.mean(audio**2))
        peak = np.abs(audio).max()

        # Estimate SNR (simplified)
        if rms > 0:
            snr_db = 20 * np.log10(peak / rms + 1e-8)
        else:
            snr_db = 0.0

        # Quality score based on RMS energy and clipping
        clipping_ratio = np.mean(np.abs(audio) > 0.99)
        silence_ratio = np.mean(np.abs(audio) < 0.01)

        quality = 1.0
        quality -= clipping_ratio * 0.5  # Penalize clipping
        quality -= max(0, silence_ratio - 0.3) * 0.5  # Penalize too much silence
        quality = max(0.1, min(1.0, quality))

        return quality, snr_db

    def _mock_extract(self, audio: np.ndarray) -> np.ndarray:
        """Generate mock embedding for testing."""
        # Create deterministic embedding based on audio characteristics
        np.random.seed(int(np.abs(audio[:100]).sum() * 1000) % (2**31))
        embedding = np.random.randn(256).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _resemblyzer_extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract using resemblyzer."""
        from resemblyzer import preprocess_wav

        # Preprocess for resemblyzer
        audio = preprocess_wav(audio)
        embedding = self._model.embed_utterance(audio)
        return embedding

    def _speechbrain_extract(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract using speechbrain."""
        import torch

        waveform = torch.tensor(audio).unsqueeze(0)
        embedding = self._model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def _wespeaker_extract(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract using wespeaker."""
        embedding = self._model.extract_embedding(audio)
        return np.array(embedding)


def extract_embedding(
    audio: Union[Path, str, np.ndarray],
    backend: Union[str, ExtractorBackend] = ExtractorBackend.MOCK,
    **kwargs,
) -> VoiceEmbedding:
    """
    Convenience function to extract embedding from audio.

    Args:
        audio: Audio file path or numpy array
        backend: Extraction backend to use
        **kwargs: Additional arguments for VoiceExtractor

    Returns:
        VoiceEmbedding instance
    """
    if isinstance(backend, str):
        backend = ExtractorBackend(backend)

    extractor = VoiceExtractor(backend=backend, **kwargs)
    return extractor.extract(audio)
